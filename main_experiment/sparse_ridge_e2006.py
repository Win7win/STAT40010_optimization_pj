#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sparse Ridge Project Runner (E2006-ready)
- Reads svmlight (supports .bz2) train/test
- Pads train/test to same feature dimension
- Restricted Greedy (Schur/rank-1 inverse update) on top-m candidates
- IHT / Projected GD for l0 ridge, with optional debias refit on support
- FISTA for Lasso / ElasticNet
  * either run once (no k dependency)
  * or tune lambda1 by sweep to match nnz≈k (on internal val split), then test

Intercept handling:
- For evaluation and refit: exact ridge-with-intercept by centering small support design (dense n×k), feasible for k<=few hundreds.

Dependencies: numpy, scipy, scikit-learn, tqdm
"""

import os
import sys
import json
import time
import math
import bz2
import argparse
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# ----------------------------
# Logging
# ----------------------------
def setup_logger(outdir: str, level: str = "INFO") -> logging.Logger:
    os.makedirs(outdir, exist_ok=True)
    logger = logging.getLogger("sparse_ridge")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(outdir, "run.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ----------------------------
# Data I/O
# ----------------------------
def load_data(path: str, fmt: str):
    if fmt == "svmlight":
        # supports .bz2
        if path.endswith(".bz2"):
            with bz2.open(path, "rb") as f:
                X, y = load_svmlight_file(f)
        else:
            X, y = load_svmlight_file(path)
        X = X.tocsr()
        y = y.astype(np.float64)
        return X, y
    elif fmt == "npz":
        d = np.load(path, allow_pickle=True)
        if "X_data" in d:
            X = sp.csr_matrix((d["X_data"], d["X_indices"], d["X_indptr"]), shape=tuple(d["X_shape"]))
            y = d["y"].astype(np.float64)
            return X, y
        raise ValueError("npz expects keys: X_data, X_indices, X_indptr, X_shape, y")
    else:
        raise ValueError(f"Unknown format: {fmt}")


def _pad_to_p(X: sp.csr_matrix, p: int) -> sp.csr_matrix:
    if X.shape[1] == p:
        return X
    if X.shape[1] > p:
        return X[:, :p]
    extra = sp.csr_matrix((X.shape[0], p - X.shape[1]), dtype=X.dtype)
    return sp.hstack([X, extra], format="csr")


def maybe_train_test(train_path: Optional[str], test_path: Optional[str], data_path: Optional[str],
                    fmt: str, test_size: float, seed: int, logger: logging.Logger):
    if train_path and test_path:
        Xtr, ytr = load_data(train_path, fmt)
        Xte, yte = load_data(test_path, fmt)

        p = max(Xtr.shape[1], Xte.shape[1])
        Xtr = _pad_to_p(Xtr, p)
        Xte = _pad_to_p(Xte, p)

        logger.info(f"Loaded train: X={Xtr.shape}, nnz={Xtr.nnz}; test: X={Xte.shape}, nnz={Xte.nnz}")
        return Xtr, ytr, Xte, yte

    if not data_path:
        raise ValueError("Need either (--train & --test) or --data")

    X, y = load_data(data_path, fmt)
    logger.info(f"Loaded data: X={X.shape}, nnz={X.nnz}")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed)
    logger.info(f"Split: train={Xtr.shape}, test={Xte.shape}")
    return Xtr.tocsr(), ytr, Xte.tocsr(), yte


# ----------------------------
# Preprocess utilities
# ----------------------------
def col_l2_norms_sq(X: sp.csr_matrix) -> np.ndarray:
    return np.asarray(X.power(2).sum(axis=0)).ravel()


def apply_colnorm(Xtr: sp.csr_matrix, Xte: sp.csr_matrix, mode: str, eps: float = 1e-12):
    if mode == "none":
        return Xtr, Xte, None

    if mode == "l2":
        ntr = Xtr.shape[0]
        norms = np.sqrt(col_l2_norms_sq(Xtr) / max(ntr, 1))
        scale = 1.0 / np.maximum(norms, eps)
    elif mode == "maxabs":
        absX = abs(Xtr)
        m = np.asarray(absX.max(axis=0)).ravel()
        scale = 1.0 / np.maximum(m, eps)
    else:
        raise ValueError("colnorm must be one of: none, l2, maxabs")

    D = sp.diags(scale, offsets=0, format="csr")
    return (Xtr @ D).tocsr(), (Xte @ D).tocsr(), scale


def predict_with_intercept(X: sp.csr_matrix, beta: np.ndarray, b: float) -> np.ndarray:
    return (X @ beta) + b


# ----------------------------
# Exact ridge refit with intercept on a support (dense n×k; k small)
# ----------------------------
def ridge_refit_with_intercept_on_support(
    X: sp.csr_matrix,
    y: np.ndarray,
    S: np.ndarray,
    lambda2: float
) -> Tuple[np.ndarray, float]:
    """
    Solve ridge with intercept restricted to support S:
      min_{beta_S,b} 0.5/n ||y - X_S beta_S - b||^2 + 0.5*lambda2 ||beta_S||^2

    Exact solution via centering X_S (dense) and y:
      beta_S = (Xc^T Xc / n + lambda2 I)^{-1} Xc^T yc / n
      b = y_mean - x_mean^T beta_S
    """
    n, p = X.shape
    beta = np.zeros(p, dtype=np.float64)

    if S.size == 0:
        return beta, float(y.mean())

    XS = X[:, S].toarray().astype(np.float64)  # n×k, dense small
    y_mean = float(y.mean())
    x_mean = XS.mean(axis=0)                   # k
    yc = y - y_mean
    Xc = XS - x_mean

    # A = Xc^T Xc / n + lambda2 I
    A = (Xc.T @ Xc) / max(n, 1) + lambda2 * np.eye(S.size, dtype=np.float64)
    rhs = (Xc.T @ yc) / max(n, 1)

    beta_S = np.linalg.solve(A, rhs)
    b = y_mean - float(x_mean @ beta_S)

    beta[S] = beta_S
    return beta, b


# ----------------------------
# Power iteration for spectral norm
# ----------------------------
def power_iteration_spectral_norm_sq(X: sp.csr_matrix, n_iter: int = 50, seed: int = 0,
                                     logger: Optional[logging.Logger] = None) -> float:
    rng = np.random.default_rng(seed)
    p = X.shape[1]
    v = rng.normal(size=p).astype(np.float64)
    v /= (np.linalg.norm(v) + 1e-12)

    for _ in range(n_iter):
        Xv = X @ v
        w = X.T @ Xv
        w = np.asarray(w).ravel()
        nw = np.linalg.norm(w) + 1e-12
        v = w / nw

    Xv = X @ v
    w = X.T @ Xv
    w = np.asarray(w).ravel()
    lam = float(v @ w)
    lam = max(lam, 1e-12)
    if logger:
        logger.info(f"Estimated ||X||_2^2 ≈ {lam:.6e}")
    return lam


# ----------------------------
# Restricted Greedy (Schur / rank-1 inverse update)
# ----------------------------
@dataclass
class GreedyResult:
    support: List[int]
    beta: np.ndarray
    intercept: float


def restricted_greedy_schur(
    X: sp.csr_matrix,
    y: np.ndarray,
    k: int,
    m_cand: int,
    lambda2: float,
    seed: int,
    logger: logging.Logger,
    show_tqdm: bool = True,
    refit: bool = True
) -> GreedyResult:
    """
    Greedy selection using gain Δ_j = γ_j^2 / α_j with Schur complement updates.
    Selection uses y0 = y - mean(y) for screening and internal math; final refit is exact ridge+intercept on support.
    """
    n, p = X.shape
    y0 = y - y.mean()

    t0 = time.time()
    corr = X.T @ y0
    corr = np.asarray(corr).ravel()
    abs_corr = np.abs(corr)

    m = min(m_cand, p)
    cand_idx = np.argpartition(-abs_corr, kth=m - 1)[:m]
    cand_idx = cand_idx[np.argsort(-abs_corr[cand_idx])]

    logger.info(f"[Greedy] k={k}, m_cand={m}, lambda2={lambda2:g}. Screening done in {time.time()-t0:.2f}s")

    Xcand = X[:, cand_idx]  # n×m
    col_norm2 = np.asarray(Xcand.power(2).sum(axis=0)).ravel()
    a = col_norm2 / n + lambda2
    c = np.asarray(Xcand.T @ y0).ravel() / n

    support_cand_pos: List[int] = []
    support_feat: List[int] = []
    Ainv = np.zeros((0, 0), dtype=np.float64)
    g = np.zeros((0,), dtype=np.float64)

    active = np.ones(m, dtype=bool)

    pbar = tqdm(range(k), disable=not show_tqdm, desc=f"Greedy(k={k})", ncols=110)
    for _ in pbar:
        active_pos = np.where(active)[0]
        if active_pos.size == 0:
            logger.warning("[Greedy] No active candidates left.")
            break

        if len(support_cand_pos) == 0:
            Delta = (c[active_pos] ** 2) / np.maximum(a[active_pos], 1e-12)
            best_local = active_pos[int(np.argmax(Delta))]
            best_delta = float(np.max(Delta))
            alpha_best = float(a[best_local])
            gamma_best = float(c[best_local])
            d_best = None
        else:
            Spos = np.array(support_cand_pos, dtype=int)
            XS = X[:, cand_idx[Spos]]
            Xact = X[:, cand_idx[active_pos]]

            M = (XS.T @ Xact).toarray() / n  # s×ma (dense)
            v = Ainv @ g
            u = Ainv @ M
            dAinv_d = np.sum(M * u, axis=0)

            alpha = np.maximum(a[active_pos] - dAinv_d, 1e-12)
            gamma = c[active_pos] - (M.T @ v)

            Delta = (gamma ** 2) / alpha
            j_star = int(np.argmax(Delta))
            best_local = int(active_pos[j_star])
            best_delta = float(Delta[j_star])
            alpha_best = float(alpha[j_star])
            gamma_best = float(gamma[j_star])
            d_best = M[:, j_star]

        feat_id = int(cand_idx[best_local])
        support_feat.append(feat_id)
        support_cand_pos.append(best_local)
        active[best_local] = False

        # Update inverse
        if Ainv.shape[0] == 0:
            Ainv = np.array([[1.0 / alpha_best]], dtype=np.float64)
            g = np.array([c[best_local]], dtype=np.float64)
        else:
            d = d_best.reshape(-1, 1)
            Ainv_d = Ainv @ d
            top_left = Ainv + (Ainv_d @ Ainv_d.T) / alpha_best
            top_right = -Ainv_d / alpha_best
            Ainv = np.block([
                [top_left, top_right],
                [top_right.T, np.array([[1.0 / alpha_best]], dtype=np.float64)],
            ])
            g = np.concatenate([g, [c[best_local]]], axis=0)

        pbar.set_postfix({"Δ": f"{best_delta:.2e}", "s": len(support_feat)})

    s = len(support_feat)
    if not refit:
        beta = np.zeros(p, dtype=np.float64)
        if s > 0:
            beta_S = Ainv @ g
            beta[np.array(support_feat, dtype=int)] = beta_S
        b = float(y.mean() - (X @ beta).mean())
    else:
        S = np.array(support_feat, dtype=int)
        beta, b = ridge_refit_with_intercept_on_support(X, y, S, lambda2)

    logger.info(f"[Greedy] done. |S|={s}, time={time.time()-t0:.2f}s")
    return GreedyResult(support=support_feat, beta=beta, intercept=b)


# ----------------------------
# IHT / Projected GD for l0+r2
# ----------------------------
@dataclass
class IHTResult:
    support: List[int]
    beta: np.ndarray
    intercept: float
    n_iter: int


def hard_threshold_topk(v: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.zeros_like(v)
    if k >= v.size:
        return v.copy()
    idx = np.argpartition(-np.abs(v), kth=k - 1)[:k]
    out = np.zeros_like(v)
    out[idx] = v[idx]
    return out


def iht_projected_gd(
    X: sp.csr_matrix,
    y: np.ndarray,
    k: int,
    lambda2: float,
    max_iter: int,
    tol: float,
    step: str,
    seed: int,
    logger: logging.Logger,
    show_tqdm: bool = True,
    debias: bool = True,
    stop_patience: int = 5,
    min_iter: int = 10
) -> IHTResult:
    """
    IHT: beta <- H_k(beta - eta * grad), with optional debias refit (ridge+intercept) on support each iter.
    step:
      - "auto": eta = 1 / (||X||_2^2/n + lambda2) (safe-ish)
      - "bt": backtracking (slower)
      - "niht": support-adaptive step size on the current candidate support (simple NIHT-style)
    """
    n, p = X.shape
    beta = np.zeros(p, dtype=np.float64)
    b = float(y.mean())

    Lx = power_iteration_spectral_norm_sq(X, n_iter=40, seed=seed, logger=logger) / max(n, 1)
    L = Lx + lambda2
    eta0 = 1.0 / max(L, 1e-12)
    logger.info(f"[IHT] k={k}, lambda2={lambda2:g}, L≈{L:.3e}, eta≈{eta0:.3e}, debias={debias}")

    def f_val(beta_vec: np.ndarray, b0: float) -> float:
        r = y - (X @ beta_vec) - b0
        return 0.5 * float((r @ r) / n) + 0.5 * lambda2 * float(beta_vec @ beta_vec)

    prev = f_val(beta, b)
    pbar = tqdm(range(max_iter), disable=not show_tqdm, desc=f"IHT(k={k})", ncols=110)
    it_used = 0
    prev_support: Optional[Tuple[int, ...]] = None
    stable_count = 0

    for it in pbar:
        it_used = it + 1

        # current intercept optimal for current beta (for gradient)
        b = float(y.mean() - (X @ beta).mean())
        r = (X @ beta) + b - y
        grad = np.asarray(X.T @ r).ravel() / n + lambda2 * beta

        if step == "auto":
            eta = eta0
            beta_tmp = hard_threshold_topk(beta - eta * grad, k)
        elif step == "niht":
            # Simple NIHT-style: pick a provisional support, then compute a support-adaptive step
            beta_prov = hard_threshold_topk(beta - eta0 * grad, k)
            Sprov = np.where(beta_prov != 0)[0].astype(int)
            if Sprov.size > 0:
                gS = grad[Sprov]
                # denom = ||X_S gS||^2 / n + lambda2 ||gS||^2
                Xg = (X[:, Sprov] @ gS)
                denom = float((Xg @ Xg) / n + lambda2 * (gS @ gS))
                eta = float((gS @ gS) / max(denom, 1e-12))
            else:
                eta = eta0
            beta_tmp = hard_threshold_topk(beta - eta * grad, k)
        elif step == "bt":
            eta = eta0
            f0 = prev
            beta_tmp = None
            for _ in range(20):
                cand = hard_threshold_topk(beta - eta * grad, k)
                # evaluate with its optimal intercept
                b_cand = float(y.mean() - (X @ cand).mean())
                f1 = f_val(cand, b_cand)
                if f1 <= f0:
                    beta_tmp = cand
                    break
                eta *= 0.5
            if beta_tmp is None:
                beta_tmp = hard_threshold_topk(beta - eta * grad, k)
        else:
            raise ValueError("step must be auto, niht or bt")

        # debias: refit ridge+intercept on support for better quality
        if debias:
            S = np.where(beta_tmp != 0)[0].astype(int)
            beta_new, b_new = ridge_refit_with_intercept_on_support(X, y, S, lambda2)
        else:
            beta_new = beta_tmp
            b_new = float(y.mean() - (X @ beta_new).mean())

        # support-stability based stopping (debias often reaches a fixed point quickly)
        cur_support = tuple(np.where(beta_new != 0)[0].tolist())
        if prev_support is not None and cur_support == prev_support:
            stable_count += 1
        else:
            stable_count = 0
        prev_support = cur_support

        cur = f_val(beta_new, b_new)
        rel = abs(prev - cur) / max(abs(prev), 1e-12)
        pbar.set_postfix({
            "f": f"{cur:.3e}",
            "rel": f"{rel:.2e}",
            "nnz": int(np.count_nonzero(beta_new)),
            "stb": stable_count,
        })

        beta, b = beta_new, b_new
        # Only stop after a minimum number of iterations and the support has been stable for a few rounds
        if it_used >= min_iter and stable_count >= stop_patience:
            prev = cur
            break
        prev = cur

    supp = np.where(beta != 0)[0].tolist()
    logger.info(f"[IHT] done. nnz={len(supp)}, iters={it_used}")
    return IHTResult(support=supp, beta=beta, intercept=b, n_iter=it_used)


# ----------------------------
# FISTA for Lasso / Elastic Net
# ----------------------------
@dataclass
class FISTAResult:
    beta: np.ndarray
    intercept: float
    n_iter: int


def soft_threshold(x: np.ndarray, thr: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - thr, 0.0)


def fista_l1(
    X: sp.csr_matrix,
    y: np.ndarray,
    lambda1: float,
    lambda2: float,
    max_iter: int,
    tol: float,
    seed: int,
    logger: logging.Logger,
    show_tqdm: bool = True
) -> FISTAResult:
    """
    Minimize: 0.5/n||y - Xb - b0||^2 + lambda1||b||_1 + 0.5*lambda2||b||^2
    """
    n, p = X.shape
    beta = np.zeros(p, dtype=np.float64)
    z = beta.copy()
    t = 1.0

    Lx = power_iteration_spectral_norm_sq(X, n_iter=40, seed=seed, logger=None) / max(n, 1)
    L = Lx + lambda2
    step = 1.0 / max(L, 1e-12)

    def obj(beta_vec: np.ndarray) -> float:
        # exact intercept for given beta
        b0 = float(y.mean() - (X @ beta_vec).mean())
        r = y - (X @ beta_vec) - b0
        return 0.5 * float((r @ r) / n) + lambda1 * float(np.sum(np.abs(beta_vec))) + 0.5 * lambda2 * float(beta_vec @ beta_vec)

    prev = obj(beta)
    pbar = tqdm(range(max_iter), disable=not show_tqdm, desc=f"FISTA(l1={lambda1:g})", ncols=110)
    it_used = 0

    for it in pbar:
        it_used = it + 1
        b0 = float(y.mean() - (X @ z).mean())
        r = (X @ z) + b0 - y
        grad = np.asarray(X.T @ r).ravel() / n + lambda2 * z

        beta_new = soft_threshold(z - step * grad, lambda1 * step)

        t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        z = beta_new + ((t - 1.0) / t_new) * (beta_new - beta)
        t = t_new

        cur = obj(beta_new)
        rel = abs(prev - cur) / max(abs(prev), 1e-12)
        pbar.set_postfix({"obj": f"{cur:.3e}", "rel": f"{rel:.2e}", "nnz": int(np.count_nonzero(beta_new))})

        beta = beta_new
        if rel < tol:
            prev = cur
            break
        prev = cur

    b = float(y.mean() - (X @ beta).mean())
    return FISTAResult(beta=beta, intercept=b, n_iter=it_used)


def tune_lambda1_match_k(
    Xtr: sp.csr_matrix, ytr: np.ndarray,
    lambda2: float,
    k_target: int,
    lambda1_grid: np.ndarray,
    val_size: float,
    seed: int,
    tune_iter: int,
    tol: float,
    logger: logging.Logger
) -> Tuple[float, Dict[str, float]]:
    """
    Split train into fit/val. For each lambda1, run short FISTA on fit, select lambda1 that:
      1) minimizes |nnz - k_target|
      2) then minimizes val MSE
    Returns best lambda1 and diagnostic dict.
    """
    Xfit, Xval, yfit, yval = train_test_split(Xtr, ytr, test_size=val_size, random_state=seed)

    best = None
    best_diag = None

    logger.info(f"[Tune] target nnz≈{k_target}, grid={len(lambda1_grid)} lambdas, val_size={val_size}, tune_iter={tune_iter}")

    for lam1 in tqdm(lambda1_grid, desc=f"Tune(k={k_target})", ncols=110):
        fr = fista_l1(Xfit, yfit, lambda1=float(lam1), lambda2=lambda2,
                      max_iter=tune_iter, tol=tol, seed=seed, logger=logger, show_tqdm=False)
        nnz = int(np.count_nonzero(fr.beta))
        yhat_val = predict_with_intercept(Xval, fr.beta, fr.intercept)
        mse_val = float(mean_squared_error(yval, yhat_val))

        dist = abs(nnz - k_target)
        cand = (dist, mse_val, float(lam1), nnz)
        if best is None or cand < best:
            best = cand
            best_diag = {"dist": dist, "mse_val": mse_val, "lambda1": float(lam1), "nnz": nnz}

    assert best_diag is not None
    logger.info(f"[Tune] best lambda1={best_diag['lambda1']:.3e}, nnz={best_diag['nnz']}, dist={best_diag['dist']}, mse_val={best_diag['mse_val']:.4g}")
    return float(best_diag["lambda1"]), best_diag


# ----------------------------
# Eval + runner
# ----------------------------
def evaluate(Xtr, ytr, Xte, yte, beta, b) -> Dict[str, float]:
    yhat_tr = predict_with_intercept(Xtr, beta, b)
    yhat_te = predict_with_intercept(Xte, beta, b)

    out = {
        "mse_tr": float(mean_squared_error(ytr, yhat_tr)),
        "r2_tr": float(r2_score(ytr, yhat_tr)),
        "mse_te": float(mean_squared_error(yte, yhat_te)),
        "r2_te": float(r2_score(yte, yhat_te)),
        "nnz": int(np.count_nonzero(beta)),
    }

    base = np.full_like(yte, fill_value=float(ytr.mean()))
    out["mse_te_const"] = float(mean_squared_error(yte, base))
    return out


def save_jsonl(path: str, rows: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None, help="single dataset path, random split")
    ap.add_argument("--train", type=str, default=None, help="train path (.bz2 ok)")
    ap.add_argument("--test", type=str, default=None, help="test path (.bz2 ok)")
    ap.add_argument("--format", type=str, default="svmlight", choices=["svmlight", "npz"])
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--methods", nargs="+", default=["greedy", "iht", "fista"],
                    choices=["greedy", "iht", "fista"])
    ap.add_argument("--k-list", nargs="+", type=int, default=[10, 20, 50])
    ap.add_argument("--m-cand", type=int, default=5000)

    ap.add_argument("--lambda2", type=float, default=1e-2)
    ap.add_argument("--lambda1", type=float, default=5e-4)

    ap.add_argument("--iht-max-iter", type=int, default=200)
    ap.add_argument("--iht-step", type=str, default="auto", choices=["auto", "niht", "bt"])
    ap.add_argument("--iht-debias", action="store_true", help="enable ridge+intercept refit on support each iter")
    ap.add_argument("--no-iht-debias", dest="iht_debias", action="store_false")
    ap.set_defaults(iht_debias=True)
    ap.add_argument("--iht-stop-patience", type=int, default=5, help="stop if support unchanged for this many consecutive iterations (after min-iter)")
    ap.add_argument("--iht-min-iter", type=int, default=10, help="minimum iterations before allowing early stop")
    

    ap.add_argument("--fista-max-iter", type=int, default=300)
    ap.add_argument("--tol", type=float, default=1e-5)

    ap.add_argument("--colnorm", type=str, default="none", choices=["none", "l2", "maxabs"])

    # FISTA tuning to match nnz≈k
    ap.add_argument("--fista-match-k", action="store_true",
                    help="tune lambda1 by sweep to match nnz≈k for each k (uses internal val split)")
    ap.add_argument("--lambda1-min", type=float, default=1e-6)
    ap.add_argument("--lambda1-max", type=float, default=1.0)
    ap.add_argument("--lambda1-grid-n", type=int, default=20)
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--tune-iter", type=int, default=0, help="iterations used during lambda1 tuning; 0 means use --fista-max-iter")

    ap.add_argument("--outdir", type=str, default="out_sparse_ridge")
    ap.add_argument("--loglevel", type=str, default="INFO")
    

    args = ap.parse_args()

    logger = setup_logger(args.outdir, args.loglevel)
    logger.info("==== Sparse Ridge Project Runner ====")
    logger.info(f"Args: {vars(args)}")

    Xtr, ytr, Xte, yte = maybe_train_test(args.train, args.test, args.data, args.format, args.test_size, args.seed, logger)

    # optional column normalization (recommended for tfidf)
    Xtr, Xte, scale = apply_colnorm(Xtr, Xte, args.colnorm)
    if scale is not None:
        logger.info(f"Applied colnorm={args.colnorm}")

    results: List[Dict] = []

    # ---- Greedy / IHT are k-dependent
    for k in args.k_list:
        if "greedy" in args.methods:
            t0 = time.time()
            gr = restricted_greedy_schur(
                Xtr, ytr, k=k, m_cand=args.m_cand, lambda2=args.lambda2,
                seed=args.seed, logger=logger, show_tqdm=True, refit=True
            )
            met = evaluate(Xtr, ytr, Xte, yte, gr.beta, gr.intercept)
            met.update({"method": "greedy_schur_refit", "k": k, "time_sec": time.time() - t0,
                        "lambda2": args.lambda2, "m_cand": args.m_cand, "colnorm": args.colnorm})
            logger.info(f"[Result] {met}")
            results.append(met)

        if "iht" in args.methods:
            t0 = time.time()
            ir = iht_projected_gd(
                Xtr, ytr, k=k, lambda2=args.lambda2,
                max_iter=args.iht_max_iter, tol=args.tol, step=args.iht_step,
                seed=args.seed, logger=logger, show_tqdm=True, debias=args.iht_debias,
                stop_patience=args.iht_stop_patience, min_iter=args.iht_min_iter
            )
            met = evaluate(Xtr, ytr, Xte, yte, ir.beta, ir.intercept)
            met.update({"method": f"iht_{args.iht_step}" + ("_debias" if args.iht_debias else ""),
                        "k": k, "time_sec": time.time() - t0,
                        "lambda2": args.lambda2, "colnorm": args.colnorm, "iters": ir.n_iter})
            logger.info(f"[Result] {met}")
            results.append(met)

        # ---- FISTA tuning per k (match nnz≈k)
        if "fista" in args.methods and args.fista_match_k:
            lambda1_grid = np.logspace(math.log10(args.lambda1_min), math.log10(args.lambda1_max), args.lambda1_grid_n)
            tune_iter = args.tune_iter if args.tune_iter and args.tune_iter > 0 else args.fista_max_iter
            best_l1, diag = tune_lambda1_match_k(
                Xtr, ytr, lambda2=args.lambda2, k_target=k,
                lambda1_grid=lambda1_grid, val_size=args.val_size,
                seed=args.seed, tune_iter=tune_iter, tol=args.tol,
                logger=logger
            )
            t0 = time.time()
            fr = fista_l1(
                Xtr, ytr, lambda1=best_l1, lambda2=args.lambda2,
                max_iter=args.fista_max_iter, tol=args.tol,
                seed=args.seed, logger=logger, show_tqdm=True
            )
            met = evaluate(Xtr, ytr, Xte, yte, fr.beta, fr.intercept)
            met.update({"method": "fista_l1_matchk", "k": k, "time_sec": time.time() - t0,
                        "lambda1": best_l1, "lambda2": args.lambda2, "colnorm": args.colnorm,
                        "iters": fr.n_iter, "tune_dist": diag["dist"], "tune_mse_val": diag["mse_val"], "tune_nnz": diag["nnz"]})
            logger.info(f"[Result] {met}")
            results.append(met)

    # ---- FISTA plain run ONCE (no k-dependency)
    if "fista" in args.methods and (not args.fista_match_k):
        t0 = time.time()
        fr = fista_l1(
            Xtr, ytr, lambda1=args.lambda1, lambda2=args.lambda2,
            max_iter=args.fista_max_iter, tol=args.tol,
            seed=args.seed, logger=logger, show_tqdm=True
        )
        met = evaluate(Xtr, ytr, Xte, yte, fr.beta, fr.intercept)
        met.update({"method": "fista_l1", "k": None, "time_sec": time.time() - t0,
                    "lambda1": args.lambda1, "lambda2": args.lambda2, "colnorm": args.colnorm, "iters": fr.n_iter})
        logger.info(f"[Result] {met}")
        results.append(met)

    out_jsonl = os.path.join(args.outdir, "results.jsonl")
    save_jsonl(out_jsonl, results)
    logger.info(f"Saved: {out_jsonl}")

    out_csv = os.path.join(args.outdir, "results.csv")
    if results:
        keys = sorted({k for r in results for k in r.keys()})
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for r in results:
                f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
        logger.info(f"Saved: {out_csv}")

    logger.info("==== Done ====")


if __name__ == "__main__":
    main()