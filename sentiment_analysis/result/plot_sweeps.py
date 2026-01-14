#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def read_results(outdir: str) -> pd.DataFrame:
    """Read results.jsonl (preferred) or results.csv from outdir."""
    jsonl_path = os.path.join(outdir, "results.jsonl")
    csv_path = os.path.join(outdir, "results.csv")

    if os.path.exists(jsonl_path):
        rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)

    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    raise FileNotFoundError(f"Cannot find results.jsonl or results.csv in: {outdir}")


def pick_method_row(df: pd.DataFrame, k: int, method_contains: str) -> pd.Series:
    """Pick one row where k matches and method contains substring."""
    d = df.copy()
    if "k" not in d.columns or "method" not in d.columns:
        raise ValueError("results file must contain columns: k, method")

    d = d[d["k"].fillna(-1).astype(int) == int(k)]
    d = d[d["method"].astype(str).str.contains(method_contains, regex=False)]
    if len(d) == 0:
        raise ValueError(f"No row found for k={k}, method_contains='{method_contains}'")
    return d.iloc[0]


def plot_lambda2_sweep(outdirs, k: int, out_png: str):
    recs = []
    for od in outdirs:
        df = read_results(od)

        # greedy
        r_g = pick_method_row(df, k=k, method_contains="greedy")
        # niht
        r_n = pick_method_row(df, k=k, method_contains="niht")

        lam2 = float(r_g.get("lambda2", r_n.get("lambda2")))
        recs.append({"lambda2": lam2, "method": "Greedy+refit", "mse_te": float(r_g["mse_te"]), "time_sec": float(r_g["time_sec"])})
        recs.append({"lambda2": lam2, "method": "NIHT+debias", "mse_te": float(r_n["mse_te"]), "time_sec": float(r_n["time_sec"])})

    data = pd.DataFrame(recs).sort_values("lambda2")

    fig, ax = plt.subplots()

    for method, g in data.groupby("method"):
        g = g.sort_values("lambda2")
        ax.plot(g["lambda2"], g["mse_te"], marker="o", label=f"{method} (MSE)")

    ax.set_xscale("log")
    ax.set_xlabel("lambda2 (log scale)")
    ax.set_ylabel("Test MSE")

    # 合并 legend
    h1, l1 = ax.get_legend_handles_labels()
    # ax.legend(h1 + h2, l1 + l2, loc="best")
    ax.legend(h1, l1, loc="best")

    # ax.set_title(f"Lambda2 sweep (k={k})")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    print(f"[Saved] {out_png}")


def plot_mcand_sweep(outdirs, k: int, out_png: str):
    recs = []
    for od in outdirs:
        df = read_results(od)
        r_g = pick_method_row(df, k=k, method_contains="greedy")

        mcand = int(r_g.get("m_cand"))
        recs.append({"m_cand": mcand, "mse_te": float(r_g["mse_te"]), "time_sec": float(r_g["time_sec"])})

    data = pd.DataFrame(recs).sort_values("m_cand")

    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    ax.plot(data["m_cand"], data["mse_te"], marker="o", label="Greedy+refit (MSE)")
    ax2.plot(data["m_cand"], data["time_sec"], marker="x", linestyle="--", label="Greedy+refit (time)")

    ax.set_xlabel("m_cand")
    ax.set_ylabel("Test MSE")
    ax2.set_ylabel("time_sec")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best")

    # ax.set_title(f"Candidate size sweep (k={k})")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    print(f"[Saved] {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=20)

    ap.add_argument("--lambda2-glob", type=str, default="out_lambda2_*",
                    help="glob pattern for lambda2 sweep folders")
    ap.add_argument("--mcand-glob", type=str, default="out_cand_*",
                    help="glob pattern for m_cand sweep folders")

    ap.add_argument("--out-lambda2", type=str, default="plot_lambda2_sweep.png")
    ap.add_argument("--out-mcand", type=str, default="plot_mcand_sweep.png")
    args = ap.parse_args()

    lam_dirs = sorted([d for d in glob.glob(args.lambda2_glob) if os.path.isdir(d)])
    mc_dirs = sorted([d for d in glob.glob(args.mcand_glob) if os.path.isdir(d)])

    if lam_dirs:
        plot_lambda2_sweep(lam_dirs, k=args.k, out_png=args.out_lambda2)
    else:
        print(f"[Skip] No lambda2 dirs matched: {args.lambda2_glob}")

    if mc_dirs:
        plot_mcand_sweep(mc_dirs, k=args.k, out_png=args.out_mcand)
    else:
        print(f"[Skip] No m_cand dirs matched: {args.mcand_glob}")


if __name__ == "__main__":
    main()