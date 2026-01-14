#!/usr/bin/env bash
set -euo pipefail

# === 数据路径===
TRAIN="../data/e2006/E2006.train.bz2"
TEST="../data/e2006/E2006.test.bz2"

# === 输出目录 ===
OUTDIR="result_greedy_iht_fista"

mkdir -p "${OUTDIR}"

python sparse_ridge_e2006.py \
  --train "${TRAIN}" \
  --test  "${TEST}" \
  --colnorm l2 \
  --lambda2 0.01 \
  --k-list 10 20 50 \
  --methods greedy iht fista \
  --m-cand 5000 \
  --iht-step auto \
  --iht-min-iter 20 \
  --iht-stop-patience 5 \
  --iht-debias \
  --fista-match-k \
  --fista-max-iter 300 \
  --tol 1e-5 \
  --outdir "${OUTDIR}" \
  --seed 0

OUTDIR="result_greedy_niht"

mkdir -p "${OUTDIR}"

python sparse_ridge_e2006.py \
  --train "${TRAIN}" \
  --test  "${TEST}" \
  --colnorm l2 \
  --lambda2 0.01 \
  --k-list 10 20 50 \
  --methods greedy iht \
  --m-cand 5000 \
  --iht-step niht \
  --iht-min-iter 20 \
  --iht-stop-patience 5 \
  --iht-debias \
  --tol 1e-5 \
  --outdir "${OUTDIR}" \
  --seed 0