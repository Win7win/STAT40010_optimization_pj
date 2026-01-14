# STAT40010 最优化理论 期末PJ

本仓库为 STAT40010《最优化理论》期末 Project 代码与实验记录。课题为 **Sparse Ridge Regression（稀疏岭回归，ℓ0 约束 + ℓ2 正则）** 的算法复现与实证对比，复现核心参考文献为 Xie & Deng (2020, SIAM)。

本文在 **E2006-TFIDF** 高维稀疏回归数据集上，对比了：
- **Restricted Greedy + Schur / Sherman–Morrison 更新 + 支持集 ridge refit**（复现重点）
- **IHT / NIHT + debias(refit)**（投影梯度类非凸近似）
- **Elastic Net 的 FISTA（match-k 调参）**（凸松弛基线）

并给出主实验结果（随 k 变化的 Test MSE / 运行时间）以及两类敏感性分析（λ2 sweep、候选集规模 m_cand sweep）。

---

## 目录结构

data/
e2006/                       # E2006 数据（需自行下载/放置）

main_experiment/
sparse_ridge_e2006.py        # 主实验脚本（对比 k={10,20,50}）
start_main.sh                # 主实验一键运行脚本
result/
combined_results.jsonl     # 主实验汇总结果（jsonl）
draw_pic.py                # 画主实验图脚本
mse_vs_k.png               # 主实验：Test MSE vs k
time_vs_k.png              # 主实验：Time vs k

sentiment_analysis/
sparse_ridge_e2006.py        # 敏感性实验脚本（λ2 / m_cand sweep）
start_sentiment.sh           # 敏感性实验一键运行脚本
result/
plot_sweeps.py             # 画敏感性图脚本
plot_lambda2_sweep.png     # 敏感性：λ2 sweep (k=20)
plot_mcand_sweep.png       # 敏感性：m_cand sweep (k=20)

README.md

---

## 环境依赖

建议使用 Python 3.10+。

主要依赖（均为常见科学计算包）：
- numpy
- scipy
- scikit-learn
- matplotlib
- tqdm（可选）

安装：

```bash
pip install numpy scipy scikit-learn matplotlib tqdm
```

⸻

数据集：E2006-TFIDF

数据来源：LIBSVM regression datasets（E2006-tfidf）
- https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html

请下载 `E2006.train.bz2` 与 `E2006.test.bz2`，解压/放置到：
`data/e2006/`

⸻

运行方式

1) 主实验（k = 10/20/50 对比）

进入 main_experiment/：

bash start_main.sh

2) 敏感性实验（λ2 sweep / m_cand sweep）

进入 sentiment_analysis/：

bash start_sentiment.sh

⸻

实验设置简介
	•	数据集：E2006-TFIDF
	•	稀疏度：k ∈ {10, 20, 50}
	•	列归一化：colnorm = l2
	•	截距：实现中显式截距（用于预测与 refit，避免强制过原点）
	•	评测：Test MSE、Test R²、nnz、wall time；并记录 mean baseline（mse_te_const）

敏感性实验：
	•	固定 k = 20，测试不同 λ2（如 1e-3~1e1）
	•	固定 k = 20，测试不同候选集规模 m_cand（如 1k~20k）

