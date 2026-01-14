import json
import pandas as pd
import matplotlib.pyplot as plt

results_path = "combined_results.jsonl" 

rows = []
with open(results_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

df = pd.DataFrame(rows)

name_map = {
    "greedy_schur_refit": "Greedy+refit",
    "iht_auto_debias": "IHT(auto)+debias",
    "iht_niht_debias": "NIHT+debias",
    "fista_l1_matchk": "FISTA(match-k)",
    "fista_l1": "FISTA",
}
df["method_show"] = df["method"].map(lambda x: name_map.get(x, x))

df = df[df["k"].notna()].copy()
df["k"] = df["k"].astype(int)

# 3) 图1：Test MSE vs k
plt.figure()
for m, g in df.sort_values("k").groupby("method_show"):
    plt.plot(g["k"], g["mse_te"], marker="o", label=m)
plt.xlabel("k")
plt.ylabel("Test MSE")
# plt.title("Test MSE vs k")
plt.legend()
plt.tight_layout()
plt.savefig("mse_vs_k.png", dpi=200)

# 4) 图2：time_sec vs k
plt.figure()
for m, g in df.sort_values("k").groupby("method_show"):
    if "time_sec" in g.columns:
        plt.plot(g["k"], g["time_sec"], marker="o", label=m)
plt.xlabel("k")
plt.ylabel("Wall time (sec)")
# plt.title("Time vs k")
plt.legend()
plt.tight_layout()
plt.savefig("time_vs_k.png", dpi=200)

print("Saved: mse_vs_k.png, time_vs_k.png")