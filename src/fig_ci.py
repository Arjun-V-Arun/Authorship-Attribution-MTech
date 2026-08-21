import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd, ast

df = pd.read_csv("results/results_rigor.csv")
df["ci"] = df["ci95"].apply(ast.literal_eval)
df["lo"] = df["ci"].apply(lambda c: c[0])
df["hi"] = df["ci"].apply(lambda c: c[1])
df = df.sort_values("honest_macroF1")

xerr = [df["honest_macroF1"] - df["lo"], df["hi"] - df["honest_macroF1"]]
colors = ["#c44" if "func" in m or "word 1" in m else
          ("#2a7" if "MuRIL" in m else "#468") for m in df["method"]]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.errorbar(df["honest_macroF1"], df["method"], xerr=xerr, fmt="o",
            color="black", ecolor="gray", elinewidth=1.5, capsize=4, zorder=3)
ax.scatter(df["honest_macroF1"], df["method"], c=colors, s=90, zorder=4)
for m, v in zip(df["method"], df["honest_macroF1"]):
    ax.text(v, m, f"  {v:.3f}", va="center", ha="left", fontsize=8)
ax.axvline(0.5707, ls="--", color="gray", lw=1, zorder=1)
ax.set_xlabel("honest macro-F1 (leave-one-book-out, 95% CI)")
ax.set_xlim(0.30, 0.70)
ax.set_title("Leakage-free Hindi authorship: method comparison with 95% CIs")
fig.tight_layout()
fig.savefig("results/fig_ci.png", dpi=140)
print("wrote results/fig_ci.png")