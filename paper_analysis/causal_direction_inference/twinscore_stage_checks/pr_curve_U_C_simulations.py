"""Precision-recall of |U| and |C| for detecting the edge: positives = A_to_B (single state) + A_to_B (two states), negatives = A_B (single) + A_B (two states).
Replicates ranked by |score| (100 per simulation type). Reports raw AUPRC (maximum 1, chance = base rate) and AUPRC / base rate (1.0 = chance). usage: python pr_curve_U_C_simulations.py DIR  (DIR = flowchart_clone_weighted output)"""
import sys, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
D = sys.argv[1]; r = pd.read_csv(f"{D}/flowchart_zscores_with_calls.csv")
r["pos"] = r.sim.str.startswith("A_to_B")
subsets = {"all four simulations": r.sim.notna(), "single state (A_to_B vs A_B)": r.sim.isin(["A_to_B", "A_B"]), "two states (A_to_B vs A_B)": r.sim.isin(["A_to_B two states", "A_B two states"])}
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1a1a19", "#5b5a55", "#e3e2dd"
fig, axs = plt.subplots(2, 3, figsize=(12.5, 7.4), sharey=True); rows = []
for i, tm in enumerate(("t2", "t1")):
    for j, (nm, m) in enumerate(subsets.items()):
        ax = axs[i, j]; d = r[m]; base = d.pos.mean()
        ax.axhline(base, color=GRID, lw=1.2, zorder=1); ax.text(0.01, base + 0.012, f"chance = base rate = {base:.2f}", color=MUTED, fontsize=8, va="bottom")
        for sc, col in (("U", BLUE), ("C", ORANGE)):
            s = d[f"{sc}_{tm}"].abs().to_numpy(); p, rc, _ = precision_recall_curve(d.pos, s); ap = average_precision_score(d.pos, s)
            ax.plot(rc, p, color=col, lw=2, label=f"|{sc}|  AUPRC = {ap:.2f}", zorder=3)
            rows.append(dict(time=tm, subset=nm, score=f"|{sc}|", n=len(d), base_rate=base, AUPRC=ap, AUPRC_over_base=ap / base))
        ax.set_title(f"{nm} - {tm}", fontsize=10.5, color=INK, loc="left"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.03)
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9); ax.legend(loc="lower left", fontsize=8.5, frameon=False, labelcolor=INK)
        if i == 1: ax.set_xlabel("recall (true edges found)", color=INK)
        if j == 0: ax.set_ylabel("precision", color=INK)
fig.suptitle("U vs C as an edge score: A_to_B replicates (edge) vs A_B replicates (no edge), 100 replicates per simulation", fontsize=11.5, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.96)); out = f"{D}/pr_curve_U_C.png"; fig.savefig(out, dpi=150, facecolor="#fcfcfb"); 
t = pd.DataFrame(rows); t.to_csv(f"{D}/pr_curve_U_C_auprc.csv", index=False)
print(t.pivot_table(index=["time", "subset"], columns="score", values="AUPRC_over_base").round(2).to_string()); print(out)
