"""Precision-recall of |U| and |C| (sample B, 20 h) on the 3-gene motifs, 100 replicates per motif (figure_4 1k_per_motif_v2, Figure-2 design 1,500 twins + 3,000 singletons).
Positive = true edge (8 of the 9 pairs per replicate); the only non-edge is Fan_out 2-3 (shared driver 1). Reports raw AUPRC (maximum 1, chance = base rate) and AUPRC / base rate (1.0 = chance).
usage: python pr_curve_U_C_motifs.py DIR (DIR = results_motif_design1500_100reps)"""
import sys, glob, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
D = sys.argv[1]; r = pd.concat([pd.read_csv(f) for f in glob.glob(f"{D}/*_rep_*.csv")], ignore_index=True)
r["pos"] = ~r.truth.str.startswith("none"); r.to_csv(f"{D}/motif_pairs_100reps.csv", index=False)
print(len(r), "pair rows;", r.groupby("motif").rep.nunique().to_dict(), "replicates per motif; positives", int(r.pos.sum()), "negatives", int((~r.pos).sum()))
print(r.groupby(["motif", "pair", "truth"])[["S", "C", "U", "rho", "z_rho", "z_C", "z_U"]].median().round(3).to_string())
subsets = {"all 3 motifs (8 edges + shared-driver pair)": r.index == r.index, "Fan_out only (1-2, 1-3 vs 2-3)": r.motif == "Fan_out"}
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1a1a19", "#5b5a55", "#e3e2dd"
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True); rows = []
for ax, (nm, m) in zip(axs, subsets.items()):
    d = r[m]; base = d.pos.mean()
    ax.axhline(base, color=GRID, lw=1.2); ax.text(0.01, base - 0.045, f"chance = base rate = {base:.2f}", color=MUTED, fontsize=8)
    for sc, col in (("U", BLUE), ("C", ORANGE)):
        s = d[sc].abs().to_numpy(); p, rc, _ = precision_recall_curve(d.pos, s); ap = average_precision_score(d.pos, s)
        ax.plot(rc, p, color=col, lw=2, label=f"|{sc}|  AUPRC = {ap:.2f}")
        rows.append(dict(subset=nm, score=f"|{sc}|", n=len(d), n_neg=int((~d.pos).sum()), base_rate=base, AUPRC=ap, AUPRC_over_base=ap / base))
    ax.set_title(nm, fontsize=10.5, color=INK, loc="left"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.03); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9); ax.legend(loc="lower left", fontsize=8.5, frameon=False, labelcolor=INK); ax.set_xlabel("recall (true edges found)", color=INK)
axs[0].set_ylabel("precision", color=INK)
fig.suptitle("U vs C as an edge score on the 3-gene motifs (sample B, 20 h; 100 replicates per motif)", fontsize=11.5, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(f"{D}/pr_curve_U_C_motifs.png", dpi=150, facecolor="#fcfcfb")
t = pd.DataFrame(rows); t.to_csv(f"{D}/pr_curve_U_C_motifs_auprc.csv", index=False); print(t.round(3).to_string(index=False))
