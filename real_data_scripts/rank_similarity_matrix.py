#!/usr/bin/env python3
"""Similarity of the FULL rankings of one TF's listed targets across all methods (Spearman correlation of the ranks; Kendall tau also written).
Input: <TF>_all_methods_listed_pairs.csv from eomes_all_methods_table.py. Ranks are by |score| (high = best); TwinScore z_s is ranked by its signed value (high = shared/inherited).
Heatmap: sequential blue (0 = unrelated, 1 = identical; negative values clip to the lightest shade, numbers printed in the cells), rows ordered by average-linkage clustering.
usage: rank_similarity_matrix.py [TF=EOMES]"""
import sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr, kendalltau
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
TF = sys.argv[1] if len(sys.argv) > 1 else "EOMES"
D = "/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/hPSC_20260927/data/twinscore_pdf_candidates_yscher_endo"
E = pd.read_csv(f"{D}/{TF}_all_methods_listed_pairs.csv")
# U and S are left out (U ranks like |z_U|; S is folded into z_s): the TwinScore columns are |z_U| and z_s
cols = {"TwinScore |z_U|": "rank_TwinScore |z_U|", "TwinScore z_s": "rank_TwinScore z_s", "weighted ρ": "rank_weighted rho",
        "Pearson": "rank_Pearson", "|Spearman|": "rank_|Spearman|", "GRNBoost2": "rank_GRNBoost2", "GENIE3": "rank_GENIE3", "ppcor": "rank_ppcor", "ppcor (shrunk)": "rank_ppcor_shrunk"}
cols = {k: v for k, v in cols.items() if v in E.columns}; names = list(cols); R = E[list(cols.values())].to_numpy()
S = np.array([[spearmanr(R[:, i], R[:, j])[0] for j in range(len(names))] for i in range(len(names))])
K = np.array([[kendalltau(R[:, i], R[:, j])[0] for j in range(len(names))] for i in range(len(names))])
for nm, M in (("spearman", S), ("kendall", K)): pd.DataFrame(M, index=names, columns=names).round(3).to_csv(f"{D}/{TF}_rank_similarity_{nm}.csv")
order = leaves_list(linkage(squareform(np.clip(1 - S, 0, None), checks=False), "average")); n = [names[i] for i in order]; M = S[np.ix_(order, order)]
# --- heatmap (reference-palette sequential blue 100 -> 700; page/figure surface #fcfcfb, text ink dark gray)
cmap = LinearSegmentedColormap.from_list("seqblue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
fig, ax = plt.subplots(figsize=(8.0, 6.8), facecolor="#fcfcfb"); ax.set_facecolor("#fcfcfb")
im = ax.imshow(np.clip(M, 0, 1), cmap=cmap, vmin=0, vmax=1)
for i in range(len(n)):
    for j in range(len(n)):
        v = M[i, j]; ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="#fcfcfb" if np.clip(v, 0, 1) > 0.55 else "#2b2b29")
ax.set_xticks(range(len(n))); ax.set_yticks(range(len(n))); ax.set_xticklabels(n, rotation=40, ha="right", fontsize=9, color="#2b2b29"); ax.set_yticklabels(n, fontsize=9, color="#2b2b29")
for s in ax.spines.values(): s.set_visible(False)
ax.set_xticks(np.arange(-.5, len(n), 1), minor=True); ax.set_yticks(np.arange(-.5, len(n), 1), minor=True); ax.grid(which="minor", color="#fcfcfb", linewidth=2); ax.tick_params(which="both", length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("Spearman correlation of the full rankings (negatives shown as 0)", color="#2b2b29", fontsize=9); cb.outline.set_visible(False); cb.ax.tick_params(colors="#2b2b29", labelsize=8)
ax.set_title(f"{TF} targets: how similar are the methods' rankings?  (n = {len(E)} listed targets)", loc="left", fontsize=11, color="#2b2b29")
fig.tight_layout(); import os; os.makedirs(f"{D}/plots", exist_ok=True); fig.savefig(f"{D}/plots/{TF}_method_rank_similarity.png", dpi=170, facecolor=fig.get_facecolor())
print(f"wrote {D}/plots/{TF}_method_rank_similarity.png and {TF}_rank_similarity_spearman.csv / _kendall.csv")
print(pd.DataFrame(M, index=n, columns=n).round(2).to_string())
