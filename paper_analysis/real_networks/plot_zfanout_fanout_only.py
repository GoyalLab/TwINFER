"""z_fanout_cross for the Fan_out motif only, t1=1h vs t1=10h, showing the confound-signature
decay: at t1=1h the confounded non-edge (2,3|1) sits clearly above the two true edges; by
t1=10h the whole scale has collapsed and the separation is gone.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/real_networks'
df = pd.read_csv(f"{HERE}/zfanout_cross_per_edge_figure4.csv")
df = df[df.motif == "Fan_out"].copy()
df["pair"] = df["pair"].apply(eval)

LABEL = {("gene_1", "gene_2"): "1,2|3", ("gene_1", "gene_3"): "1,3|2", ("gene_2", "gene_3"): "2,3|1"}
ORDER = [("gene_1", "gene_2"), ("gene_1", "gene_3"), ("gene_2", "gene_3")]
GRN, GRY = "#2ca02c", "#9aa0a6"

fig, axes = plt.subplots(1, 2, figsize=(8, 5.5), sharey=False)
for ax, T1 in zip(axes, (1, 10)):
    sub = df[df.T1 == T1]
    for pos, pr in enumerate(ORDER):
        v = sub[sub.pair == pr]
        col = GRN if v.is_true.iloc[0] else GRY
        vals = v.zfan.to_numpy()
        ax.boxplot(vals, positions=[pos], widths=0.6, patch_artist=True, showfliers=False,
                   medianprops=dict(color=col, lw=2.2),
                   boxprops=dict(facecolor=col, alpha=0.25, edgecolor=col, lw=1.3),
                   whiskerprops=dict(color=col, lw=1.2), capprops=dict(color=col, lw=1.2))
        rng = np.random.default_rng(0)
        ax.scatter(pos + rng.uniform(-0.2, 0.2, len(vals)), vals, s=22, color=col,
                   alpha=0.65, edgecolor="none", zorder=3)
    ax.set_xticks(range(3)); ax.set_xticklabels([LABEL[p] for p in ORDER], fontsize=11)
    ax.axhline(0, color="0.85", lw=0.8)
    ax.set_title(f"t1={T1}h", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("z_fanout_cross(x,y) = max$_C$ min(z$_{cross}$(C,x), z$_{cross}$(C,y))")
handles = [Patch(facecolor=GRN, alpha=.5, label="true edge"),
           Patch(facecolor=GRY, alpha=.5, label="confounded non-edge (2,3)")]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=10,
           bbox_to_anchor=(0.5, -0.05))
fig.suptitle("Fan-out motif: z_fanout_cross per pair, t1=1h vs t1=10h\n"
             "(confound signature is clean and separated early, decays and overlaps by t1=10h)",
             fontsize=12)
fig.tight_layout(rect=[0, 0.04, 1, 0.92])
out = f"{HERE}/zfanout_cross_fanout_only.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
