# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Per network: all 7 TwINFER z-scores on the x-axis, paired box+dots for
true directed edges (green) vs non-edges (grey).  Permutation z from the
no-filter ranked_edges, mean over replicates.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RN = f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/real_networks/twinfer_inference_nofilter'
TOPO = f'{TWINFER_PROJECT_ROOT}/input_data/real_world_networks'
NETS = {"GSD": "GSD.txt", "HSC": "HSC.txt", "VSC": "VSC.txt", "mCAD": "mCAD.txt",
        "EMT": "EMT.txt", "Pluripotent": "Pluripotent.txt",
        "B_cell_activation": "B_cell.txt", "Circadian_cycle": "circadian.txt"}
ZC = ["z_abs_rho_t1", "z_abs_rho_t2", "z_abs_rho_change", "z_het", "z_div", "z_d_het", "z_gamma"]
ZLAB = [r"$z_{|\rho|,t_1}$", r"$z_{|\rho|,t_2}$", r"$z_{|\Delta\rho|}$", r"$z_{\rm het}$",
        r"$z_{\rm div}$", r"$z_{d,\rm het}$", r"$z_\gamma$"]
GRN, GRY = "#2ca02c", "#9aa0a6"

rng = np.random.default_rng(0)
fig, axes = plt.subplots(4, 2, figsize=(17, 20))
for ax, (net, topo) in zip(axes.flat, NETS.items()):
    M = np.loadtxt(f"{TOPO}/{topo}", delimiter=",", dtype=int); n = M.shape[0]
    g = [f"gene_{i+1}" for i in range(n)]
    true = {(g[i], g[j]) for i in range(n) for j in range(n) if i != j and M[i, j] != 0}
    fr = []
    for jf in sorted(glob.glob(f"{RN}/{net}_rep_*_all_results.json")):
        d = json.load(open(jf))
        red = pd.DataFrame(d["ranked_edges"]["data"], columns=d["ranked_edges"]["columns"])
        fr.append(red[["gene_1", "gene_2"] + ZC])
    df = pd.concat(fr).groupby(["gene_1", "gene_2"], as_index=False)[ZC].mean()
    df["t"] = [(a, b) in true for a, b in zip(df.gene_1, df.gene_2)]
    nT = int(df.t.sum())

    for k, zc in enumerate(ZC):
        for j, (isT, col) in enumerate([(False, GRY), (True, GRN)]):
            v = df.loc[df.t == isT, zc].replace([np.inf, -np.inf], np.nan).dropna().values
            if not len(v):
                continue
            pos = k + (0.19 if isT else -0.19)
            ax.boxplot(v, positions=[pos], widths=0.32, showfliers=False, patch_artist=True,
                       medianprops=dict(color=col, lw=2.2),
                       boxprops=dict(facecolor=col, alpha=0.22, edgecolor=col, lw=1.2),
                       whiskerprops=dict(color=col, lw=1.1), capprops=dict(color=col, lw=1.1))
            vv = v if len(v) <= 150 else rng.choice(v, 150, replace=False)
            ax.scatter(pos + rng.uniform(-0.14, 0.14, len(vv)), vv, s=8, color=col,
                       alpha=0.55, edgecolor="none", zorder=3)
    ax.axhline(0, color="0.8", lw=0.8)
    for t in (2.576, -2.576):
        ax.axhline(t, ls=":", lw=0.9, color="0.6")
    # full y-axis -- no clipping (z_het extremes on VSC/mCAD/Circadian dominate
    # the scale; the other z-scores compress toward 0).
    ax.margins(y=0.05)
    ax.set_xticks(range(len(ZC))); ax.set_xticklabels(ZLAB, fontsize=12)
    ax.set_xlim(-0.6, len(ZC) - 0.4)
    ax.set_ylabel("permutation z"); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{net.replace('_activation','').replace('_cycle','')}   "
                 f"({nT} true / {len(df)} directed pairs)", fontsize=12)

fig.legend(handles=[Patch(facecolor=GRY, alpha=.5, label="non-edge"),
                    Patch(facecolor=GRN, alpha=.5, label="true directed edge")],
           loc="lower center", ncol=2, frameon=False, fontsize=12, bbox_to_anchor=(0.5, 0.005))
fig.suptitle("TwINFER permutation z-scores by network — true edges vs non-edges "
             "(no-filter, mean over reps; dotted = $\\pm$2.576)", fontsize=14)
fig.tight_layout(rect=[0, 0.02, 1, 0.98])
out = f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/real_networks/zscore_box_per_network.png'
fig.savefig(out, dpi=115, bbox_inches="tight")
print("wrote", out)
