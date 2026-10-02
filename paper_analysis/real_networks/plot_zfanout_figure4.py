"""Box+dots of z_fanout(x,y) per undirected gene pair, for each figure_4 3-gene motif
(Fan_out / Feed_forward / Mutual_regulation), at t1=1h and t1=10h (separate panels, not pooled).
z_fanout is symmetric (same value for (x,y) and (y,x)), so one value per undirected pair per file.
True edge (green, exists in either direction) vs non-edge (grey).
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = f'{TWINFER_PROJECT_ROOT}'
SIM_DIR = f"{ROOT}/simulation_data/figure_4"
# HERE = os.path.dirname(os.path.abspath(__file__))   [2026-09-30 replaced: this script used to sit in the data dir analysis_data/paper_analysis/real_networks and read/write next to itself; that dir is now referenced explicitly]
HERE = f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/real_networks'

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, f"{ROOT}/analysis_data/network_sweep_final")
from twinfer.scoring.analytic_core import full_table_signed_general, s

N_GENES = 3
GENES = [f"gene_{i+1}" for i in range(N_GENES)]
POSS = [(a, b) for a in GENES for b in GENES if a != b]
T2 = 20

TRUE_EDGES = {
    "Fan_out": {("gene_1", "gene_2"), ("gene_1", "gene_3")},
    "Feed_forward": {("gene_1", "gene_2"), ("gene_1", "gene_3"), ("gene_2", "gene_3")},
    "Mutual_regulation": {("gene_1", "gene_2"), ("gene_1", "gene_3"),
                          ("gene_2", "gene_3"), ("gene_3", "gene_2")},
}
MOTIF_ORDER = ["Fan_out", "Feed_forward", "Mutual_regulation"]
UNDIR_PAIRS = [("gene_1", "gene_2"), ("gene_1", "gene_3"), ("gene_2", "gene_3")]
LABEL = {("gene_1", "gene_2"): "1,2|3", ("gene_1", "gene_3"): "1,3|2", ("gene_2", "gene_3"): "2,3|1"}


def motif_type(fname):
    for t in TRUE_EDGES:
        if t in fname:
            return t
    return None


def is_true(motif, a, b):
    return (a, b) in TRUE_EDGES[motif] or (b, a) in TRUE_EDGES[motif]


def main():
    files = sorted(glob.glob(f"{SIM_DIR}/*.csv"))
    rows = []
    t0 = time.time()
    for f in files:
        mt = motif_type(os.path.basename(f))
        if mt is None:
            continue
        for T1 in (1, 10):
            tsi = full_table_signed_general(f, T1, T2, GENES)
            if tsi is None:
                continue
            tab = tsi.reindex(POSS)
            n = len(GENES); gx = {g: i for i, g in enumerate(GENES)}
            Z = np.zeros((n, n))
            for (a, b), v in zip(POSS, tab.z_abs_rho_t1.to_numpy()):
                Z[gx[a], gx[b]] = v
            Z = np.maximum(Z, Z.T); np.fill_diagonal(Z, -np.inf)
            for (a, b) in UNDIR_PAIRS:
                i, j = gx[a], gx[b]
                ci, cj = Z[i, :].copy(), Z[j, :].copy()
                ci[j] = -np.inf; cj[i] = -np.inf
                zfan = np.max(np.minimum(ci, cj))
                rows.append(dict(motif=mt, T1=T1, pair=(a, b), zfan=zfan, is_true=is_true(mt, a, b)))
        print(f"[{time.time()-t0:.0f}s] {os.path.basename(f)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{HERE}/zfanout_per_edge_figure4.csv", index=False)

    GRN, GRY = "#2ca02c", "#9aa0a6"
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)
    for ax, T1 in zip(axes, (1, 10)):
        sub = df[df.T1 == T1]
        xt, xl = [], []
        pos = 0
        for mt in MOTIF_ORDER:
            for pr in UNDIR_PAIRS:
                v = sub[(sub.motif == mt) & (sub.pair == pr)]
                if not len(v):
                    continue
                col = GRN if v.is_true.iloc[0] else GRY
                vals = v.zfan.to_numpy()
                ax.boxplot(vals, positions=[pos], widths=0.6, patch_artist=True, showfliers=False,
                           medianprops=dict(color=col, lw=2),
                           boxprops=dict(facecolor=col, alpha=0.25, edgecolor=col, lw=1.2),
                           whiskerprops=dict(color=col, lw=1.1), capprops=dict(color=col, lw=1.1))
                rng = np.random.default_rng(0)
                ax.scatter(pos + rng.uniform(-0.2, 0.2, len(vals)), vals, s=14, color=col,
                           alpha=0.6, edgecolor="none", zorder=3)
                xt.append(pos); xl.append(LABEL[pr])
                pos += 1
            pos += 0.8
        ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=10)
        ax.axhline(0, color="0.85", lw=0.8)
        ax.set_title(f"t1={T1}h", fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("z_fanout(x,y) = max$_C$ min(z(C,x), z(C,y))")

    # motif group labels
    for ax in axes:
        pass
    fig.text(0.5, -0.02, "pairs labeled x,y|C  (C = the third gene, candidate confounder)",
              ha="center", fontsize=9, color="0.4")
    handles = [Patch(facecolor=GRN, alpha=.5, label="true edge (either direction)"),
               Patch(facecolor=GRY, alpha=.5, label="non-edge")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("z_fanout per undirected gene pair, by motif  (Fan_out | Feed_forward | Mutual_regulation)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    out = f"{HERE}/zfanout_per_edge_figure4.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
