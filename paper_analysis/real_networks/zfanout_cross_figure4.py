"""z_fanout using CROSS-CORRELATION (not co-expression) -- the "(or cross corr)" alternative from
the original definition. z_c(C,x) = max(|rho_cross_xy(C,x)|, |rho_cross_xy(x,C)|) / SD_cross
(direction-agnostic magnitude of cross-time coupling between C and x); z_fanout_cross(x,y) =
max_C min(z_c(C,x), z_c(C,y)). Verified: unlike the co-expression version, this one DOES put the
Fan_out confound pair (gene_2,gene_3 | gene_1) at the top (24/25 files) -- matching the reference
figure's pattern. This script (1) plots it per edge, same style as plot_zfanout_figure4.py, and
(2) re-runs the add/delete weight sweep on existdir + w*z_fanout_cross.
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
from twinfer.scoring.analytic_zscores import DEFAULT_SD
from benchmarks.network_benchmarks.score.formula_search.regdetector_vs_current import score

N_GENES = 3
GENES = [f"gene_{i+1}" for i in range(N_GENES)]
POSS = [(a, b) for a in GENES for b in GENES if a != b]
T2 = 20
SD_CROSS = DEFAULT_SD["cross"]

TRUE_EDGES = {
    "Fan_out": {("gene_1", "gene_2"), ("gene_1", "gene_3")},
    "Feed_forward": {("gene_1", "gene_2"), ("gene_1", "gene_3"), ("gene_2", "gene_3")},
    "Mutual_regulation": {("gene_1", "gene_2"), ("gene_1", "gene_3"),
                          ("gene_2", "gene_3"), ("gene_3", "gene_2")},
}
MOTIF_ORDER = ["Fan_out", "Feed_forward", "Mutual_regulation"]
UNDIR_PAIRS = [("gene_1", "gene_2"), ("gene_1", "gene_3"), ("gene_2", "gene_3")]
LABEL = {("gene_1", "gene_2"): "1,2|3", ("gene_1", "gene_3"): "1,3|2", ("gene_2", "gene_3"): "2,3|1"}
WS = [-2.0, -1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0]


def motif_type(fname):
    for t in TRUE_EDGES:
        if t in fname:
            return t
    return None


def is_true(motif, a, b):
    return (a, b) in TRUE_EDGES[motif] or (b, a) in TRUE_EDGES[motif]


def zc_lookup(tab):
    return {(a, b): abs(tab.loc[(a, b)].rho_cross_xy) / SD_CROSS for a, b in POSS}


def zcross(zc, c, x):
    return max(zc.get((c, x), 0.0), zc.get((x, c), 0.0))


def main():
    files = sorted(glob.glob(f"{SIM_DIR}/*.csv"))
    edge_rows, sweep_rows = [], []
    t0 = time.time()
    for f in files:
        mt = motif_type(os.path.basename(f))
        if mt is None:
            continue
        true = TRUE_EDGES[mt]
        for T1 in (1, 10):
            tsi = full_table_signed_general(f, T1, T2, GENES)
            if tsi is None:
                continue
            tab = tsi.reindex(POSS)
            zc = zc_lookup(tab)

            for a, b in UNDIR_PAIRS:
                c = [g for g in GENES if g not in (a, b)][0]
                v = min(zcross(zc, c, a), zcross(zc, c, b))
                edge_rows.append(dict(motif=mt, T1=T1, pair=(a, b), zfan=v, is_true=is_true(mt, a, b)))

            exist = s(tab.rho_t1.abs().to_numpy()) + s(tab.rho_t2.abs().to_numpy())
            rev = {(b_, a_): abs(v_) for (a_, b_), v_ in zip(POSS, tab.rho_cross_xy.to_numpy())}
            direction = np.array([abs(tab.loc[p].rho_cross_xy) - rev[p] for p in POSS])
            s_dir = s(direction)
            zfan_full = np.zeros(len(POSS))
            for k, (a, b) in enumerate(POSS):
                c = [g for g in GENES if g not in (a, b)][0]
                zfan_full[k] = min(zcross(zc, c, a), zcross(zc, c, b))
            s_fan = s(zfan_full)
            base = exist + 1.0 * s_dir
            for w in WS:
                mag = dict(zip(POSS, base + w * s_fan))
                sc = score(mag, true, POSS)
                sweep_rows.append(dict(motif=mt, T1=T1, w=w, **sc))
        print(f"[{time.time()-t0:.0f}s] {os.path.basename(f)}", flush=True)

    edf = pd.DataFrame(edge_rows)
    edf.to_csv(f"{HERE}/zfanout_cross_per_edge_figure4.csv", index=False)
    sdf = pd.DataFrame(sweep_rows)
    sdf.to_csv(f"{HERE}/zfanout_cross_sign_check_figure4_results.csv", index=False)

    GRN, GRY = "#2ca02c", "#9aa0a6"
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)
    for ax, T1 in zip(axes, (1, 10)):
        sub = edf[edf.T1 == T1]
        xt, xl, pos = [], [], 0
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
    axes[0].set_ylabel("z_fanout_cross(x,y) = max$_C$ min(z$_{cross}$(C,x), z$_{cross}$(C,y))")
    handles = [Patch(facecolor=GRN, alpha=.5, label="true edge (either direction)"),
               Patch(facecolor=GRY, alpha=.5, label="non-edge")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("z_fanout (cross-correlation version) per undirected gene pair, by motif", fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    out = f"{HERE}/zfanout_cross_per_edge_figure4.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)

    print("\n=== add-vs-delete sweep (z_fanout_cross), mean AUPRC by motif x T1 x w ===")
    summ = sdf.groupby(["motif", "T1", "w"]).auprc.mean()
    for (mt, t1), g in summ.groupby(["motif", "T1"]):
        print(f"\n--- {mt} T1={t1} ---")
        print(g.droplevel(["motif", "T1"]).round(3).to_string())


if __name__ == "__main__":
    main()
