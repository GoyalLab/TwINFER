"""Does z_fanout need to be ADDED or DELETED (subtracted/excluded) on top of the existdir
(existence+direction) formula, on the cleanest ground-truth case available: the figure_4 3-gene
motifs (Fan_out, Feed_forward, Mutual_regulation)? Per motif type separately (not pooled -- they
have very different true-edge structure), sweep w across both signs.

score(x->y) = s(|rho_t1|) + s(|rho_t2|) + 1.0*s(direction(x->y)) + w*s(z_fanout(x,y))
  direction(x->y) = |rho_cross_xy(x,y)| - |rho_cross_xy(y,x)|
  z_fanout(x,y)   = max_C min(z_abs_rho_t1(C,x), z_abs_rho_t1(C,y))   (only 1 candidate C here)
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = f'{TWINFER_PROJECT_ROOT}'
SIM_DIR = f"{ROOT}/simulation_data/figure_4"
# HERE = os.path.dirname(os.path.abspath(__file__))   [2026-09-30 replaced: this script used to sit in the data dir analysis_data/paper_analysis/real_networks and read/write next to itself; that dir is now referenced explicitly]
HERE = f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/real_networks'

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, f"{ROOT}/analysis_data/network_sweep_final")
from twinfer.scoring.analytic_core import full_table_signed_general, s
from benchmarks.network_benchmarks.score.formula_search.regdetector_vs_current import score

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


def motif_type(fname):
    for t in TRUE_EDGES:
        if t in fname:
            return t
    return None


def main():
    files = sorted(glob.glob(f"{SIM_DIR}/*.csv"))
    WS = [-2.0, -1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
    rows = []
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
            exist = s(tab.rho_t1.abs().to_numpy()) + s(tab.rho_t2.abs().to_numpy())
            rev = {(b, a): abs(v) for (a, b), v in zip(POSS, tab.rho_cross_xy.to_numpy())}
            direction = np.array([abs(tab.loc[p].rho_cross_xy) - rev[p] for p in POSS])
            s_dir = s(direction)

            n = len(GENES); gx = {g: i for i, g in enumerate(GENES)}
            Z = np.zeros((n, n))
            for (a, b), v in zip(POSS, tab.z_abs_rho_t1.to_numpy()):
                Z[gx[a], gx[b]] = v
            Z = np.maximum(Z, Z.T); np.fill_diagonal(Z, -np.inf)
            zfan = np.zeros(len(POSS))
            for k, (a, b) in enumerate(POSS):
                i, j = gx[a], gx[b]
                ci, cj = Z[i, :].copy(), Z[j, :].copy()
                ci[j] = -np.inf; cj[i] = -np.inf
                zfan[k] = np.max(np.minimum(ci, cj))
            s_fan = s(zfan)

            base = exist + 1.0 * s_dir
            for w in WS:
                mag = dict(zip(POSS, base + w * s_fan))
                sc = score(mag, true, POSS)
                rows.append(dict(motif=mt, T1=T1, w=w, **sc))
        print(f"[{time.time()-t0:.0f}s] {os.path.basename(f)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{HERE}/zfanout_sign_check_figure4_results.csv", index=False)
    summ = df.groupby(["motif", "T1", "w"])[["auprc", "f1"]].mean()
    summ.to_csv(f"{HERE}/zfanout_sign_check_figure4_summary.csv")
    print("\n=== mean AUPRC by motif x T1 x weight w ===")
    for (mt, t1), g in summ.groupby(["motif", "T1"]):
        print(f"\n--- {mt}  T1={t1} ---")
        print(g.droplevel(["motif", "T1"]).round(3).to_string())


if __name__ == "__main__":
    main()
