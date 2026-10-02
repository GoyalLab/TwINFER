"""Two-stage score: merge existence (symmetric, undirected: is there ANY edge between x,y) then
use cross-correlation asymmetry for direction (does x->y or y->x explain the timing better),
instead of flatly summing existence- and direction-flavored terms together (as S/one_score_search
did). Tests whether explicit staging beats the flat additive combination on network_sweep_final.

  existence(x,y) = s(|rho_t1(x,y)|) + s(|rho_t2(x,y)|)      -- symmetric, same value for (x,y),(y,x)
  direction(x->y) = |rho_cross_xy(x,y)| - |rho_cross_xy(y,x)|  -- MAGNITUDES, not signed values:
                     a repressive edge has negative rho_cross_xy, so a signed difference would
                     misread it as evidence for the reverse direction. abs() first (gamma's
                     convention) makes this antisymmetric in which direction wins, not in sign
                     of regulation: direction(y->x) = -direction(x->y)
  score(x->y) = existence(x,y) + w * s(direction(x->y))

Reuses full_table() from zscore_search_network_sweep.py and true_edges()/score() from
regdetector_vs_current.py. Scored per (topology, sim rep, T1), averaged, weight grid on w.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

# HERE = os.path.dirname(os.path.abspath(__file__))   [2026-09-30 replaced: these scripts used to sit in analysis_data/network_sweep_final and read/write their CSVs next to themselves; that data dir is now referenced explicitly]
HERE = f'{TWINFER_PROJECT_ROOT}/analysis_data/network_sweep_final'
ROOT = f'{TWINFER_PROJECT_ROOT}'
SIM_DIR = f"{ROOT}/simulation_data/network_sweep_final"
TOPO_DIR = f"{ROOT}/input_data/network_sweep_final"

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, HERE)
from benchmarks.network_benchmarks.score.formula_search.zscore_search_network_sweep import GENES, GCOLS, USECOLS, N_RAND, SEED, T2
from benchmarks.network_benchmarks.score.formula_search.regdetector_vs_current import true_edges, score

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, f"{ROOT}/code/TwINFER/package")
from twinfer.inference.correlation_functions import (
    assign_twin_id, calculate_pairwise_gene_gene_correlation_matrix,
    calculate_twin_random_correlations, _build_cross_time_twins, get_cross_correlations,
)
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, f"{ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation")
from twinfer.scoring.analytic_zscores import m_effs_from_table, analytic_twin_score_inputs

WS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


def full_table_signed(csv_path, T1):
    """Same pipeline as zscore_search_network_sweep.full_table, but keeps rho_cross_xy/yx
    SIGNED (needed for the direction term) instead of collapsing to |.| immediately."""
    df = pd.read_csv(csv_path, usecols=USECOLS)
    t1_raw = df[df.time_step == T1].reset_index(drop=True)
    t2_raw = df[df.time_step == T2].reset_index(drop=True)
    if len(t1_raw) < 20 or len(t2_raw) < 20:
        return None
    t1_tw = assign_twin_id(t1_raw).reset_index(drop=True)
    t2_tw = assign_twin_id(t2_raw).reset_index(drop=True)

    M = m_effs_from_table(df, t1=T1, t2=T2)
    sd1 = 1.0 / np.sqrt(max(M["step1_t1"] - 1, 1e-6))
    sd2 = 1.0 / np.sqrt(max(M["step1_t2"] - 1, 1e-6))
    SD = dict(step1_t1=sd1, step1_t2=sd2,
              div_t1=1.0 / np.sqrt(max(M["twin_t1"] - 1, 1e-6)),
              het_t1=1.0 / np.sqrt(max(M["twin_t1"] - 1, 1e-6)),
              d=1.0 / np.sqrt(max(M["d"] - 1, 1e-6)),
              change=float(np.sqrt(sd1 ** 2 + sd2 ** 2)),
              cross=1.0 / np.sqrt(max(M["cross"] - 1, 1e-6)))

    rho = {}
    rho["rho_t1"] = calculate_pairwise_gene_gene_correlation_matrix(t1_raw, GENES, use_clone=True)
    rho["rho_t2"] = calculate_pairwise_gene_gene_correlation_matrix(t2_raw, GENES, use_clone=True)
    rho["rho_delta_t1"], _ = calculate_twin_random_correlations(t1_raw, t1_tw, GENES, random_state=SEED, unit="clone")
    rho["rho_delta_t2"], _ = calculate_twin_random_correlations(t2_raw, t2_tw, GENES, random_state=SEED, unit="clone")
    r1 = np.mean([calculate_twin_random_correlations(t1_raw, t1_tw, GENES, random_state=SEED + 100 + i, unit="clone")[1].to_numpy()
                  for i in range(N_RAND)], axis=0)
    r2 = np.mean([calculate_twin_random_correlations(t2_raw, t2_tw, GENES, random_state=SEED + 500 + i, unit="clone")[1].to_numpy()
                  for i in range(N_RAND)], axis=0)
    rho["rho_delta_random_t1"] = pd.DataFrame(r1, index=GENES, columns=GENES)
    rho["rho_delta_random_t2"] = pd.DataFrame(r2, index=GENES, columns=GENES)
    at1, at2 = _build_cross_time_twins(t1_raw, t2_raw)
    ordered = [(a, b) for a in GENES for b in GENES if a != b]
    rho["rho_cross"] = get_cross_correlations(at1, at2, gene_pairs=ordered + [(g, g) for g in GENES], unit="clone")

    tsi = analytic_twin_score_inputs(rho, SD=SD).set_index(["gene_1", "gene_2"])
    return tsi


def s(v):
    v = np.asarray(v, float); f = np.isfinite(v); o = np.zeros(v.shape)
    if f.sum() > 1:
        o[f] = (v[f] - v[f].mean()) / max(v[f].std(ddof=1), 1e-12)
    return o


def main():
    topo_files = sorted(glob.glob(f"{TOPO_DIR}/grn_n6_*.txt"))
    rows = []
    t0 = time.time()
    for tf in topo_files:
        net = os.path.basename(tf)[:-4]
        true, poss = true_edges(tf)
        sims = sorted(glob.glob(f"{SIM_DIR}/df_{net}_rep*_*.csv"))
        for sim in sims:
            simrep = os.path.basename(sim).split(f"df_{net}_")[1].split("_")[0]
            for T1 in (1, 10):
                tsi = full_table_signed(sim, T1)
                if tsi is None:
                    continue
                tab_i = tsi.reindex(poss)
                if not np.isfinite(tab_i.rho_t1.to_numpy()).all():
                    continue

                exist = s(tab_i.rho_t1.abs().to_numpy()) + s(tab_i.rho_t2.abs().to_numpy())
                # direction(x->y) = |rho_cross_xy(x,y)| - |rho_cross_xy(y,x)|  (magnitudes, not
                # signed values -- a repressive edge has negative rho_cross_xy, so a signed
                # difference would misread it as evidence for the reverse direction; abs() is
                # exactly gamma's convention elsewhere in this codebase)
                rev = {(b, a): abs(v) for (a, b), v in zip(poss, tab_i.rho_cross_xy.to_numpy())}
                direction = np.array([abs(tab_i.loc[p].rho_cross_xy) - rev[p] for p in poss])
                s_dir = s(direction)

                for w in WS:
                    mag = dict(zip(poss, exist + w * s_dir))
                    sc = score(mag, true, poss)
                    rows.append(dict(net=net, simrep=simrep, T1=T1, w=w, **sc))
        print(f"[{time.time()-t0:.0f}s] {net}  ({len(sims)} sim reps)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{HERE}/existence_then_direction_results.csv", index=False)
    print(f"\nwrote {HERE}/existence_then_direction_results.csv  ({len(df)} rows)")

    summ = df.groupby(["T1", "w"])[["auprc", "f1", "precision", "recall"]].mean()
    summ.to_csv(f"{HERE}/existence_then_direction_summary.csv")
    print("\n=== mean AUPRC / F1 / precision / recall, by t1 and weight w ===")
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
