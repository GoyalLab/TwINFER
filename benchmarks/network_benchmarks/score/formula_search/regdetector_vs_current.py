"""Compare the regulation-detector D = s(|z_gamma|) + s(|z_reg_gated(t1)|) against the CURRENT
shipped twinScore, on the network_sweep_final 6-gene sweep (24 topologies x 10 sim reps), at
t1=1h and t1=10h (t2=20h), z-scores computed FROM SCRATCH with the analytic formula (no
permutation) -- calibrated per replicate from its own clone-size distribution via
analytic_zscores.m_effs_from_table (Kish effective sample size), not the LARRY DEFAULT_SD
constants.

For each (topology, sim rep, t1):
  - load only clone_id/cell_id/time_step/gene_*_mRNA at time_step in {t1, t2}
  - weighted (use_clone=True) correlations: rho_t1, rho_t2, rho_delta_t1/t2 (twin),
    rho_delta_random_t1/t2 (mean of N_RAND re-pairings), rho_cross (directed)
  - analytic_twin_score_inputs(...) -> z_abs_rho_t1/t2, z_div, z_het, z_d_het, z_gamma
  - calculate_twin_score(...)              = CURRENT (shipped) twinScore
  - z_reg_gated from rho_same/rho_cross_2n (pooled-2n twin rows, as in
    run_analytic_twinfer_tuned.py), sd_reg = calibrated SD["het_t1"]
  - D = s(|z_gamma|) + s(|z_reg_gated|), s() over this replicate's own ordered-pair panel

Scored against the topology's true directed edges: AUPRC (full 30-pair universe), F1@k,
precision@k, recall@k (k = n_true_edges, tie-aware).

Output: regdetector_vs_current_results.csv, regdetector_vs_current_summary.csv/png
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import auc, precision_recall_curve

# HERE = os.path.dirname(os.path.abspath(__file__))   [2026-09-30 replaced: these scripts used to sit in analysis_data/network_sweep_final and read/write their CSVs next to themselves; that data dir is now referenced explicitly]
HERE = f'{TWINFER_PROJECT_ROOT}/analysis_data/network_sweep_final'
ROOT = f'{TWINFER_PROJECT_ROOT}'
SIM_DIR = f"{ROOT}/simulation_data/network_sweep_final"
TOPO_DIR = f"{ROOT}/input_data/network_sweep_final"

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, f"{ROOT}/code/TwINFER/package")
from twinfer.inference.correlation_functions import (
    assign_twin_id, calculate_pairwise_gene_gene_correlation_matrix,
    calculate_twin_random_correlations, _build_cross_time_twins, get_cross_correlations,
    calculate_twin_score, split_twins,
)
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, f"{ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation")
from twinfer.scoring.analytic_zscores import (
    m_effs_from_table, analytic_twin_score_inputs, z_reg_gated as z_reg_gated_fn,
)

T2 = 20
N_RAND = int(os.environ.get("N_RAND", "20"))
SEED = 0
GATE_Z = 1.645
N_GENES = 6
GENES = [f"gene_{i+1}" for i in range(N_GENES)]
GCOLS = [f"{g}_mRNA" for g in GENES]
USECOLS = ["clone_id", "cell_id", "time_step"] + GCOLS


def s(v):
    v = np.asarray(v, float); f = np.isfinite(v); o = np.zeros(v.shape)
    if f.sum() > 1:
        o[f] = (v[f] - v[f].mean()) / max(v[f].std(ddof=1), 1e-12)
    return o


def true_edges(topo_path):
    M = np.loadtxt(topo_path, delimiter=",", dtype=int)
    true = {(GENES[i], GENES[j]) for i in range(N_GENES) for j in range(N_GENES) if i != j and M[i, j] != 0}
    poss = [(GENES[i], GENES[j]) for i in range(N_GENES) for j in range(N_GENES) if i != j]
    return true, poss


def score(mag, true, poss, seed=0):
    y = np.array([1 if p in true else 0 for p in poss], int)
    k = int(y.sum())
    if k == 0 or k == len(poss):
        return dict(auprc=np.nan, f1=np.nan, precision=np.nan, recall=np.nan)
    x = np.array([mag[p] for p in poss], float)
    prec, rec, _ = precision_recall_curve(y, x)
    auprc = auc(rec, prec)
    order = np.argsort(-x, kind="stable")
    boundary = x[order[k - 1]]
    sel = np.where(x >= boundary)[0]
    tp = y[sel].sum()
    precision = tp / len(sel); recall = tp / k
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return dict(auprc=auprc, f1=f1, precision=precision, recall=recall)


def analytic_for_file(csv_path, T1):
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

    tsi = analytic_twin_score_inputs(rho, SD=SD)
    ranked = calculate_twin_score({"twin_score_inputs": tsi})
    current = {(r.gene_1, r.gene_2): abs(float(r.twinScore)) for r in ranked.itertuples(index=False)}

    # z_reg_gated: pooled-2n rho_same / rho_cross_2n (as in run_analytic_twinfer_tuned.py)
    a0, b0 = split_twins(t1_tw)
    Xa = a0[GCOLS].to_numpy(float); Xb = b0[GCOLS].to_numpy(float)
    Xp = np.vstack([Xa, Xb])
    Rk = np.apply_along_axis(rankdata, 0, Xp); Rk -= Rk.mean(0)
    Sd = np.sqrt((Rk ** 2).sum(0)); Sd[Sd < 1e-12] = 1
    rho_same = (Rk.T @ Rk) / np.outer(Sd, Sd)
    Yc = np.vstack([Xb, Xa])
    RkY = np.apply_along_axis(rankdata, 0, Yc); RkY -= RkY.mean(0)
    SY = np.sqrt((RkY ** 2).sum(0)); SY[SY < 1e-12] = 1
    rho_cross_2n = (Rk.T @ RkY) / np.outer(Sd, SY)
    gx = {g: i for i, g in enumerate(GENES)}

    z_het_map = {(r.gene_1, r.gene_2): r.z_het for r in tsi.itertuples(index=False)}
    z_gamma_map = {(r.gene_1, r.gene_2): r.z_gamma for r in tsi.itertuples(index=False)}
    D_raw = {}
    for a, b in ordered:
        zh = z_het_map[(a, b)]
        lam = min(1.0, abs(zh) / 2.33)
        rreg = rho_same[gx[a], gx[b]] - lam * rho_cross_2n[gx[a], gx[b]]
        zrg = rreg / SD["het_t1"]
        D_raw[(a, b)] = dict(abs_zg=abs(z_gamma_map[(a, b)]), abs_zrg=abs(zrg))

    sD = s(np.array([D_raw[p]["abs_zg"] for p in ordered]))
    sR = s(np.array([D_raw[p]["abs_zrg"] for p in ordered]))
    D = {p: sD[i] + sR[i] for i, p in enumerate(ordered)}
    return current, D


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
                res = analytic_for_file(sim, T1)
                if res is None:
                    continue
                current, D = res
                sc_cur = score(current, true, poss)
                sc_D = score(D, true, poss)
                rows.append(dict(net=net, simrep=simrep, T1=T1, method="current", **sc_cur))
                rows.append(dict(net=net, simrep=simrep, T1=T1, method="D", **sc_D))
        print(f"[{time.time()-t0:.0f}s] {net}  ({len(sims)} sim reps)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{HERE}/regdetector_vs_current_results.csv", index=False)
    print(f"\nwrote {HERE}/regdetector_vs_current_results.csv  ({len(df)} rows)")

    summ = df.groupby(["T1", "method"])[["auprc", "f1", "precision", "recall"]].mean()
    summ.to_csv(f"{HERE}/regdetector_vs_current_summary.csv")
    print("\n=== mean AUPRC / F1 / precision / recall, by t1 and method ===")
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
