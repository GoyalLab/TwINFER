"""Which analytic z-score (of everything TwINFER computes) best discriminates true directed
edges from non-edges on the network_sweep_final 6-gene sweep? Reuses the same from-scratch,
per-replicate-calibrated analytic pipeline as regdetector_vs_current.py, but instead of
assembling a twinScore it dumps every per-pair quantity and scores each one's own
discriminative power (AUPRC and AUROC), both signed and |.|, per replicate then averaged.

Output: zscore_search_results.csv (long, per replicate x candidate)
        zscore_search_summary.csv  (mean AUPRC/AUROC per candidate x t1, best sign)
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
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

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
    split_twins,
)
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, f"{ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation")
from twinfer.scoring.analytic_zscores import m_effs_from_table, analytic_twin_score_inputs

T2 = 20
N_RAND = int(os.environ.get("N_RAND", "20"))
SEED = 0
N_GENES = 6
GENES = [f"gene_{i+1}" for i in range(N_GENES)]
GCOLS = [f"{g}_mRNA" for g in GENES]
USECOLS = ["clone_id", "cell_id", "time_step"] + GCOLS

CANDS = ["rho_t1", "rho_t2", "rho_change", "rho_delta_t2",
         "z_abs_rho_t1", "z_abs_rho_t2", "z_abs_rho_change", "z_rho_change",
         "z_div", "z_het", "z_d_het", "z_gamma", "z_reg_gated",
         "abs_rho_cross_xy", "min_abs_rho_cross", "max_abs_rho_cross"]


def true_edges(topo_path):
    M = np.loadtxt(topo_path, delimiter=",", dtype=int)
    true = {(GENES[i], GENES[j]) for i in range(N_GENES) for j in range(N_GENES) if i != j and M[i, j] != 0}
    poss = [(GENES[i], GENES[j]) for i in range(N_GENES) for j in range(N_GENES) if i != j]
    return true, poss


def full_table(csv_path, T1):
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

    rows = []
    for a, b in ordered:
        r = tsi.loc[(a, b)]
        zh = r.z_het
        lam = min(1.0, abs(zh) / 2.33)
        rreg = rho_same[gx[a], gx[b]] - lam * rho_cross_2n[gx[a], gx[b]]
        z_reg_gated = rreg / SD["het_t1"]
        rows.append(dict(
            gene_1=a, gene_2=b,
            rho_t1=r.rho_t1, rho_t2=r.rho_t2, rho_change=r.rho_change, rho_delta_t2=r.rho_delta_t2,
            z_abs_rho_t1=r.z_abs_rho_t1, z_abs_rho_t2=r.z_abs_rho_t2,
            z_abs_rho_change=r.z_abs_rho_change, z_rho_change=r.z_rho_change,
            z_div=r.z_div, z_het=r.z_het, z_d_het=r.z_d_het, z_gamma=r.z_gamma,
            z_reg_gated=z_reg_gated,
            abs_rho_cross_xy=abs(r.rho_cross_xy),
            min_abs_rho_cross=min(abs(r.rho_cross_xy), abs(r.rho_cross_yx)),
            max_abs_rho_cross=max(abs(r.rho_cross_xy), abs(r.rho_cross_yx)),
        ))
    return pd.DataFrame(rows)


def score_candidate(vals, y):
    best = None
    for sign, x in ((1, vals), (-1, -vals)):
        try:
            prec, rec, _ = precision_recall_curve(y, x)
            ap = auc(rec, prec)
            auroc = roc_auc_score(y, x)
        except Exception:
            continue
        if best is None or ap > best[1]:
            best = (sign, ap, auroc)
    return best


def main():
    topo_files = sorted(glob.glob(f"{TOPO_DIR}/grn_n6_*.txt"))
    rows = []
    t0 = time.time()
    for tf in topo_files:
        net = os.path.basename(tf)[:-4]
        true, poss = true_edges(tf)
        y = np.array([1 if p in true else 0 for p in poss], int)
        if y.sum() == 0 or y.sum() == len(poss):
            continue
        sims = sorted(glob.glob(f"{SIM_DIR}/df_{net}_rep*_*.csv"))
        for sim in sims:
            simrep = os.path.basename(sim).split(f"df_{net}_")[1].split("_")[0]
            for T1 in (1, 10):
                tab = full_table(sim, T1)
                if tab is None:
                    continue
                tab = tab.set_index(["gene_1", "gene_2"]).reindex(poss)
                for c in CANDS:
                    vals = tab[c].to_numpy(float)
                    if not np.isfinite(vals).all():
                        continue
                    best = score_candidate(vals, y)
                    if best is None:
                        continue
                    sign, ap, auroc = best
                    rows.append(dict(net=net, simrep=simrep, T1=T1, cand=c, sign=sign, auprc=ap, auroc=auroc))
        print(f"[{time.time()-t0:.0f}s] {net}  ({len(sims)} sim reps)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{HERE}/zscore_search_results.csv", index=False)
    print(f"\nwrote {HERE}/zscore_search_results.csv  ({len(df)} rows)")

    summ = df.groupby(["T1", "cand"]).agg(mean_auprc=("auprc", "mean"), mean_auroc=("auroc", "mean"),
                                          mode_sign=("sign", lambda x: int(np.sign(x.mean())) or 1))
    summ = summ.sort_values(["T1", "mean_auprc"], ascending=[True, False])
    summ.to_csv(f"{HERE}/zscore_search_summary.csv")
    print("\n=== mean AUPRC / AUROC per candidate, by t1 (best sign per replicate) ===")
    for t1v, g in summ.groupby("T1"):
        print(f"\n--- t1={t1v} ---")
        print(g.droplevel("T1").round(3).to_string())


if __name__ == "__main__":
    main()
