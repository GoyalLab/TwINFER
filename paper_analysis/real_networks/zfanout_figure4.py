"""Does z_fanout(x,y) = max_C min(z_rho(C,x), z_rho(C,y)) help discriminate true directed edges
from non-edges on the figure_4 3-gene motif sweep (Fan_out / Feed_forward / Mutual_regulation),
the same way it did on LARRY? z-scores computed from scratch (analytic formula, no permutation),
calibrated per-file via Kish effective sample size, exactly as for network_sweep_final.

Ground truth (connectivity_matrix[i,j] != 0 => gene i+1 -> gene j+1), from
code/TwINFER/simulation_example_input_data/connectivity_matrix_{type}.txt:
  Fan_out:          1->2, 1->3
  Feed_forward:     1->2, 1->3, 2->3
  Mutual_regulation:1->2, 1->3, 2->3, 3->2

Each file has only 3 genes (6 ordered pairs) -- too small for a per-file AUPRC, so all files'
standardized (s(), per-file) scores are pooled into one universe per T1 for a single AUROC/AUPRC.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = f'{TWINFER_PROJECT_ROOT}'
SIM_DIR = f"{ROOT}/simulation_data/figure_4"

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, f"{ROOT}/code/TwINFER/package")
from twinfer.inference.correlation_functions import (
    assign_twin_id, calculate_pairwise_gene_gene_correlation_matrix,
    calculate_twin_random_correlations, _build_cross_time_twins, get_cross_correlations,
)
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, f"{ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation")
from twinfer.scoring.analytic_zscores import m_effs_from_table, analytic_twin_score_inputs

T2 = 20
N_RAND = 20
SEED = 0
N_GENES = 3
GENES = [f"gene_{i+1}" for i in range(N_GENES)]
GCOLS = [f"{g}_mRNA" for g in GENES]
USECOLS = ["clone_id", "cell_id", "time_step"] + GCOLS

TRUE_EDGES = {
    "Fan_out": {("gene_1", "gene_2"), ("gene_1", "gene_3")},
    "Feed_forward": {("gene_1", "gene_2"), ("gene_1", "gene_3"), ("gene_2", "gene_3")},
    "Mutual_regulation": {("gene_1", "gene_2"), ("gene_1", "gene_3"),
                          ("gene_2", "gene_3"), ("gene_3", "gene_2")},
}
ORDERED = [(a, b) for a in GENES for b in GENES if a != b]


def s(v):
    v = np.asarray(v, float); f = np.isfinite(v)
    o = np.zeros(len(v))
    if f.sum() > 1:
        o[f] = (v[f] - v[f].mean()) / max(v[f].std(ddof=1), 1e-12)
    o[~f] = (o[f].min() - 1.0) if f.any() else 0.0
    return o


def motif_type(fname):
    for t in TRUE_EDGES:
        if t in fname:
            return t
    return None


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
    rho["rho_cross"] = get_cross_correlations(at1, at2, gene_pairs=ORDERED + [(g, g) for g in GENES], unit="clone")

    tsi = analytic_twin_score_inputs(rho, SD=SD).set_index(["gene_1", "gene_2"])
    return tsi


def main():
    files = sorted(glob.glob(f"{SIM_DIR}/*.csv"))
    per_t1 = {1: [], 10: []}
    for f in files:
        mt = motif_type(os.path.basename(f))
        if mt is None:
            continue
        true = TRUE_EDGES[mt]
        for T1 in (1, 10):
            tsi = full_table(f, T1)
            if tsi is None:
                continue
            zmat = {}
            for a, b in ORDERED + [(g, g) for g in GENES]:
                pass
            zmat = {(r[0], r[1]): tsi.loc[r].z_abs_rho_t1 for r in tsi.index}
            abs_rho_t1 = np.array([abs(tsi.loc[p].rho_t1) for p in ORDERED])
            zfan = np.zeros(len(ORDERED))
            for k, (x, y) in enumerate(ORDERED):
                c = [g for g in GENES if g not in (x, y)][0]
                zcx = zmat.get((c, x), zmat.get((x, c)))
                zcy = zmat.get((c, y), zmat.get((y, c)))
                zfan[k] = min(zcx, zcy)
            y = np.array([1 if p in true else 0 for p in ORDERED], int)
            per_t1[T1].append(dict(motif=mt, file=os.path.basename(f),
                                    s_rho=s(abs_rho_t1), s_zfan=s(zfan), y=y))
        print(f"done {os.path.basename(f)}", flush=True)

    for T1, rows in per_t1.items():
        S_rho = np.concatenate([r["s_rho"] for r in rows])
        S_zfan = np.concatenate([r["s_zfan"] for r in rows])
        Y = np.concatenate([r["y"] for r in rows])
        n_by_motif = pd.Series([r["motif"] for r in rows]).value_counts()
        print(f"\n=== T1={T1}h  (n_files={len(rows)}, n_pairs={len(Y)}, base_rate={Y.mean():.3f}) ===")
        print(n_by_motif.to_string())
        for w in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5):
            score = S_rho + w * S_zfan
            print(f"  w={w:.2f}  AUROC={roc_auc_score(Y, score):.3f}  AUPRC={average_precision_score(Y, score):.3f}")
        print(f"  zfan alone   AUROC={roc_auc_score(Y, S_zfan):.3f}  AUPRC={average_precision_score(Y, S_zfan):.3f}")


if __name__ == "__main__":
    main()
