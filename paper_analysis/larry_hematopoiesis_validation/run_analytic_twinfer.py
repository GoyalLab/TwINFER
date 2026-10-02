"""Shuffle-free TwINFER for every gene set: compute the clone-weighted Spearman correlations
exactly as infer.py does (same helpers, no permutation), turn them into analytic z-scores with
analytic_zscores.py, then calculate_twin_score -> ranked_edges.

  GENE_SETS   comma list, default all 9 keys of resources/gene_sets.json
  INPUT_DIR   default resources/twinfer_input_cp10k  (log1p-CP10k normalized -- see build script)
  OUT_DIR     default resources/analytic_infer
  N_RAND      fresh random-pair re-pairings averaged for the z_het / z_d_het centre (default 40)

Per gene set writes OUT_DIR/{gene_set}/{twin_score_inputs.csv, ranked_edges.csv, m_eff.json}.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
import os
import sys
import time

import numpy as np
import pandas as pd

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/home/gzu5140/TwINFER_KA/code/TwINFER/package")
from twinfer.inference.correlation_functions import (
    assign_twin_id, calculate_pairwise_gene_gene_correlation_matrix,
    calculate_twin_random_correlations, _build_cross_time_twins, get_cross_correlations,
    calculate_twin_score,
)
from twinfer.scoring.analytic_zscores import DEFAULT_SD, analytic_twin_score_inputs, m_effs_from_table

HERE = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation'
from twinfer.utils.paths import get_data_root as _res_gdr  # [2026-10-01 LARRY resources/ moved out of the code tree to analysis_data/paper_analysis/larry_hematopoiesis_validation/resources (user)]
RES_HERE = f"{_res_gdr()}/paper_analysis/larry_hematopoiesis_validation"
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] INPUT_DIR = os.environ.get("INPUT_DIR", f"{HERE}/resources/twinfer_input_cp10k")
INPUT_DIR = os.environ.get("INPUT_DIR", f"{RES_HERE}/resources/twinfer_input_cp10k")
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] OUT_DIR = os.environ.get("OUT_DIR", f"{HERE}/resources/analytic_infer")
OUT_DIR = os.environ.get("OUT_DIR", f"{RES_HERE}/resources/analytic_infer")
N_RAND = int(os.environ.get("N_RAND", "40"))
T1, T2, SEED = 2, 4, 0
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] ALL = list(json.load(open(f"{HERE}/resources/gene_sets.json")))
ALL = list(json.load(open(f"{RES_HERE}/resources/gene_sets.json")))
GENE_SETS = os.environ.get("GENE_SETS", ",".join(ALL)).split(",")
os.makedirs(OUT_DIR, exist_ok=True)


def analytic_for_geneset(gs):
    df = pd.read_csv(f"{INPUT_DIR}/{gs}.csv")
    genes = [c[:-5] for c in df.columns if c.endswith("_mRNA")]
    t1_raw = df[df.time_step == T1].reset_index(drop=True)
    t2_raw = df[df.time_step == T2].reset_index(drop=True)
    t1_tw = assign_twin_id(t1_raw).reset_index(drop=True)
    t2_tw = assign_twin_id(t2_raw).reset_index(drop=True)

    rho = {}
    rho["rho_t1"] = calculate_pairwise_gene_gene_correlation_matrix(t1_raw, genes, use_clone=True)
    rho["rho_t2"] = calculate_pairwise_gene_gene_correlation_matrix(t2_raw, genes, use_clone=True)
    rho["rho_delta_t1"], _ = calculate_twin_random_correlations(t1_raw, t1_tw, genes, random_state=SEED, unit="clone")
    rho["rho_delta_t2"], _ = calculate_twin_random_correlations(t2_raw, t2_tw, genes, random_state=SEED, unit="clone")
    # centre for z_het / z_d_het: average N_RAND fresh random-pair rho_Delta draws
    r1 = np.mean([calculate_twin_random_correlations(t1_raw, t1_tw, genes, random_state=SEED + 100 + i, unit="clone")[1].to_numpy()
                  for i in range(N_RAND)], axis=0)
    r2 = np.mean([calculate_twin_random_correlations(t2_raw, t2_tw, genes, random_state=SEED + 500 + i, unit="clone")[1].to_numpy()
                  for i in range(N_RAND)], axis=0)
    rho["rho_delta_random_t1"] = pd.DataFrame(r1, index=genes, columns=genes)
    rho["rho_delta_random_t2"] = pd.DataFrame(r2, index=genes, columns=genes)
    at1, at2 = _build_cross_time_twins(t1_raw, t2_raw)
    ordered = [(a, b) for a in genes for b in genes if a != b]
    rho["rho_cross"] = get_cross_correlations(at1, at2, gene_pairs=ordered + [(g, g) for g in genes], unit="clone")

    tsi = analytic_twin_score_inputs(rho, SD=DEFAULT_SD)
    ranked = calculate_twin_score({"twin_score_inputs": tsi}).sort_values("twinScore", ascending=False).reset_index(drop=True)

    d = f"{OUT_DIR}/{gs}"; os.makedirs(d, exist_ok=True)
    tsi.to_csv(f"{d}/twin_score_inputs.csv", index=False)
    ranked.to_csv(f"{d}/ranked_edges.csv", index=False)
    M = m_effs_from_table(df, t1=T1, t2=T2)
    json.dump({k: float(v) for k, v in M.items()}, open(f"{d}/m_eff.json", "w"), indent=2)
    return len(ranked), M


if __name__ == "__main__":
    print(f"input {INPUT_DIR}  ->  {OUT_DIR}   (N_RAND={N_RAND})")
    for gs in GENE_SETS:
        t = time.time()
        n, M = analytic_for_geneset(gs)
        print(f"  {gs:18s} {n:5d} directed edges   "
              f"m_eff: step1_t1={M['step1_t1']:.0f} twin_t1={M['twin_t1']:.0f} cross={M['cross']:.0f}   "
              f"({time.time() - t:.0f}s)", flush=True)
