"""Supplementary computation: random_delta_t1/t2 reference matrices for one gene set, needed for
the new TwinScore's "ref_Delta drift contrast" term (BEST_SCORER.md), which the original
infer_with_twinfer run computed internally (calculate_twin_random_correlations, called once at t1
and once at t2) but never saved to disk.

NOT a permutation-heavy step -- one fresh random-pair draw per timepoint, not a null distribution
-- so this is fast and runs locally without SLURM. Reuses the exact same seed/unit as the original
ALL_PAIRS run (default seed=101010, unit="clone") so the values match what that run would have
produced internally.

Run directly: python3 compute_random_delta_reference.py
Different gene set: GENE_SET=variability_low python3 compute_random_delta_reference.py
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import os
import sys

import pandas as pd

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/home/gzu5140/TwINFER_KA/code/TwINFER/package")
from twinfer.inference.correlation_functions import (
    assign_twin_id,
    calculate_twin_random_correlations,
)

HERE = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation'
from twinfer.utils.paths import get_data_root as _res_gdr  # [2026-10-01 LARRY resources/ moved out of the code tree to analysis_data/paper_analysis/larry_hematopoiesis_validation/resources (user)]
RES_HERE = f"{_res_gdr()}/paper_analysis/larry_hematopoiesis_validation"
GENE_SET = os.environ.get("GENE_SET", "correlation_high")
T1, T2 = 2, 4
SEED = 101010  # infer_with_twinfer's own default -- must match to reproduce the same run
UNIT = "clone"

# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] INPUT_CSV = os.path.join(HERE, "resources", "twinfer_input", f"{GENE_SET}.csv")
INPUT_CSV = os.path.join(RES_HERE, "resources", "twinfer_input", f"{GENE_SET}.csv")
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] OUT_DIR = os.path.join(HERE, "resources", "infer_results", f"{GENE_SET}_t2_t4_allpairs_50core")
OUT_DIR = os.path.join(RES_HERE, "resources", "infer_results", f"{GENE_SET}_t2_t4_allpairs_50core")
assert os.path.isdir(OUT_DIR), f"expected the ALL_PAIRS run's output dir to already exist: {OUT_DIR}"


def log(m):
    print(m, flush=True)


df = pd.read_csv(INPUT_CSV)
gene_list = [c[:-5] for c in df.columns if c.endswith("_mRNA")]
log(f"[{GENE_SET}] {len(gene_list)} genes")

t1_raw = df[df["time_step"] == T1].reset_index(drop=True)
t2_raw = df[df["time_step"] == T2].reset_index(drop=True)
t1_twins = assign_twin_id(t1_raw).reset_index(drop=True)
t2_twins = assign_twin_id(t2_raw).reset_index(drop=True)

# Same seed offsets infer_with_twinfer itself uses for these two calls.
_, random_delta_t1 = calculate_twin_random_correlations(
    t1_raw, t1_twins, gene_list, unit=UNIT, random_state=SEED + 271829)
_, random_delta_t2 = calculate_twin_random_correlations(
    t2_raw, t2_twins, gene_list, unit=UNIT, random_state=SEED + 271830)

random_delta_t1.to_csv(os.path.join(OUT_DIR, "corr_random_delta_t1.csv"))
random_delta_t2.to_csv(os.path.join(OUT_DIR, "corr_random_delta_t2.csv"))
log(f"wrote {OUT_DIR}/corr_random_delta_t1.csv and corr_random_delta_t2.csv")
