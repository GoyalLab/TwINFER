#!/usr/bin/env python
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_a2a3302f_2026-09-28_why_twinfer_fails; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Test: does increasing mRNA burst size (k_prod_mRNA 2 -> 10, same k_off) increase mRNA dispersion
(var/mean) and existence-detection signal on HSC, one of the near-Poisson (dispersion~1.14) networks?
HSC (11 genes) chosen over GSD (19 genes) for speed. Same topology, same everything else, only
k_prod_mRNA changes between the two runs.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import os
import sys

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/package")
from numba import set_num_threads
from twinfer.simulation.gillespie_simulations import process_param_set

MATRIX = f'{TWINFER_PROJECT_ROOT}/simulation_data/twinfer_format/HSC/interaction_matrix.txt'
PARAM_CSV = f'{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/real_networks/why_twinfer_fails/burst_size_test/test_params_hsc.csv'
OUT = f'{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/real_networks/why_twinfer_fails/burst_size_test'
os.makedirs(OUT, exist_ok=True)
os.makedirs(f"{OUT}/logs", exist_ok=True)

set_num_threads(64)

n_genes = 11
for tag, row in [("hsc_burst2_baseline", 0), ("hsc_burst10_boosted", 1)]:
    cfg = {
        "n_cells": 2000,
        "simulation_time_before_division": 300,
        "twin_simulation_time_after_division": 48,
        "twin_measurement_resolution": 1,
        "path_to_connectivity_matrix": MATRIX,
        "param_csv": PARAM_CSV,
        "rows_to_use": [[row] * n_genes],
        "output_folder": OUT,
        "log_file": f"{OUT}/logs/HSC_{tag}.log",
        "type": f"HSC_{tag}",
        "combinatorial_interaction_type": "additive",
    }
    print(f"=== running {tag} (param row {row}) ===", flush=True)
    path = process_param_set(cfg["rows_to_use"][0], tag, cfg)
    print(f"=== {tag} DONE -> {path} ===", flush=True)
