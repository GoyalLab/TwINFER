#!/usr/bin/env python
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_a2a3302f_2026-09-28_why_twinfer_fails; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Quick hypothesis test: does reducing protein_half_life (45 -> 16) make mCAD twins diverge
faster post-division, as predicted by the memory-buffer argument (protein half-life ~ observation
window -> slow forgetting of shared inherited state)?

Same topology (simulation_data/twinfer_format/mCAD/interaction_matrix.txt), same twin window
(twin_simulation_time_after_division=48, resolution=1) as the canonical mCAD analysis this session.
Only protein_half_life changes (45 baseline vs 16 test), everything else identical (same param row
otherwise, default Hill n=2 / k_add from CSV defaults). Reduced n_cells and pre-division time for speed
since this is a quick confirmatory test, not a production run.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import os
import sys

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/package")
from numba import set_num_threads
from twinfer.simulation.gillespie_simulations import process_param_set

MATRIX = f'{TWINFER_PROJECT_ROOT}/simulation_data/twinfer_format/mCAD/interaction_matrix.txt'
PARAM_CSV = f'{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/real_networks/why_twinfer_fails/protein_halflife_test/test_params.csv'
OUT = f'{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/real_networks/why_twinfer_fails/protein_halflife_test'
os.makedirs(OUT, exist_ok=True)
os.makedirs(f"{OUT}/logs", exist_ok=True)

set_num_threads(64)

n_genes = 5
for tag, row in [("halflife45", 0), ("halflife16", 1)]:
    cfg = {
        "n_cells": 2000,
        "simulation_time_before_division": 300,  # mCAD is small/fast, reaches steady state well before this
        "twin_simulation_time_after_division": 48,
        "twin_measurement_resolution": 1,
        "path_to_connectivity_matrix": MATRIX,
        "param_csv": PARAM_CSV,
        "rows_to_use": [[row] * n_genes],
        "output_folder": OUT,
        "log_file": f"{OUT}/logs/mCAD_{tag}.log",
        "type": f"mCAD_{tag}",
        "combinatorial_interaction_type": "additive",
    }
    print(f"=== running {tag} (param row {row}) ===", flush=True)
    path = process_param_set(cfg["rows_to_use"][0], tag, cfg)
    print(f"=== {tag} DONE -> {path} ===", flush=True)
