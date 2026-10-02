# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Tiny smoke test of the multistate sim config: 20 cells, 40 steps, mCAD + VSC."""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import os, time, glob
from numba import set_num_threads
from twinfer.simulation.gillespie_simulations import process_param_set

set_num_threads(4)
# [2026-10-01 commented out: ephemeral Claude scratchpad; the data was copied to clean_data/_rescued_scratchpad] OUT = "/gpfs/home/gzu5140/.claude-tmp/claude-2000104/-gpfs-projects-b1255-hzhang-TwINFER-KA/69296547-e537-4dbf-b059-2e4d21364c6b/scratchpad/simtest"
OUT = f"{TWINFER_PROJECT_ROOT}/clean_data/_rescued_scratchpad/69296547/simtest"
os.makedirs(f"{OUT}/logs", exist_ok=True)
INPUT_ROOT = f'{TWINFER_PROJECT_ROOT}/input_data'

for net, n_genes in [("mCAD", 5), ("VSC", 8), ("GSD", 19)]:
    cfg = {
        "n_cells": 20,
        "simulation_time_before_division": 40,
        "twin_simulation_time_after_division": 10,
        "twin_measurement_resolution": 1,
        "path_to_connectivity_matrix": f"{INPUT_ROOT}/real_world_networks/{net}.txt",
        "param_csv": f"{INPUT_ROOT}/network_sweep/parameters.csv",
        "rows_to_use": [[0] * n_genes],
        "output_folder": OUT,
        "log_file": f"{OUT}/logs/{net}.log",
        "type": f"{net}_smoketest",
        "combinatorial_interaction_type": "additive",
        "number_of_parallel_parameters": 1,
        "number_of_cores_per_parameter": 4,
    }
    label = "rows_" + "_".join(map(str, cfg["rows_to_use"][0]))
    t0 = time.time()
    path = process_param_set(cfg["rows_to_use"][0], label, cfg)
    print(f"{net}: OK in {time.time()-t0:.1f}s -> {os.path.basename(path)}", flush=True)

print("\nfiles written:")
for f in sorted(glob.glob(f"{OUT}/*.csv")):
    import pandas as pd
    d = pd.read_csv(f, nrows=3)
    print(f"  {os.path.basename(f)}  cols={list(d.columns)[:6]}... rows(sample) shape {d.shape}")
