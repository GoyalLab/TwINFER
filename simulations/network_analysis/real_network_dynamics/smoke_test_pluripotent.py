# [UNREVIEWED: rescued 2026-09-30 from scratchpad_807d072d_2026-09-20_real_network_sim; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-10-01 added: project root for the rescued scratchpad copies]
import os, sys
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/work_in_progress/benchmark")
from twinfer.simulation.gillespie_simulations import process_param_set
from twinfer.utils.paths import get_repo_root
from numba import set_num_threads

set_num_threads(4)
# [2026-10-01 commented out: ephemeral Claude scratchpad; the data was copied to clean_data/_rescued_scratchpad] out = "/gpfs/home/gzu5140/.claude-tmp/claude-2000104/-gpfs-projects-b1255-hzhang-TwINFER-KA/807d072d-6fc7-411c-938f-bf2f50e95066/scratchpad/smoke_out"
out = f"{TWINFER_PROJECT_ROOT}/clean_data/_rescued_scratchpad/807d072d/smoke_out"
os.makedirs(out, exist_ok=True)
os.makedirs(f"{out}/logs", exist_ok=True)

cfg = {
    "n_cells": 20,
    "simulation_time_before_division": 30,
    "twin_simulation_time_after_division": 5,
    "twin_measurement_resolution": 1,
    "path_to_connectivity_matrix": f"{get_repo_root().parent.parent}/input_data/real_world_networks/Pluripotent.txt",
    "param_csv": f"{get_repo_root().parent.parent}/input_data/network_sweep/parameters.csv",
    "rows_to_use": [[0]*36],
    "output_folder": out,
    "log_file": f"{out}/logs/Pluripotent.log",
    "type": "Pluripotent_smoketest",
    "multiple_interaction_type": "additive",
    "number_of_parallel_parameters": 1,
    "number_of_cores_per_parameter": 4,
}
label = "rows_" + "_".join(map(str, cfg["rows_to_use"][0]))
path = process_param_set(cfg["rows_to_use"][0], label, cfg)
print("SMOKE TEST OK ->", path)
