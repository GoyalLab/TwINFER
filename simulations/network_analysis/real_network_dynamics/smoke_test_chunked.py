# [UNREVIEWED: rescued 2026-09-30 from scratchpad_807d072d_2026-09-20_real_network_sim; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-10-01 added: project root for the rescued scratchpad copies]
import sys, os
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/work_in_progress/benchmark")
from benchmarks.network_benchmarks.simulate import pluripotent_sim_6000steps as mod
from numba import set_num_threads

set_num_threads(4)
mod.CHUNK_SIZE = 30
mod.N_CHUNKS = 2

# [2026-10-01 commented out: ephemeral Claude scratchpad; the data was copied to clean_data/_rescued_scratchpad] out = "/gpfs/home/gzu5140/.claude-tmp/claude-2000104/-gpfs-projects-b1255-hzhang-TwINFER-KA/807d072d-6fc7-411c-938f-bf2f50e95066/scratchpad/smoke_out_chunked"
out = f"{TWINFER_PROJECT_ROOT}/clean_data/_rescued_scratchpad/807d072d/smoke_out_chunked"
os.makedirs(out, exist_ok=True)
os.makedirs(f"{out}/logs", exist_ok=True)

cfg = {
    "n_cells": 15,
    "twin_simulation_time_after_division": 5,
    "twin_measurement_resolution": 1,
    "path_to_connectivity_matrix": f"{mod.get_repo_root().parent.parent}/input_data/real_world_networks/Pluripotent.txt",
    "param_csv": f"{mod.get_repo_root().parent.parent}/input_data/network_sweep/parameters.csv",
    "rows_to_use": [[0]*36],
    "output_folder": out,
    "log_file": f"{out}/logs/Pluripotent.log",
    "type": "Pluripotent_smoketest_chunked",
    "multiple_interaction_type": "additive",
    "number_of_parallel_parameters": 1,
    "number_of_cores_per_parameter": 4,
}
label = "rows_" + "_".join(map(str, cfg["rows_to_use"][0]))
before_path, rep_path = mod.run_chunked(cfg, cfg["rows_to_use"][0], label)
print("CHUNKED SMOKE TEST OK")
print("before:", before_path)
print("rep:", rep_path)

import pandas as pd
df = pd.read_csv(before_path)
print("\nbefore-division shape:", df.shape)
print("time_step range:", df.time_step.min(), "-", df.time_step.max())
print("unique time_steps:", sorted(df.time_step.unique()))
print("cell_id range:", df.cell_id.min(), "-", df.cell_id.max())

df2 = pd.read_csv(rep_path)
print("\ntwin df shape:", df2.shape)
print("columns:", list(df2.columns)[:8], "...")
print("replicate values:", df2.replicate.unique())
print("clone_id range:", df2.clone_id.min(), "-", df2.clone_id.max())
