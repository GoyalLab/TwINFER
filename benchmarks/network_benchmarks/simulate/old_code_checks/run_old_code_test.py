# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-10-01 added: project root for the rescued scratchpad copies]
import sys
import time
import pandas as pd
from numba import set_num_threads

# [2026-10-01 commented out: ephemeral Claude scratchpad; the data was copied to clean_data/_rescued_scratchpad] OLD_DIR = "/gpfs/home/gzu5140/.claude-tmp/claude-2000104/-gpfs-projects-b1255-hzhang-TwINFER-KA/3f20d120-6551-47bb-b46a-4cc9df254990/scratchpad/TwINFER-1.0/TwINFER-1.0"
OLD_DIR = f"{TWINFER_PROJECT_ROOT}/clean_data/_rescued_scratchpad/3f20d120/TwINFER-1.0/TwINFER-1.0"
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, OLD_DIR)

# from TwINFER_function_scripts.gillespie_script_variations import process_param_set   # [2026-09-30 ported: TwINFER_function_scripts was merged into the twinfer package (git 58f3442); old import kept above as a comment]
from twinfer.simulation.gillespie_simulations import process_param_set

# [2026-10-01 commented out: ephemeral Claude scratchpad; the data was copied to clean_data/_rescued_scratchpad] OUT_DIR = "/gpfs/home/gzu5140/.claude-tmp/claude-2000104/-gpfs-projects-b1255-hzhang-TwINFER-KA/3f20d120-6551-47bb-b46a-4cc9df254990/scratchpad/old_code_test"
OUT_DIR = f"{TWINFER_PROJECT_ROOT}/clean_data/_rescued_scratchpad/3f20d120/old_code_test"

set_num_threads(8)

for rep_id in range(2):
    cfg = {
        "n_cells": 6000,
        "simulation_time_before_division": 1500,
        "twin_simulation_time_after_division": 48,
        "twin_measurement_resolution": 1,
        "path_to_connectivity_matrix": f"{OLD_DIR}/simulation_example_input_data/connectivity_matrix_A_rep_B.txt",
        "param_csv": f"{OLD_DIR}/simulation_example_input_data/median_parameter.csv",
        "output_folder": OUT_DIR,
        "log_file": f"{OUT_DIR}/log.jsonl",
        "type": f"A_rep_B_oldcode_rep_{rep_id}",
        "multiple_interaction_type": "additive",
        "number_of_parallel_parameters": 1,
        "number_of_cores_per_parameter": 8,
    }
    t0 = time.time()
    path = process_param_set([4, 4], "rows_4_4", cfg)
    print(f"rep {rep_id}: elapsed={time.time()-t0:.0f}s -> {path}", flush=True)
    df = pd.read_csv(path)
    t1 = df[df["time_step"] == 1]
    print(f"rep {rep_id}: gene_2_mRNA mean={t1['gene_2_mRNA'].mean():.4f} std={t1['gene_2_mRNA'].std():.4f}", flush=True)
    print(f"rep {rep_id}: gene_1_mRNA mean={t1['gene_1_mRNA'].mean():.4f} std={t1['gene_1_mRNA'].std():.4f}", flush=True)

print("done", flush=True)
