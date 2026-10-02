# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""
Runs the ORIGINAL, UNMODIFIED figure_3_simulations.py logic for config_index=1
(A_rep_B) from the TwINFER-1.0 release, exactly as the release script would,
except: path_to_code_repo/output paths redirected to scratch, num_cores_available
lowered to 8, and only 2 replicates instead of 10 (for speed). Everything else
(base_configs[1] dict, process_param_set call) is copied verbatim.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-10-01 added: project root for the rescued scratchpad copies]
import sys
import os
import time
import pandas as pd
from numba import set_num_threads, get_num_threads

# [2026-10-01 commented out: ephemeral Claude scratchpad; the data was copied to clean_data/_rescued_scratchpad] OLD_REPO = "/gpfs/home/gzu5140/.claude-tmp/claude-2000104/-gpfs-projects-b1255-hzhang-TwINFER-KA/3f20d120-6551-47bb-b46a-4cc9df254990/scratchpad/TwINFER-1.0/TwINFER-1.0"
OLD_REPO = f"{TWINFER_PROJECT_ROOT}/clean_data/_rescued_scratchpad/3f20d120/TwINFER-1.0/TwINFER-1.0"
if OLD_REPO not in sys.path:
    pass
    # [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
    # sys.path.insert(0, OLD_REPO)

# [2026-10-01 commented out: ephemeral Claude scratchpad; the data was copied to clean_data/_rescued_scratchpad] OUT_DIR = "/gpfs/home/gzu5140/.claude-tmp/claude-2000104/-gpfs-projects-b1255-hzhang-TwINFER-KA/3f20d120-6551-47bb-b46a-4cc9df254990/scratchpad/official_script_test"
OUT_DIR = f"{TWINFER_PROJECT_ROOT}/clean_data/_rescued_scratchpad/3f20d120/official_script_test"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f"{OUT_DIR}/logs", exist_ok=True)

num_cores_available = 8

# ==========================================================
# base_configs[1] copied VERBATIM from figure_3_simulations.py, only
# path_to_code_repo, output_folder, log_file, number_of_cores_per_parameter redirected
# ==========================================================
base_config = {
    'n_cells': 6000,
    'simulation_time_before_division': 1500,
    'twin_simulation_time_after_division': 48,
    'twin_measurement_resolution': 1,
    "path_to_connectivity_matrix": f"{OLD_REPO}/simulation_example_input_data/connectivity_matrix_A_rep_B.txt",
    "param_csv": f"{OLD_REPO}/simulation_example_input_data/median_parameter.csv",
    "rows_to_use": [[4] * 2],
    "output_folder": f"{OUT_DIR}/A_rep_B/",
    "log_file": f"{OUT_DIR}/logs/Figure3.log",
    "type": "A_rep_B",
    "multiple_interaction_type": "additive",
    "number_of_parallel_parameters": 1,
    "number_of_cores_per_parameter": num_cores_available,
}

set_num_threads(base_config["number_of_cores_per_parameter"])
print(f"Using {get_num_threads()} threads for config: {base_config['type']}", flush=True)

# from TwINFER_function_scripts import gillespie_script_variations   # [2026-09-30 ported: TwINFER_function_scripts was merged into the twinfer package (git 58f3442); old import kept above as a comment]
from twinfer.simulation import gillespie_simulations as gillespie_script_variations
# from TwINFER_function_scripts.gillespie_script_variations import process_param_set   # [2026-09-30 ported: TwINFER_function_scripts was merged into the twinfer package (git 58f3442); old import kept above as a comment]
from twinfer.simulation.gillespie_simulations import process_param_set

os.makedirs(base_config['output_folder'], exist_ok=True)

rows_to_use = base_config['rows_to_use']
labels = ["rows_" + "_".join(map(str, row)) for row in rows_to_use]

original_type = base_config["type"]
last_path = None

# Only 2 replicates instead of the original script's 10, for speed
for i in range(2):
    cfg = dict(base_config)
    cfg["type"] = f"{original_type}_rep_{i}"
    print(f"Running replicate {i} with type={cfg['type']}", flush=True)
    t0 = time.time()
    path_to_simulation_file = process_param_set(rows_to_use[0], labels[0], cfg)
    print(f"rep {i}: elapsed={time.time()-t0:.0f}s -> {path_to_simulation_file}", flush=True)
    df = pd.read_csv(path_to_simulation_file)
    t1 = df[df["time_step"] == 1]
    print(f"rep {i}: gene_2_mRNA mean={t1['gene_2_mRNA'].mean():.4f} std={t1['gene_2_mRNA'].std():.4f}", flush=True)
    print(f"rep {i}: gene_1_mRNA mean={t1['gene_1_mRNA'].mean():.4f} std={t1['gene_1_mRNA'].std():.4f}", flush=True)
    last_path = path_to_simulation_file

print(f"Last simulation file saved as: {last_path}", flush=True)
