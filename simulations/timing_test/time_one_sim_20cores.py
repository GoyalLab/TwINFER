from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import time
from numba import set_num_threads
from twinfer.simulation.gillespie_simulations import process_param_set
from twinfer.utils.paths import get_repo_root

path_to_code_repo = str(get_repo_root())
set_num_threads(20)

base_config = {
    'n_cells': 6000,
    'simulation_time_before_division': 1500,
    'twin_simulation_time_after_division': 48,
    'twin_measurement_resolution': 1,
    "path_to_connectivity_matrix": f"{path_to_code_repo}/simulation_example_input_data/connectivity_matrix_A_rep_B.txt",
    "param_csv": f"{path_to_code_repo}/simulation_example_input_data/median_parameter.csv",
    "rows_to_use": [[4] * 2],
    "output_folder": f'{TWINFER_PROJECT_ROOT}/simulation_data/_timing_test/output/',
    "log_file": f'{TWINFER_PROJECT_ROOT}/simulation_data/_timing_test/output/log.jsonl',
    "type": "A_rep_B_timing_test",
    "multiple_interaction_type": "additive",
    "number_of_parallel_parameters": 1,
    "number_of_cores_per_parameter": 20,
}

import os
os.makedirs(base_config["output_folder"], exist_ok=True)

t0 = time.perf_counter()
path = process_param_set([4, 4], "rows_4_4", base_config)
dt = time.perf_counter() - t0
print(f"one replicate took {dt:.1f}s -> {path}")
