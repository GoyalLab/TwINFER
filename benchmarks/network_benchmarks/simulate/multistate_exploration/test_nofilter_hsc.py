from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import time, warnings
warnings.filterwarnings("ignore")
import sys
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/package")
from twinfer.inference.infer import infer_with_twinfer

path = f'{TWINFER_PROJECT_ROOT}/simulation_data/real_data/df_rows_0_0_0_0_0_0_0_0_0_0_0_12072026_020848_ncells_6000_HSC_balanced_0_0_6bff4482.csv'
base_config = {
    "n_cells": 6000,
    "simulation_time_before_division": 6000,
    "twin_simulation_time_after_division": 48,
    "twin_measurement_resolution": 1,
    "path_to_connectivity_matrix": f'{TWINFER_PROJECT_ROOT}/input_data/real_world_networks/HSC.txt',
    "param_csv": f'{TWINFER_PROJECT_ROOT}/input_data/network_sweep/parameters.csv',
    "rows_to_use": [[0]*11],
    "type": "HSC",
}

t0 = time.time()
results = infer_with_twinfer(
    path, merge_to_multiple_states=False, base_config=base_config, t1=1, t2=20,
    match_sim_details=False, check_for_steady_state=False, seed=101010, n_cores=15,
    alpha_gene_gene_corr=0.999999, alpha_stage3=0.999999,
    z_score_threshold_two_states=0, z_score_threshold_cross_correlation=0,
    corr_threshold_cross_correlation=0, fan_out_z_score_threshold=0,
    ranked_list=True,
)
print("elapsed:", time.time()-t0)
red = results["ranked_edges"]
print("ranked_edges rows:", len(red))
print("n possible directed pairs:", 11*10)
