import os
import re
import glob
import json
import warnings
from pathlib import Path
from itertools import permutations, combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from joblib import Parallel, delayed
from scipy.stats import norm
from scipy.optimize import brentq
from sklearn.mixture import GaussianMixture
from sklearn.metrics import auc, precision_recall_curve


# Path to TwINFER code repository
path_to_code_repo = "/home/gzu5140/Keerthana_b1042/TwINFER/code/TwINFER/"
path_to_simulation_data = "/home/gzu5140/Keerthana_b1042/TwINFER/simulation_data/network_sweep_final/"
path_to_ground_truth_dir = "/home/gzu5140/Keerthana_b1042/TwINFER/input_data/network_sweep_final/"

# NEW output dir for the fan-out rerun -- kept separate from the original cached
# results (analysis_data/network_sweep_final/network_inference/) so existing
# notebooks/CSVs depending on the old cache are left untouched.
path_to_twinfer_fanout_output = "/home/gzu5140/Keerthana_b1042/TwINFER/analysis_data/network_sweep_final/network_inference_fanout/"
os.makedirs(path_to_twinfer_fanout_output, exist_ok=True)

# BEELINE (7 methods) cached outputs -- no rerun needed, nothing changed for these.
BEELINE_INPUT_ROOT = Path("/home/gzu5140/Keerthana_b1042/TwINFER/code/Beeline/inputs/network_sweep_final")
BEELINE_OUTPUT_ROOT = Path("/home/gzu5140/Keerthana_b1042/TwINFER/analysis_data/network_sweep_final/beeline_inference")

import sys
sys.path.append(str(path_to_code_repo))
from TwINFER_function_scripts.infer_with_twinfer import infer_with_twinfer

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy scalar/array types infer_with_twinfer's result dict contains."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def make_json_safe(obj):
    """Recursively converts infer_with_twinfer's result dict (DataFrames, sets, tuple-keyed
    dicts) into a JSON-serializable structure."""
    if isinstance(obj, pd.DataFrame):
        return {
            "__type__": "DataFrame",
            "index": [str(i) for i in obj.index.tolist()],
            "columns": [str(c) for c in obj.columns.tolist()],
            "data": obj.values.tolist(),
        }
    if isinstance(obj, pd.Series):
        return {"__type__": "Series", "index": [str(i) for i in obj.index.tolist()], "data": obj.values.tolist()}
    if isinstance(obj, dict):
        return {
            ("__".join(map(str, k)) if isinstance(k, tuple) else str(k)): make_json_safe(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (set, frozenset)):
        return [make_json_safe(x) for x in obj]
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(x) for x in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def extract_run_id(filename):
    """Extracts '{config_index}_{replicate}' from '..._{config_index}_{replicate}_{8hex}.csv'."""
    fname = os.path.basename(filename)
    match = re.search(r'_(\d+_\d+)_[0-9a-fA-F]{8}\.csv$', fname)
    return match.group(1) if match else None


def run_twinfer_with_fanout(path_to_simulation_file, sim_type, rep_id, base_config, t1, t2, output_path):
    """
    Runs infer_with_twinfer with separate_fan_outs_from_mutual_regulation_flag=True and
    saves the full result dict to JSON, same convention as the original
    infer_network_simulation_network_sweep.py's build_simulation_record, plus the two new
    fan-out kwargs.
    """
    analysis_key = f"{sim_type}_rep_{rep_id}"
    f_result_path = os.path.join(output_path, f"{analysis_key}_all_results.json")
    if os.path.exists(f_result_path):
        return f_result_path  # skip -- already computed

    results = infer_with_twinfer(
        path_to_simulation_file,
        merge_to_multiple_states=False,
        base_config=base_config,
        t1=t1,
        t2=t2,
        check_for_steady_state=False,
        show_scrambled_distribution_gene_correlation=True,
        plot_correlation_matrices_as_heatmap=False,
        return_gene_corr_thresholds=False,
        match_sim_details=False,
        seed=101010,
        n_cores=15,
        z_score_threshold_two_states=10,
        infer_direction_for_which_edges="all-edges",
        ranked_list=True,
        separate_fan_outs_from_mutual_regulation_flag=True,
        fan_out_z_score_threshold=8,
    )

    n_genes = None
    for v in results.values():
        if isinstance(v, pd.DataFrame):
            n_genes = v.shape[0]
            break
    gene_names = [f"g{i+1}" for i in range(n_genes)] if n_genes is not None else None

    record = {
        "sim_type": sim_type, "rep_id": rep_id, "analysis_key": analysis_key,
        "gene_names": gene_names, "n_genes": n_genes,
        **make_json_safe(results),
    }
    os.makedirs(output_path, exist_ok=True)
    with open(f_result_path, "w") as f:
        json.dump(record, f, cls=NumpyEncoder, indent=2)
    return f_result_path

# Same base_config as infer_network_simulation_network_sweep.py -- the connectivity
# matrix here is only used to read off n_genes (all sweep networks are 6-gene), so it
# doesn't need to match each variant's true topology; match_sim_details=False below
# skips the assertions that would otherwise require it to.
base_configs = {
    'n_cells': 6000,
    'simulation_time_before_division': 6000,
    'twin_simulation_time_after_division': 48,
    'twin_measurement_resolution': 1,
    "path_to_connectivity_matrix": "/home/gzu5140/Keerthana_b1042/TwINFER/input_data/network_sweep/grn_n6_e5_pos50_density_rep0.txt",
    "param_csv": "/home/gzu5140/Keerthana_b1042/TwINFER/input_data/network_sweep/parameters.csv",
    "rows_to_use": [[0] * 6],
    "output_folder": f"{path_to_code_repo}/simulation_example_output_data/",
    "log_file": f"{path_to_code_repo}/simulation_example_output_data/logs/HSC.jsonl",
    "type": "HSC_balanced",
    "combinatorial_interaction_type": "additive",
    "number_of_parallel_parameters": 1,
    "number_of_cores_per_parameter": 56,
    "log_pi_on": False,
    "ranked_list": True,
}
t1, t2 = 1, 20

pattern = os.path.join(path_to_simulation_data, "df_grn_n6_*_ncells_6000_grn_n6_*.csv")
files = sorted(glob.glob(pattern), reverse=True)
variant_re = re.compile(r"df_grn_n6_(.+?)_ncells_6000_grn_n6_")

tasks = []
for f in files:
    fname = os.path.basename(f)
    match = variant_re.search(fname)
    if not match:
        continue
    variant_full = match.group(1)
    rep_id = extract_run_id(fname)
    tasks.append((f, variant_full, rep_id))

print(f"Collected {len(tasks)} TwINFER rerun tasks.")

LIMIT_TASKS_FOR_TESTING = None  # set to a small int for a quick smoke test
if LIMIT_TASKS_FOR_TESTING is not None:
    tasks = tasks[:LIMIT_TASKS_FOR_TESTING]
    print(f"LIMIT_TASKS_FOR_TESTING set -- only running {len(tasks)} task(s).")

result_paths = Parallel(n_jobs=4, backend="loky")(
    delayed(run_twinfer_with_fanout)(path, sim_type, rep_id, base_configs, t1, t2, path_to_twinfer_fanout_output)
    for path, sim_type, rep_id in tasks
)
print(f"Done. {len(result_paths)} result file(s) available in {path_to_twinfer_fanout_output}")
