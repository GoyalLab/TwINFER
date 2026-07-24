import pandas as pd
import numpy as np
import os
from pathlib import Path
from scipy.stats import spearmanr, rankdata
from joblib import Parallel, delayed
import warnings
import gc
import argparse
from tqdm.auto import tqdm
import glob
import matplotlib.pyplot as plt
import networkx as nx
import re
import sys

#Path to TwINFER code repository
path_to_code_repo = "/home/gzu5140/Keerthana_b1042/TwINFER/code/TwINFER/"
path_to_simulation_data = "/home/gzu5140/Keerthana_b1042/TwINFER/simulation_data/network_sweep_final/"
path_to_save_plot_data = "/home/gzu5140/Keerthana_b1042/TwINFER/analysis_data/network_sweep_final/network_inference/"
os.makedirs(path_to_save_plot_data, exist_ok = True)
print(path_to_save_plot_data)
#Common path to data files
path_to_input_data = f"{path_to_code_repo}/simulation_example_input_data/"
path_to_output_folder = f"{path_to_code_repo}/simulation_example_output_data/"
# Note: This configuration need not match the exact simulation parameters.
# However, ensure the following requirements are met:
# 1. The connectivity matrix has the same number of genes as the simulations being analyzed
#    (multiple base_configs can be created if needed for analyzing networks with different number of genes)
# 2. The twin simulation duration matches the time specified in base_config
# 3. The number of twin pairs in the simulation equals n_cell (the number of parent cells)

base_configs = {
        'n_cells': 6000,
        'simulation_time_before_division': 6000,
        'twin_simulation_time_after_division': 48,
        'twin_measurement_resolution': 1,
        "path_to_connectivity_matrix": "/home/gzu5140/Keerthana_b1042/TwINFER/input_data/network_sweep/grn_n6_e5_pos50_density_rep0.txt", #path to the connectivity matrix specifying the GRN to simulate
        "param_csv": f"/home/gzu5140/Keerthana_b1042/TwINFER/input_data/network_sweep/parameters.csv", #Path to the parameters for all genes and interaction terms
        "rows_to_use": [[0]*6],
        "output_folder": f"{path_to_output_folder}",
        "log_file": f"{path_to_output_folder}/logs/HSC.jsonl",
        "type": "HSC_balanced",
        "combinatorial_interaction_type": "additive",
        "number_of_parallel_parameters": 1,
        "number_of_cores_per_parameter": 56,
        "log_pi_on": False,
        "ranked_list": True,
    }
#Default settings for when the two samples are measured
t1 = 1
t2 = 20

# Calculation functions
import sys
sys.path.append(str(path_to_code_repo))
import importlib
from TwINFER_function_scripts import correlation_analysis_functions
from TwINFER_function_scripts import correlation_analysis_helpers
from TwINFER_function_scripts import infer_with_twinfer

importlib.reload(correlation_analysis_functions)
importlib.reload(correlation_analysis_helpers)
importlib.reload(infer_with_twinfer)

# Helper functions
from TwINFER_function_scripts.correlation_analysis_helpers import (
    read_input_matrix,
    split_and_merge_simulations
)
from TwINFER_function_scripts.correlation_analysis_functions import (
    generate_random_shuffle

)
from TwINFER_function_scripts.infer_with_twinfer import (
    infer_with_twinfer
)

import os
import json
import numpy as np
import pandas as pd


class NumpyEncoder(json.JSONEncoder):
    """
    JSON encoder that handles numpy scalar and array types, which the
    standard json module cannot serialize natively (np.int64, np.float64,
    np.ndarray all raise TypeError otherwise).
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def make_json_safe(obj):
    """
    Recursively converts an arbitrary nested object into a JSON-serializable
    structure. Must be recursive: results dicts from infer_with_twinfer
    contain DataFrames, sets, and dicts with tuple keys, sometimes nested
    inside other dicts/lists (e.g. gene_lists is a dict of lists of tuples),
    so a single top-level type check is not sufficient.
 
    Args:
        obj: Arbitrary object, typically a value (or the full dict) returned
            by infer_with_twinfer.
 
    Returns:
        A structure containing only JSON-native types.
    """
    if isinstance(obj, pd.DataFrame):
        return {
            "__type__": "DataFrame",
            "index": [str(i) for i in obj.index.tolist()],
            "columns": [str(c) for c in obj.columns.tolist()],
            "data": obj.values.tolist(),
        }
    if isinstance(obj, pd.Series):
        return {
            "__type__": "Series",
            "index": [str(i) for i in obj.index.tolist()],
            "data": obj.values.tolist(),
        }
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
 
 
def build_simulation_record(path_to_simulation_file, sim_type, rep_id, base_config, t1, t2, output_path, gene_names=None):
    """
    Runs a simulation through TwINFER and saves the ENTIRE results dict
    (all matrices, gene classifications, thresholds, edges -- nothing
    dropped) to a single JSON file, plus returns the same record in memory.
 
    Matrices are NOT flattened into gene_i_gene_j columns, since flattening
    only produces a consistent schema when every record has the same number
    and order of genes. Keeping matrices nested (with their own index/columns
    preserved) lets each record stand alone with its own gene count and gene
    names, which is required when different simulations have different
    numbers of genes.
 
    Parameters
    ----------
    path_to_simulation_file : str or list
        Path or list of paths to simulation file(s).
    sim_type : str
        Label describing the simulation condition (e.g., "A_to_B_low_kon").
    rep_id : int
        Replicate identifier for this simulation run.
    base_config : dict
        Configuration dictionary passed directly to TwINFER.
    t1, t2 : int or float
        Time points used for extracting twin measurements.
    output_path : str
        Directory to save the JSON file in.
    gene_names : list[str], optional
        Names of genes for this simulation. If None, generic names
        (g1, g2, ...) are generated based on matrix size.
 
    Returns
    -------
    record : dict
        Dictionary containing simulation metadata plus every key returned
        by infer_with_twinfer (DataFrames replaced with a JSON-safe nested
        form; this is the SAME dict structure written to disk).
    """
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
        z_score_threshold_two_states=12,
        infer_direction_for_which_edges="all-edges",
        ranked_list=True,

    )
 
    # infer gene count/names from whichever DataFrame is present in results
    n_genes = None
    for v in results.values():
        if isinstance(v, pd.DataFrame):
            n_genes = v.shape[0]
            break
    if gene_names is None and n_genes is not None:
        gene_names = [f"g{i+1}" for i in range(n_genes)]
 
    analysis_key = f"{sim_type}_rep_{rep_id}"
 
    record = {
        "sim_type": sim_type,
        "rep_id": rep_id,
        "analysis_key": analysis_key,
        "gene_names": gene_names,
        "n_genes": n_genes,
        **make_json_safe(results),  # every key infer_with_twinfer returned, sanitized
    }
 
    os.makedirs(output_path, exist_ok=True)
    f_result_path = os.path.join(output_path, f"{analysis_key}_all_results.json")
    with open(f_result_path, "w") as f:
        json.dump(record, f, cls=NumpyEncoder, indent=2)
 
    print(f"✅ Saved record to {f_result_path}")
    return record

import re

def extract_run_id(filename):
    """
    Extracts the '{config_index}_{replicate}' identifier from a simulation
    output filename of the form
    '..._{config_index}_{replicate}_{8charhex}.csv'.

    Args:
        filename (str): Filename or full path.

    Returns:
        str or None: The extracted ID (e.g. '1_0'), or None if the pattern
                     doesn't match.
    """
    fname = os.path.basename(filename)
    match = re.search(r'_(\d+_\d+)_[0-9a-fA-F]{8}\.csv$', fname)
    return match.group(1) if match else None

#Collect all the simulations needed for figure 2: The 4 scenarios.
tasks = []

sim_folder = f"{path_to_simulation_data}"
pattern = os.path.join(sim_folder, "df_grn_n6_*_ncells_6000_grn_n6_*.csv")
files = sorted(glob.glob(pattern), reverse=True)

# Extract the variant string (everything between "grn_n6_" and "_ncells_6000")
# e.g. "e5_pos50_density_rep0_rep0" -> variant="e5_pos50_density_rep0", tech_rep="rep0"
variant_re = re.compile(r"df_grn_n6_(.+?)_ncells_6000_grn_n6_")

tasks = []
skipped = []

for f in files:
    fname = os.path.basename(f)
    match = variant_re.search(fname)
    if not match:
        skipped.append(fname)
        continue
    variant_full = match.group(1)  # e.g. "e5_pos50_density_rep0_rep0"
    rep_id = extract_run_id(fname)
    tasks.append((f, variant_full, rep_id, base_configs))

print(f"Collected {len(tasks)} grn tasks across all variants.")
if skipped:
    print(f"[warning] {len(skipped)} files didn't match the expected pattern, e.g.: {skipped[:3]}")

# # # ---------- Run all tasks in parallel and save the results in plot_data folder
print("Starting parallel processing...")

results_list = Parallel(n_jobs=4, backend="loky")(
    delayed(build_simulation_record)(path, sim_type, rep_id, base_config, t1, t2, path_to_save_plot_data)
    for path, sim_type, rep_id,base_config in tasks
)

# # ---------- Save results to CSV ----------
print("\n💾 Saving results to JSON files...")
