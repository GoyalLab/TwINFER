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
import json

#Path to TwINFER code repository
path_to_code_repo = "/home/gzu5140/Keerthana_b1042/TwINFER/code/TwINFER/"
path_to_simulation_data = "/home/gzu5140/Keerthana_b1042/TwINFER/simulation_data/twinfer_format/"
path_to_save_plot_data = "/home/gzu5140/Keerthana_b1042/TwINFER/analysis_data/boolode_sims/twinfer_inference/"
os.makedirs(path_to_save_plot_data, exist_ok=True)

# Calculation functions
sys.path.append(str(path_to_code_repo))
import importlib
from TwINFER_function_scripts import correlation_analysis_functions
from TwINFER_function_scripts import correlation_analysis_helpers
from TwINFER_function_scripts import infer_with_twinfer

importlib.reload(correlation_analysis_functions)
importlib.reload(correlation_analysis_helpers)
importlib.reload(infer_with_twinfer)

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

# --------------------------------------------------------------------------
# GSD/HSC (simulation_time=8, 800 steps, branch at step 400) share one
# (t1, t2); mCAD/VSC (simulation_time=5, 500 steps, branch at step 250)
# share the other. t1 is the first post-branch step SAVED IN THE SIM DATA
# (not the branch step itself -- at the branch step twin_A/twin_B are
# still identical by construction, giving zero-variance deltas and NaN
# twin correlations); t2 is the final saved step. Same t_early/t_final
# convention used in convert_twins_to_beeline.py's twin_paired scheme.
# Unlike that script, there's no proportional-equivalence judgment call
# needed here since infer_with_twinfer just needs t1/t2 to literally
# exist in the 'time_step' column -- it doesn't need them to be a
# specific fraction.
#
# n_cells/twin_simulation_time_after_division/twin_measurement_resolution/
# rows_to_use are only ever read inside infer_with_twinfer's
# `if match_sim_details:` block (n_cells is technically read once outside
# it too, but only ever *compared* inside it) -- since every call below
# passes match_sim_details=False, their exact values don't matter. Kept
# here anyway, set to the real values where known, for clarity/robustness
# if match_sim_details is ever flipped back on for a real check later.
NETWORK_TIMEPOINTS = {
    'GSD':  dict(t1=500, t2=799),
    'HSC':  dict(t1=500, t2=799),
    'mCAD': dict(t1=300, t2=499),
    'VSC':  dict(t1=300, t2=499),
}


def build_base_config(network):
    net_dir = Path(path_to_simulation_data, network)
    return {
        'n_cells': 6000,
        'twin_simulation_time_after_division': NETWORK_TIMEPOINTS[network]['t2'],
        'twin_measurement_resolution': 100,  # our TRAJECTORY_SAVE_STRIDE -- unused, match_sim_details=False
        'path_to_connectivity_matrix': str(net_dir / "interaction_matrix.txt"),
        'param_csv': str(net_dir / "param_placeholder.csv"),  # placeholder, not real params -- see boolode_to_twinfer_format.py
        'rows_to_use': [[0]],  # unused, match_sim_details=False
        'output_folder': path_to_save_plot_data,
        'log_file': os.path.join(path_to_save_plot_data, "logs", f"{network}.jsonl"),
        'type': network,
        'combinatorial_interaction_type': 'additive',
        'number_of_parallel_parameters': 1,
        'number_of_cores_per_parameter': 15,
        'log_pi_on': False,
    }


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


def build_simulation_record(path_to_simulation_file, sim_type, rep_id, base_config, t1, t2, output_path, gene_names):
    """
    Runs a simulation through TwINFER and saves the ENTIRE results dict
    (all matrices, gene classifications, thresholds, edges -- nothing
    dropped) to a single JSON file, plus returns the same record in memory.

    Same structure as infer_network_simulation_network_sweep.py's version,
    with two differences:
      1. check_for_steady_state=False AND match_sim_details=False here --
         our BoolODE-based simulation is a fundamentally different system
         from TwINFER's native stochastic promoter-state simulator (2
         variables/gene vs. 4), so the internal steady-state check and the
         file/base_config consistency asserts (which expect a complete
         fixed-resolution time grid and matching param-row metadata we
         don't have) are bypassed rather than faked. We've already
         independently validated convergence for these networks (e.g. the
         2x-simulation-time checks done during the twin-similarity
         analysis), so skipping the built-in check is not skipping
         validation entirely, just this particular redundant one.
      2. gene_list_given=gene_names is REQUIRED here, unlike the reference
         script: our simulation CSVs have real gene name columns (e.g.
         'UGR', 'GATA4'), not the network_sweep data's 'gene_1','gene_2'
         convention infer_with_twinfer defaults to when gene_list_given is
         omitted. Passing the wrong (default) gene list would look up
         nonexistent columns and fail.

    Parameters
    ----------
    path_to_simulation_file : str or list
        Path or list of paths to simulation file(s).
    sim_type : str
        Label describing the simulation condition (here: the network name).
    rep_id : int
        Replicate identifier for this simulation run.
    base_config : dict
        Configuration dictionary passed directly to TwINFER.
    t1, t2 : int or float
        Time points used for extracting twin measurements.
    output_path : str
        Directory to save the JSON file in.
    gene_names : list[str]
        Gene order matching interaction_matrix.txt's rows/columns (from
        gene_order.txt) and the simulation file's gene columns.

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
        match_sim_details=False,
        gene_list_given=gene_names,
        show_scrambled_distribution_gene_correlation=False,
        plot_correlation_matrices_as_heatmap=False,
        return_gene_corr_thresholds=False,
        seed=101010,
        n_cores=15,
        z_score_threshold_two_states=12,
        infer_direction_for_which_edges="all-edges",
        ranked_list = True

    )

    n_genes = len(gene_names)
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


def find_available_replicates(network):
    reps = []
    net_dir = Path(path_to_simulation_data, network)
    for f in sorted(net_dir.glob("replicate_*_simulation.csv")):
        m = re.match(r"replicate_(\d+)_simulation\.csv$", f.name)
        if m:
            reps.append(int(m.group(1)))
    return sorted(reps)


# ---------- Collect one task per (network, replicate) ----------
NETWORKS = ['GSD', 'HSC', 'mCAD', 'VSC']
tasks = []

for network in NETWORKS:
    net_dir = Path(path_to_simulation_data, network)
    gene_order_path = net_dir / "gene_order.txt"
    if not gene_order_path.exists():
        print(f"[skip] {network}: no gene_order.txt found at {gene_order_path}")
        continue
    gene_names = gene_order_path.read_text().split()

    base_config = build_base_config(network)
    t1 = NETWORK_TIMEPOINTS[network]['t1']
    t2 = NETWORK_TIMEPOINTS[network]['t2']

    reps = find_available_replicates(network)
    for rep_id in reps:
        sim_path = str(net_dir / f"replicate_{rep_id}_simulation.csv")
        tasks.append((sim_path, network, rep_id, base_config, t1, t2, gene_names))

print(f"Collected {len(tasks)} (network, replicate) tasks across {len(NETWORKS)} networks.")

# ---------- Run all tasks in parallel and save the results in plot_data folder ----------
print("Starting parallel processing...")

results_list = Parallel(n_jobs=4, backend="loky")(
    delayed(build_simulation_record)(path, sim_type, rep_id, base_config, t1, t2, path_to_save_plot_data, gene_names)
    for path, sim_type, rep_id, base_config, t1, t2, gene_names in tasks
)

print("\n\U0001F4BE All records saved as individual JSON files.")
