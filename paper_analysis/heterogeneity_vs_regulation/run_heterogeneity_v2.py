"""
Standalone SLURM-batch driver for the heterogeneity_vs_regulation Figure-2
d-vs-null pipeline (see analysis_v2.ipynb -- this script mirrors that
notebook's process_replicate() logic verbatim, so it's driven by SLURM the
same way rerun_twinfer_150.py / run_twinfer_150_rerun.sh already drive the
synthetic_network_analysis benchmark).

Output: one {sim_type}_rep_{rep_id}.json per task, under --output-dir.
Skips (does not recompute) any task whose output file already exists, so
this script is safe to resume after a partial/timed-out run.
"""
import argparse
import glob
import json
import os
import re
import time

import numpy as np
import pandas as pd
import numba
from threadpoolctl import threadpool_limits

from twinfer.inference.infer import infer_with_twinfer
from twinfer.inference.correlation_functions import (
    read_input_matrix,
    split_and_merge_simulations,
    assign_twin_id,
    calculate_twin_random_correlations,
    identify_reg_if_multiple_states,
)
from twinfer.utils.paths import get_repo_root

T1, T2 = 1, 20

REPO_ROOT = get_repo_root()
PATH_TO_INPUT_DATA = f"{REPO_ROOT}/simulation_example_input_data"

BASE_CONFIG = {
    'n_cells': 6000,
    'simulation_time_before_division': 1000,
    'twin_simulation_time_after_division': 48,
    'twin_measurement_resolution': 1,
    "path_to_connectivity_matrix": f"{PATH_TO_INPUT_DATA}/connectivity_matrix_A_to_B.txt",
    "param_csv": f"{PATH_TO_INPUT_DATA}/median_parameter.csv",
    "rows_to_use": [[0, 1]],
}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _load_simulation(path_to_simulation_file):
    """Load a single simulation CSV, or merge a list of them (2-state case)."""
    if isinstance(path_to_simulation_file, str):
        return pd.read_csv(path_to_simulation_file)
    return split_and_merge_simulations(path_to_simulation_file)


def process_simulation_through_infer_with_twinfer(simulation, sim_type, rep_id, base_config, t1, t2, gene_names=None, n_cores=1):
    """Verbatim port of analysis_v2.ipynb's function of the same name."""
    results = infer_with_twinfer(
        None,
        data=simulation,
        is_simulation_data=True,
        merge_to_multiple_states=False,
        base_config=base_config,
        t1=t1,
        t2=t2,
        corr_threshold_cross_correlation=0.0421,
        z_score_threshold_two_states=4.51,
        use_scramble_cross_correlation=False,
        check_for_steady_state=False,
        match_sim_details=False,
        seed=101010,
        plot=False,
        verbose=False,
        ranked_list=False,
        return_diagnostics=True,
        n_cores=n_cores,
    )

    null_mean, null_std = results["diagnostics"]["step1"]["null_stats"][("gene_1", "gene_2")]
    z_critical = results["diagnostics"]["step1"]["z_critical"]
    gene_gene_threshold = null_mean + z_critical * null_std

    # Step 2 (twin-vs-random gate) already computes the null distribution's
    # mean/std/median and the twin z-score for (gene_1, gene_2) internally --
    # pull it out here instead of re-deriving it later by re-reading every
    # raw simulation CSV a second time (the old process_all_reps_with_twins
    # path).
    step2_pair = ("gene_1", "gene_2")
    step2_null_stats = results["diagnostics"]["step2"]["null_stats"].get(step2_pair)
    step2_z_score = results["diagnostics"]["step2"]["z_scores"].get(step2_pair)
    if step2_null_stats is not None:
        step2_null_mean, step2_null_std, step2_null_median = step2_null_stats
    else:
        step2_null_mean = step2_null_std = step2_null_median = None

    record = {
        "sim_type": sim_type,
        "rep_id": rep_id,
        "analysis_key": f"{sim_type}_rep_{rep_id}",
        "gene_gene_threshold": gene_gene_threshold,
        "step2_null_mean": step2_null_mean,
        "step2_null_std": step2_null_std,
        "step2_null_median": step2_null_median,
        "step2_z_score": step2_z_score,
    }

    def matrix_to_gene_pair_columns(matrix, gene_names, prefix=""):
        columns = {}
        if matrix is not None:
            matrix_array = np.array(matrix)
            n_genes = len(gene_names) if gene_names else matrix_array.shape[0]
            if gene_names is None:
                gene_names = [f"g{i+1}" for i in range(n_genes)]
            for i in range(matrix_array.shape[0]):
                for j in range(matrix_array.shape[1]):
                    key = f"{prefix}{gene_names[i]}_{gene_names[j]}"
                    columns[key] = matrix_array[i, j]
        return columns

    correlations = results["correlations"]
    matrix_data = {
        "directional": correlations["direction"],
        "gene_gene": correlations["gene_t1"],
        "random_t1": correlations["random_delta_t1"],
        "random": correlations["random_delta_t2"],
        "twin_t1": correlations["twin_delta_t1"],
        "twin_t2": correlations["twin_delta_t2"],
    }
    for mtype, matrix in matrix_data.items():
        record.update(matrix_to_gene_pair_columns(matrix, gene_names, f"{mtype}_"))
    return record


def compute_d_vs_null_for_replicate(simulation, sim_type, rep_id, base_config, t1, t2, gene_pair=("gene_1", "gene_2"), alpha=0.01, n_shuffles=None, unit="clone", seed=101010, n_cores=1):
    """Verbatim port of analysis_v2.ipynb's function of the same name."""
    n_genes, _ = read_input_matrix(base_config["path_to_connectivity_matrix"])
    gene_list = [f"gene_{i}" for i in np.arange(1, n_genes + 1)]

    rng = np.random.default_rng(seed)
    clone_ids = simulation["clone_id"].drop_duplicates().to_numpy()
    clone_ids_shuffled = rng.permutation(clone_ids)
    n1 = n2 = len(clone_ids_shuffled) // 4
    t1_clones = clone_ids_shuffled[:n1]
    t2_clones = clone_ids_shuffled[n1:n1 + n2]

    t1_twins_raw = simulation[simulation["clone_id"].isin(t1_clones) & (simulation["time_step"] == t1)].copy()
    t2_twins_raw = simulation[simulation["clone_id"].isin(t2_clones) & (simulation["time_step"] == t2)].copy()

    t1_twins = assign_twin_id(t1_twins_raw).reset_index(drop=True)
    t2_twins = assign_twin_id(t2_twins_raw).reset_index(drop=True)

    twin_delta_t1, random_delta_t1 = calculate_twin_random_correlations(None, t1_twins, gene_list, unit=unit)
    twin_delta_t2, random_delta_t2 = calculate_twin_random_correlations(None, t2_twins, gene_list, unit=unit)

    _, _, stage3_details = identify_reg_if_multiple_states(
        twin_delta_t1, twin_delta_t2, random_delta_t1, random_delta_t2,
        multiple_states_gene_pairs=[gene_pair],
        gene_list=gene_list,
        t1_twins=t1_twins,
        t2_twins=t2_twins,
        alpha=alpha,
        n_shuffles=n_shuffles,
        unit=unit,
        n_cores_to_use=n_cores,
    )
    details = stage3_details[tuple(gene_pair)]
    return {
        "sim_type": sim_type,
        "rep_id": rep_id,
        "analysis_key": f"{sim_type}_rep_{rep_id}",
        "d_obs": details["d"],
        "null_mean": details["null_mean"],
        "null_std": details["null_std"],
        "z_d": details["z_d"],
    }


def process_replicate(path_to_simulation_file, sim_type, rep_id, base_config, t1, t2, n_cores=1):
    """Verbatim port of analysis_v2.ipynb's function of the same name."""
    numba.set_num_threads(1)
    with threadpool_limits(limits=1):
        simulation = _load_simulation(path_to_simulation_file)
        matrix_record = process_simulation_through_infer_with_twinfer(
            simulation, sim_type, rep_id, base_config, t1, t2, n_cores=n_cores,
        )
        d_vs_null_record = compute_d_vs_null_for_replicate(
            simulation, sim_type, rep_id, base_config, t1, t2, n_cores=n_cores,
        )
    return matrix_record, d_vs_null_record


def collect_tasks(path_to_simulation_data):
    """Verbatim port of analysis_v2.ipynb's task-collection cell."""
    tasks = []

    sim_folder = f"{path_to_simulation_data}/A_to_B/"
    pattern = os.path.join(sim_folder, "df_rows_0_1_*_ncells_6000_A_to_B_rep_*.csv")
    for f in sorted(glob.glob(pattern)):
        rep_id = re.search(r"_rep_(\d+)", os.path.basename(f)).group(1)
        tasks.append((f, "A_to_B", rep_id))

    sim_folder = f"{path_to_simulation_data}/A_B/"
    pattern = os.path.join(sim_folder, "df_rows_0_1_*_ncells_6000_A_B_rep*.csv")
    for f in sorted(glob.glob(pattern)):
        rep_id = re.search(r"_rep_(\d+)", os.path.basename(f)).group(1)
        tasks.append((f, "A_B", rep_id))

    for name, folder_hi, folder_lo in [
        ("A_to_B_2_states", "A_to_B_high_k_on", "A_to_B_low_k_on"),
        ("A_B_2_states", "A_B_high_k_on", "A_B_low_k_on"),
    ]:
        sim_folder_1 = f"{path_to_simulation_data}/{folder_hi}/"
        sim_folder_2 = f"{path_to_simulation_data}/{folder_lo}/"
        files_1 = [f for f in glob.glob(os.path.join(sim_folder_1, "*.csv")) if os.path.basename(f).startswith('df_')]
        files_2 = [f for f in glob.glob(os.path.join(sim_folder_2, "*.csv")) if os.path.basename(f).startswith('df_')]

        pairs = {}
        for f in files_1:
            m = re.search(r"rep_(\d+)", os.path.basename(f))
            if m:
                pairs.setdefault((name, m.group(1)), {'high': None, 'low': None})['high'] = f
        for f in files_2:
            m = re.search(r"rep_(\d+)", os.path.basename(f))
            if m:
                pairs.setdefault((name, m.group(1)), {'high': None, 'low': None})['low'] = f

        for (sim_type, rep_id), pair in pairs.items():
            if pair['high'] and pair['low']:
                tasks.append(([pair['high'], pair['low']], sim_type, rep_id))

    return tasks


def run_one(path, sim_type, rep_id, output_dir, n_cores):
    out_path = os.path.join(output_dir, f"{sim_type}_rep_{rep_id}.json")
    if os.path.exists(out_path):
        return out_path, "skipped (already exists)"

    matrix_record, d_vs_null_record = process_replicate(path, sim_type, rep_id, BASE_CONFIG, T1, T2, n_cores=n_cores)
    record = {"matrix_record": matrix_record, "d_vs_null_record": d_vs_null_record}
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(record, f, cls=NumpyEncoder, indent=2)
    return out_path, "computed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cores", type=int, default=1)
    parser.add_argument("--sim-folder", type=str, default="/projects/b1255/yscher/Transcriptomic Distance/simulation_data/figure_2_simulations_1000/")
    parser.add_argument("--output-dir", type=str, default=None,
                         help="required unless --list-tasks is given")
    parser.add_argument("--only", type=str, default=None,
                         help="comma-separated sim_type:rep_id to run (for a quick test); default = all tasks")
    parser.add_argument("--list-tasks", action="store_true",
                         help="print sim_type:rep_id for every available task, one per line, then exit "
                              "(used by the SLURM wrapper to build batches -- honors --limit)")
    parser.add_argument("--limit", type=int, default=None,
                         help="with --list-tasks, print only the first N task ids")
    args = parser.parse_args()

    if not args.list_tasks and not args.output_dir:
        parser.error("--output-dir is required unless --list-tasks is given")

    tasks = collect_tasks(args.sim_folder)

    if args.list_tasks:
        ids = [f"{sim_type}:{rep_id}" for _, sim_type, rep_id in tasks]
        if args.limit is not None:
            ids = ids[:args.limit]
        print("\n".join(ids))
        raise SystemExit(0)

    if args.only:
        wanted = set(args.only.split(","))
        tasks = [t for t in tasks if f"{t[1]}:{t[2]}" in wanted]

    print(f"Running {len(tasks)} task(s), output -> {args.output_dir}", flush=True)
    for i, (path, sim_type, rep_id) in enumerate(tasks):
        t0 = time.time()
        out_path, status = run_one(path, sim_type, rep_id, args.output_dir, args.n_cores)
        print(f"[{i+1}/{len(tasks)}] {sim_type} rep {rep_id}: {status} ({time.time()-t0:.1f}s) -> {out_path}", flush=True)
