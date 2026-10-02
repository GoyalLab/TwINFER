"""
Standalone SLURM-runnable version of analysis.ipynb.

Same logic as the notebook, with two changes to make it finish on a single
many-core node instead of a laptop:
  1. Step 1 (process_simulation_through_infer_with_twinfer over `tasks`) uses
     joblib Parallel with n_jobs from --n_jobs / SLURM_CPUS_PER_TASK instead
     of the notebook's hardcoded n_jobs=2.
  2. Step 2 (process_all_reps_with_twins), which the notebook runs as a plain
     sequential for-loop over ~1000 reps per scenario, is parallelized
     the same way.
"""
import argparse
import glob
import os
import re
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

import twinfer  # noqa: F401
from twinfer.utils.paths import get_repo_root, stage_dir
from twinfer.inference.infer import infer_with_twinfer
from twinfer.inference.correlation_functions import (
    read_input_matrix,
    split_and_merge_simulations,
    generate_random_shuffle,
)


def default_n_jobs():
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_JOB_CPUS_PER_NODE"):
        val = os.environ.get(var)
        if val:
            try:
                return int(val.split("(")[0])
            except ValueError:
                pass
    return os.cpu_count() or 1


parser = argparse.ArgumentParser()
parser.add_argument("--n_jobs", type=int, default=default_n_jobs())
args = parser.parse_args()
N_JOBS = max(1, args.n_jobs)
print(f"Using n_jobs={N_JOBS}", flush=True)

path_to_code_repo = get_repo_root()
path_to_simulation_data = "/projects/b1255/yscher/Transcriptomic Distance/simulation_data/figure_2_simulations_1000/"
path_to_save_plot_data = stage_dir("heterogeneity_vs_regulation_1k_merge_tp", "analysis")
print("path_to_save_plot_data:", path_to_save_plot_data, flush=True)

path_to_input_data = f"{path_to_code_repo}/simulation_example_input_data/"
base_config = {
    "n_cells": 6000,
    "simulation_time_before_division": 1000,
    "twin_simulation_time_after_division": 48,
    "twin_measurement_resolution": 1,
    "path_to_connectivity_matrix": f"{path_to_input_data}/connectivity_matrix_A_to_B.txt",
    "param_csv": f"{path_to_input_data}/median_parameter.csv",
    "rows_to_use": [[0, 1]],
    "output_folder": f"{path_to_simulation_data}",
    "log_file": f"{path_to_code_repo}/example_simulation_output/example_log.jsonl",
    "type": "A_to_B",
    "number_of_parallel_parameters": 1,
    "number_of_cores_per_parameter": 10,
}
t1 = 1
t2 = 20


# --------------------------------------------------------------------------
# Helper functions (identical logic to the notebook)
# --------------------------------------------------------------------------
def get_many_random_pair_corr(
    path_to_simulation_file,
    base_config,
    t1,
    t2,
    threshold_gene_gene_corr=0.04,
    check_for_steady_state=True,
    plot_correlation_matrices_as_heatmap=True,
    have_any_output=True,
    random_seed=42,
):
    try:
        if isinstance(path_to_simulation_file, str):
            simulation = pd.read_csv(path_to_simulation_file)
        elif isinstance(path_to_simulation_file, (list, tuple)):
            if len(path_to_simulation_file) == 1:
                simulation = pd.read_csv(path_to_simulation_file[0])
            elif len(path_to_simulation_file) >= 2:
                simulation = split_and_merge_simulations(path_to_simulation_file)
            else:
                raise ValueError("List of simulation files is empty.")
        else:
            raise TypeError("path_to_simulation_file must be a string or list/tuple.")
    except Exception as e:
        raise RuntimeError(f"Error reading simulation file(s): {e}")

    path_to_connectivity_matrix = base_config["path_to_connectivity_matrix"]
    param_df = pd.read_csv(base_config["param_csv"], index_col=0)

    n_genes, interaction_matrix = read_input_matrix(path_to_connectivity_matrix)
    gene_list = [f"gene_{i}" for i in np.arange(1, n_genes + 1)]

    n_clones_simulation = simulation["clone_id"].nunique()

    np.random.seed(random_seed)
    clone_ids_shuffled = np.random.permutation(n_clones_simulation)

    n1 = n2 = n_clones_simulation // 4
    t1_clones = clone_ids_shuffled[:n1]
    t2_clones = clone_ids_shuffled[n1 : n1 + n2]
    across_t_clones = clone_ids_shuffled[n1 + n2 :]

    t1_twins = simulation[(simulation["clone_id"].isin(t1_clones)) & (simulation["time_step"] == t1)]
    t2_twins = simulation[(simulation["clone_id"].isin(t2_clones)) & (simulation["time_step"] == t2)]
    across_t_twin1 = simulation[
        (simulation["clone_id"].isin(across_t_clones))
        & (simulation["time_step"] == t1)
        & (simulation["replicate"] == 1)
    ]
    across_t_twin2 = simulation[
        (simulation["clone_id"].isin(across_t_clones))
        & (simulation["time_step"] == t2)
        & (simulation["replicate"] == 2)
    ]

    t1_twins = t1_twins.reset_index(drop=True)
    t2_twins = t2_twins.reset_index(drop=True)
    across_t_twin1 = across_t_twin1.reset_index(drop=True)
    across_t_twin2 = across_t_twin2.reset_index(drop=True)

    all_t1_measurements = pd.concat([t1_twins, across_t_twin1], ignore_index=True)

    scrambled_random_corr = generate_random_shuffle(
        all_t1_measurements,
        gene_list,
        n_shuffles=10000,
        random_state=42,
    )

    all_correlations = scrambled_random_corr[("gene_1", "gene_2")]

    random_stats = {
        "all_values": all_correlations.flatten(),
        "mean_per_pair": np.mean(all_correlations, axis=0),
        "std_per_pair": np.std(all_correlations, axis=0),
        "percentile_95": np.percentile(np.abs(all_correlations.flatten()), 95),
        "percentile_100": np.percentile(np.abs(all_correlations.flatten()), 100),
        "global_mean": np.mean(all_correlations),
        "global_std": np.std(all_correlations),
    }

    return random_stats


def process_simulation_through_infer_with_twinfer(path_to_simulation_file, sim_type, rep_id, base_config, t1, t2, gene_names=None):
    results = infer_with_twinfer(
        path_to_simulation_file,
        merge_to_multiple_states=True,
        base_config=base_config,
        t1=t1,
        t2=t2,
        # use_scramble=False,   [2026-09-30 ported to current twinfer.inference.infer: keyword no longer exists]
        # threshold_gene_gene_corr=0.0126,   [2026-09-30 ported to current twinfer.inference.infer: keyword no longer exists]
        corr_threshold_cross_correlation=0.0421,
        z_score_threshold_two_states=4.501,
        use_scramble_cross_correlation=False,
        # merge_time_points=True, #COMMENT: this has been turned on temporarily to validate if it works   [2026-09-30 ported to current twinfer.inference.infer: keyword no longer exists]
        check_for_steady_state=False,
        # show_scrambled_distribution_gene_correlation=True,   [2026-09-30 ported to current twinfer.inference.infer: keyword no longer exists]
        # plot_correlation_matrices_as_heatmap=False,   [2026-09-30 ported to current twinfer.inference.infer: keyword no longer exists]
        # return_gene_corr_thresholds=True,   [2026-09-30 ported to current twinfer.inference.infer: keyword no longer exists]
        return_diagnostics=True,   # [2026-09-30 added: provides diagnostics['step1'] (null SD per pair + z_critical), the replacement of the removed gene_corr_thresholds]
        match_sim_details=False,
        seed=101010,
    )

    # [2026-09-30 ported to the current nested result schema (owner confirmed the statistics are equivalent: the old twin/random pair correlation
    #  matrices are the new rho_Delta matrices -- the package keeps `calculate_twin_random_pair_correlations = calculate_twin_random_correlations` --, and the old
    #  scramble threshold on gene-gene rho is the alpha-based Step-1 critical value). Mapping old key -> new key:
    #   gene_corr_thresholds[("gene_1","gene_2")] -> diagnostics["step1"]["z_critical"] * diagnostics["step1"]["null_stats"][("gene_1","gene_2")][1] (null SD)
    #   direction_matrix                          -> correlations["direction"]
    #   pairwise_gene_gene_correlation_matrix     -> correlations["gene_t1"]
    #   random_pair_correlation_matrix_t2         -> correlations["random_delta_t2"]
    #   twin_pair_correlation_matrix_t1 / _t2     -> correlations["twin_delta_t1"] / ["twin_delta_t2"]  ]
    step1 = results["diagnostics"]["step1"]
    corr = results["correlations"]
    record = {
        "sim_type": sim_type,
        "rep_id": rep_id,
        "analysis_key": f"{sim_type}_rep_{rep_id}",
        "gene_gene_threshold": float(step1["z_critical"]) * float(step1["null_stats"][("gene_1", "gene_2")][1]),
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

    # [2026-09-30 commented out: old flat keys]
    #     matrix_data = {
    #         "directional": results["direction_matrix"],
    #         "gene_gene": results["pairwise_gene_gene_correlation_matrix"],
    #         "random": results["random_pair_correlation_matrix_t2"],
    #         "twin_t1": results["twin_pair_correlation_matrix_t1"],
    #         "twin_t2": results["twin_pair_correlation_matrix_t2"],
    #     }
    matrix_data = {
        "directional": corr["direction"],
        "gene_gene": corr["gene_t1"],
        "random": corr["random_delta_t2"],
        "twin_t1": corr["twin_delta_t1"],
        "twin_t2": corr["twin_delta_t2"],
    }

    for mtype, matrix in matrix_data.items():
        record.update(matrix_to_gene_pair_columns(matrix, gene_names, f"{mtype}_"))
    return record


def process_simulation_through_infer_with_twinfer_safe(path, sim_type, rep_id, base_config, t1, t2):
    try:
        return process_simulation_through_infer_with_twinfer(path, sim_type, rep_id, base_config, t1, t2)
    except Exception as e:
        print(f"Error processing {sim_type} rep {rep_id} ({path}): {e}", flush=True)
        return None


def save_results_to_csv(results_list, output_dir="results"):
    df = pd.DataFrame(results_list)

    os.makedirs(output_dir, exist_ok=True)
    matrix_types = ["directional", "gene_gene", "random", "twin_t1", "twin_t2"]

    for matrix_type in matrix_types:
        matrix_columns = [c for c in df.columns if c.startswith(f"{matrix_type}_")]

        if matrix_columns:
            metadata_cols = ["sim_type", "rep_id", "analysis_key"]
            subset_cols = metadata_cols + matrix_columns
            subset_df = df[subset_cols].copy()

            rename_dict = {col: col.replace(f"{matrix_type}_", "") for col in matrix_columns}
            subset_df.rename(columns=rename_dict, inplace=True)

            file_path = os.path.join(output_dir, f"{matrix_type}_matrix_results.csv")
            subset_df.to_csv(file_path, index=False)
            print(f"Saved {matrix_type} matrix to {file_path}", flush=True)
    return df


Z_STAR = 4.501  # match z_score_threshold_two_states used in process_simulation_through_infer_with_twinfer

def _process_one_rep(file_or_pair, rep_num, sim_type_name, base_config, t1, t2, twin_data_df, random_correlation_data):
    try:
        random_correlation = get_many_random_pair_corr(
            file_or_pair, base_config, t1, t2,
            check_for_steady_state=False, have_any_output=False,
            plot_correlation_matrices_as_heatmap=False,
        )["all_values"]

        median_corr = np.median(random_correlation)

        twin_row = twin_data_df[(twin_data_df["sim_type"] == sim_type_name) & (twin_data_df["rep_id"] == rep_num)]
        random_row = random_correlation_data[
            (random_correlation_data["sim_type"] == sim_type_name) & (random_correlation_data["rep_id"] == rep_num)
        ]
        if not twin_row.empty:
            twin_vals = twin_row["g2_g1"].values[0]
            random_vals = random_row["g2_g1"].values[0]
            random_mean = np.mean(random_correlation)
            random_std = np.std(random_correlation)
            z_score = (twin_vals - random_mean) / random_std
            z_score_rand = (random_vals - random_mean) / random_std

            # Two-sided: report BOTH edges of the calibrated threshold in
            # correlation units, so downstream plotting can pick whichever
            # edge is relevant to that group's sign, instead of baking in
            # a single hardcoded direction.
            z_threshold_upper = Z_STAR * random_std + random_mean
            z_threshold_lower = -Z_STAR * random_std + random_mean

            return median_corr, z_score, z_score_rand, (z_threshold_lower, z_threshold_upper)
        else:
            print(f"Warning: No twin data found for {sim_type_name} rep {rep_num}", flush=True)
            return median_corr, np.nan, None, None
    except Exception as e:
        print(f"Error processing {sim_type_name} rep {rep_num} ({file_or_pair}): {e}", flush=True)
        return None

def process_all_reps_with_twins(file_pattern_or_folders, base_config, t1, t2, twin_data_df, random_correlation_data, sim_type_name, is_two_state=False, n_jobs=1):
    work_items = []
    if is_two_state:
        high_k_files = glob.glob(f"{file_pattern_or_folders[0]}/df_rows_*_rep_*.csv")
        low_k_files = glob.glob(f"{file_pattern_or_folders[1]}/df_rows_*_rep_*.csv")

        for high_file in high_k_files:
            match = re.search(r"rep_(\d+)", high_file)
            if match is None:
                continue
            rep_num = int(match.group(1))
            matching_low_file = [f for f in low_k_files if re.search(rf"rep_{rep_num}\D", f)]
            if matching_low_file:
                work_items.append(([high_file, matching_low_file[0]], rep_num))
    else:
        files = glob.glob(file_pattern_or_folders)
        for file_path in files:
            rep_num = int(re.search(r"rep_(\d+)", file_path).group(1))
            work_items.append((file_path, rep_num))

    print(f"[{sim_type_name}] {len(work_items)} reps to process", flush=True)

    raw_results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_process_one_rep)(item, rep_num, sim_type_name, base_config, t1, t2, twin_data_df, random_correlation_data)
        for item, rep_num in work_items
    )

    all_medians, z_scores, z_score_rand_list, z_threshold_list = [], [], [], []
    for r in raw_results:
        if r is None:
            continue
        median_corr, z_score, z_score_rand, z_threshold = r
        all_medians.append(median_corr)
        z_scores.append(z_score)
        if z_score_rand is not None:
            z_score_rand_list.append(z_score_rand)
        if z_threshold is not None:
            z_threshold_list.append(z_threshold)

    return all_medians, z_scores, z_score_rand_list, z_threshold_list


# --------------------------------------------------------------------------
# Step 1: build task list (identical to notebook cell 12)
# --------------------------------------------------------------------------
def build_tasks():
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

    sim_folder_1 = f"{path_to_simulation_data}/A_to_B_high_k_on/"
    sim_folder_2 = f"{path_to_simulation_data}/A_to_B_low_k_on/"
    files_1 = [f for f in glob.glob(os.path.join(sim_folder_1, "df*.csv")) if os.path.basename(f).startswith("df_")]
    files_2 = [f for f in glob.glob(os.path.join(sim_folder_2, "df*.csv")) if os.path.basename(f).startswith("df_")]
    pairs = {}
    for f in files_1:
        rep_match = re.search(r"rep_(\d+)", os.path.basename(f))
        if rep_match:
            rep_id = rep_match.group(1)
            pairs.setdefault(("A_to_B_2_states", rep_id), {"high": None, "low": None})["high"] = f
    for f in files_2:
        rep_match = re.search(r"rep_(\d+)", os.path.basename(f))
        if rep_match:
            rep_id = rep_match.group(1)
            pairs.setdefault(("A_to_B_2_states", rep_id), {"high": None, "low": None})["low"] = f
    for (sim_type, rep_id), pair in pairs.items():
        if pair["high"] and pair["low"]:
            tasks.append(([pair["high"], pair["low"]], sim_type, rep_id))

    sim_folder_1 = f"{path_to_simulation_data}/A_B_high_k_on/"
    sim_folder_2 = f"{path_to_simulation_data}/A_B_low_k_on/"
    files_1 = [f for f in glob.glob(os.path.join(sim_folder_1, "*.csv")) if os.path.basename(f).startswith("df_")]
    files_2 = [f for f in glob.glob(os.path.join(sim_folder_2, "*.csv")) if os.path.basename(f).startswith("df_")]
    pairs = {}
    for f in files_1:
        rep_match = re.search(r"rep_(\d+)", os.path.basename(f))
        if rep_match:
            rep_id = rep_match.group(1)
            pairs.setdefault(("A_B_2_states", rep_id), {"high": None, "low": None})["high"] = f
    for f in files_2:
        rep_match = re.search(r"rep_(\d+)", os.path.basename(f))
        if rep_match:
            rep_id = rep_match.group(1)
            pairs.setdefault(("A_B_2_states", rep_id), {"high": None, "low": None})["low"] = f
    for (sim_type, rep_id), pair in pairs.items():
        if pair["high"] and pair["low"]:
            tasks.append(([pair["high"], pair["low"]], sim_type, rep_id))

    return tasks


def main():
    t_start = time.time()

    tasks = build_tasks()
    print(f"Total step-1 tasks: {len(tasks)}", flush=True)

    print("Starting parallel processing (step 1)...", flush=True)
    results_list = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(process_simulation_through_infer_with_twinfer_safe)(path, sim_type, rep_id, base_config, t1, t2)
        for path, sim_type, rep_id in tasks
    )
    results_list = [r for r in results_list if r is not None]

    print("Saving step-1 results to CSV files...", flush=True)
    df = save_results_to_csv(results_list, output_dir=path_to_save_plot_data)
    print(f"Step 1 done in {time.time() - t_start:.1f}s. Total simulations: {len(df)}", flush=True)
    print(f"Simulation types: {list(df['sim_type'].unique())}", flush=True)

    # ----------------------------------------------------------------------
    # Step 2: z-score analysis (notebook cells 15-17)
    # ----------------------------------------------------------------------
    t_step2 = time.time()
    random_correlation_path = f"{path_to_save_plot_data}/random_matrix_results.csv"
    twin_correlation_t1_path = f"{path_to_save_plot_data}/twin_t1_matrix_results.csv"
    random_correlation_data = pd.read_csv(random_correlation_path)
    twin_correlation_t1_data = pd.read_csv(twin_correlation_t1_path)

    A_to_B_medians, A_to_B_z_scores, A_to_B_z_rand_scores, A_to_B_z_threshold_list = process_all_reps_with_twins(
        f"{path_to_simulation_data}/A_to_B/df_rows_*_A_to_B_rep_*.csv",
        base_config, t1, t2, twin_correlation_t1_data, random_correlation_data, "A_to_B",
        is_two_state=False, n_jobs=N_JOBS,
    )

    A_B_2_states_medians, A_B_2_states_z_scores, A_B_2_states_z_rand_scores, A_B_2_states_z_threshold_list = process_all_reps_with_twins(
        [f"{path_to_simulation_data}/A_B_high_k_on", f"{path_to_simulation_data}/A_B_low_k_on"],
        base_config, t1, t2, twin_correlation_t1_data, random_correlation_data, "A_B_2_states",
        is_two_state=True, n_jobs=N_JOBS,
    )

    A_to_B_2_states_medians, A_to_B_2_states_z_scores, A_to_B_2_states_z_rand_scores, A_to_B_2_states_z_threshold_list = process_all_reps_with_twins(
        [f"{path_to_simulation_data}/A_to_B_high_k_on", f"{path_to_simulation_data}/A_to_B_low_k_on"],
        base_config, t1, t2, twin_correlation_t1_data, random_correlation_data, "A_to_B_2_states",
        is_two_state=True, n_jobs=N_JOBS,
    )
    print(f"Step 2 done in {time.time() - t_step2:.1f}s", flush=True)



    data_dict = {"network_type": [], "metric": [], "values": []}
    entries = [
        ("A_to_B", "medians", A_to_B_medians),
        ("A_to_B", "z_scores", A_to_B_z_scores),
        ("A_to_B", "z_rand_scores", A_to_B_z_rand_scores),
        ("A_to_B", "z_threshold_list", A_to_B_z_threshold_list),
        ("A_to_B_2_states", "medians", A_to_B_2_states_medians),
        ("A_to_B_2_states", "z_scores", A_to_B_2_states_z_scores),
        ("A_to_B_2_states", "z_rand_scores", A_to_B_2_states_z_rand_scores),
        ("A_to_B_2_states", "z_threshold_list", A_to_B_2_states_z_threshold_list),
        ("A_B_2_states", "medians", A_B_2_states_medians),
        ("A_B_2_states", "z_scores", A_B_2_states_z_scores),
        ("A_B_2_states", "z_rand_scores", A_B_2_states_z_rand_scores),
        ("A_B_2_states", "z_threshold_list", A_B_2_states_z_threshold_list),
    ]
    for network, metric, values in entries:
        for v in values:
            data_dict["network_type"].append(network)
            data_dict["metric"].append(metric)
            data_dict["values"].append(v)

    out_df = pd.DataFrame(data_dict)
    output_csv = f"{path_to_save_plot_data}/twins_random_zscore_summary.csv"
    out_df.to_csv(output_csv, index=False)
    print("Saved to:", output_csv, flush=True)
    print(f"Total wall time: {time.time() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
