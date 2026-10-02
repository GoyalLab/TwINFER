"""
Standalone SLURM-batch driver for the causal_direction_inference Figure-3
2-gene direction inference (6 conditions), mirroring
heterogeneity_vs_regulation/run_heterogeneity_v2.py's structure.

Per replicate:
  1. Run the main infer_with_twinfer matrices, with
     rescue_opposite_sign_pairs=True (see infer_with_twinfer_updated.py) --
     for the negative-feedback conditions (e.g. A_rep_B_B_to_A) where the
     same-time gene-gene correlation washes out to ~0 but the two directional
     cross-correlations are opposite-signed.
  2. Compute the direct across-time cross-correlation significance test for
     (gene_1, gene_2) / (gene_2, gene_1) via get_cross_correlations +
     identify_actual_directed_edges -- this REPLACES the old notebook's
     hand-rolled analyze_single_file/cross_correlation_data_across_replicates
     (single source of truth for the significance test, not a parallel
     reimplementation). Always computed regardless of Step 1's verdict, since
     the box-plot panel needs full per-replicate coverage.

Output: one {condition}_rep_{rep_id}.json per task, under --output-dir.
Skips (does not recompute) any task whose output file already exists.
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

# infer_with_twinfer_updated no longer exists in the current twinfer
# package; the current infer.py's infer_with_twinfer keeps the same
# rescue_opposite_sign_pairs / return_diagnostics kwargs this script uses.
from twinfer.inference.infer import infer_with_twinfer
from twinfer.inference.correlation_functions import (
    get_cross_correlations,
    identify_actual_directed_edges,
    _build_cross_time_twins,
    assign_twin_id,
    calculate_pairwise_gene_gene_correlation_matrix,
    check_gene_gene_correlation_threshold,
    calculate_twin_random_correlations,
    differentiate_single_state_reg_and_multiple_states,
    identify_reg_if_multiple_states,
    calculate_gated_regulation_statistic,
)

T1, T2 = 1, 20
SEED = 101010
GENE_LIST = ["gene_1", "gene_2"]
PAIR = ("gene_1", "gene_2")

# notebook condition name -> actual data folder (folder names changed at the
# new data location; see the migration plan for the full mapping table)
CONDITIONS = {
    "A_B": "A_B",
    "A_rep_B": "A_rep_B",
    "A_and_B_both_repress": "A_rep_B_B_rep_A",
    "A_rep_B_B_to_A": "A_rep_B_B_to_A",
    "A_to_B": "A_to_B",
    "A_and_B": "A_to_B_B_to_A",
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


def run_main_matrices(simulation, condition, rep_id, n_cores=1):
    """Part A item 2: the main infer_with_twinfer call, rescue enabled."""
    results = infer_with_twinfer(
        None,
        data=simulation,
        is_simulation_data=True,
        t1=T1,
        t2=T2,
        check_for_steady_state=False,
        match_sim_details=False,
        seed=SEED,
        rescue_opposite_sign_pairs=True,
        plot=False,
        verbose=False,
        ranked_list=False,
        return_diagnostics=True,
        n_cores=n_cores,
    )

    null_mean, null_std = results["diagnostics"]["step1"]["null_stats"][("gene_1", "gene_2")]
    z_critical = results["diagnostics"]["step1"]["z_critical"]
    gene_gene_threshold = null_mean + z_critical * null_std

    pair = ("gene_1", "gene_2")
    diag = results["diagnostics"]
    record = {
        "condition": condition,
        "rep_id": rep_id,
        "analysis_key": f"{condition}_rep_{rep_id}",
        "gene_gene_threshold": gene_gene_threshold,
        "no_regulation": [list(p) for p in results["classification"]["no_regulation"]],
        "directed_edges": [list(e) for e in results["directed_edges"]],
        "rescued_from_no_regulation": [list(p) for p in results["direction"]["rescued_from_no_regulation"]],
        # Step 1: gene-gene correlation z-score vs cell-scramble null
        "step1_z": diag["step1"]["z_scores"].get(pair),
        "step1_z_critical": z_critical,
        # Step 2: twin heterogeneity z-score and fixed-delta divergence z-score
        "step2_z_het": diag["step2"]["z_het"].get(pair),
        "step2_z_div": diag["step2"]["z_div"].get(pair),
        "step2_z_reg_gated": diag["step2"].get("z_reg_gated", {}).get(pair),
        # Step 4: signed cross-correlation z-score, one per ordered gene pair
        **{f"step4_z_{g1}_{g2}": z for (g1, g2), z in results["direction"]["z_scores"].items()},
    }

    correlations = results["correlations"]
    matrix_data = {
        "directional": correlations["direction"],
        "gene_gene": correlations["gene_t1"],
        "twin_t1": correlations["twin_delta_t1"],
        "twin_t2": correlations["twin_delta_t2"],
    }
    for mtype, matrix in matrix_data.items():
        record.update(matrix_to_gene_pair_columns(matrix, GENE_LIST, f"{mtype}_"))
    return record


def _pv(d, pair=PAIR):
    """Look up a (gene_1, gene_2)-keyed dict value, either orientation."""
    if d is None:
        return None
    for k in (pair, (pair[1], pair[0])):
        if k in d:
            return d[k]
    return None


def _partition(simulation, seed=SEED):
    """Reproduce infer.py's simulation clone split (infer.py's own clone
    partition): seed -> permute clones -> first 1/4 = t1 twins, second 1/4
    = t2 twins, second 1/2 = cross-time (rep-0 cell at t1 paired to rep-1
    cell at t2)."""
    rng = np.random.default_rng(seed)
    clone_ids = simulation["clone_id"].drop_duplicates().to_numpy()
    sh = rng.permutation(clone_ids)
    n = len(sh) // 4
    t1c, t2c, ac = sh[:n], sh[n:2 * n], sh[2 * n:]
    reps = sorted(simulation["replicate"].unique())

    t1_twins_raw = simulation[simulation["clone_id"].isin(t1c) & (simulation["time_step"] == T1)].copy()
    t2_twins_raw = simulation[simulation["clone_id"].isin(t2c) & (simulation["time_step"] == T2)].copy()
    ac_left = simulation[simulation["clone_id"].isin(ac) & (simulation["time_step"] == T1)
                          & (simulation["replicate"] == reps[0])].copy()
    ac_right = simulation[simulation["clone_id"].isin(ac) & (simulation["time_step"] == T2)
                           & (simulation["replicate"] == reps[1])].copy()

    return {
        "t1_twins": assign_twin_id(t1_twins_raw).reset_index(drop=True),
        "t2_twins": assign_twin_id(t2_twins_raw).reset_index(drop=True),
        "rho_t1": pd.concat([t1_twins_raw, ac_left], ignore_index=True),
        "rho_t2": pd.concat([t2_twins_raw, ac_right], ignore_index=True),
        "ac_left": ac_left.reset_index(drop=True),
        "ac_right": ac_right.reset_index(drop=True),
    }


def _forced_step1(rho_meas, base_seed, n_shuffles):
    mat = calculate_pairwise_gene_gene_correlation_matrix(rho_meas, GENE_LIST, use_clone=True)
    _, _, null_stats, z = check_gene_gene_correlation_threshold(
        rho_meas, mat, GENE_LIST, use_scramble=True, z_score_threshold=2.576,
        verbose=False, use_clone=True, n_shuffles=n_shuffles, base_seed=base_seed,
    )
    ns = null_stats.get(PAIR)
    return {"z": _pv(z), "rho": float(mat.loc[PAIR]),
            "null_mean": float(ns[0]) if ns else None,
            "null_std": float(ns[1]) if ns else None}


def _forced_step2(rho_meas, twins, rand_seed, div_seed, n_cores, n_shuffles):
    twin_mat, rand_mat = calculate_twin_random_correlations(
        rho_meas, twins, GENE_LIST, random_state=rand_seed, unit="clone")
    _, _, null_stats, z_het, div = differentiate_single_state_reg_and_multiple_states(
        rho_meas, [PAIR], twin_mat, rand_mat, GENE_LIST,
        z_score_threshold=1e9, verbose=False, unit="clone",
        n_shuffles=n_shuffles, n_cores_to_use=n_cores,
        divergence_random_state=div_seed, return_divergence_details=True,
    )
    dv = _pv(div) or {}
    return {"z_het": _pv(z_het), "z_div": dv.get("z_div")}, twin_mat, rand_mat


def _forced_step4(ac_left, ac_right, n_cores, n_shuffles):
    at1, at2 = _build_cross_time_twins(ac_left, ac_right)
    dpairs = [("gene_1", "gene_2"), ("gene_2", "gene_1"),
              ("gene_1", "gene_1"), ("gene_2", "gene_2")]
    dmat = get_cross_correlations(at1, at2, gene_pairs=dpairs, unit="clone")
    _, zc, _ = identify_actual_directed_edges(
        at1, at2, dmat, gene_pairs=dpairs, z_score_threshold=2.5,
        use_scramble=True, n_shuffles=n_shuffles, n_cores_to_use=n_cores,
        verbose=False, return_z_scores=True, return_rho_cross_null=True,
        prepare_rho_cross_null=True, unit="clone",
    )
    return {"forced_step4_z_1to2": zc.get(("gene_1", "gene_2")), "forced_step4_z_2to1": zc.get(("gene_2", "gene_1"))}


def run_forced_zscores(simulation, condition, rep_id, n_cores=1, n_shuffles=10000):
    """
    Faithful, ungated reproduction of infer.py's Steps 1-4 for the
    gene_1-gene_2 pair, evaluated at both t1 and t2 for Steps 1-2 and
    always for Step 4 -- unlike run_main_matrices' pipeline-derived
    step*_z fields (which are None/NaN whenever an earlier step's gate
    routes the pair away from a later step), every value here is computed
    unconditionally so no z-score is ever missing because "the step
    failed" (see [[project_drift_zscore_analysis]] for why the gated
    pipeline leaves later steps unset, and run_6scenario_zscores.py for
    the original version of this reproduction on other scenarios).
    """
    P = _partition(simulation)
    record = {"condition": condition, "rep_id": rep_id}

    s1a = _forced_step1(P["rho_t1"], SEED + 1, n_shuffles)
    s1b = _forced_step1(P["rho_t2"], SEED + 2, n_shuffles)
    record["forced_step1_z_t1"], record["forced_step1_z_t2"] = s1a["z"], s1b["z"]
    record["forced_rho_t1"], record["forced_rho_t2"] = s1a["rho"], s1b["rho"]

    s2a, twin_t1, rand_t1 = _forced_step2(P["rho_t1"], P["t1_twins"], SEED + 271829, SEED + 271832, n_cores, n_shuffles)
    s2b, twin_t2, rand_t2 = _forced_step2(P["rho_t2"], P["t2_twins"], SEED + 271830, SEED + 271833, n_cores, n_shuffles)
    record["forced_step2_z_het_t1"], record["forced_step2_z_het_t2"] = s2a["z_het"], s2b["z_het"]
    record["forced_z_div_t1"], record["forced_z_div_t2"] = s2a["z_div"], s2b["z_div"]

    g_t1 = calculate_gated_regulation_statistic(
        P["t1_twins"], PAIR[0], PAIR[1], s2a["z_het"], n_shuffles=n_shuffles, random_state=SEED + 271840)
    g_t2 = calculate_gated_regulation_statistic(
        P["t2_twins"], PAIR[0], PAIR[1], s2b["z_het"], n_shuffles=n_shuffles, random_state=SEED + 271841)
    record["forced_z_reg_gated_t1"], record["forced_z_reg_gated_t2"] = g_t1["z_reg_gated"], g_t2["z_reg_gated"]

    _, _, stage3 = identify_reg_if_multiple_states(
        twin_t1, twin_t2, rand_t1, rand_t2, [PAIR], GENE_LIST,
        P["t1_twins"], P["t2_twins"], t1_raw=P["rho_t1"], t2_raw=P["rho_t2"],
        alpha=0.01, n_shuffles=n_shuffles, shuffle_seed=SEED + 271831,
        unit="clone", n_cores_to_use=n_cores,
    )
    s3 = _pv(stage3) or {}
    record["forced_step3_z_d"] = s3.get("z_d")
    record["forced_step3_z_d_div"] = s3.get("z_d_div")

    record.update(_forced_step4(P["ac_left"], P["ac_right"], n_cores, n_shuffles))
    return record


def run_box_plot_significance_test(simulation, condition, rep_id, n_shuffles=10000):
    """
    Part A item 3: replaces the old hand-rolled analyze_single_file. Rebuilds
    the across-time twin frames the same way infer_with_twinfer_updated.py
    does internally (same seed), then calls get_cross_correlations +
    identify_actual_directed_edges directly for full per-replicate coverage
    regardless of Step 1's verdict (the box plot needs every replicate's
    observed correlations and null-derived thresholds, not just a
    significance call).
    """
    rng = np.random.default_rng(SEED)
    clone_ids = simulation["clone_id"].drop_duplicates().to_numpy()
    clone_ids_shuffled = rng.permutation(clone_ids)
    n1 = n2 = len(clone_ids_shuffled) // 4
    across_t_clones = clone_ids_shuffled[n1 + n2:]

    replicates = simulation["replicate"].drop_duplicates().sort_values().to_numpy()

    across_t_left_raw = simulation[
        simulation["clone_id"].isin(across_t_clones)
        & (simulation["time_step"] == T1)
        & (simulation["replicate"] == replicates[0])
    ].copy()
    across_t_right_raw = simulation[
        simulation["clone_id"].isin(across_t_clones)
        & (simulation["time_step"] == T2)
        & (simulation["replicate"] == replicates[1])
    ].copy()

    across_t_twin1, across_t_twin2 = _build_cross_time_twins(across_t_left_raw, across_t_right_raw)

    gene_pairs = [("gene_1", "gene_2"), ("gene_2", "gene_1")]
    direction_matrix = get_cross_correlations(across_t_twin1, across_t_twin2, gene_pairs=gene_pairs, unit="clone")

    _, _, rho_cross_null = identify_actual_directed_edges(
        across_t_twin1,
        across_t_twin2,
        direction_matrix,
        gene_pairs=gene_pairs,
        z_score_threshold=2.5,
        use_scramble=True,
        n_shuffles=n_shuffles,
        n_cores_to_use=1,
        verbose=False,
        base_seed=SEED,
        return_z_scores=True,
        return_rho_cross_null=True,
    )

    null_12 = np.asarray(rho_cross_null[("gene_1", "gene_2")]["null_values"], dtype=float)
    null_21 = np.asarray(rho_cross_null[("gene_2", "gene_1")]["null_values"], dtype=float)
    gene_1_to_2 = float(rho_cross_null[("gene_1", "gene_2")]["observed_rho_cross"])
    gene_2_to_1 = float(rho_cross_null[("gene_2", "gene_1")]["observed_rho_cross"])

    # Two-tailed empirical p-value, matching the paper's stated two-tailed
    # p=0.02 threshold and the old analyze_single_file's exact computation.
    p_plus_12 = float(np.mean(null_12 >= gene_1_to_2))
    p_minus_12 = float(np.mean(null_12 <= gene_1_to_2))
    p_value_12 = min(2 * p_plus_12, 2 * p_minus_12, 1.0)

    p_plus_21 = float(np.mean(null_21 >= gene_2_to_1))
    p_minus_21 = float(np.mean(null_21 <= gene_2_to_1))
    p_value_21 = min(2 * p_plus_21, 2 * p_minus_21, 1.0)

    threshold_12 = float(max(abs(np.percentile(null_12, 1)), abs(np.percentile(null_12, 99))))
    threshold_21 = float(max(abs(np.percentile(null_21, 1)), abs(np.percentile(null_21, 99))))
    combined_null = np.concatenate([null_12, null_21])
    threshold_combined = float(max(abs(np.percentile(combined_null, 1)), abs(np.percentile(combined_null, 99))))

    return {
        "condition": condition,
        "rep_id": rep_id,
        "gene_1_to_gene_2": gene_1_to_2,
        "gene_2_to_gene_1": gene_2_to_1,
        "pvalue_12": p_value_12,
        "pvalue_21": p_value_21,
        "threshold_12": threshold_12,
        "threshold_21": threshold_21,
        "threshold_combined": threshold_combined,
    }


def process_replicate(path_to_simulation_file, condition, rep_id, n_cores=1, n_shuffles=10000):
    numba.set_num_threads(1)
    with threadpool_limits(limits=1):
        simulation = pd.read_csv(path_to_simulation_file)
        matrix_record = run_main_matrices(simulation, condition, rep_id, n_cores=n_cores)
        forced_record = run_forced_zscores(simulation, condition, rep_id, n_cores=n_cores, n_shuffles=n_shuffles)
        matrix_record.update({k: v for k, v in forced_record.items() if k not in ("condition", "rep_id")})
        box_plot_record = run_box_plot_significance_test(simulation, condition, rep_id, n_shuffles=n_shuffles)
    return matrix_record, box_plot_record


def collect_tasks(sim_folder):
    tasks = []
    for condition, folder in CONDITIONS.items():
        folder_path = os.path.join(sim_folder, folder)
        if not os.path.isdir(folder_path):
            continue
        # 4 of 6 condition folders contain a full parallel set of
        # "test_df_..." files -- exact duplicates of the real "df_..."
        # files, not distinct replicates. Excluding by prefix (matching the
        # "df_" convention used elsewhere in this codebase) avoids both
        # double-counting replicates and rep_id collisions between a real
        # file and its test_ duplicate.
        pattern = os.path.join(folder_path, "df_*.csv")
        # Some conditions (e.g. figure_3_1k's A_and_B) have leftover
        # duplicate files per rep_id from a prior rerun/race in the
        # simulation job -- keep only the most-recently-written file per
        # rep_id so each condition contributes exactly one task per rep.
        by_rep_id = {}
        for f in sorted(glob.glob(pattern)):
            m = re.search(r"rep_(\d+)", os.path.basename(f))
            rep_id = m.group(1) if m else os.path.splitext(os.path.basename(f))[0]
            mtime = os.path.getmtime(f)
            if rep_id not in by_rep_id or mtime > by_rep_id[rep_id][1]:
                by_rep_id[rep_id] = (f, mtime)
        for rep_id, (f, _) in sorted(by_rep_id.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]):
            tasks.append((f, condition, rep_id))
    return tasks


def run_one(path, condition, rep_id, output_dir, n_cores, n_shuffles):
    out_path = os.path.join(output_dir, f"{condition}_rep_{rep_id}.json")
    if os.path.exists(out_path):
        return out_path, "skipped (already exists)"

    matrix_record, box_plot_record = process_replicate(path, condition, rep_id, n_cores=n_cores, n_shuffles=n_shuffles)
    record = {"matrix_record": matrix_record, "box_plot_record": box_plot_record}
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(record, f, cls=NumpyEncoder, indent=2)
    return out_path, "computed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cores", type=int, default=1)
    parser.add_argument("--n-shuffles", type=int, default=10000)
    parser.add_argument("--sim-folder", type=str,
                         default="/projects/b1255/yscher/Transcriptomic Distance/simulation_data/figure_3_simulations")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--only", type=str, default=None,
                         help="comma-separated condition:rep_id to run; default = all tasks")
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not args.list_tasks and not args.output_dir:
        parser.error("--output-dir is required unless --list-tasks is given")

    tasks = collect_tasks(args.sim_folder)

    if args.list_tasks:
        ids = [f"{condition}:{rep_id}" for _, condition, rep_id in tasks]
        if args.limit is not None:
            ids = ids[:args.limit]
        print("\n".join(ids))
        raise SystemExit(0)

    if args.only:
        wanted = set(args.only.split(","))
        tasks = [t for t in tasks if f"{t[1]}:{t[2]}" in wanted]

    print(f"Running {len(tasks)} task(s), output -> {args.output_dir}", flush=True)
    for i, (path, condition, rep_id) in enumerate(tasks):
        t0 = time.time()
        out_path, status = run_one(path, condition, rep_id, args.output_dir, args.n_cores, args.n_shuffles)
        print(f"[{i+1}/{len(tasks)}] {condition} rep {rep_id}: {status} ({time.time()-t0:.1f}s) -> {out_path}", flush=True)
