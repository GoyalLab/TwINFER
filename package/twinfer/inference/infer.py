# Calculation functions
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numba
import tqdm
import scipy
import seaborn
import os
import sys
import joblib
from itertools import product
import importlib

from twinfer.inference.correlation_functions import (
    calculate_pairwise_gene_gene_correlation_matrix,
    check_system_in_steady_state,
    check_gene_gene_correlation_threshold,
    calculate_twin_random_pair_correlations,
    differentiate_single_state_reg_and_multiple_states,
    identify_reg_if_multiple_states,
    get_cross_correlations,
    identify_actual_directed_edges,
    separate_fan_outs_from_mutual_regulation,
    extract_param_index,
    read_input_matrix,
    get_param_data, 
    split_and_merge_simulations,
    plot_matrix_as_heatmap,
    print_summary
)

def infer_with_twinfer(path_to_simulation_file=None,
                        merge_to_multiple_states=False,
                        base_config=None,
                        t1=None, t2=None,
                        gene_list_given=None,
                        match_sim_details=True,
                        check_for_steady_state=True,
                        remove_twin_structure=False,
                        seed=101010,
                        merge_time_points=True,
                        threshold_gene_gene_corr=0.04,
                        use_scramble=True,
                        p_val_threshold_scrambled_gene_correlation=0.01,
                        show_scrambled_distribution_gene_correlation=True,
                        n_cores=4,
                        return_gene_corr_thresholds=False,
                        z_score_threshold_two_states=10,
                        infer_direction_for_which_edges="single-state",
                        p_value_threshold_cross_correlation=0.01,
                        separate_fan_outs_from_mutual_regulation_flag=False,
                        fan_out_z_score_threshold=8,
                        plot_correlation_matrices_as_heatmap=True,
                        have_any_output=True,
                        ranked_list=False):
    """
    Infer gene regulatory interactions from simulated or experimental twin-cell data
    using the TwINFER pipeline.

    This function processes a single simulation (or equivalent experimental dataset)
    to:
      1. Check system steady state at an early timepoint.
      2. Compute gene–gene correlations at early and late timepoints.
      3. Classify candidate regulations as single-state or multiple-state.
      4. Infer directionality of single-state interactions from across-time twin pairs.
      5. Optionally visualize intermediate matrices and the inferred network.

    The approach uses twin cell pairs (descended from the same mother cell) and
    compares their gene expression correlations at early and late post-division
    times, as well as across-time twin measurements, to determine regulation type
    and directionality.

    Parameters are listed below in the order the pipeline uses them: input/config,
    steady-state and twin-structure setup, Step 1 (gene-gene correlation), Step 3
    (single- vs multiple-state classification), Step 5 (directionality), the
    optional fan-out step, and finally display/output options.

    Parameters
    ----------
    path_to_simulation_file : str or list of str
        Path to the CSV file containing simulation or experimental output, or a
        list of such paths when `merge_to_multiple_states=True`. The file should
        have one row per cell per timepoint, with at least:
        - 'clone_id': integer clone identifier.
        - 'cell_id': unique cell identifier.
        - 'time_step': time (in hours) post-division.
        - gene expression columns for each gene.

    merge_to_multiple_states : bool, default=False
        If True, `path_to_simulation_file` is treated as multiple underlying
        states to combine: a list of paths is merged via `split_and_merge_simulations`,
        while a single string path is used as-is (with a printed note, since there
        is nothing to merge).

    base_config : dict
        Dictionary specifying simulation metadata and parameter sources:
            - "n_cells" : int
                Expected number of twin clones.
            - "twin_simulation_time_after_division" : int or float
                Duration after division covered in the simulation (hours).
            - "twin_measurement_resolution" : int or float
                Sampling resolution (hours).
            - "path_to_connectivity_matrix" : str
                File path to the interaction (connectivity) matrix.
            - "param_csv" : str
                File path to the parameter CSV file.
            - "rows_to_use" : list[list[int]]
                Parameter row indices corresponding to this simulation.

    t1 : int or float
        Early timepoint (hours) used for initial gene–gene correlation analysis.

    t2 : int or float
        Late timepoint (hours) used for twin vs random correlation comparison and
        across-time directionality inference.

    gene_list_given : list of str, optional
        Explicit gene names to use in place of the default `gene_1, gene_2, ...`
        naming derived from the connectivity matrix's gene count.

    match_sim_details : bool, default=True
        If True, cross-checks the simulation file's clone count, sampled
        timepoints, and parameter-row index against `base_config` before
        proceeding (raises an AssertionError on mismatch); this also gates
        whether `check_for_steady_state` runs. Set False for data with no
        matching `base_config` (e.g. real/experimental data).

    check_for_steady_state : bool, default=True
        If True (and `match_sim_details=True`), verifies that the system is in
        steady state at t1 using a mean and slope threshold; raises ValueError
        if not steady.

    remove_twin_structure : bool, default=False
        If True, scrambles clone identity for every replicate after the first
        (via a derangement), so unrelated cells are treated as twins instead —
        used as a negative control for the twin-pair signal.

    seed : int, default=101010
        Random seed for the clone-id shuffle that splits clones into the
        t1-only, t2-only, and across-time subsets (and for `remove_twin_structure`
        when enabled).

    merge_time_points : bool, default=True
        Should cells be merged to get gene-gene correlation and random correlation between the two time points. Set it to be True if the population is in steady state.

    threshold_gene_gene_corr : float, default=0.04
        Absolute correlation threshold above which gene–gene pairs are considered
        potential regulations.

    use_scramble : bool, default=True
        Passed to the Step 1 significance test (`check_gene_gene_correlation_threshold`).

    p_val_threshold_scrambled_gene_correlation : float, default=0.01
        p-value cutoff for the Step 1 scramble-based significance test (used only
        when `use_scramble=True`).

    show_scrambled_distribution_gene_correlation : bool, default=True
        If True, plots the scrambled null distribution for each significant gene
        pair during Step 1.

    n_cores : int, default=4
        Number of cores used for the numba-parallelized scramble/permutation
        steps (gene-gene correlation thresholding and directionality inference).

    return_gene_corr_thresholds : bool, default=False
        If True, adds "gene_corr_thresholds" and "gene_gene_corr_p_values" to the
        returned dict (the per-pair thresholds/p-values computed during Step 1).

    z_score_threshold_two_states : float, default=10
        z-score threshold used in Step 3 to classify a `potential_regulation`
        pair as multiple-state (heterogeneous) vs single-state, from the
        difference between twin and random-pair correlations.

    infer_direction_for_which_edges : {"single-state", "all-potential-regulation", "all-edges"}, default="single-state"
        Which candidate edge set Step 5 tests for directionality: only
        single-state-regulation pairs, all potential-regulation pairs
        (single- and multiple-state), or every gene pair.

    p_value_threshold_cross_correlation : float, default=0.01
        p-value cutoff for the Step 5 across-time directionality test.

    separate_fan_outs_from_mutual_regulation_flag : bool, default=False
        If True, after final_directed_edges is computed, checks every gene pair (A, B) that
        shares a common regulator C (C->A and C->B both in final_directed_edges) for whether
        A<->B (both A->B and B->A present) is true mutual regulation or a fan-out artifact of
        being co-driven by C. Uses the undirected twin-vs-random z-score for the (A, B) pair
        (same statistic as the Step 3 single/multiple-state classification): |z| above
        fan_out_z_score_threshold removes both A->B and B->A as a fan-out; otherwise both
        edges are kept as mutual regulation. Pairs with no common regulator, no cross-edges,
        or only one direction (feed-forward loop) are left untouched.

    fan_out_z_score_threshold : float, default=8
        |z|-score threshold used by separate_fan_outs_from_mutual_regulation_flag above.

    plot_correlation_matrices_as_heatmap : bool, default=True
        If True, generates heatmaps for:
            - Gene–gene correlations at t1
            - Twin and random correlations at t2
            - Directionality matrix

    have_any_output : bool, default=True
        If True, prints a summary of inferred regulations and shows network plots.

    ranked_list : bool, default=False
        If True, adds "ranked_edge_list" to the returned dict: a DataFrame with columns
        gene_1, gene_2, directional_correlation, p_val, built from every non-zero entry
        of unfiltered_direction_matrix (self-pairs excluded), for whichever candidate set
        infer_direction_for_which_edges actually tested this call (single-state,
        all-potential-regulation, or all-edges -- nothing extra is computed). Ranked by
        |directional_correlation| magnitude descending (strongest first), with p_val
        ascending as the tie-break.

    Returns
    -------
    dict
        Dictionary containing:
            - "direction_matrix" : pd.DataFrame
                Normalized directional correlation matrix (t1 → t2) for single-state regulations.
            - "direction_raw_matrix" : pd.DataFrame
                Raw directional correlation differences without thresholding.
            - "pairwise_gene_gene_correlation_matrix" : pd.DataFrame
                Gene–gene Spearman correlation matrix at t1.
            - "twin_pair_correlation_matrix_t2" : pd.DataFrame
                Twin-cell correlation matrix at t2.
            - "random_pair_correlation_matrix_t2" : pd.DataFrame
                Random-cell correlation matrix at t2.
            - "twin_pair_correlation_matrix_t1" : pd.DataFrame
                Twin-cell correlation matrix at t1.
            - "random_pair_correlation_matrix_t1" : pd.DataFrame
                Random-cell correlation matrix at t1.
            - "stage3_details" : dict
                Keyed by (gene_i, gene_j) for every pair in multiple_states_gene_pairs;
                point estimate, standard error, confidence interval, and
                conclusive/inconclusive verdict for the Stage III regulation test.
                See identify_reg_if_multiple_states's Returns for the per-pair schema.
                Empty dict if no pairs were classified as multiple-state.

    Raises
    ------
    AssertionError
        If the number of clones or sampled timepoints in the simulation file does 
        not match `base_config`.
    ValueError
        If required timepoints t1 or t2 are missing from the data.
        If steady state is required and not reached.

    Notes
    -----
    - Clones are split into three disjoint sets for t1-only, t2-only, and across-time
      measurements in a 1:1:2 ratio.
    - Gene-gene correlations uses all cell measurements at both time t1 and t2.
    - Across-time twin pairs are sampled by selecting one cell per clone at t1 and 
      one different cell at t2 from the same clone.
    - Single-state vs multiple-state regulation classification is based on the 
      difference between twin and random correlations at t1.
    - Directionality inference uses correlation differences between across-time 
      twin pairs at t1 and t2.
    """
    # Load simulation data
    if merge_to_multiple_states:
        if isinstance(path_to_simulation_file, str):
            print("Only one simulation file was provided while merge_to_multiple_states was set to True. The file will be used as-is.")
            simulation = pd.read_csv(path_to_simulation_file)
        else:
            # It must be a list/tuple/etc.
            simulation = split_and_merge_simulations(path_to_simulation_file)
    else:
        simulation = pd.read_csv(path_to_simulation_file)


    # Load connectivity matrix and parameter set
    path_to_connectivity_matrix = base_config["path_to_connectivity_matrix"]
    path_to_parameter_csv = base_config["param_csv"]
    param_df = pd.read_csv(path_to_parameter_csv, index_col=0)

    # --- Basic sanity checks ---
    # Assert number of clones in simulation file matches config
    n_clones_simulation = simulation['clone_id'].nunique()
    n_clones_base_config = base_config["n_cells"]

    # Assert time points match expected resolution
    time_points_simulations = simulation['time_step'].unique()
    time_points_base_config = np.arange(
        0, 
        base_config['twin_simulation_time_after_division'] + base_config['twin_measurement_resolution'], 
        base_config['twin_measurement_resolution']
    )


    if match_sim_details:
        # Assert parameter row identity matches
        param_index_from_file_name = extract_param_index(path_to_simulation_file)
        param_index_from_base_config = "_".join(map(str, base_config["rows_to_use"][0]))
        assert n_clones_simulation == n_clones_base_config, \
            "Number of twin pairs in the simulation file does not match n_cells in base_config."
        assert set(time_points_simulations) == set(time_points_base_config), \
            "The sampling time points in the simulation file do not match those specified in base_config."
        assert param_index_from_file_name == param_index_from_base_config, \
            f"Simulation parameters ({param_index_from_file_name}) must match the details (parameter rows) in  ({param_index_from_base_config})."

    # Load gene parameters and connectivity structure
    n_genes, interaction_matrix = read_input_matrix(path_to_connectivity_matrix)
    if gene_list_given:
        gene_list = gene_list_given
    else:
        gene_list = [f"gene_{i}" for i in np.arange(1, n_genes + 1)]
    try:
        gene_params = get_param_data(param_df, param_index_from_file_name, n_genes)
        print(gene_params)
    except:
        gene_params = None
        print("Could not ascertain corresponding parameter rows to check for gene parameters")

    valid_options = ["single-state", "all-edges", "all-potential-regulation"]
    if infer_direction_for_which_edges not in valid_options:
        raise ValueError(f"infer_direction_for_which_edges must be one of {valid_options}, got '{infer_direction_for_which_edges}'")
        
    # --- Check for steady state at t1 (optional) ---
    if check_for_steady_state and match_sim_details:
        is_system_in_steady_state, steady_state_summary = check_system_in_steady_state(simulation, gene_params, interaction_matrix, gene_list,
                                  relative_diff_threshold=0.01, relative_slope_threshold=0.01)
        if not is_system_in_steady_state:
            print(steady_state_summary)
            raise ValueError(
                "The system is not in steady state. "
                "You can override this by setting check_for_steady_state=False."
            )

    # Ensure the time points t1 and t2 exist in the simulation data
    unique_timepoints = simulation['time_step'].unique()

    if t1 not in unique_timepoints:
        raise ValueError(f"Time point t1={t1} not found in simulation['time_step'].")
    if t2 not in unique_timepoints:
        raise ValueError(f"Time point t2={t2} not found in simulation['time_step'].")

    # If remove_twin_structure is set to True, random pairs of cells are used as "pairs of twins"
    # --- Break twin structure but preserve within-cell continuity ---
    replicates = simulation["replicate"].drop_duplicates().sort_values().to_numpy()

    if remove_twin_structure:
        rng = np.random.default_rng(12345)

        # sort so the RNG draw is reproducible regardless of row order in the frame
        unique_clones = simulation["clone_id"].drop_duplicates().sort_values().to_numpy()
        

        if len(unique_clones) < 2:
            raise ValueError("need at least 2 clones to build a derangement")

        # first replicate stays as the reference; every other one gets its own derangement
        for rep in replicates[1:]:
            shuffled = unique_clones.copy()
            while np.any(shuffled == unique_clones):
                rng.shuffle(shuffled)

            shuffle_map = dict(zip(unique_clones, shuffled))
            mask = simulation["replicate"] == rep
            simulation.loc[mask, "clone_id"] = simulation.loc[mask, "clone_id"].map(shuffle_map)


    # Subset the simulation at the desired timepoints

    # Shuffle all clone IDs
    np.random.seed(seed)
    clone_ids_shuffled = np.random.permutation(n_clones_simulation)

    # Split into 1:1:2 ratio
    n1 = n2 = n_clones_simulation // 4
    t1_clones = clone_ids_shuffled[:n1]
    t2_clones = clone_ids_shuffled[n1:n1 + n2]
    across_t_clones = clone_ids_shuffled[n1 + n2:]

    # Subset directly
    t1_twins = simulation[(simulation['clone_id'].isin(t1_clones)) & (simulation['time_step'] == t1)]
    t2_twins = simulation[simulation['clone_id'].isin(t2_clones) & (simulation['time_step'] == t2)]

    # Across_t: pick exactly one random twin per clone_id
    # One cell per clone at t1
    
    across_t_twin1 = (
        simulation[(simulation['clone_id'].isin(across_t_clones)) & (simulation['time_step'] == t1) & (simulation['replicate'] == replicates[0])]
    )
    
    across_t_twin2 = (
        simulation[(simulation['clone_id'].isin(across_t_clones)) & (simulation['time_step'] == t2) & (simulation['replicate'] == replicates[1])]
    )

    # Reset index for cleanliness
    t1_twins = t1_twins.reset_index(drop=True)
    t2_twins = t2_twins.reset_index(drop=True)
    across_t_twin1 = across_t_twin1.reset_index(drop=True)
    across_t_twin2 = across_t_twin2.reset_index(drop=True)

    all_t1_t2_measurements = pd.concat(
    [t1_twins, t2_twins, across_t_twin1, across_t_twin2],
    ignore_index=True
    )
    all_t1_measurements = pd.concat(
        [t1_twins, across_t_twin1],
        ignore_index=True
    )
    all_t2_measurements = pd.concat(
        [t2_twins, across_t_twin2],
        ignore_index=True
    )
    #TODO Merge timepoints for correlation - need to be removed 
    #Step 1: Calculate pairwise gene correlations to check for existence of an edge
    if merge_time_points == True:
        # --- Step 1: Pairwise gene-gene correlations at t1 ---
        pairwise_gene_gene_correlation_matrix = calculate_pairwise_gene_gene_correlation_matrix(
            all_t1_t2_measurements, gene_list
        )
        no_regulation, potential_regulation, gene_corr_thresholds, p_values  = check_gene_gene_correlation_threshold(
            all_t1_t2_measurements, pairwise_gene_gene_correlation_matrix, gene_list,  threshold = threshold_gene_gene_corr, use_scramble = True, 
            p_val_threshold = p_val_threshold_scrambled_gene_correlation, verbose = show_scrambled_distribution_gene_correlation, n_cores_to_use = n_cores, return_gene_corr_thresholds = return_gene_corr_thresholds
        )
    else:
        pairwise_gene_gene_correlation_matrix = calculate_pairwise_gene_gene_correlation_matrix(
            all_t1_measurements, gene_list
        )
           
        no_regulation, potential_regulation, gene_corr_thresholds, p_values = check_gene_gene_correlation_threshold(
            all_t2_measurements, pairwise_gene_gene_correlation_matrix, gene_list,  threshold = threshold_gene_gene_corr, use_scramble = True, 
            p_val_threshold = p_val_threshold_scrambled_gene_correlation, verbose = show_scrambled_distribution_gene_correlation, n_cores_to_use = n_cores, return_gene_corr_thresholds = return_gene_corr_thresholds
        )

    if plot_correlation_matrices_as_heatmap:
        if merge_time_points == True:
            title = r"Gene correlations $\rho$ with cells from both two points"
        else:
            title = rf"Gene correlations $\rho$ with cells from time {t1}"
        plot_matrix_as_heatmap(corr_matrix=pairwise_gene_gene_correlation_matrix, gene_list=gene_list, no_regulation=no_regulation, potential_regulation=potential_regulation,
            title=title, add_gene_labels=True, add_time=False, gray_out_no_reg=False, black_out_self = True
        )

    #Step 2: Calculate z-score by comparing z-score of twin correlation as compared to a distribution of random-pair correlations for every gene pair
    title_random_plot = rf"Random-pair difference correlation $\rho_{{\Delta}}$ using cells at time {t1}"
    twin_pair_correlation_matrix_t1, random_pair_correlation_matrix_t1 = calculate_twin_random_pair_correlations(
            all_t1_measurements, t1_twins, gene_list
        )

    if plot_correlation_matrices_as_heatmap:
        plot_matrix_as_heatmap( corr_matrix=twin_pair_correlation_matrix_t1, gene_list=gene_list, no_regulation=no_regulation, potential_regulation=potential_regulation,
            title=rf"Twin pair correlations $\hat{{\rho}}_{{\Delta}}(t_1)$ at time {t1}h", add_gene_labels=True, add_time=False, time=[t1], gray_out_no_reg=True, black_out_self = True, symmetric = True
        )
        
        plot_matrix_as_heatmap(corr_matrix=random_pair_correlation_matrix_t1, gene_list=gene_list, no_regulation=no_regulation, potential_regulation=potential_regulation,
            title=title_random_plot, add_gene_labels=True, add_time=False, time=[t1], gray_out_no_reg=True, black_out_self = True, symmetric = True
        )

    multiple_states_gene_pairs, single_state_regulation = differentiate_single_state_reg_and_multiple_states(
            all_t1_measurements, potential_regulation, twin_pair_correlation_matrix_t1, random_pair_correlation_matrix_t1, gene_list, z_score_threshold=z_score_threshold_two_states
        )
    twin_pair_correlation_matrix_t2, random_pair_correlation_matrix_t2 = calculate_twin_random_pair_correlations(
                    all_t2_measurements, t2_twins, gene_list
        )
    #Step 3: Identify if there is multiple states and regulation by comparing twin-correlation across time
    if len(multiple_states_gene_pairs) > 0:
        multiple_states_no_reg, multiple_states_and_reg, stage3_details = identify_reg_if_multiple_states(
            twin_pair_correlation_matrix_t1,twin_pair_correlation_matrix_t2,random_pair_correlation_matrix_t1,
            random_pair_correlation_matrix_t2,multiple_states_gene_pairs,gene_list,
            t1_twins,t2_twins
            )
    else:
        multiple_states_no_reg, multiple_states_and_reg, stage3_details = [], [], {}

    #Print summary of results ---
    all_gene_pairs = list(product(gene_list, repeat=2))
    if have_any_output:
        print_summary(no_regulation, single_state_regulation, multiple_states_no_reg, multiple_states_and_reg)
    
    #Step 4: Identify direction of regulation using cross-correlation of expression between twins separated across the two time points
    direction_matrix = pd.DataFrame()
    final_directed_edges = set()
    directed_p_values = {}

    if infer_direction_for_which_edges == "single-state" :
        if len(single_state_regulation) > 0:
            bidirectional_pairs = {(a, b) for (a, b) in single_state_regulation} | \
                      {(b, a) for (a, b) in single_state_regulation}

            # Add self-pairs
            genes = {g for pair in single_state_regulation for g in pair}
            self_pairs = {(g, g) for g in genes}

            # Final
            all_gene_pairs = bidirectional_pairs | self_pairs
            all_gene_pairs = list(all_gene_pairs)

            direction_matrix = get_cross_correlations(across_t_twin1, across_t_twin2, gene_pairs=all_gene_pairs)

            final_directed_edges, directed_p_values = identify_actual_directed_edges(across_t_twin1, across_t_twin2, direction_matrix, gene_pairs=all_gene_pairs, threshold = p_value_threshold_cross_correlation, n_cores_to_use = n_cores, verbose = True, return_p_values = True)

    elif infer_direction_for_which_edges == "all-potential-regulation":
        if len(single_state_regulation) > 0 or len(multiple_states_and_reg) > 0 or len(multiple_states_gene_pairs) > 0:
                combined_list = single_state_regulation + multiple_states_and_reg + multiple_states_no_reg
                bidirectional_pairs = {(a, b) for (a, b) in combined_list} | \
                      {(b, a) for (a, b) in combined_list}
                genes = {g for pair in combined_list for g in pair}
                self_pairs = {(g, g) for g in genes}

                # Final
                all_gene_pairs_all_reg = bidirectional_pairs | self_pairs
                all_gene_pairs_all_reg = list(all_gene_pairs_all_reg)

                direction_matrix = get_cross_correlations(across_t_twin1, across_t_twin2, gene_pairs=all_gene_pairs_all_reg)
                final_directed_edges, directed_p_values = identify_actual_directed_edges(across_t_twin1, across_t_twin2, direction_matrix, gene_pairs=all_gene_pairs_all_reg, threshold = p_value_threshold_cross_correlation, n_cores_to_use = n_cores, verbose = True, return_p_values = True)
        else:
                final_directed_edges = []
                direction_matrix = pd.DataFrame(
                    np.zeros((len(gene_list), len(gene_list))),
                    index=gene_list,
                    columns=gene_list
                )
    else:
        direction_matrix = get_cross_correlations(across_t_twin1, across_t_twin2, gene_pairs=all_gene_pairs)
        final_directed_edges, directed_p_values = identify_actual_directed_edges(across_t_twin1, across_t_twin2, direction_matrix, gene_pairs=all_gene_pairs, threshold = p_value_threshold_cross_correlation, n_cores_to_use = n_cores, verbose = True, return_p_values = True)

    direction_matrix = direction_matrix.reindex(
    index=gene_list,
    columns=gene_list,
    fill_value=0
    )
    unfiltered_direction_matrix = direction_matrix.copy()
    for i in direction_matrix.index:
        for j in direction_matrix.columns:
            if i != j and (i, j) not in final_directed_edges:
                    direction_matrix.loc[i,j] = 0

    #Step 5: Separate fan-outs (C->A, C->B) from true mutual regulation (A<->B) ---
    fan_out_log = None
    final_directed_edges = set(final_directed_edges)
    if separate_fan_outs_from_mutual_regulation_flag:
        twin_matrix_for_fan_out = twin_pair_correlation_matrix_t1
        measurements_for_fan_out = all_t1_t2_measurements if merge_time_points else all_t1_measurements
        final_directed_edges, directed_p_values, direction_matrix, fan_out_log = separate_fan_outs_from_mutual_regulation(
            measurements_for_fan_out, twin_matrix_for_fan_out, gene_list,
            final_directed_edges, directed_p_values, direction_matrix,
            z_score_threshold=fan_out_z_score_threshold
        )

    if plot_correlation_matrices_as_heatmap and not direction_matrix.empty:
        all_gene_pairs = list(product(gene_list, repeat=2))
        no_reg_pairs = [pair for pair in all_gene_pairs if pair not in final_directed_edges]
        if infer_direction_for_which_edges == "all-potential-regulation" and multiple_states_and_reg:
            plot_matrix_as_heatmap(
                corr_matrix=direction_matrix,
                gene_list=gene_list,
                no_regulation=no_reg_pairs,                   
                potential_regulation=final_directed_edges,     
                title=r"Twin cross-correlation $\hat{\rho}^{\dagger}_{x(t_{1}) \to y(t_{2})}$",
                add_gene_labels=True,
                add_time=False,
                time=[t1, t2],
                gray_out_no_reg=True,
                black_out_self = True,
                symmetric = False,
                draw_diagonal_multi_state_reg = True,
                multi_state_reg_edges = multiple_states_gene_pairs
            )
        elif infer_direction_for_which_edges == "single-state" and multiple_states_and_reg:
            plot_matrix_as_heatmap(
                corr_matrix=direction_matrix,
                gene_list=gene_list,
                no_regulation=no_reg_pairs,                   
                potential_regulation=final_directed_edges,     
                title=r"Twin cross-correlation $\hat{\rho}^{\dagger}_{x(t_{1}) \to y(t_{2})}$",
                add_gene_labels=True,
                add_time=False,
                time=[t1, t2],
                gray_out_no_reg=True,
                black_out_self = True,
                symmetric = False,
                draw_diagonal_multi_state_reg = True,
                multi_state_reg_edges = multiple_states_gene_pairs
            )
        else:
            plot_matrix_as_heatmap(
                corr_matrix=direction_matrix,
                gene_list=gene_list,
                no_regulation=no_reg_pairs,                   
                potential_regulation=final_directed_edges,     
                title=r"Twin cross-correlation $\hat{\rho}^{\dagger}_{x(t_{1}) \to y(t_{2})}$",
                add_gene_labels=True,
                add_time=False,
                time=[t1, t2],
                gray_out_no_reg=True,
                black_out_self = True,
                symmetric = False
            )

    # Step 6: #TODO use the new plot network function to visualize the inferred network
    # if (len(single_state_regulation) >= 0):
    #     if have_any_output:
    #         if (len(final_directed_edges) > 0):
    #             plot_network(direction_matrix, gene_list, final_directed_edges)
    #         else:
    #             plot_network(direction_matrix, gene_list, final_directed_edges)

    # --- Optional: ranked edge list from every non-zero entry of unfiltered_direction_matrix ---
    # Reflects whichever candidate set was actually tested this call (single-state,
    # all-potential-regulation, or all-edges) -- nothing is recomputed here, just
    # reshaped. Self-pairs (gene_1 == gene_2) are excluded: they're not regulatory
    # edges. Ranked by magnitude of cross-correlations (most significant first), with the p-value as a tie-breaker if necessary
    ranked_edge_list = None
    if ranked_list:
        rank_rows = []
        for (gi, gj), p in (directed_p_values or {}).items():
            if gi == gj:
                continue
            if gi not in unfiltered_direction_matrix.index or gj not in unfiltered_direction_matrix.columns:
                continue
            score = unfiltered_direction_matrix.loc[gi, gj]
            if pd.isna(score) or score == 0:
                continue
            rank_rows.append({"gene_1": gi, "gene_2": gj, "directional_correlation": score, "p_val": p})
        ranked_edge_list = pd.DataFrame(rank_rows, columns=["gene_1", "gene_2", "directional_correlation", "p_val"])
        if not ranked_edge_list.empty:
            ranked_edge_list["_abs_corr"] = ranked_edge_list["directional_correlation"].abs()
            ranked_edge_list = (
                ranked_edge_list.sort_values(["_abs_corr", "p_val"], ascending=[False, True])
                .drop(columns="_abs_corr")
                .reset_index(drop=True)
            )

    try:
        result =  {
            "all_gene_pairs": all_gene_pairs,
            "gene_lists": {"no_regulation":no_regulation, "single_state_regulation":single_state_regulation, "multiple_states_no_reg": multiple_states_no_reg, "multiple_states_and_reg": multiple_states_and_reg},
            "potential_regulation": potential_regulation,
            "final_directed_edges": final_directed_edges,
            "directed_p_values": directed_p_values,
            "direction_matrix": direction_matrix,
            "unfiltered_direction_matrix": unfiltered_direction_matrix,
            "pairwise_gene_gene_correlation_matrix": pairwise_gene_gene_correlation_matrix,
            "twin_pair_correlation_matrix_t2": twin_pair_correlation_matrix_t2,
            "random_pair_correlation_matrix_t2": random_pair_correlation_matrix_t2,
            "twin_pair_correlation_matrix_t1": twin_pair_correlation_matrix_t1,
            "random_pair_correlation_matrix_t1": random_pair_correlation_matrix_t1,
            "fan_out_log": fan_out_log,
            "stage3_details": stage3_details
        }
    except:
        result = {
            "all_gene_pairs": all_gene_pairs,
            "gene_lists": {"no_regulation":no_regulation, "single_state_regulation":single_state_regulation, "multiple_states_no_reg": multiple_states_no_reg, "multiple_states_and_reg": multiple_states_and_reg},
            "potential_regulation": potential_regulation,
            "final_directed_edges": None,
            "directed_p_values": None,
            "direction_matrix": None,
            "unfiltered_direction_matrix": None,
            "pairwise_gene_gene_correlation_matrix": pairwise_gene_gene_correlation_matrix,
            "random_pair_correlation_matrix_t2": random_pair_correlation_matrix_t2,
            "twin_pair_correlation_matrix_t2": twin_pair_correlation_matrix_t2,
            "twin_pair_correlation_matrix_t1": twin_pair_correlation_matrix_t1,
            "random_pair_correlation_matrix_t1": random_pair_correlation_matrix_t1,
            "stage3_details": stage3_details
        }
    if return_gene_corr_thresholds:
        result['gene_corr_thresholds'] = gene_corr_thresholds
        result['gene_gene_corr_p_values'] = p_values
    if ranked_list:
        result['ranked_edge_list'] = ranked_edge_list
    return result