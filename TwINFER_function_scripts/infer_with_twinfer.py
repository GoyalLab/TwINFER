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

from TwINFER_function_scripts.correlation_analysis_functions import (
    
    calculate_pairwise_gene_gene_correlation_matrix,
    check_system_in_steady_state,
    check_gene_gene_correlation_threshold,
    calculate_twin_random_pair_correlations,
    differentiate_single_state_reg_and_multiple_states,
    identify_reg_if_multiple_states,
    get_cross_correlations,
    identify_actual_directed_edges
)

# Helper functions
from TwINFER_function_scripts.correlation_analysis_helpers import (
    extract_param_index,
    read_input_matrix,
    get_param_data, 
    split_and_merge_simulations,
    plot_matrix_as_heatmap,
    print_summary,
    plot_network
)

def infer_with_twinfer(path_to_simulation_file= None, 
                        merge_to_multiple_states = False,
                        base_config=None, t1 = None, t2 = None, 
                        check_for_steady_state=True, 
                        merge_time_points=True,
                        threshold_gene_gene_corr=0.04, use_scramble = True, 
                        p_val_threshold_scrambled_gene_correlation = 0.01,
                        show_scrambled_distribution_gene_correlation = True,
                        z_score_threshold_two_states = 10,
                        p_value_threshold_cross_correlation = 0.01,
                        plot_correlation_matrices_as_heatmap=True,
                        have_any_output=True,
                        seed = 101010,
                        infer_direction_for_which_edges = "single-state",
                        remove_twin_structure = False,
                        return_gene_corr_thresholds = False,
                        match_sim_details = True,
                        n_cores = 4):
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

    Parameters
    ----------
    path_to_simulation_file : str
        Path to the CSV file containing simulation or experimental output.
        The file should have one row per cell per timepoint, with at least:
        - 'clone_id': integer clone identifier.
        - 'cell_id': unique cell identifier.
        - 'time_step': time (in hours) post-division.
        - gene expression columns for each gene.

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

    merge_time_points : bool, default=True
        Should cells be merged to get gene-gene correlation and random correlation between the two time points. Set it to be True if the population is in steady state.

    threshold_gene_gene_corr : float, default=0.04
        Absolute correlation threshold above which gene–gene pairs are considered
        potential regulations.

    check_for_steady_state : bool, default=True
        If True, verifies that the system is in steady state at t1 using a mean and 
        slope threshold; raises ValueError if not steady.

    plot_correlation_matrices_as_heatmap : bool, default=True
        If True, generates heatmaps for:
            - Gene–gene correlations at t1
            - Twin and random correlations at t2
            - Directionality matrix

    have_any_output : bool, default=True
        If True, prints a summary of inferred regulations and shows network plots.

    remove_twin_structure : bool, default=False
        If True, scrambles twin structure and random pairs of cells are labelled as twins instead.
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
    - Gene-gene and random-pair correlations uses all cell measurements at both time t1 and t2.
    - Across-time twin pairs are sampled by selecting one cell per clone at t1 and 
      one different cell at t2 from the same clone.
    - Single-state vs multiple-state regulation classification is based on the 
      difference between twin and random correlations at t2.
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
        is_system_in_steady_state = check_system_in_steady_state(simulation, gene_params, interaction_matrix, gene_list,
                                  relative_diff_threshold=0.01, relative_slope_threshold=0.01)
        if not is_system_in_steady_state:
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
    if remove_twin_structure:
        rng = np.random.default_rng(12345)
        unique_clones = np.array(simulation["clone_id"].unique())

        # --- Generate a derangement (no clone keeps its original ID) ---
        shuffled = unique_clones.copy()
        while np.any(shuffled == unique_clones):
            rng.shuffle(shuffled)

        shuffle_map = dict(zip(unique_clones, shuffled))

        # --- Apply mapping ONLY to replicate 2 ---
        mask_rep2 = simulation["replicate"] == 2
        simulation.loc[mask_rep2, "clone_id"] = simulation.loc[mask_rep2, "clone_id"].map(shuffle_map)


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
        simulation[(simulation['clone_id'].isin(across_t_clones)) & (simulation['time_step'] == t1) & (simulation['replicate'] == 1)]
    )
    
    across_t_twin2 = (
        simulation[(simulation['clone_id'].isin(across_t_clones)) & (simulation['time_step'] == t2) & (simulation['replicate'] == 2)]
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
           
        no_regulation, potential_regulation, gene_corr_thresholds = check_gene_gene_correlation_threshold(
            all_t2_measurements, pairwise_gene_gene_correlation_matrix, gene_list,  threshold = threshold_gene_gene_corr, use_scramble = True, 
            p_val_threshold = p_val_threshold_scrambled_gene_correlation, verbose = show_scrambled_distribution_gene_correlation, n_cores_to_use = n_cores, return_gene_corr_thresholds = return_gene_corr_thresholds
        )
    # print(no_regulation)
    if plot_correlation_matrices_as_heatmap:
        if merge_time_points == True:
            title = r"Gene correlations $\rho$ with cells from both two points"
        else:
            title = rf"Gene correlations $\rho$ with cells from time {t1}"
        plot_matrix_as_heatmap(corr_matrix=pairwise_gene_gene_correlation_matrix, gene_list=gene_list, no_regulation=no_regulation, potential_regulation=potential_regulation,
            title=title, add_gene_labels=True, add_time=False, gray_out_no_reg=False, black_out_self = True
        )

    # --- Step 2: Twin/random correlations at t2 ---
    if merge_time_points:
        twin_pair_correlation_matrix_t1, random_pair_correlation_matrix_t1 = calculate_twin_random_pair_correlations(
            all_t1_t2_measurements, t1_twins, gene_list
        )
        title_random_plot = r"Random-pair difference correlation $\rho_{\Delta}$ using cells across both timepoints"
    else:
        title_random_plot = rf"Random-pair difference correlation $\rho_{{\Delta}}$ using cells at time {t1}"
        twin_pair_correlation_matrix_t1, random_pair_correlation_matrix_t1 = calculate_twin_random_pair_correlations(
            all_t1_measurements, t1_twins, gene_list
        )
    # print(twin_pair_correlation_matrix_t2)
    if plot_correlation_matrices_as_heatmap:
        plot_matrix_as_heatmap( corr_matrix=twin_pair_correlation_matrix_t1, gene_list=gene_list, no_regulation=no_regulation, potential_regulation=potential_regulation,
            title=rf"Twin pair correlations $\hat{{\rho}}_{{\Delta}}(t_1)$ at time {t1}h", add_gene_labels=True, add_time=True, time=[t1], gray_out_no_reg=True, black_out_self = True, symmetric = True
        )
        
        plot_matrix_as_heatmap(corr_matrix=random_pair_correlation_matrix_t1, gene_list=gene_list, no_regulation=no_regulation, potential_regulation=potential_regulation,
            title=title_random_plot, add_gene_labels=True, add_time=False, time=[t1], gray_out_no_reg=True, black_out_self = True, symmetric = True
        )

    # --- Step 3: Classify regulation type: single-state vs multiple-states ---
    if merge_time_points:
        multiple_states_gene_pairs, single_state_regulation = differentiate_single_state_reg_and_multiple_states(
            all_t1_t2_measurements, potential_regulation, twin_pair_correlation_matrix_t1, random_pair_correlation_matrix_t1, gene_list, z_score_threshold=z_score_threshold_two_states
        )
        twin_pair_correlation_matrix_t2, random_pair_correlation_matrix_t2 = calculate_twin_random_pair_correlations(
                    all_t1_t2_measurements, t2_twins, gene_list
                )
    else:
        multiple_states_gene_pairs, single_state_regulation = differentiate_single_state_reg_and_multiple_states(
            all_t1_measurements, potential_regulation, twin_pair_correlation_matrix_t1, random_pair_correlation_matrix_t1, gene_list, z_score_threshold=z_score_threshold_two_states
        )
        twin_pair_correlation_matrix_t2, random_pair_correlation_matrix_t2 = calculate_twin_random_pair_correlations(
                    all_t2_measurements, t2_twins, gene_list
                )
    if len(multiple_states_gene_pairs) > 0:

        multiple_states_no_reg, multiple_states_and_reg = identify_reg_if_multiple_states(
            twin_pair_correlation_matrix_t1,twin_pair_correlation_matrix_t2,random_pair_correlation_matrix_t1,
            random_pair_correlation_matrix_t2,multiple_states_gene_pairs,gene_list
            )
    else:
        multiple_states_no_reg, multiple_states_and_reg = [], []

    # --- Step 4: Print summary of results ---
    all_gene_pairs = list(product(gene_list, repeat=2))
    if have_any_output:
        print_summary(no_regulation, single_state_regulation, multiple_states_no_reg, multiple_states_and_reg)
    direction_matrix = pd.DataFrame()
    final_directed_edges = None
    # --- Step 5: Infer directionality of single-state interactions ---
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
            
            final_directed_edges = identify_actual_directed_edges(across_t_twin1, across_t_twin2, direction_matrix, gene_pairs=all_gene_pairs, threshold = p_value_threshold_cross_correlation, n_cores_to_use = n_cores, verbose = True)
        
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
                final_directed_edges = identify_actual_directed_edges(across_t_twin1, across_t_twin2, direction_matrix, gene_pairs=all_gene_pairs_all_reg, threshold = p_value_threshold_cross_correlation, n_cores_to_use = n_cores, verbose = True)
        else:
                final_directed_edges = []
                direction_matrix = pd.DataFrame(
                    np.zeros((len(gene_list), len(gene_list))),
                    index=gene_list,
                    columns=gene_list
                )
    else:
        direction_matrix = get_cross_correlations(across_t_twin1, across_t_twin2, gene_pairs=all_gene_pairs)
        final_directed_edges = identify_actual_directed_edges(across_t_twin1, across_t_twin2, direction_matrix, gene_pairs=all_gene_pairs, threshold = p_value_threshold_cross_correlation, n_cores_to_use = n_cores, verbose = True)
    print(final_directed_edges)
    # print(pre_threshold_direction_matrix)
    direction_matrix = direction_matrix.reindex(
    index=gene_list,
    columns=gene_list,
    fill_value=0
    )
    unfiltered_direction_matrix = direction_matrix
    if final_directed_edges:
        for i in direction_matrix.index:
            for j in direction_matrix.columns:
                if i != j and (i, j) not in final_directed_edges:
                    direction_matrix.loc[i,j] = 0
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

    # --- Step 6: Visualize the inferred network ---
    # if (len(single_state_regulation) >= 0):
    #     if have_any_output:
    #         if (len(final_directed_edges) > 0):
    #             plot_network(direction_matrix, gene_list, final_directed_edges)
    #         else:
    #             plot_network(direction_matrix, gene_list, final_directed_edges)
    try:
        result =  {
            "all_gene_pairs": all_gene_pairs,
            "gene_lists": {"no_regulation":no_regulation, "single_state_regulation":single_state_regulation, "multiple_states_no_reg": multiple_states_no_reg, "multiple_states_and_reg": multiple_states_and_reg},
            "potential_regulation": potential_regulation,
            "final_directed_edges": final_directed_edges,
            "direction_matrix": direction_matrix, 
            "unfiltered_direction_matrix": unfiltered_direction_matrix, 
            "pairwise_gene_gene_correlation_matrix": pairwise_gene_gene_correlation_matrix,
            "twin_pair_correlation_matrix_t2": twin_pair_correlation_matrix_t2,
            "random_pair_correlation_matrix_t2": random_pair_correlation_matrix_t2,
            "twin_pair_correlation_matrix_t1": twin_pair_correlation_matrix_t1,
            "random_pair_correlation_matrix_t1": random_pair_correlation_matrix_t2
        }
    except:
        result = {
            "all_gene_pairs": all_gene_pairs,
            "gene_lists": {"no_regulation":no_regulation, "single_state_regulation":single_state_regulation, "multiple_states_no_reg": multiple_states_no_reg, "multiple_states_and_reg": multiple_states_and_reg},
            "potential_regulation": potential_regulation,
            "final_directed_edges": None,
            "direction_matrix": None, 
            "unfiltered_direction_matrix": None, 
            "pairwise_gene_gene_correlation_matrix": pairwise_gene_gene_correlation_matrix,
            "random_pair_correlation_matrix_t2": random_pair_correlation_matrix_t2,
            "twin_pair_correlation_matrix_t2": twin_pair_correlation_matrix_t2,
            "twin_pair_correlation_matrix_t1": twin_pair_correlation_matrix_t1,
            "random_pair_correlation_matrix_t1": random_pair_correlation_matrix_t2,
        }
    if return_gene_corr_thresholds:
        result['gene_corr_thresholds'] = gene_corr_thresholds
        result['gene_gene_corr_p_values'] = p_values
    return result