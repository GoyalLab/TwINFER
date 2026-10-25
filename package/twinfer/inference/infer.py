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
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from itertools import product, combinations
import importlib

from twinfer.inference.correlation_functions import (
    _resolve_unit,
    get_conservative_null_draw_count,
    assign_twin_id,
    calculate_pairwise_gene_gene_correlation_matrix,
    check_system_in_steady_state,
    check_gene_gene_correlation_threshold,
    calculate_twin_random_correlations,
    generate_random_shuffle,
    generate_divergence_shuffle,
    differentiate_single_state_reg_and_multiple_states,
    calculate_gated_regulation_statistic,
    identify_reg_if_multiple_states,
    get_cross_correlations,
    identify_actual_directed_edges,
    separate_fan_outs_from_mutual_regulation,
    extract_param_index,
    read_input_matrix, 
    split_and_merge_simulations,
    plot_matrix_as_heatmap,
    print_summary,
    _build_cross_time_twins,
    _prepare_global_clone_order_maps,
    _validate_canonical_input,
    generate_step1_permutation_null,
    generate_rho_change_time_shuffle_null,
    calculate_signed_rho_change_z,
    calculate_twin_score,
)


def infer_with_twinfer(
    path_to_simulation_file=None,
    data=None,
    is_simulation_data=True,
    merge_to_multiple_states=False,
    base_config=None,
    t1=None,
    t2=None,
    gene_list_given=None,
    match_sim_details=True,
    check_for_steady_state=True,
    steady_state_tol=0.01,
    steady_state_flat_tol=None,
    steady_state_min_r2=0.80,
    steady_state_slope_tol=None,
    steady_state_min_stable_points=5,
    remove_twin_structure=False,
    seed=101010,
    use_clone=True,
    unit="clone",
    alpha_gene_gene_corr=0.01,
    n_shuffles_step1=None,
    z_score_threshold_two_states=5,
    alpha_stage3=0.01,
    n_shuffles_step2=None,
    n_shuffles_stage3=None,
    z_score_threshold_cross_correlation=2.5,
    n_shuffles_direction=None,
    n_shuffles_fanout=None,
    null_alpha=None,
    expected_null_rejections=50,
    minimum_null_draws=2000,
    use_scramble_cross_correlation=True,
    corr_threshold_cross_correlation=0.0421,
    rescue_opposite_sign_pairs=False,
    separate_fan_outs_from_mutual_regulation_flag=False,
    fan_out_z_score_threshold=8,
    n_cores=4,
    plot=False,
    verbose=False,
    return_diagnostics=False,
    ranked_list=True,
    return_raw_nulls=False,
    bypass_stage3_gate=False,
):
    """
    Run TwINFER on single-cell gene expression data.

    Input:
      Supply exactly one of `data=<DataFrame>` or
      `path_to_simulation_file=<CSV path>`.

      Required raw columns:
        clone_id, cell_id, time_step, {gene}_mRNA

      Simulation additionally requires:
        replicate

      Do not supply twin_id/pair_id. TwINFER builds twins internally.

    Steady-state check:
      When enabled for simulation data, steady_state_tol defines the relative
      steady-state band. steady_state_flat_tol controls the allowed terminal
      deviation from the detected plateau. steady_state_slope_tol controls the
      allowed directional terminal drift; if None, it defaults to
      steady_state_tol inside check_system_in_steady_state.
      steady_state_min_stable_points is the minimum number of terminal
      observations required to establish an empirical stable suffix.
      steady_state_min_r2 is retained only for the diagnostic fallback fit.

    Twin construction:
      - simulation: clone == twin;
      - real Step 2: all unordered within-time cell pairs in each clone;
      - real Step 4: all t1 x t2 cell combinations in each clone.

    use_clone:
      Controls Step 1 only.
      - False: ordinary unweighted cell-level rho + analytic cell-level null.
      - True: clone-weighted rho + clone-aware random-matching null.

    unit:
      Controls twin-derived Steps 2+ only.
      - "twin": each enumerated twin has weight 1;
      - "clone": each biological clone has total weight 1.

    Step 1 clone null when use_clone=True:
      Derange clones, then draw one target cell per source cell from the
      matched clone. No all-Cartesian-pairs averaging is used.

    Null draw count:
      Automatic mode is deliberately conservative:
          B = max(minimum_null_draws,
                  ceil(expected_null_rejections / alpha)).
      With alpha=0.01, expected_null_rejections=50 gives 5,000 draws.
      There is no adaptive early stopping. Explicit n_shuffles_* values override
      the automatic count.

    rho_Delta / d references:
      Heterogeneity (het) is the earlier fresh-random-pair null: twins are
      replaced by unrelated cell pairs sampled clone-first, then cell-within-
      clone. Its within-time score is z_het. This remains the null used by the
      existing Step-2 classification and by the existing Stage-3 regulation
      call, where the corresponding change score is z_d_het.

      Divergence (div) is Yuval's fixed-Delta shuffle: every observed Delta_X
      and Delta_Y is kept exactly, Delta_X stays fixed, and Delta_Y is re-paired
      one-to-one across biological clones. Its within-time score is z_div. For
      d_div, the same fixed-Delta null is applied independently at t1 and t2
      and the null difference rho_Delta_div(t2)-rho_Delta_div(t1) is used to
      calculate z_d_div.

      Adding div does not silently change the existing classification flow;
      the old generic Step-2/Step-3 fields are retained as aliases for het.

    Direction nulls:
      The raw signed rho_cross null is prepared and returned separately even
      when use_scramble_cross_correlation=False. gamma uses the derived null
      |rho_xy_null| - |rho_yx_null|.

    rescue_opposite_sign_pairs (default False):
      A pair that fails Step 1 (no_regulation) can still hide a real
      negative-feedback relationship (e.g. A represses B, B activates A):
      the same-time gene-gene correlation can wash out to ~0 even though
      both directional relationships are real. When True, every
      no_regulation pair's two raw signed cross-correlations
      (rho_{a(t1)->b(t2)}, rho_{b(t1)->a(t2)}) are computed; pairs whose
      cross-correlations are opposite-signed (and both finite/nonzero) are
      added to Step 4's direction_candidates -- bypassing Step 2's
      twin-vs-random gate entirely for these pairs -- so they go through
      the same permutation null test (identify_actual_directed_edges) as
      every other candidate. Rescued pairs that come back significant are
      added to final_directed_edges/direction_matrix like any other edge.
      result["direction"]["rescued_from_no_regulation"] only reports pairs
      that qualified for the opposite-sign screen AND came back significant
      (i.e. ended up in final_directed_edges) -- a pair that entered the
      candidate pool but failed Step 4 was not actually rescued. Default
      False so this does not change behavior for existing callers.

    ranked_list:
      If True, calculate the latest TwinScore for every non-self directed
      candidate tested in Step 4. For the rho and gamma terms, the code first
      computes the permutation-null Z-score z(statistic), then standardizes
      those Z-scores across the specific candidate panel as s(z(statistic)).
      The score also uses the canonical z_div (Yuval fixed-Delta), z_het
      (fresh random-pair), and z_d_het terms; literal pi; and signed gamma =
      |rho_cross_xy|-|rho_cross_yx|. Rows are ranked by signed TwinScore
      descending and negative scores are retained. The additional signed
      z(rho(t2)-rho(t1)) and s(z(rho(t2)-rho(t1))) are returned as
      z_rho_change and s_z_rho_change but do not alter TwinScore.

    return_raw_nulls (default False, purely additive -- omitting it changes nothing):
      If True, adds a "raw_nulls" key to the output with the full raw
      permutation-null draws already computed internally for the expensive
      steps, keyed by "{gene_a}__{gene_b}" (npz-friendly, since npz can't key
      on tuples): step1 (Stage-I co-expression null matrix, only present when
      use_clone=True), rho_delta_het_t1 (z_het's null, generate_random_shuffle
      at t1), rho_delta_div_t1 / rho_delta_div_t2 (z_div's null,
      generate_divergence_shuffle at t1/t2), d_het / d_div (z_d_het's /
      z_d_div's null for d = rho_Delta(t2)-rho_Delta(t1) -- the same
      fresh-random-pair construction as z_het/z_div, just random pairs
      instead of twin pairs, jointly resampled across t1/t2 -- only present
      for pairs that reached Stage 3), and, only when ranked_list=True and
      there is a non-empty directed panel, rho_t1_for_twinscore /
      rho_t2_for_twinscore (the Stage-I null reused/regenerated for TwinScore's
      |rho(t1)|/|rho(t2)| terms), and rho_change_for_twinscore (the signed
      rho_pseudo(t2)-rho_pseudo(t1) draws from cell-level time-label
      permutations; the absolute change term uses the absolute values of these
      same draws). The direction/gamma null
      (rho_cross_null_details) is already returned uncondensed under
      result["direction"]["rho_cross_null"] and is not duplicated here.
      Regenerating any of these from scratch is exactly what makes a rerun
      slow, so save this dict (e.g. to .npz) if you want it without rerunning.

    Output:
      Stable nested dict with settings, classification, directed_edges,
      correlations, direction, heterogeneity, divergence, gated_regulation,
      stage3, fan_out, twin_score_inputs, ranked_edges, optional diagnostics,
      and optional raw_nulls (see return_raw_nulls above).
    """

    # Load and validate raw input.
    if data is not None and path_to_simulation_file is not None:
        raise ValueError("Supply either data= or path_to_simulation_file=, not both")

    if data is not None:
        if merge_to_multiple_states:
            raise ValueError(
                "merge_to_multiple_states is only supported for file-path input"
            )
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        single_cell_data = data.copy()
    else:
        if path_to_simulation_file is None:
            raise ValueError("Supply either data= or path_to_simulation_file=")

        if merge_to_multiple_states:
            if isinstance(path_to_simulation_file, (str, os.PathLike)):
                single_cell_data = pd.read_csv(path_to_simulation_file)
            else:
                single_cell_data = split_and_merge_simulations(
                    path_to_simulation_file
                )
        else:
            if not isinstance(path_to_simulation_file, (str, os.PathLike)):
                raise TypeError(
                    "path_to_simulation_file must be one CSV path when "
                    "merge_to_multiple_states=False"
                )
            single_cell_data = pd.read_csv(path_to_simulation_file)

    _validate_canonical_input(single_cell_data, is_simulation_data)

    if not isinstance(use_clone, (bool, np.bool_)):
        raise ValueError("use_clone must be True or False")

    # `unit` is intentionally independent of Step 1.
    # It controls twin-derived statistics in Steps 2+ only.
    unit = _resolve_unit(unit)

    # Automatic null sizes use the strictest alpha already requested in the
    # pipeline. Explicit n_shuffles_* integers override this.
    if null_alpha is None:
        null_alpha = min(
            float(alpha_gene_gene_corr),
            float(alpha_stage3),
        )

    null_draws = {
        "step1": get_conservative_null_draw_count(
            n_shuffles_step1,
            alpha=alpha_gene_gene_corr,
            expected_rejections=expected_null_rejections,
            minimum=minimum_null_draws,
        ),
        "step2": get_conservative_null_draw_count(
            n_shuffles_step2,
            alpha=null_alpha,
            expected_rejections=expected_null_rejections,
            minimum=minimum_null_draws,
        ),
        "stage3": get_conservative_null_draw_count(
            n_shuffles_stage3,
            alpha=alpha_stage3,
            expected_rejections=expected_null_rejections,
            minimum=minimum_null_draws,
        ),
        "direction": get_conservative_null_draw_count(
            n_shuffles_direction,
            alpha=null_alpha,
            expected_rejections=expected_null_rejections,
            minimum=minimum_null_draws,
        ),
        "fan_out": get_conservative_null_draw_count(
            n_shuffles_fanout,
            alpha=null_alpha,
            expected_rejections=expected_null_rejections,
            minimum=minimum_null_draws,
        ),
    }

    n_clones_data = int(single_cell_data["clone_id"].nunique())
    time_points_data = single_cell_data["time_step"].unique()

    if gene_list_given:
        gene_list = list(gene_list_given)
    else:
        gene_list = [
            c[:-5] for c in single_cell_data.columns
            if c.endswith("_mRNA")
        ]
    if not gene_list:
        raise ValueError("No *_mRNA gene-expression columns were found.")

    unique_timepoints = single_cell_data["time_step"].unique()
    if t1 not in unique_timepoints:
        raise ValueError(f"Time point t1={t1} not found in input['time_step'].")
    if t2 not in unique_timepoints:
        raise ValueError(f"Time point t2={t2} not found in input['time_step'].")

    if is_simulation_data:
        # Simulation-only metadata/sanity checks. These checks do not change
        # the canonical input schema used by the inference calculations.
        if match_sim_details:
            if base_config is None:
                raise ValueError(
                    "base_config is required when is_simulation_data=True "
                    "and match_sim_details=True."
                )

            n_clones_base_config = int(base_config["n_cells"])
            time_points_base_config = np.arange(
                0,
                base_config["twin_simulation_time_after_division"]
                + base_config["twin_measurement_resolution"],
                base_config["twin_measurement_resolution"],
            )

            assert n_clones_data == n_clones_base_config, (
                "Number of biological clones in the input does not match "
                "n_cells in base_config."
            )
            assert set(time_points_data) == set(time_points_base_config), (
                "The input time points do not match base_config."
            )

            if data is None and isinstance(path_to_simulation_file, str):
                param_index_from_file_name = extract_param_index(
                    path_to_simulation_file
                )
                param_index_from_base_config = "_".join(
                    map(str, base_config["rows_to_use"][0])
                )
                assert param_index_from_file_name == param_index_from_base_config, (
                    f"Simulation parameters ({param_index_from_file_name}) must "
                    f"match parameter rows ({param_index_from_base_config})."
                )

        if check_for_steady_state and match_sim_details:
            is_system_in_steady_state, steady_state_summary = check_system_in_steady_state(
                simulation_df=single_cell_data,
                gene_list=gene_list,
                steady_tol=steady_state_tol,
                flat_tol=steady_state_flat_tol,
                min_r2=steady_state_min_r2,
                slope_tol=steady_state_slope_tol,
                min_stable_points=steady_state_min_stable_points,
            )
            if not is_system_in_steady_state:
                print(steady_state_summary)
                raise ValueError(
                    "The system is not in steady state. "
                    "You can override this by setting check_for_steady_state=False."
                )

        if "replicate" not in single_cell_data.columns:
            raise ValueError(
                "Canonical simulation input must contain 'replicate' so the two "
                "simulated sibling lineages can be followed across time."
            )

        replicates = (
            single_cell_data["replicate"]
            .drop_duplicates()
            .sort_values()
            .to_numpy()
        )
        if len(replicates) < 2:
            raise ValueError("Simulation input requires at least two replicate labels.")

        if remove_twin_structure:
            rng = np.random.default_rng(seed)
            unique_clones = (
                single_cell_data["clone_id"]
                .drop_duplicates()
                .sort_values()
                .to_numpy()
            )
            if len(unique_clones) < 2:
                raise ValueError("Need at least 2 clones to build a derangement.")

            # Keep the first replicate as reference; derange every other replicate
            # consistently across time so within-cell temporal continuity is preserved.
            for rep in replicates[1:]:
                shuffled = unique_clones.copy()
                while np.any(shuffled == unique_clones):
                    rng.shuffle(shuffled)
                shuffle_map = dict(zip(unique_clones, shuffled))
                mask = single_cell_data["replicate"] == rep
                single_cell_data.loc[mask, "clone_id"] = (
                    single_cell_data.loc[mask, "clone_id"].map(shuffle_map)
                )

        # Use actual clone labels, not an assumed 0..N-1 integer range.
        rng = np.random.default_rng(seed)
        clone_ids = single_cell_data["clone_id"].drop_duplicates().to_numpy()
        clone_ids_shuffled = rng.permutation(clone_ids)

        n1 = n2 = len(clone_ids_shuffled) // 4
        t1_clones = clone_ids_shuffled[:n1]
        t2_clones = clone_ids_shuffled[n1:n1 + n2]
        across_t_clones = clone_ids_shuffled[n1 + n2:]

        t1_twins_raw = single_cell_data[
            single_cell_data["clone_id"].isin(t1_clones)
            & (single_cell_data["time_step"] == t1)
        ].copy()
        t2_twins_raw = single_cell_data[
            single_cell_data["clone_id"].isin(t2_clones)
            & (single_cell_data["time_step"] == t2)
        ].copy()

        # Simulation: clone == twin. For the across-time subset, use one selected\n        # sibling at t1 and the other at t2, giving one cross-time twin per clone.
        across_t_left_raw = single_cell_data[
            single_cell_data["clone_id"].isin(across_t_clones)
            & (single_cell_data["time_step"] == t1)
            & (single_cell_data["replicate"] == replicates[0])
        ].copy()
        across_t_right_raw = single_cell_data[
            single_cell_data["clone_id"].isin(across_t_clones)
            & (single_cell_data["time_step"] == t2)
            & (single_cell_data["replicate"] == replicates[1])
        ].copy()

        across_t_twin1, across_t_twin2 = _build_cross_time_twins(
            across_t_left_raw,
            across_t_right_raw,
        )

        # Ordinary rho uses raw cells only. The cross-time twin expansion is
        # deliberately NOT concatenated here.
        rho_t1_measurements = pd.concat(
            [t1_twins_raw, across_t_left_raw],
            ignore_index=True,
        )
        rho_t2_measurements = pd.concat(
            [t2_twins_raw, across_t_right_raw],
            ignore_index=True,
        )

    else:
        # Experimental/real data must already have been converted upstream to
        # the same canonical clone_id/cell_id/time_step schema. No LARRY-specific
        # column guessing, suffix parsing, or clone-map loading happens here.
        if remove_twin_structure:
            cell_rank = (
                single_cell_data[["clone_id", "cell_id"]]
                .drop_duplicates()
                .sort_values(["clone_id", "cell_id"])
                .assign(_slot=lambda d: d.groupby("clone_id").cumcount())
                .set_index("cell_id")["_slot"]
            )
            single_cell_data["_slot"] = single_cell_data["cell_id"].map(cell_rank)
            slots = (
                single_cell_data["_slot"]
                .drop_duplicates()
                .sort_values()
                .to_numpy()
            )

            rng = np.random.default_rng(seed)
            unique_clones = (
                single_cell_data["clone_id"]
                .drop_duplicates()
                .sort_values()
                .to_numpy()
            )
            if len(unique_clones) < 2:
                raise ValueError("Need at least 2 clones to build a derangement.")

            for ts in single_cell_data["time_step"].drop_duplicates():
                time_mask = single_cell_data["time_step"] == ts
                for slot in slots[1:]:
                    shuffled = unique_clones.copy()
                    while np.any(shuffled == unique_clones):
                        rng.shuffle(shuffled)
                    shuffle_map = dict(zip(unique_clones, shuffled))
                    mask = time_mask & (single_cell_data["_slot"] == slot)
                    single_cell_data.loc[mask, "clone_id"] = (
                        single_cell_data.loc[mask, "clone_id"].map(shuffle_map)
                    )
            single_cell_data = single_cell_data.drop(columns="_slot")

        t1_twins_raw = single_cell_data[
            single_cell_data["time_step"] == t1
        ].copy()
        t2_twins_raw = single_cell_data[
            single_cell_data["time_step"] == t2
        ].copy()

        # Real-data cross-time observations are the Cartesian product of t1 and
        # t2 cells within each biological clone, with a fresh canonical twin_id.
        across_t_twin1, across_t_twin2 = _build_cross_time_twins(
            t1_twins_raw,
            t2_twins_raw,
        )

        # Crucial: rho is calculated on the original one-row-per-cell table,
        # never on the twin-expanded cross-time table.
        rho_t1_measurements = t1_twins_raw.reset_index(drop=True)
        rho_t2_measurements = t2_twins_raw.reset_index(drop=True)

    if t1_twins_raw.empty or t2_twins_raw.empty:
        raise ValueError("No within-time cells remain for t1 or t2 after subsetting.")

    if is_simulation_data:
        # Simulation: clone == twin, so each selected within-time clone must
        # contain exactly the two sibling cells.
        for name, frame in (("t1", t1_twins_raw), ("t2", t2_twins_raw)):
            counts = frame.groupby("clone_id", sort=False).size()
            bad = counts[counts != 2]
            if not bad.empty:
                clone_id = bad.index[0]
                raise ValueError(
                    f"Simulation {name}: clone_id={clone_id!r} has "
                    f"{int(bad.iloc[0])} cells; expected exactly 2 "
                    "because simulation clone == twin."
                )

    # Build within-time twins internally.
    # Simulation: one twin per clone.
    # Real data: all unordered within-clone cell pairs.
    t1_twins = assign_twin_id(t1_twins_raw).reset_index(drop=True)
    t2_twins = assign_twin_id(t2_twins_raw).reset_index(drop=True)
    across_t_twin1 = across_t_twin1.reset_index(drop=True)
    across_t_twin2 = across_t_twin2.reset_index(drop=True)
    rho_t1_measurements = rho_t1_measurements.reset_index(drop=True)
    rho_t2_measurements = rho_t2_measurements.reset_index(drop=True)

    # Step 1: gene-gene correlation / edge-existence screen
    pairwise_gene_gene_correlation_matrix = calculate_pairwise_gene_gene_correlation_matrix(
        rho_t1_measurements,
        gene_list,
        use_clone=use_clone,
    )
           
    # -------------------------------------------------------------------------
    # Official name: Gene-expression correlation at time t (TwINFER Stage I)
    # LaTeX: \rho(t)
    # Permutation method: when use_clone=True, keep gene X and the source-cell
    # weights fixed, randomly derange the biological clones, and for each source
    # cell sample one cell from its assigned different clone to supply gene Y.
    # Recompute the clone-weighted Spearman \rho(t) for every shuffle and compare
    # the observed \rho(t) with that null. When use_clone=False, Step I uses the
    # analytic no-correlation null with mean 0 and SD 1/sqrt(N-1).

    step1_z_critical = float(
        scipy.stats.norm.ppf(1.0 - alpha_gene_gene_corr / 2.0)
    )
    step1_result = check_gene_gene_correlation_threshold(
        rho_t1_measurements,
        pairwise_gene_gene_correlation_matrix,
        gene_list,
        use_scramble=True,
        z_score_threshold=step1_z_critical,
        verbose=verbose,
        use_clone=use_clone,
        n_shuffles=null_draws["step1"],
        return_null_matrix=bool(ranked_list),
    )
    if ranked_list:
        (
            no_regulation,
            potential_regulation,
            gene_corr_null_stats,
            z_scores,
            step1_t1_null_matrix,
        ) = step1_result
    else:
        (
            no_regulation,
            potential_regulation,
            gene_corr_null_stats,
            z_scores,
        ) = step1_result
        step1_t1_null_matrix = None
    if plot:
        rho_label = "clone-weighted " if unit == "clone" else ""
        title = rf"{rho_label}Gene correlations $\rho$ with cells from time {t1}"
        plot_matrix_as_heatmap(corr_matrix=pairwise_gene_gene_correlation_matrix, gene_list=gene_list, no_regulation=no_regulation, potential_regulation=potential_regulation,
            title=title, add_gene_labels=True, add_time=False, gray_out_no_reg=False, black_out_self = True
        )

    # -------------------------------------------------------------------------
    # Official name: Random-pair difference correlation
    # LaTeX: \rho_{\Delta}(t)
    # Permutation method: for every observed twin slot, draw a fresh unrelated
    # pair by choosing clone A uniformly, choosing a different clone B uniformly,
    # then choosing one cell uniformly from each clone. Use that same random cell
    # pair to form both \Delta X and \Delta Y, recompute the weighted Spearman
    # difference correlation across all slots, and repeat to build the null.
    # In clone mode, the source twin slots carry weights so each biological clone
    # has total weight 1. The t1 draws are reused by z_het and the fan-out check.
    # (same random-pair construction as z_d_het at each time point)
    delta_null_draws = max(
        null_draws["step2"],
        null_draws["fan_out"],
    )
    t1_delta_null_full = generate_random_shuffle(
        t1_twins,
        gene_list=gene_list,
        gene_pairs=potential_regulation,
        n_shuffles=delta_null_draws,
        unit=unit,
        raw_cells=rho_t1_measurements,
        random_state=int(seed) + 271828,
        n_cores_to_use=n_cores,
    )

    # -------------------------------------------------------------------------
    # Official name: Twin difference correlation
    # LaTeX: \hat{\rho}_{\Delta}(t)
    # Z-score: z_div
    # Permutation method: keep every observed \Delta X and \Delta Y value exactly
    # as measured, hold the \Delta X rows fixed, and reassign the complete
    # \Delta Y rows one-to-one across twin rows, never to a row from the same
    # biological clone. Every \Delta Y row is used exactly once in each shuffle.
    # The t1 and t2 fixed-Delta nulls are generated with independent RNG streams.
    # (same with z_d_div at each time point)
    div_delta_null_draws = max(
        null_draws["step2"],
        null_draws["stage3"],
    )
    t1_div_null_full = generate_divergence_shuffle(
        t1_twins,
        gene_list=gene_list,
        gene_pairs=potential_regulation,
        n_shuffles=div_delta_null_draws,
        unit=unit,
        random_state=int(seed) + 271832,
        n_cores_to_use=n_cores,
    )
    t2_div_null_full = generate_divergence_shuffle(
        t2_twins,
        gene_list=gene_list,
        gene_pairs=potential_regulation,
        n_shuffles=div_delta_null_draws,
        unit=unit,
        random_state=int(seed) + 271833,
        n_cores_to_use=n_cores,
    )

    # Step 2: compare twin difference-correlation with the random reference
    title_random_plot = rf"Random-reference difference correlation $\rho_{{\Delta}}$ using cells at time {t1}"
    twin_delta_correlation_matrix_t1, random_delta_reference_matrix_t1 = calculate_twin_random_correlations(
            rho_t1_measurements,
            t1_twins,
            gene_list,
            unit=unit,
            random_state=int(seed) + 271829,
        )

    if plot:
        plot_matrix_as_heatmap( corr_matrix=twin_delta_correlation_matrix_t1, gene_list=gene_list, no_regulation=no_regulation, potential_regulation=potential_regulation,
            title=rf"Twin correlations $\hat{{\rho}}_{{\Delta}}(t_1)$ at time {t1}h", add_gene_labels=True, add_time=False, time=[t1], gray_out_no_reg=True, black_out_self = True, symmetric = True
        )
        
        plot_matrix_as_heatmap(corr_matrix=random_delta_reference_matrix_t1, gene_list=gene_list, no_regulation=no_regulation, potential_regulation=potential_regulation,
            title=title_random_plot, add_gene_labels=True, add_time=False, time=[t1], gray_out_no_reg=True, black_out_self = True, symmetric = True
        )

    twin_delta_correlation_matrix_t2, random_delta_reference_matrix_t2 = calculate_twin_random_correlations(
            rho_t2_measurements,
            t2_twins,
            gene_list,
            unit=unit,
            random_state=int(seed) + 271830,
        )
    
    # -------------------------------------------------------------------------
    # Official name: Twin difference correlation at time t
    # LaTeX: observed twins \hat{\rho}_{\Delta}(t); random-pair reference
    # \rho_{\Delta}(t)
    # Z-score: z_het
    # Permutation method: for every observed twin slot, draw a fresh unrelated
    # pair by choosing clone A uniformly, choosing a different clone B uniformly,
    # then choosing one cell uniformly from each clone. Use that same random cell
    # pair to form both \Delta X and \Delta Y, recompute weighted Spearman across
    # all slots, and repeat to form the random-pair \rho_{\Delta}(t) distribution.
    # z_het compares the observed \hat{\rho}_{\Delta}(t) with that distribution.
    # (same with the fan-out check; same random-pair construction as z_d_het
    # at each time point)

    (
        multiple_states_gene_pairs,
        single_state_regulation,
        multi_state_null_stats,
        z_het_scores,
        divergence_details,
    ) = differentiate_single_state_reg_and_multiple_states(
            rho_t1_measurements, potential_regulation, twin_delta_correlation_matrix_t1, random_delta_reference_matrix_t1, gene_list,
            z_score_threshold=z_score_threshold_two_states,
            verbose=verbose,
            unit=unit,
            n_shuffles=null_draws["step2"],
            null_alpha=null_alpha,
            expected_null_rejections=expected_null_rejections,
            minimum_null_draws=minimum_null_draws,
            precomputed_null=t1_delta_null_full,
            n_cores_to_use=n_cores,
            divergence_random_state=int(seed) + 271832,
            precomputed_divergence_null=t1_div_null_full,
            return_divergence_details=True,
        )
    # -------------------------------------------------------------------------
    # Official name: Twin difference correlation
    # LaTeX: \hat{\rho}_{\Delta}(t)
    # Z-score: z_div
    # Permutation method: keep every observed \Delta X and \Delta Y value exactly
    # as measured, hold \Delta X fixed, and reassign the complete \Delta Y rows
    # one-to-one across other biological clones. Recompute weighted Spearman for
    # every shuffle and Z-score the observed \hat{\rho}_{\Delta}(t) against it.
    # (same with z_d_div at each time point)
    z_div = {
        pair: details["z_div"]
        for pair, details in divergence_details.items()
    }

    # -------------------------------------------------------------------------
    # Official name: Heterogeneity-gated regulation statistic
    # LaTeX:
    # \hat{\rho}_{reg}(\lambda)
    #     = \rho_{same} - \lambda \rho_{cross},
    # \lambda = \min(1, |z_{het}| / 2.33).
    # Z-score: z_reg_gated
    # Permutation method: keep every X sibling block fixed and reassign the
    # complete Y sibling blocks (y^a, y^b) one-to-one across twin rows, never
    # to a row from the same biological clone. Randomly flip the two members
    # of each assigned Y block because within-time twin orientation is
    # arbitrary. Use the same reassignment for rho_same_null and
    # rho_cross_null, then calculate
    # rho_reg_null = rho_same_null - lambda * rho_cross_null.
    # Twin mode gives every enumerated twin weight 1.
    # Clone mode gives every biological clone total weight 1, keeps the source
    # clone weights fixed, and recomputes exact weighted Y ranks after every
    # reassignment. (same cross-clone reassignment principle as z_div)
    #
    # z_het and z_reg_gated use the same inference unit. This statistic is
    # returned as an additional diagnostic and does not silently change the
    # existing Step-2 or Step-3 classification calls.
    # 2026-09-17: this loop was serial (one gene pair at a time, each pair's
    # n_shuffles permutations also computed in a plain Python for-loop inside
    # calculate_gated_regulation_statistic) regardless of n_cores -- measured
    # CPU efficiency 2.10% of a 50-core allocation. Pairs are embarrassingly
    # parallel (each call only reads the shared, unmutated t1_twins and seeds
    # its own np.random.default_rng fresh -- no cross-pair state), so this now
    # farms pairs out across n_cores threads, matching the
    # Parallel(prefer="threads", require="sharedmem") + threadpool_limits
    # pattern already used for the shuffle-chunking in correlation_functions.py.
    # Reusing the same random_state for every pair (see below) makes each
    # pair's result independent of which other pairs ran alongside it or in
    # what order, so this is bit-identical to the old serial loop.
    # old serial version:
    # gated_regulation_details = {}
    # for gene_i, gene_j in potential_regulation:
    #     pair = (gene_i, gene_j)
    #     sorted_pair = tuple(sorted(pair))
    #     z_het_for_pair = z_het_scores.get(
    #         pair,
    #         z_het_scores.get(sorted_pair),
    #     )
    #     gated_regulation_details[pair] = (
    #         calculate_gated_regulation_statistic(
    #             t1_twins,
    #             gene_i,
    #             gene_j,
    #             z_het=z_het_for_pair,
    #             n_shuffles=null_draws["step2"],
    #             random_state=int(seed) + 271835,
    #             unit=unit,
    #         )
    #     )
    def _one_gated_regulation(gene_i, gene_j):
        pair = (gene_i, gene_j)
        sorted_pair = tuple(sorted(pair))
        z_het_for_pair = z_het_scores.get(
            pair,
            z_het_scores.get(sorted_pair),
        )
        # Reusing the same seed intentionally gives all gene pairs the same
        # matched sequence of Y-block reassignments.
        return pair, calculate_gated_regulation_statistic(
            t1_twins,
            gene_i,
            gene_j,
            z_het=z_het_for_pair,
            n_shuffles=null_draws["step2"],
            random_state=int(seed) + 271835,
            unit=unit,
        )

    potential_regulation_list = list(potential_regulation)
    n_jobs = max(1, min(int(n_cores), len(potential_regulation_list))) if potential_regulation_list else 1
    if n_jobs == 1:
        pair_results = [
            _one_gated_regulation(gene_i, gene_j)
            for gene_i, gene_j in potential_regulation_list
        ]
    else:
        with threadpool_limits(limits=1, user_api="blas"):
            pair_results = Parallel(
                n_jobs=n_jobs,
                prefer="threads",
                require="sharedmem",
            )(
                delayed(_one_gated_regulation)(gene_i, gene_j)
                for gene_i, gene_j in potential_regulation_list
            )
    gated_regulation_details = dict(pair_results)

    z_reg_gated = {
        pair: details["z_reg_gated"]
        for pair, details in gated_regulation_details.items()
    }
    
    # -------------------------------------------------------------------------
    # Official name: Change in twin difference correlation across time
    # LaTeX: \hat{\rho}_{\Delta}(t_1), \hat{\rho}_{\Delta}(t_2)
    # LaTeX:
    # d = \hat{\rho}_{\Delta}(t_2) - \hat{\rho}_{\Delta}(t_1)
    #
    # Z-score: z_d_het
    # Permutation method: jointly resample source-clone blocks across t1 and t2
    # when the same biological clone appears at both times. Then, independently
    # at each time point, replace every selected twin slot with a fresh unrelated
    # pair: choose clone A uniformly, choose clone B != A uniformly, and choose
    # one cell uniformly within each clone. Compute weighted random-pair
    # \rho_{\Delta}(t_1) and \rho_{\Delta}(t_2), subtract t1 from t2 for each
    # draw, and Z-score the observed d against that difference distribution.
    # Twin mode gives each twin weight 1; clone mode gives each source clone total
    # weight 1. (same random-pair construction as z_het at each time point)
    #
    # Z-score: z_d_div
    # Permutation method: at each time point, keep all observed \Delta X and
    # \Delta Y values fixed, hold \Delta X fixed, and reassign the complete
    # \Delta Y rows one-to-one across other biological clones. Recompute weighted
    # Spearman at t1 and t2, subtract t1 from t2, and Z-score the observed d
    # against that difference distribution. (same with z_div at each time point)
    # Stage 3's observed quantity is d = rho_Delta(t2) - rho_Delta(t1). When
    # t1 == t2 (single-timepoint data, e.g. no second time available), t1_twins
    # and t2_twins are built from the identical set of cells, so d is exactly
    # 0.000000 for every pair and z_d_het never clears alpha_stage3 -- Stage 3
    # would silently call every pair "no regulation" regardless of true signal,
    # leaving multiple_states_and_reg permanently empty. In that case, resolve
    # these pairs instead with z_reg_gated (calculate_gated_regulation_statistic,
    # computed above from t1_twins alone -- already single-timepoint-native,
    # not a t1-vs-t2 comparison), at the same alpha Stage 3 would have used.
    # z_d_het/z_d/d are left absent for these pairs (read as NaN downstream),
    # which correctly zeroes out TwinScore's "- z_het * I(z_d_het > 0)" term --
    # that penalty is itself a temporal-change measure and has no single-
    # timepoint analog.
    single_timepoint = (t1 == t2)
    stage3_raw_nulls = {}
    if len(multiple_states_gene_pairs) > 0:
        if single_timepoint:
            z_critical_gated = float(scipy.stats.norm.ppf(1.0 - alpha_stage3))
            multiple_states_no_reg = []
            multiple_states_and_reg = []
            stage3_details = {}
            for pair in multiple_states_gene_pairs:
                sorted_pair = tuple(sorted(pair))
                z = z_reg_gated.get(pair, z_reg_gated.get(sorted_pair))
                # 2026-09-22 user request ("remove all gates", analogous to Stage I's
                # alpha_gene_gene_corr relaxation): the normal test is one-sided
                # (z > z_critical_gated), so pairs with a strong NEGATIVE z_reg_gated
                # (real signal, opposite sign) are structurally excluded no matter how
                # permissive alpha_stage3 is -- confirmed empirically (208/1122 FM06
                # correlation_high pairs excluded at alpha_stage3=0.999, ALL with
                # negative z_reg_gated, range -3.05 to -39.4). bypass_stage3_gate=True
                # skips this significance test entirely so every pair reaching Stage 3
                # proceeds to Step 4 and gets a real computed twinScore, regardless of
                # z_reg_gated's sign or magnitude.
                if bypass_stage3_gate:
                    passed = z is not None and np.isfinite(z)
                else:
                    passed = z is not None and np.isfinite(z) and z > z_critical_gated
                (multiple_states_and_reg if passed else multiple_states_no_reg).append(pair)
                stage3_details[pair] = {
                    "method": "single_timepoint_gated_regulation",
                    "z_reg_gated": z,
                    "z_critical": z_critical_gated,
                    "call": "regulation" if passed else "no regulation",
                    "gate_bypassed": bool(bypass_stage3_gate),
                }
                if verbose:
                    print(
                        f"gene 1: {pair[0]}, gene 2: {pair[1]}, single-timepoint "
                        f"(t1=t2={t1}): z_reg_gated={z}, critical={z_critical_gated:.4f}, "
                        f"call={'regulation' if passed else 'no regulation'}"
                    )
        else:
            stage3_result = identify_reg_if_multiple_states(
                twin_delta_correlation_matrix_t1,
                twin_delta_correlation_matrix_t2,
                random_delta_reference_matrix_t1,
                random_delta_reference_matrix_t2,
                multiple_states_gene_pairs,
                gene_list,
                t1_twins,
                t2_twins,
                t1_raw=rho_t1_measurements,
                t2_raw=rho_t2_measurements,
                alpha=alpha_stage3,
                n_shuffles=null_draws["stage3"],
                shuffle_seed=int(seed) + 271831,
                unit=unit,
                expected_null_rejections=expected_null_rejections,
                minimum_null_draws=minimum_null_draws,
                precomputed_t1_null=None,
                shuffle_seeds=None,
                global_clone_map=None,
                n_cores_to_use=n_cores,
                divergence_shuffle_seed=int(seed) + 271834,
                precomputed_divergence_t1_null=t1_div_null_full,
                precomputed_divergence_t2_null=t2_div_null_full,
                return_raw_nulls=return_raw_nulls,
            )
            if return_raw_nulls:
                (
                    multiple_states_no_reg,
                    multiple_states_and_reg,
                    stage3_details,
                    stage3_raw_nulls,
                ) = stage3_result
            else:
                multiple_states_no_reg, multiple_states_and_reg, stage3_details = stage3_result
    else:
        multiple_states_no_reg, multiple_states_and_reg, stage3_details = [], [], {}

    #Print summary of results ---
    all_gene_pairs = list(product(gene_list, repeat=2))
    if verbose:
        print_summary(no_regulation, single_state_regulation, multiple_states_no_reg, multiple_states_and_reg)
    
    # Step 4: infer direction for every pair that RETAINS regulation evidence.
    # The same direction method is used for single_state_regulation and
    # multiple_states_and_reg. multiple_states_no_reg stops after Step 3.
    direction_matrix = pd.DataFrame()
    final_directed_edges = set()
    directed_z_scores = {}
    direction_diff_details = {}
    rho_cross_null_details = {}

    # Pairs that failed Step 1 (no_regulation) can still hide a real
    # negative-feedback relationship whose same-time correlation washes out
    # to ~0. Screen them by their raw signed cross-correlations: opposite
    # signs are the toggle-switch/negative-feedback signature. Pairs that
    # pass this screen are folded into direction_candidates so they go
    # through the exact same permutation null test as every other
    # candidate below -- no separate significance-testing code path.
    rescued_from_no_regulation = []
    if rescue_opposite_sign_pairs and no_regulation:
        no_reg_bidirectional = list({
            (a, b) for (a, b) in no_regulation
        } | {
            (b, a) for (a, b) in no_regulation
        })
        no_reg_cross_matrix = get_cross_correlations(
            across_t_twin1,
            across_t_twin2,
            gene_pairs=no_reg_bidirectional,
            unit=unit,
        )
        for a, b in no_regulation:
            rho_ab = float(no_reg_cross_matrix.loc[a, b])
            rho_ba = float(no_reg_cross_matrix.loc[b, a])
            if (
                np.isfinite(rho_ab) and np.isfinite(rho_ba)
                and rho_ab != 0 and rho_ba != 0
                and np.sign(rho_ab) != np.sign(rho_ba)
            ):
                rescued_from_no_regulation.append((a, b))

    direction_candidates = list(dict.fromkeys(
        single_state_regulation + multiple_states_and_reg + rescued_from_no_regulation
    ))

    if direction_candidates:
        bidirectional_pairs = {
            (a, b) for (a, b) in direction_candidates
        } | {
            (b, a) for (a, b) in direction_candidates
        }

        genes = {g for pair in direction_candidates for g in pair}
        self_pairs = {(g, g) for g in genes}
        direction_gene_pairs = list(bidirectional_pairs | self_pairs)

        direction_matrix = get_cross_correlations(
            across_t_twin1,
            across_t_twin2,
            gene_pairs=direction_gene_pairs,
            unit=unit,
        )

        # ---------------------------------------------------------------------
        # Official name: Twin cross-correlation for direction x -> y
        # LaTeX: \hat{\rho}^{\dagger}_{x(t_1)\to y(t_2)}
        # Permutation method: keep the t1 rows fixed and reassign the complete t2
        # rows one-to-one across the cross-time twin table, with the rule that a
        # t1 row can never receive a t2 row from the same biological clone.
        # Recompute the weighted x(t1)-to-y(t2) Spearman correlation for every
        # shuffle; z_rho_cross compares the observed twin cross-correlation with
        # this random-pair cross-correlation distribution.
        # (same with the reverse direction and gamma)
        #
        # Official name: Directional asymmetry
        # LaTeX: \gamma =
        # |\hat{\rho}^{\dagger}_{x(t_1)\to y(t_2)}|
        # - |\hat{\rho}^{\dagger}_{y(t_1)\to x(t_2)}|
        # Permutation method: use the same paired cross-clone permutation draws
        # from the two directional cross-correlations; for every draw calculate
        # |rho_xy_null| - |rho_yx_null| and Z-score the observed signed gamma
        # against that null. (same with z_rho_cross in both directions)

        (
            final_directed_edges,
            directed_z_scores,
            direction_diff_details,
            rho_cross_null_details,
        ) = identify_actual_directed_edges(
            across_t_twin1,
            across_t_twin2,
            direction_matrix,
            gene_pairs=direction_gene_pairs,
            z_score_threshold=z_score_threshold_cross_correlation,
            use_scramble=use_scramble_cross_correlation,
            corr_threshold=corr_threshold_cross_correlation,
            n_shuffles=null_draws["direction"],
            n_cores_to_use=n_cores,
            # 2026-09-29: was hardcoded True, ignoring infer_with_twinfer's own `verbose`
            # arg -- caused every direction-stage pair to build (and never close) a
            # matplotlib figure regardless of caller intent, the dominant wall-clock cost
            # at thousand-pair scale. Now forwards the real caller setting.
            verbose=verbose,
            multiple_states_gene_pairs=multiple_states_and_reg,
            return_z_scores=True,
            return_direction_diff=True,
            return_rho_cross_null=True,
            prepare_rho_cross_null=True,
            unit=unit,
        )
    else:
        direction_matrix = pd.DataFrame(
            np.zeros((len(gene_list), len(gene_list))),
            index=gene_list,
            columns=gene_list,
        )

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
        twin_matrix_for_fan_out = twin_delta_correlation_matrix_t1
        measurements_for_fan_out = rho_t1_measurements
        # ---------------------------------------------------------------------
        # Official name: Twin difference correlation at time t
        # LaTeX: observed twins \hat{\rho}_{\Delta}(t); random-pair reference
        # \rho_{\Delta}(t)
        # Permutation method: for every observed twin slot, draw a fresh unrelated
        # pair by choosing clone A uniformly, choosing a different clone B uniformly,
        # then choosing one cell uniformly from each clone. Use that same random cell
        # pair to form both \Delta X and \Delta Y, recompute weighted Spearman across
        # all slots, and repeat to form the random-pair \rho_{\Delta}(t) distribution.
        # The Z-score compares the observed \hat{\rho}_{\Delta}(t) with that distribution.
        # (same with z_het)
        final_directed_edges, directed_z_scores, direction_matrix, fan_out_log = separate_fan_outs_from_mutual_regulation(
            measurements_for_fan_out, twin_matrix_for_fan_out, gene_list,
            final_directed_edges, directed_z_scores, direction_matrix,
            z_score_threshold=fan_out_z_score_threshold,
            unit=unit,
            precomputed_null=t1_delta_null_full,
            n_shuffles=null_draws["fan_out"],
            n_cores_to_use=n_cores,
        )

    # A pair only qualifies as "rescued" if it both passed the opposite-sign
    # screen (rescued_from_no_regulation, computed at Step 4 above) AND came
    # back significant in the permutation null test -- i.e. it actually made
    # it into final_directed_edges. Pairs that qualified for the screen but
    # failed the significance test were never really rescued.
    rescued_from_no_regulation_significant = [
        (a, b) for (a, b) in rescued_from_no_regulation
        if (a, b) in final_directed_edges or (b, a) in final_directed_edges
    ]

    if plot and not direction_matrix.empty:
        all_gene_pairs = list(product(gene_list, repeat=2))
        no_reg_pairs = [pair for pair in all_gene_pairs if pair not in final_directed_edges]

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
            black_out_self=True,
            symmetric=False,
            draw_diagonal_multi_state_reg=bool(multiple_states_and_reg),
            multi_state_reg_edges=multiple_states_and_reg,
        )

    # Step 6: #TODO use the new plot network function to visualize the inferred network
    # if (len(single_state_regulation) >= 0):
    #     if verbose:
    #         if (len(final_directed_edges) > 0):
    #             plot_network(direction_matrix, gene_list, final_directed_edges)
    #         else:
    #             plot_network(direction_matrix, gene_list, final_directed_edges)

    # ------------------------------------------------------------------
    # TwinScore inputs for the directed candidate panel.
    #
    # The panel is exactly the non-self candidate directions tested in Step 4.
    # This includes pairs admitted by the existing rescue_opposite_sign_pairs
    # flag, without creating a second rescue path.
    # ------------------------------------------------------------------
    ranked_edge_list = None
    twin_score_inputs = None
    pairwise_gene_gene_correlation_matrix_t2 = None
    # Only assigned below when ranked_list=True and the directed panel is
    # non-empty; initialized here so return_raw_nulls can check for None
    # instead of risking a NameError in the empty-panel case.
    rho_t1_null = None
    rho_t2_null = None
    rho_change_null_by_pair = {}
    signed_rho_change_details = {}

    if ranked_list:
        directed_panel = []
        for a, b in direction_candidates:
            if a != b:
                directed_panel.append((a, b))
                directed_panel.append((b, a))
        directed_panel = list(dict.fromkeys(directed_panel))

        if directed_panel:
            undirected_panel = list(dict.fromkeys(
                tuple(sorted(pair))
                for pair in direction_candidates
                if pair[0] != pair[1]
            ))

            # ----------------------------------------------------------
            # rho(t2) observed values.
            # ----------------------------------------------------------
            pairwise_gene_gene_correlation_matrix_t2 = (
                calculate_pairwise_gene_gene_correlation_matrix(
                    rho_t2_measurements,
                    gene_list,
                    use_clone=use_clone,
                )
            )

            # ----------------------------------------------------------
            # Current TwinScore term: co-expression magnitude at t1
            # LaTeX: s(z(|\rho(t_1)|))
            # Permutation method: use the Stage-I co-expression null. With
            # use_clone=True, keep gene X and source-cell weights fixed, derange
            # biological clones, and sample one cell from each assigned different
            # clone to supply gene Y. With use_clone=False, permute cells directly.
            # Take the absolute observed/null correlations before computing
            # z(|\rho(t_1)|); TwinScore later applies the panel standardization s.
            # (same with s(z(|\rho(t_2)|)))
            # ----------------------------------------------------------
            if use_clone and step1_t1_null_matrix is not None:
                all_step1_pairs = list(combinations(gene_list, 2))
                step1_pos = {
                    tuple(sorted(pair)): k
                    for k, pair in enumerate(all_step1_pairs)
                }
                rho_t1_null = {
                    pair: np.asarray(
                        step1_t1_null_matrix[:, step1_pos[pair]],
                        dtype=float,
                    )
                    for pair in undirected_panel
                }
            else:
                rho_t1_null = generate_step1_permutation_null(
                    rho_t1_measurements,
                    gene_list,
                    gene_pairs=undirected_panel,
                    use_clone=use_clone,
                    n_shuffles=null_draws["step1"],
                    random_state=int(seed) + 314159,
                )

            # ----------------------------------------------------------
            # Current TwinScore term: co-expression magnitude at t2
            # LaTeX: s(z(|\rho(t_2)|))
            # Permutation method: use the Stage-I co-expression null. With
            # use_clone=True, keep gene X and source-cell weights fixed, derange
            # biological clones, and sample one cell from each assigned different
            # clone to supply gene Y. With use_clone=False, permute cells directly.
            # Take the absolute observed/null correlations before computing
            # z(|\rho(t_2)|); TwinScore later applies the panel standardization s.
            # (same with s(z(|\rho(t_1)|)))
            # ----------------------------------------------------------
            rho_t2_null = generate_step1_permutation_null(
                rho_t2_measurements,
                gene_list,
                gene_pairs=undirected_panel,
                use_clone=use_clone,
                n_shuffles=null_draws["step1"],
                random_state=int(seed) + 314160,
            )

            # ----------------------------------------------------------
            # Official name: Change in gene-expression correlation across time
            # LaTeX: \Delta\rho = \rho(t_2) - \rho(t_1)
            # Z-scores: z_rho_change and z_abs_rho_change
            # Permutation method: pool the complete raw-cell rows used for
            # rho(t1) and rho(t2), then shuffle only their time labels while
            # preserving the observed n_t1 and n_t2. Recompute rho in both
            # pseudo-time groups and subtract pseudo-t1 from pseudo-t2.
            # Twin mode (use_clone=False): give every raw-cell row weight 1 and
            # recompute ordinary unweighted Spearman rho in each group.
            # Clone mode (use_clone=True): keep clone_id attached to each cell's
            # complete expression row and recompute weights within each group,
            # so every represented clone has total weight 1.
            # The signed and absolute scores use the same signed null draws;
            # only z_abs_rho_change applies abs before its null comparison.
            # ----------------------------------------------------------
            rho_change_null_by_pair = (
                generate_rho_change_time_shuffle_null(
                    rho_t1_measurements,
                    rho_t2_measurements,
                    gene_list,
                    gene_pairs=undirected_panel,
                    use_clone=use_clone,
                    n_shuffles=null_draws["step1"],
                    random_state=int(seed) + 314161,
                    n_cores_to_use=n_cores,
                )
            )

            def _null_unit(observed, null_values, use_absolute):
                arr = np.asarray(null_values, dtype=float)
                arr = arr[np.isfinite(arr)]

                obs = float(observed)
                if use_absolute:
                    obs = abs(obs)
                    arr = np.abs(arr)

                if arr.size < 2:
                    return np.nan

                mu = float(np.mean(arr))
                sd = float(np.std(arr, ddof=1))
                if not np.isfinite(sd) or sd <= 0:
                    return np.nan
                return float((obs - mu) / sd)

            rescued_set = {
                tuple(sorted(pair))
                for pair in rescued_from_no_regulation
            }

            # TwinScore uses the canonical within-time and change Z scores
            # directly (they are already standardized against their own nulls).
            z_het_by_pair = {
                tuple(sorted(pair)): (
                    np.nan if value is None else float(value)
                )
                for pair, value in z_het_scores.items()
            }
            z_div_by_pair = {
                tuple(sorted(pair)): float(details.get("z_div", np.nan))
                for pair, details in divergence_details.items()
            }
            stage3_by_pair = {
                tuple(sorted(pair)): details
                for pair, details in stage3_details.items()
            }

            symmetric_components = {}

            for a, b in undirected_panel:
                key = tuple(sorted((a, b)))

                rho_t1 = float(
                    pairwise_gene_gene_correlation_matrix.loc[a, b]
                )
                rho_t2 = float(
                    pairwise_gene_gene_correlation_matrix_t2.loc[a, b]
                )
                rho_change = float(rho_t2 - rho_t1)
                rho_delta_t2 = float(
                    twin_delta_correlation_matrix_t2.loc[a, b]
                )

                null_t1 = np.asarray(
                    rho_t1_null[key],
                    dtype=float,
                )
                null_t2 = np.asarray(
                    rho_t2_null[key],
                    dtype=float,
                )
                rho_change_null = np.asarray(
                    rho_change_null_by_pair[key],
                    dtype=float,
                )

                # ----------------------------------------------------------
                # Official name: Absolute change in gene-expression correlation
                # LaTeX: |\Delta\rho| = |\rho(t_2) - \rho(t_1)|
                # Z-score: z_abs_rho_change
                # Permutation method: pool complete raw-cell rows across t1 and
                # t2 and shuffle only their time labels, preserving n_t1 and
                # n_t2. Recompute signed Delta rho for every draw, then compare
                # |Delta rho observed| with |Delta rho null|.
                # Twin mode (use_clone=False): every raw-cell row has weight 1.
                # Clone mode (use_clone=True): keep clone_id with its cell row
                # and recompute within-group weights so each represented clone
                # has total weight 1. TwinScore later applies panel s.

                # ----------------------------------------------------------
                # Official name: Signed change in gene-expression correlation
                # LaTeX: \Delta\rho = \rho(t_2) - \rho(t_1)
                # Z-score: z_rho_change
                # Permutation method: use the same cell-level time-label draws
                # as z_abs_rho_change, but keep signed Delta rho without abs.
                # Twin mode (use_clone=False): every raw-cell row has weight 1.
                # Clone mode (use_clone=True): keep clone_id with its cell row
                # and recompute within-group weights so each represented clone
                # has total weight 1.
                signed_rho_change_details[key] = (
                    calculate_signed_rho_change_z(
                        rho_t1,
                        rho_t2,
                        rho_change_null,
                        use_clone=use_clone,
                    )
                )

                stage3_for_pair = stage3_by_pair.get(key, {})
                z_d_het = float(
                    stage3_for_pair.get("z_d_het", np.nan)
                )

                symmetric_components[key] = {
                    "rho_t1": rho_t1,
                    "rho_t2": rho_t2,
                    "rho_delta_t2": rho_delta_t2,
                    "rho_change": rho_change,
                    "z_abs_rho_t1": _null_unit(
                        rho_t1,
                        null_t1,
                        use_absolute=True,
                    ),
                    "z_abs_rho_t2": _null_unit(
                        rho_t2,
                        null_t2,
                        use_absolute=True,
                    ),
                    "z_abs_rho_change": _null_unit(
                        rho_change,
                        rho_change_null,
                        use_absolute=True,
                    ),
                    # Signed permutation-null Z-score. calculate_twin_score later
                    # panel-standardizes it as s_z_rho_change; both remain
                    # additional diagnostics and are not inserted into TwinScore.
                    "z_rho_change": signed_rho_change_details[
                        key
                    ]["z_rho_change"],
                    # Canonical TwinScore Z inputs are already Z-scored
                    # against their own nulls, so they enter TwinScore directly.
                    # z_div: fixed-Delta cross-clone re-pairing described above.
                    # z_het: fresh random-pair null described above.
                    # z_d_het: t2-minus-t1 fresh random-pair change null described
                    # above. Do not panel-standardize these three again here.
                    "z_div": float(z_div_by_pair.get(key, np.nan)),
                    "z_het": float(z_het_by_pair.get(key, np.nan)),
                    "z_d_het": z_d_het,
                    # Backward-readable generic alias for the z_d used by
                    # the current TwinScore formula.
                    "z_d": z_d_het,
                    "rescued_from_no_regulation": (
                        key in rescued_set
                    ),
                }

            rows = []

            for x, y in directed_panel:
                key = tuple(sorted((x, y)))
                common = symmetric_components[key]

                rho_xy = float(
                    unfiltered_direction_matrix.loc[x, y]
                )
                rho_yx = float(
                    unfiltered_direction_matrix.loc[y, x]
                )

                # ----------------------------------------------------------
                # Official name: Directional asymmetry
                # LaTeX: \gamma = |\hat{\rho}^{\dagger}_{x(t_1)\to y(t_2)}|
                #                    - |\hat{\rho}^{\dagger}_{y(t_1)\to x(t_2)}|
                # Positive gamma favors x->y; negative gamma favors y->x.
                # Permutation method: use the same paired cross-clone permutation
                # draws as z_rho_cross in the two directions. For each draw compute
                # |rho_xy_null|-|rho_yx_null|; z_gamma is the observed signed gamma
                # expressed in that null's Z units. TwinScore later applies s across
                # the specific candidate panel. (same direction-step gamma null)
                gamma = float(abs(rho_xy) - abs(rho_yx))

                xy_null_details = rho_cross_null_details.get((x, y))
                yx_null_details = rho_cross_null_details.get((y, x))

                if (
                    xy_null_details is None
                    or yx_null_details is None
                ):
                    z_gamma = np.nan
                else:
                    null_xy = np.asarray(
                        xy_null_details["null_values"],
                        dtype=float,
                    )
                    null_yx = np.asarray(
                        yx_null_details["null_values"],
                        dtype=float,
                    )
                    n_gamma = min(
                        len(null_xy),
                        len(null_yx),
                    )
                    gamma_null = (
                        np.abs(null_xy[:n_gamma])
                        - np.abs(null_yx[:n_gamma])
                    )
                    z_gamma = _null_unit(
                        gamma,
                        gamma_null,
                        use_absolute=False,
                    )

                rows.append({
                    "gene_1": x,
                    "gene_2": y,
                    **common,
                    "rho_cross_xy": rho_xy,
                    "rho_cross_yx": rho_yx,
                    "gamma": gamma,
                    "z_gamma": z_gamma,
                    "is_final_directed_edge": (
                        (x, y) in final_directed_edges
                    ),
                })

            twin_score_inputs = pd.DataFrame(rows)

        else:
            twin_score_inputs = pd.DataFrame(
                columns=[
                    "gene_1",
                    "gene_2",
                    "rho_t1",
                    "rho_t2",
                    "rho_delta_t2",
                    "rho_change",
                    "z_abs_rho_t1",
                    "z_abs_rho_t2",
                    "z_abs_rho_change",
                    "z_rho_change",
                    "z_div",
                    "z_het",
                    "z_d_het",
                    "z_d",
                    "rho_cross_xy",
                    "rho_cross_yx",
                    "gamma",
                    "z_gamma",
                    "rescued_from_no_regulation",
                    "is_final_directed_edge",
                ]
            )

    # Stable, compact output.
    result = {
        "settings": {
            "use_clone": bool(use_clone),
            "unit": unit,
            "steady_state_tol": float(steady_state_tol),
            "steady_state_flat_tol": (
                None
                if steady_state_flat_tol is None
                else float(steady_state_flat_tol)
            ),
            "steady_state_min_r2": float(steady_state_min_r2),
            "steady_state_slope_tol": (
                None
                if steady_state_slope_tol is None
                else float(steady_state_slope_tol)
            ),
            "steady_state_min_stable_points": int(
                steady_state_min_stable_points
            ),
            "rho_delta_het_null_type": "fresh_random_cell_pairs_clone_first",
            "rho_delta_div_null_type": "fixed_delta_y_cross_clone_permutation",
            "z_reg_gated_null_type": "y_sibling_block_cross_clone_permutation",
            "stage3_d_het_null_type": "fresh_random_cell_pairs_with_joint_source_clone_bootstrap",
            "stage3_d_div_null_type": "fixed_delta_y_cross_clone_permutation_independent_timepoints",
            "signed_rho_change_null_type": (
                "clone_weighted_cell_time_label_permutation_t2_minus_t1"
                if use_clone
                else "unweighted_cell_time_label_permutation_t2_minus_t1"
            ),
            "absolute_rho_change_null_type": (
                "clone_weighted_cell_time_label_permutation_abs_t2_minus_t1"
                if use_clone
                else "unweighted_cell_time_label_permutation_abs_t2_minus_t1"
            ),
            "gamma_definition": "abs(rho_cross_xy)-abs(rho_cross_yx)",
            # Aliases for the same statistics.
            "rho_delta_null_type": "fresh_random_cell_pairs_clone_first",
            "z_score_div_null_type": "fixed_delta_y_cross_clone_permutation",
            "stage3_d_null_type": "fresh_random_cell_pairs_with_joint_source_clone_bootstrap",
            "null_alpha": float(null_alpha),
            "expected_null_rejections": int(expected_null_rejections),
            "minimum_null_draws": int(minimum_null_draws),
            "null_draws": dict(null_draws),
            "is_simulation_data": bool(is_simulation_data),
            "t1": t1,
            "t2": t2,
            "n_clones": n_clones_data,
            "genes": list(gene_list),
        },
        "classification": {
            "no_regulation": list(no_regulation),
            "single_state_regulation": list(single_state_regulation),
            "multiple_states_no_reg": list(multiple_states_no_reg),
            "multiple_states_and_reg": list(multiple_states_and_reg),
        },
        "directed_edges": sorted(final_directed_edges),
        "correlations": {
            "gene_t1": pairwise_gene_gene_correlation_matrix,
            "gene_t2": pairwise_gene_gene_correlation_matrix_t2,
            "twin_delta_t1": twin_delta_correlation_matrix_t1,
            "random_delta_t1": random_delta_reference_matrix_t1,
            "twin_delta_t2": twin_delta_correlation_matrix_t2,
            "random_delta_t2": random_delta_reference_matrix_t2,
            "direction": direction_matrix,
            # Canonical statistic aliases.
            "rho_t1": pairwise_gene_gene_correlation_matrix,
            "rho_t2": pairwise_gene_gene_correlation_matrix_t2,
            "rho_delta_t1": twin_delta_correlation_matrix_t1,
            "rho_delta_t2": twin_delta_correlation_matrix_t2,
            "rho_cross": unfiltered_direction_matrix,
        },
        "direction": {
            "z_scores": directed_z_scores,
            "rho_cross_null": rho_cross_null_details,
            "gamma_details": direction_diff_details,
            "difference_details": direction_diff_details,
            "unfiltered_matrix": unfiltered_direction_matrix,
            "rescued_from_no_regulation": rescued_from_no_regulation_significant,
        },
        "stage3": stage3_details,
        "heterogeneity": {
            "timepoint": t1,
            "z_het": z_het_scores,
            "null_stats": multi_state_null_stats,
            "null_type": "fresh_random_cell_pairs_clone_first",
        },
        "divergence": {
            "timepoint": t1,
            "z_div": z_div,
            # Alias for the same statistic.
            "z_score_div": z_div,
            "details": divergence_details,
            "null_type": "fixed_delta_y_cross_clone_permutation",
        },
        "gated_regulation": {
            "timepoint": t1,
            "z_reg_gated": z_reg_gated,
            "details": gated_regulation_details,
            "null_type": "y_sibling_block_cross_clone_permutation",
        },
        "statistics": {
            "rho": {
                "t1": pairwise_gene_gene_correlation_matrix,
                "t2": pairwise_gene_gene_correlation_matrix_t2,
                "signed_change": signed_rho_change_details,
            },
            "rho_delta": {
                "t1": twin_delta_correlation_matrix_t1,
                "t2": twin_delta_correlation_matrix_t2,
                "z_het_t1": z_het_scores,
                "z_div_t1": z_div,
            },
            "d": stage3_details,
            "rho_reg_gated": gated_regulation_details,
            "rho_cross": unfiltered_direction_matrix,
            "gamma": direction_diff_details,
        },
        "fan_out": fan_out_log,
        "twin_score_inputs": twin_score_inputs,
        "ranked_edges": None,
    }

    if ranked_list:
        ranked_edge_list = calculate_twin_score(result)
        result["ranked_edges"] = ranked_edge_list

    if return_diagnostics:
        result["diagnostics"] = {
            "step1": {
                "null_stats": gene_corr_null_stats,
                "z_scores": z_scores,
                "alpha": float(alpha_gene_gene_corr),
                "z_critical": float(
                    scipy.stats.norm.ppf(
                        1.0 - alpha_gene_gene_corr / 2.0
                    )
                ),
            },
            "step2": {
                "null_stats": multi_state_null_stats,
                "z_het": z_het_scores,
                "z_div": z_div,
                "z_reg_gated": z_reg_gated,
                # Aliases for the same statistics.
                "z_scores": z_het_scores,
                "z_score_div": z_div,
                "divergence_details": divergence_details,
                "gated_regulation_details": gated_regulation_details,
            },
            "step3": {
                "alpha": float(alpha_stage3),
            },
            "rho_change": {
                "signed_details": signed_rho_change_details,
                "null_type": (
                    "clone_weighted_cell_time_label_permutation_t2_minus_t1"
                    if use_clone
                    else "unweighted_cell_time_label_permutation_t2_minus_t1"
                ),
            },
        }

    if return_raw_nulls:
        def _flatten_pair_dict(d):
            if not d:
                return {}
            return {
                f"{a}__{b}": np.asarray(v, dtype=float)
                for (a, b), v in d.items()
            }

        raw_nulls = {
            # z_het's null: generate_random_shuffle at t1 (fresh random-pair,
            # clone-first). Reused for the fan-out check when enabled.
            "rho_delta_het_t1": _flatten_pair_dict(t1_delta_null_full),
            # z_div's null: generate_divergence_shuffle (Yuval fixed-Delta,
            # cross-clone Y reassignment), independently at t1 and t2.
            "rho_delta_div_t1": _flatten_pair_dict(t1_div_null_full),
            "rho_delta_div_t2": _flatten_pair_dict(t2_div_null_full),
        }
        # Stage-I co-expression null matrix, only computed when
        # use_clone=True and ranked_list=True (return_null_matrix above).
        # Columns follow combinations(gene_list, 2), the same order
        # check_gene_gene_correlation_threshold uses to build it.
        if step1_t1_null_matrix is not None:
            all_step1_pairs = list(combinations(gene_list, 2))
            raw_nulls["step1"] = {
                f"{a}__{b}": np.asarray(step1_t1_null_matrix[:, k], dtype=float)
                for k, (a, b) in enumerate(all_step1_pairs)
            }
        # TwinScore's rho(t1)/rho(t2) magnitude nulls and its separate
        # rho-change null -- only computed when ranked_list=True and the
        # directed panel is non-empty. rho_change_for_twinscore contains signed
        # pseudo-t2-minus-pseudo-t1 correlations from cell-level time-label
        # permutations; z_abs_rho_change takes abs of these same draws.
        if rho_t1_null is not None:
            raw_nulls["rho_t1_for_twinscore"] = _flatten_pair_dict(rho_t1_null)
        if rho_t2_null is not None:
            raw_nulls["rho_t2_for_twinscore"] = _flatten_pair_dict(rho_t2_null)
        if rho_change_null_by_pair:
            raw_nulls["rho_change_for_twinscore"] = _flatten_pair_dict(
                rho_change_null_by_pair
            )
        # z_d_het's / z_d_div's null: d = rho_Delta(t2) - rho_Delta(t1) under the
        # same fresh-random-pair ("random pairs instead of twin pairs") construction
        # as z_het/z_div, but jointly across t1 and t2 -- only computed for pairs
        # that reached Stage 3 (multiple_states_gene_pairs was non-empty).
        if stage3_raw_nulls.get("d_het"):
            raw_nulls["d_het"] = _flatten_pair_dict(stage3_raw_nulls["d_het"])
        if stage3_raw_nulls.get("d_div"):
            raw_nulls["d_div"] = _flatten_pair_dict(stage3_raw_nulls["d_div"])
        result["raw_nulls"] = raw_nulls

    return result
