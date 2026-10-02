import numpy as np
import pandas as pd
from scipy.stats import spearmanr, linregress, pearsonr
from scipy.stats import rankdata
from itertools import combinations, permutations
import os
from joblib import Parallel, delayed
from scipy import stats
import numba
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.colors import Normalize, LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
import networkx as nx
import seaborn as sns
from itertools import cycle
from pathlib import Path

def check_system_in_steady_state(
    simulation_df,
    gene_list,
    steady_tol=0.01,
    flat_tol=None,
    min_r2=0.80,
):
    """
    Determine whether the population-mean protein trajectory of each gene
    has reached steady state.

    For each gene:

    1. Calculate the mean protein level across cells at every time point.

    2. If the observed trajectory is already flat,

           (max(P) - min(P)) / mean(P) <= flat_tol

       classify it as steady directly.

    3. Otherwise fit the trajectory to

           P(t) = P_inf + (P0 - P_inf) * exp(-(t / tau)**beta)

       where P_inf is the extrapolated steady-state protein level.

    4. Calculate the time at which the fitted trajectory becomes
       steady_tol-close to P_inf:

           |P(t) - P_inf| / |P_inf| <= steady_tol

       For steady_tol=0.01, this means within 1% of P_inf.

    5. A fitted trajectory is classified as steady if:
           - R2 >= min_r2
           - the fitted steady-state arrival time is no later than
             the final observed time.

    The system is steady only if all genes are steady.

    Parameters
    ----------
    simulation_df : pd.DataFrame
        Simulation output containing 'time_step' and
        '{gene}_protein' columns.

    gene_list : list of str
        Ordered list of genes, for example:
        ['gene_1', 'gene_2'].

    steady_tol : float, default=0.01
        Relative distance from the fitted asymptotic level used to
        define steady state.

    flat_tol : float or None, default=None
        Maximum relative range across the observed trajectory for it
        to be classified directly as flat. If None, uses steady_tol.

    min_r2 : float, default=0.80
        Minimum R2 required when extrapolation is needed.

    Returns
    -------
    is_steady : bool
        True if all genes are steady.

    summary_df : pd.DataFrame
        Per-gene steady-state results.
    """

    from scipy.optimize import curve_fit

    if flat_tol is None:
        flat_tol = steady_tol

    if "time_step" not in simulation_df.columns:
        raise ValueError("simulation_df must contain a 'time_step' column.")

    if steady_tol <= 0:
        raise ValueError("steady_tol must be greater than 0.")

    if flat_tol <= 0:
        raise ValueError("flat_tol must be greater than 0.")

    if not 0 <= min_r2 <= 1:
        raise ValueError("min_r2 must be between 0 and 1.")

    for gene in gene_list:
        protein_col = f"{gene}_protein"

        if protein_col not in simulation_df.columns:
            raise ValueError(
                f"{protein_col} not found in simulation_df."
            )

    # Population-mean protein level at each time point
    mean_df = (
        simulation_df
        .groupby("time_step", sort=True)[
            [f"{gene}_protein" for gene in gene_list]
        ]
        .mean()
    )

    time_points = mean_df.index.to_numpy(dtype=float)

    if len(time_points) < 5:
        raise ValueError(
            "Need at least 5 time points for steady-state checking."
        )

    if np.any(np.diff(time_points) <= 0):
        raise ValueError(
            "time_step values must be strictly increasing."
        )

    # Shift fitting time so the first observation is t = 0
    t = time_points - time_points[0]

    results = []

    for gene in gene_list:

        y = mean_df[f"{gene}_protein"].to_numpy(dtype=float)

        mean_level = float(np.mean(y))
        min_level = float(np.min(y))
        max_level = float(np.max(y))

        relative_range = (
            (max_level - min_level)
            / max(abs(mean_level), 1e-12)
        )

        # ----------------------------------------------------------
        # Case 1: observed trajectory is already flat
        # ----------------------------------------------------------

        if relative_range <= flat_tol:

            results.append({
                "Gene": gene,
                "Method": "flat window",
                "P_inf": np.nan,
                "Tau": np.nan,
                "Beta": np.nan,
                "R2": np.nan,
                "t_steady": time_points[0],
                "Relative Range": relative_range,
                "Final Distance": np.nan,
                "Steady State?": True,
            })

            continue

        # ----------------------------------------------------------
        # Case 2: fit and extrapolate
        # ----------------------------------------------------------

        P0 = float(y[0])

        def relaxation_model(t_fit, P_inf, tau, beta):

            return (
                P_inf
                + (P0 - P_inf)
                * np.exp(
                    -np.power(
                        np.maximum(t_fit, 0.0) / tau,
                        beta,
                    )
                )
            )

        tail_n = max(3, len(y) // 5)

        tail_mean = float(
            np.mean(y[-tail_n:])
        )

        median_dt = float(
            np.median(np.diff(t))
        )

        tau_guess = max(
            float(t[-1]) / 4.0,
            median_dt,
        )

        observed_scale = max(
            abs(min_level),
            abs(max_level),
            abs(mean_level),
            1.0,
        )

        try:

            popt, _ = curve_fit(
                relaxation_model,
                t,
                y,
                p0=[
                    max(tail_mean, 0.0),
                    tau_guess,
                    1.0,
                ],
                bounds=(
                    [
                        0.0,
                        1e-8,
                        0.2,
                    ],
                    [
                        10.0 * observed_scale,
                        100.0 * max(float(t[-1]), median_dt),
                        5.0,
                    ],
                ),
                maxfev=50000,
            )

            P_inf, tau, beta = map(float, popt)

            fitted = relaxation_model(
                t,
                P_inf,
                tau,
                beta,
            )

            # ------------------------------------------------------
            # Fit quality
            # ------------------------------------------------------

            ss_res = float(
                np.sum(
                    (y - fitted) ** 2
                )
            )

            ss_tot = float(
                np.sum(
                    (y - np.mean(y)) ** 2
                )
            )

            if ss_tot > 0:
                r2 = 1.0 - ss_res / ss_tot
            else:
                r2 = 1.0

            # ------------------------------------------------------
            # Calculate time to steady state
            #
            # Require:
            #
            # |P(t) - P_inf| / |P_inf| <= steady_tol
            # ------------------------------------------------------

            initial_distance = abs(
                P0 - P_inf
            )

            allowed_distance = (
                steady_tol
                * max(abs(P_inf), 1e-12)
            )

            if initial_distance <= allowed_distance:

                t_steady_relative = 0.0

            else:

                ratio = (
                    initial_distance
                    / allowed_distance
                )

                t_steady_relative = (
                    tau
                    * np.power(
                        np.log(ratio),
                        1.0 / beta,
                    )
                )

            t_steady = (
                time_points[0]
                + t_steady_relative
            )

            final_distance = (
                abs(fitted[-1] - P_inf)
                / max(abs(P_inf), 1e-12)
            )

            gene_steady = bool(
                np.isfinite(t_steady)
                and np.isfinite(r2)
                and r2 >= min_r2
                and t_steady <= time_points[-1]
            )

            results.append({
                "Gene": gene,
                "Method": "fit",
                "P_inf": P_inf,
                "Tau": tau,
                "Beta": beta,
                "R2": r2,
                "t_steady": t_steady,
                "Relative Range": relative_range,
                "Final Distance": final_distance,
                "Steady State?": gene_steady,
            })

        except (
            RuntimeError,
            ValueError,
            FloatingPointError,
        ):

            results.append({
                "Gene": gene,
                "Method": "fit failed",
                "P_inf": np.nan,
                "Tau": np.nan,
                "Beta": np.nan,
                "R2": np.nan,
                "t_steady": np.nan,
                "Relative Range": relative_range,
                "Final Distance": np.nan,
                "Steady State?": False,
            })

    summary_df = pd.DataFrame(results)

    is_steady = bool(
        summary_df["Steady State?"].all()
    )

    return is_steady, summary_df
    
def check_system_in_steady_state(simulation_df, gene_params, interaction_matrix, gene_list,
                                  relative_diff_threshold=0.01, relative_slope_threshold=0.01):
    """
    Determines if each gene in the system has reached steady state based on empirical vs theoretical protein levels.

    Args:
        simulation_df (pd.DataFrame): Simulation output with columns like 'time_step' and 'gene_{i}_protein'.
        gene_params (dict): Parameter dictionary for gene kinetics.
        interaction_matrix (np.ndarray): Regulatory matrix (n_genes x n_genes).
        gene_list (list): List of gene names, e.g., ['gene_1', 'gene_2'].
        relative_diff_threshold (float): Threshold for max allowable relative error between empirical and theoretical protein level.
        relative_slope_threshold (float): Threshold for max allowable slope of protein level over time.

    Returns:
        is_steady (bool): True if all genes are in steady state.
        summary_df (pd.DataFrame): Per-gene summary of steady state check.
    """

    n_genes = len(gene_list)
    t_list = sorted(simulation_df['time_step'].unique())
    mean_val = [[] for _ in range(n_genes)]
    gene_means = [[] for _ in range(n_genes)]

    for t in t_list:
        sim_data_t = simulation_df[simulation_df['time_step'] == t]
        steady_state_with_sim_data = steady_state_calc(gene_params, interaction_matrix, gene_list, sim_data=sim_data_t)

        for i in range(n_genes):
            gene_means[i].append(steady_state_with_sim_data[i])
            mean_val[i].append(sim_data_t[f'{gene_list[i]}_protein'].mean())

    t_array = np.array(t_list)
    relative_diffs = []
    relative_slopes = []
    steady_state_flags = []

    for i in range(n_genes):
        empirical = np.array(mean_val[i])
        theoretical = np.array(gene_means[i])

        with np.errstate(divide='ignore', invalid='ignore'):
            relative_diff = np.abs(empirical - theoretical) / theoretical
            relative_diff = np.nan_to_num(relative_diff)
        relative_diffs.append(relative_diff)

        slope, _, _, _, _ = linregress(t_array, empirical)
        final_mean = np.mean(empirical)
        relative_slope = np.abs(slope / final_mean) if final_mean != 0 else 0
        relative_slopes.append(relative_slope)

        is_steady = np.all(relative_diff < relative_diff_threshold) and relative_slope < relative_slope_threshold
        steady_state_flags.append(is_steady)

    summary_df = pd.DataFrame({
        "Gene": [f"Gene {i + 1}" for i in range(n_genes)],
        "Max Relative Diff": [np.max(rd) for rd in relative_diffs],
        "Relative Slope": relative_slopes,
        "Steady State?": steady_state_flags
    })

    return all(steady_state_flags), summary_df

def calculate_pairwise_gene_gene_correlation_matrix(simulation_at_t1, gene_list):
    """
    Gene-gene Spearman correlation matrix across cells at a single timepoint.

    Parameters
    ----------
    simulation_at_t1 : pd.DataFrame
        One row per cell, must contain '{gene}_mRNA' for each gene in gene_list.
    gene_list : list of str
        Gene names (without '_mRNA' suffix).

    Returns
    -------
    correlation_matrix : pd.DataFrame
        Gene x gene Spearman correlation matrix.
    """
    correlations = {}
    for gene_1 in gene_list:
        for gene_2 in gene_list:
            gene_gene_corr = spearmanr(simulation_at_t1[f"{gene_1}_mRNA"], simulation_at_t1[f"{gene_2}_mRNA"]).correlation
            correlations[f"{gene_1}-{gene_2}"] = gene_gene_corr
    correlation_matrix = dict_to_matrix(correlations, gene_list)
    return correlation_matrix


def get_correlations(correlation_dict, gene_i, gene_j):
   """Look up the correlation for (gene_i, gene_j) in a dict keyed by sorted gene-pair tuples."""
   return correlation_dict[tuple(sorted([gene_i, gene_j]))]

def generate_random_shuffle(simulation_data, gene_list, n_shuffles=10000, random_state=42):
    """
    Random-pair difference-correlation null distribution sized to the number of true
    twin pairs in simulation_data (clone_ids with exactly 2 cells), not to
    floor(n_cells/2) -- those only coincide when every clone_id in the pool has
    exactly 2 cells. Every shuffle draws a single permutation of the whole cell
    pool, splits it into two parts of that target size, and pairs them
    positionally -- so each shuffle yields (up to) that many random-pair deltas,
    matching the N true twin-pair deltas that twin_correlation_matrix (the
    statistic being compared against in differentiate_single_state_reg_and_multiple_states)
    was computed from. No assumption is made about which column/labels distinguish
    the two twins -- only that 'clone_id' identifies which rows are each other's
    twin, so a position that would pair a cell with its own twin can be dropped
    (see _half_split_diff_null_kernel) instead of leaking real twin correlation
    into the "random" null.

    Parameters
    ----------
    simulation_data : pd.DataFrame
        Must contain 'clone_id' and '{gene}_mRNA' columns for gene_list.
    gene_list : list of str
        Gene base names (without '_mRNA' suffix).
    n_shuffles : int, default=10000
        Number of random shuffles to perform.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    correlation_dict : dict
        Mapping {(gene_i, gene_j): np.ndarray of n_shuffles shuffled correlations}, same
        shape as generate_random_shuffle's return value.
    """
    gene_cols = [f"{gene}_mRNA" for gene in gene_list]
    sub = simulation_data[gene_cols + ['clone_id']].dropna(subset=gene_cols)
    expr = sub[gene_cols].to_numpy(dtype=np.float64)
    clone_codes = pd.factorize(sub['clone_id'])[0].astype(np.int64)

    # Target sample size = number of clone_ids with exactly 2 cells here, i.e. the
    # true twin-pair count -- not n_cells // 2, which only matches when every
    # clone_id in the pool happens to have exactly 2 cells.
    _, clone_counts = np.unique(clone_codes, return_counts=True)
    n_twin_pairs = int(np.sum(clone_counts == 2))
    if n_twin_pairs == 0:
        raise ValueError("No clone_id with exactly 2 cells found in simulation_data; cannot size the random-pair null.")
    print(f"n_cells_in_pool={len(clone_codes)}, n_unique_clones={len(clone_counts)}, "
      f"n_twin_pairs(target)={n_twin_pairs}, n_cells//2={len(clone_codes)//2}")
    n_genes = expr.shape[1]
    triu_i, triu_j = np.triu_indices(n_genes, k=1)
    gene_pairs = [(gene_list[i], gene_list[j]) for i, j in zip(triu_i, triu_j)]

    seeds = _spawn_independent_seeds(random_state, n_shuffles)
    all_correlations = _half_split_diff_null_kernel(
        expr, clone_codes, seeds, triu_i.astype(np.int64), triu_j.astype(np.int64), n_twin_pairs
    )

    correlation_dict = {
        tuple(sorted((gi, gj))): all_correlations[:, k]
        for k, (gi, gj) in enumerate(gene_pairs)
    }

    return correlation_dict


def compute_correlation_matrix(gene_matrix_1, gene_matrix_2, gene_list, gene_pairs=None):
   """Compute Spearman correlations between gene expression matrices."""
   n_genes = len(gene_list)
   gene_to_idx = {gene: i for i, gene in enumerate(gene_list)}
   raw_matrix = np.zeros((n_genes, n_genes))
   
   # Determine which pairs to compute
   if gene_pairs is None:
       pairs_to_compute = [(i, j) for i in range(n_genes) for j in range(n_genes)]
   else:
       pairs_to_compute = []
       for gene_1, gene_2 in gene_pairs:
           if gene_1 in gene_to_idx and gene_2 in gene_to_idx:
               i, j = gene_to_idx[gene_1], gene_to_idx[gene_2]
               pairs_to_compute.append((i, j))
   
   # Compute correlations
   for i, j in pairs_to_compute:
       corr = spearmanr(gene_matrix_1[i, :], gene_matrix_2[j, :]).correlation
       raw_matrix[i, j] = corr
   
   return pd.DataFrame(raw_matrix, index=gene_list, columns=gene_list)

def single_cell_shuffle(gene_matrix_1, gene_matrix_2, gene_list, shuffle_pairs, seed=101010):
            """
            One shuffle of the correlation-permutation null: randomly permutes the
            cell order of gene_matrix_2 and recomputes the gene-pair correlation
            matrix against the unshuffled gene_matrix_1.

            Parameters
            ----------
            gene_matrix_1, gene_matrix_2 : np.ndarray
                Shape (n_genes, n_cells); gene_matrix_2's columns (cells) are shuffled.
            gene_list : list of str
                Gene names, in row order matching the matrices.
            shuffle_pairs : list of tuple
                Gene-index pairs to compute correlations for.
            seed : int, default=101010
                Random seed for the cell permutation.

            Returns
            -------
            pd.DataFrame
                Gene x gene correlation matrix for this one shuffle.
            """
            rng = np.random.default_rng(seed)
            n_cells = gene_matrix_1.shape[1]
            shuffled_indices = rng.permutation(n_cells)
            return compute_correlation_matrix(gene_matrix_1, gene_matrix_2[:, shuffled_indices], gene_list, shuffle_pairs)


def _assert_no_nan(gene_matrix, caller_name):
    """
    Raise a clear error if gene_matrix contains NaN, rather than silently computing a wrong
    (not NaN) correlation. scipy.stats.spearmanr degrades gracefully to NaN in the presence of
    NaN inputs; the rank-precompute kernels below do not get this for free, since ranking a
    column that contains NaN does not reliably propagate to a clean NaN downstream.
    """
    if np.isnan(gene_matrix).any():
        raise ValueError(
            f"NaN values found in the gene expression data passed to {caller_name}. "
            "Remove or impute missing values (e.g. via .dropna()) before calling this function."
        )


def _spawn_independent_seeds(seed, n_shuffles):
    """
    Generate n_shuffles seeds for independent parallel shuffles using
    np.random.SeedSequence.spawn -- the numpy-recommended mechanism for statistically
    independent parallel streams. Drawing n_shuffles plain integers from a single
    generator instead has a small but real collision chance (empirically ~2% for
    n_shuffles=10000 drawn from a 2**31 range) that would silently duplicate a null sample;
    SeedSequence.spawn is specifically designed to avoid this.
    """
    children = np.random.SeedSequence(seed).spawn(n_shuffles)
    return np.array([int(c.generate_state(1)[0]) for c in children], dtype=np.int64)


def _rank_center_and_sumsq(gene_matrix):
    """
    Rank each gene's row (axis=1, across cells) with average-tie ranking, center by the constant
    (n_cells+1)/2 -- exact regardless of ties, since average-tie ranks always sum to the same
    total -- and return the per-gene sum of squared centered ranks (the correlation denominator
    is built from this). gene_matrix shape: (n_genes, n_cells).
    """
    n_cells = gene_matrix.shape[1]
    R = rankdata(gene_matrix, axis=1)
    Rc = R - (n_cells + 1) / 2.0
    s2 = np.sum(Rc ** 2, axis=1)
    return Rc, s2


@numba.njit(cache=True)
def _rankdata_numba(a):
    """Average-tie rank of a 1D array; matches scipy.stats.rankdata(method='average') exactly."""
    n = a.size
    order = np.argsort(a)
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        start = i
        val = a[order[i]]
        while i + 1 < n and a[order[i + 1]] == val:
            i += 1
        end = i
        avg_rank = 0.5 * (start + end) + 1
        for k in range(start, end + 1):
            ranks[order[k]] = avg_rank
        i += 1
    return ranks


@numba.njit(parallel=True, fastmath=True)
def _rank_permutation_null_kernel(Rc1, Rc2, denom, seeds, pair_i, pair_j):
    """
    Null distribution of Spearman correlations where Rc1 (n_genes1 x n_cells) is fixed and
    Rc2 (n_genes2 x n_cells) has its cell-columns permuted each shuffle. Exact under ties,
    because ranks are computed once upstream (in _rank_center_and_sumsq) -- permuting a
    rank vector's columns and then correlating is identical to ranking a permuted array,
    for average-tie ranking. Pass Rc2=Rc1 and a symmetric denom for the undirected
    gene-gene case (check_gene_gene_correlation_threshold); pass distinct matrices for the
    directed cross-time case (identify_actual_directed_edges).
    """
    n_cells = Rc1.shape[1]
    n_shuffles = seeds.shape[0]
    n_pairs = pair_i.shape[0]
    out = np.empty((n_shuffles, n_pairs), dtype=np.float64)
    for s in numba.prange(n_shuffles):
        np.random.seed(seeds[s])
        idx = np.random.permutation(n_cells)
        Rc2_perm = Rc2[:, idx]
        N = Rc1 @ Rc2_perm.T
        for k in range(n_pairs):
            i = pair_i[k]
            j = pair_j[k]
            d = denom[i, j]
            if d > 0:
                out[s, k] = N[i, j] / d
            else:
                out[s, k] = np.nan
    return out


@numba.njit(parallel=True, fastmath=True)
def _two_permutation_diff_null_kernel(expr, seeds, triu_i, triu_j):
    """
    Random-pair difference-correlation null. Replicates generate_random_shuffle's original
    algorithm exactly (two independent permutations per shuffle, self-match pairs filtered
    via idx_1 != idx_2), just executed inside a numba-parallel loop instead of a Python loop
    with np.apply_along_axis(rankdata, ...). Unlike _rank_permutation_null_kernel, the values
    here genuinely change every shuffle (they are differences of permuted cells), so ranking
    must be redone each shuffle -- there is no precompute-once shortcut for this piece.
    expr: (n_cells, n_genes).
    """
    n_cells, n_genes = expr.shape
    n_shuffles = seeds.shape[0]
    n_pairs = triu_i.shape[0]
    out = np.full((n_shuffles, n_pairs), np.nan, dtype=np.float64)

    for s in numba.prange(n_shuffles):
        np.random.seed(seeds[s])
        idx_1 = np.random.permutation(n_cells)
        idx_2 = np.random.permutation(n_cells)

        n_used = 0
        keep = np.empty(n_cells, dtype=np.int64)
        for k in range(n_cells):
            if idx_1[k] != idx_2[k]:
                keep[n_used] = k
                n_used += 1

        if n_used < 3:
            raise ValueError("Could not create random pairs of cells to generate null distribution.")
            continue

        deltas = np.empty((n_used, n_genes), dtype=np.float64)
        for k in range(n_used):
            kk = keep[k]
            a_idx = idx_1[kk]
            b_idx = idx_2[kk]
            for g in range(n_genes):
                deltas[k, g] = expr[a_idx, g] - expr[b_idx, g]

        R = np.empty((n_used, n_genes), dtype=np.float64)
        for g in range(n_genes):
            R[:, g] = _rankdata_numba(deltas[:, g])

        m = (n_used + 1) / 2.0
        Rc = R - m
        s2 = np.empty(n_genes, dtype=np.float64)
        for g in range(n_genes):
            acc = 0.0
            for k in range(n_used):
                acc += Rc[k, g] * Rc[k, g]
            s2[g] = acc

        for p_idx in range(n_pairs):
            i = triu_i[p_idx]
            j = triu_j[p_idx]
            num = 0.0
            for k in range(n_used):
                num += Rc[k, i] * Rc[k, j]
            d = np.sqrt(s2[i] * s2[j])
            if d > 0:
                out[s, p_idx] = num / d
    return out


@numba.njit(parallel=True, fastmath=True)
def _half_split_diff_null_kernel(expr, clone_codes, seeds, triu_i, triu_j, n_pairs_target):
    """
    Random-pair difference-correlation null sized to n_pairs_target (the true twin-pair
    count, passed in by the caller), not the ~2N pairs _two_permutation_diff_null_kernel
    draws via two independent permutations of the full pool. Each shuffle draws a SINGLE
    permutation and splits it into two parts of size n_pairs_target, pairing them
    positionally -- every cell used is used at most once per part, so there's no
    near-self-match to filter, and no assumption about which two labels distinguish the
    twins (unlike splitting by a 'replicate' column). Positions where the split happens to
    pair a cell with its own twin (same clone_codes value) are dropped instead, since that
    would leak the real twin correlation into the "random" null.
    """
    n_cells, n_genes = expr.shape
    if n_pairs_target > n_cells:
        raise ValueError("n_pairs_target cannot exceed the number of cells in expr")
    half = n_pairs_target
    # Tolerance for how many of the `half` drawn positions may be dropped as
    # same-twin clashes before a shuffle is discarded as unrepresentative.
    # 2% matches the original literal band (1470-1500) at half=1500 exactly;
    # generalized here so the kernel isn't silently all-NaN for any half != 1500
    # (e.g. this dataset's half=6000, where the old fixed band rejected every draw).
    tol = max(1, int(0.02 * half))
    n_shuffles = seeds.shape[0]
    n_pairs = triu_i.shape[0]
    out = np.full((n_shuffles, n_pairs), np.nan, dtype=np.float64)

    for s in numba.prange(n_shuffles):
        np.random.seed(seeds[s])
        # Draw two independent random parts from the full cell pool.
        # Each part contains unique cells internally, but the two parts may overlap.
        perm_a = np.random.permutation(n_cells)
        perm_b = np.random.permutation(n_cells)
        idx_a = perm_a[:half]
        idx_b = perm_b[:half]

        n_used = 0
        keep = np.empty(half, dtype=np.int64)
        for k in range(half):
            a_idx = idx_a[k]
            b_idx = idx_b[k]
            #if clone_codes[idx_a[k]] != clone_codes[idx_b[k]]:
            if a_idx != b_idx and clone_codes[a_idx] != clone_codes[b_idx]:
                keep[n_used] = k
                n_used += 1

        if n_used < half - tol or n_used > half:
            continue

        deltas = np.empty((n_used, n_genes), dtype=np.float64)
        for k in range(n_used):
            kk = keep[k]
            a_idx = idx_a[kk]
            b_idx = idx_b[kk]
            for g in range(n_genes):
                deltas[k, g] = expr[a_idx, g] - expr[b_idx, g]

        R = np.empty((n_used, n_genes), dtype=np.float64)
        for g in range(n_genes):
            R[:, g] = _rankdata_numba(deltas[:, g])

        m = (n_used + 1) / 2.0
        Rc = R - m
        s2 = np.empty(n_genes, dtype=np.float64)
        for g in range(n_genes):
            acc = 0.0
            for k in range(n_used):
                acc += Rc[k, g] * Rc[k, g]
            s2[g] = acc

        for p_idx in range(n_pairs):
            i = triu_i[p_idx]
            j = triu_j[p_idx]
            num = 0.0
            for k in range(n_used):
                num += Rc[k, i] * Rc[k, j]
            d = np.sqrt(s2[i] * s2[j])
            if d > 0:
                out[s, p_idx] = num / d
    return out


def plot_qq_distribution(shuffled_full, obs_value, gene_pair_name, verbose =True):
    """
    Plots a histogram (with fitted normal curve) and Q-Q plot of a shuffled
    correlation null distribution against the observed value, and checks
    whether the null looks normal enough (Q-Q R^2 > 0.90) to trust.

    Parameters
    ----------
    shuffled_full : array-like
        Correlation values from the permutation/scramble null.
    obs_value : float
        The observed (unshuffled) correlation, marked on the plot.
    gene_pair_name : str
        Label used in plot titles.
    verbose : bool, default=True
        If True, draws and shows the histogram/Q-Q figure. If False, the
        normality decision is still computed (and returned) but nothing is
        plotted.

    Returns
    -------
    bool
        True if the Q-Q fit R^2 exceeds 0.90. False if the fit is poor, or
        early (without plotting, regardless of `verbose`) if there are too
        few finite values or the shuffle has zero variance to judge
        normality at all.
    """
    shuffled_full = np.asarray(shuffled_full)
    shuffled = shuffled_full[np.isfinite(shuffled_full)]
        
    # with 10k shuffles, anything under 9000 usable values is suspicious
    if len(shuffled) < 0.09*len(shuffled_full):
        return False
        
    # zero variance → degenerate distribution → not normal
    if np.std(shuffled) == 0:
        return False
    (osm, osr), (slope, intercept, r) = stats.probplot(
            shuffled, dist="norm"
        )
    if verbose:
        # Fit normal distribution (only needed for the plot below)
        mu, sigma = stats.norm.fit(shuffled)
        x = np.linspace(shuffled.min(), shuffled.max(), 100)
        fitted_normal = stats.norm.pdf(x, mu, sigma)

        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 1. Histogram with fitted normal distribution
        ax1 = axes[0]
        n, bins, patches = ax1.hist(shuffled, bins=50, density=True, alpha=0.7,
                                    edgecolor='black', label='Shuffled data')

        ax1.plot(x, fitted_normal, 'r-', linewidth=2, label=f'Normal fit\nμ={mu:.4f}, σ={sigma:.4f}')
        # Mark observed value
        ax1.axvline(obs_value, color='green', linestyle='--', linewidth=2, 
                    label=f'Actual correlation = {obs_value:.4f}')
        
        ax1.set_xlabel('Correlation coefficient')
        ax1.set_ylabel('Density')
        ax1.set_title(f'Histogram with Normal Fit\n{gene_pair_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]


        ax2.scatter(osm, osr, s=12, alpha=0.6, label='Quantiles')
        ax2.plot(
            osm,
            slope * osm + intercept,
            'r--',
            label=f'QQ fit (R² = {r**2:.4f})'
        )

        ax2.set_title(f'Q-Q Plot\n{gene_pair_name}')
        ax2.set_xlabel('Theoretical quantiles')
        ax2.set_ylabel('Sample quantiles')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # =========================
    # 3. Decision (QQ-based)
    # =========================
    qq_r2 = r ** 2
    return qq_r2 > 0.90


def check_gene_gene_correlation_threshold(measurements,
                                          pairwise_gene_gene_correlation_matrix, 
                                          gene_list, 
                                          threshold=0.04,
                                          use_scramble=True,
                                          z_score_threshold=2.5,
                                          n_shuffles=10000,
                                          verbose=False,
                                          n_cores_to_use = 4,
                                          base_seed = 101010):
    """
    Splits gene-gene pairs into "no_regulation" vs "potential_regulation" by testing
    each observed gene-gene Spearman correlation against a null distribution built by
    permuting cells (when `use_scramble=True`), or against a fixed absolute-correlation
    cutoff (when `use_scramble=False`).

    Parameters
    ----------
    measurements : pd.DataFrame
        One row per cell, must contain '{gene}_mRNA' for each gene in gene_list.
    pairwise_gene_gene_correlation_matrix : pd.DataFrame
        Gene x gene observed correlation matrix (e.g. from
        `calculate_pairwise_gene_gene_correlation_matrix`), computed from the same
        cells as `measurements` -- the null below is built at that same sample size,
        so a mismatched source table would silently compare correlations computed at
        different n.
    gene_list : list of str
        Gene names (without '_mRNA' suffix) to test, pairwise.
    threshold : float, default=0.04
        Absolute correlation cutoff used only when `use_scramble=False`.
    use_scramble : bool, default=True
        If True, significance is a two-sided z-test against a permutation null
        (see z_score_threshold). If False, significance is `abs(corr) > abs(threshold)`
        with no permutation test and no normality check.
    z_score_threshold : float, default=2.5
        Two-sided z-score cutoff for significance when `use_scramble=True`:
        a pair is significant if `abs(z) > z_score_threshold`, where z is the
        observed correlation standardized against the permutation null's mean/std.
    n_shuffles : int, default=10000
        Number of independent cell-permutations used to build the null distribution.
    verbose : bool, default=False
        If True, plots the null histogram (and, for significant pairs, the Q-Q
        normality check via `plot_qq_distribution`) for each gene pair.
    n_cores_to_use : int, default=4
    base_seed : int, default=101010
        Seed for generating the independent per-shuffle permutation seeds.

    Returns
    -------
    no_regulation : list of (str, str)
        Gene pairs classified as not significantly correlated (or, when
        `use_scramble=True`, significant but with a non-normal-looking null).
    potential_regulation : list of (str, str)
        Gene pairs classified as significantly correlated.
    null_stats : dict[(str, str), tuple[float, float] or float]
        Per pair, `(null_mean, null_std)` of the permutation null when
        `use_scramble=True` -- lets a caller derive a correlation-scale threshold at
        any z-score cutoff (`null_mean +/- z * null_std`) without re-running the
        permutation test. When `use_scramble=False`, holds the plain `threshold`
        value instead.
    z_score_calc : dict[(str, str), float or None]
        Per pair, the z-score of the observed correlation against the permutation
        null when `use_scramble=True`; `None` when `use_scramble=False`.
    """
    
    # Extract gene matrices from DataFrame
    gene_matrix = []
    
    for gene in gene_list:
        # Look for gene columns (adapt this to your column naming)
        gene_col = f"{gene}_mRNA"  # Adjust this pattern as needed
        if gene_col in measurements.columns:
            # Split by time point or condition - adjust this logic for your data structure
            gene_data = measurements[gene_col].values
            gene_matrix.append(gene_data)
        else:
            raise ValueError(f"Could not find column for gene: {gene}")
    
    gene_matrix = np.array(gene_matrix)  # Shape: (n_genes, n_cells)
    all_pairs = list(combinations(gene_list, 2))  # Unique undirected pairs

    pair_correlations = {(gi, gj): pairwise_gene_gene_correlation_matrix.loc[gi, gj] for gi, gj in all_pairs}

    if use_scramble:
        _assert_no_nan(gene_matrix, "check_gene_gene_correlation_threshold")

        numba.set_num_threads(max(1, n_cores_to_use))
        Rc, s2 = _rank_center_and_sumsq(gene_matrix)
        denom = np.sqrt(np.outer(s2, s2))
        denom[denom == 0] = np.nan

        gene_to_idx = {g: i for i, g in enumerate(gene_list)}
        pair_i = np.array([gene_to_idx[gi] for gi, gj in all_pairs], dtype=np.int64)
        pair_j = np.array([gene_to_idx[gj] for gi, gj in all_pairs], dtype=np.int64)

        # Generate null distribution
        seeds = _spawn_independent_seeds(base_seed, n_shuffles)

        null_matrix = _rank_permutation_null_kernel(Rc, Rc, denom, seeds, pair_i, pair_j)
        # null_matrix shape: (n_shuffles, n_pairs), columns aligned with all_pairs order

        # Null mean/std per pair, used to convert each observed correlation into a z-score.
        null_mean = np.mean(null_matrix, axis=0)
        null_std = np.std(null_matrix, axis=0, ddof=1)
        observed = np.array([pair_correlations[(gi, gj)] for gi, gj in all_pairs], dtype=float)
        z_score = (observed - null_mean) / null_std

    no_regulation, potential_regulation = [], []
    null_stats = {}
    z_score_calc = {}
    is_significant = False
    is_relatively_normal = True
    for pos, (gi, gj) in enumerate(all_pairs):
        is_significant = False
        corr_val = pair_correlations[(gi, gj)]
        if use_scramble:
            shuffled_vals = null_matrix[:, pos]
            z_val = z_score[pos]

            is_significant = not np.isnan(z_val) and abs(z_val) > z_score_threshold
            print(f"For gene {gi}, gene {gj}, observed correlation: {corr_val:.4f}, z-score: {z_val:.4f}")
            if verbose:
                try:
                    direction_str = "-"
                    plt.figure(figsize=(6, 4))
                    plt.hist(shuffled_vals, bins=50, color="skyblue", alpha=0.7, edgecolor="k")
                    plt.axvline(corr_val, color="black", linestyle="-", label=f"actual={(corr_val):.3f}")
                    plt.title(f"Gene correlation: {gi} {direction_str} {gj}, z-score = {z_val:.3f}")
                    plt.xlabel(r"gene correlation $\rho$")
                    plt.ylabel("number of scrambles")
                    plt.legend()
                    plt.tight_layout()
                    plt.show()
                except:
                    print(f"Error encountered when calculating correlation for Gene correlation: {gi} {direction_str} {gj}")
                    no_regulation.append((gi, gj))
                    continue
                
            if is_significant:
                gene_pair_name = f"{gi}-{gj}"
                is_relatively_normal = plot_qq_distribution(shuffled_vals, corr_val, gene_pair_name, verbose = verbose)
                print(f"For gene {gi}, gene {gj}, null distribution is normal: {is_relatively_normal}")
        else:
            if abs(corr_val) > abs(threshold):
                is_significant = True
        # Classify pairs
        if use_scramble:
            # (null_mean, null_std) per pair -- lets a caller derive a correlation-scale
            # threshold at any z_score_threshold (null_mean +/- z * null_std) without
            # re-running the permutation test.
            null_stats[(gi, gj)] = (null_mean[pos], null_std[pos])
            z_score_calc[(gi, gj)] = z_val
        else:
            null_stats[(gi, gj)] = threshold
            z_score_calc[(gi, gj)] = None  # No z-score calculated without scramble
        if is_significant and is_relatively_normal:
            potential_regulation.append((gi, gj))
        else:
            no_regulation.append((gi, gj))
    return no_regulation, potential_regulation, null_stats, z_score_calc

def calculate_pair_correlation(rep_0, rep_1, gene_list, type_comparison="twin"):
    """
    Computes gene-wise pairwise Spearman correlations between delta values across two replicates.

    Parameters
    ----------
    rep_0 : pd.DataFrame
        DataFrame for replicate 1, must include 'clone_id' and '{gene}_mRNA' columns.

    rep_1 : pd.DataFrame
        DataFrame for replicate 2, same structure as rep_0.

    gene_list : list of str
        List of gene names (without "_mRNA" suffix) to analyze.

    type_comparison : str, optional
        Type of comparison:
        - "twin": requires exact matching of `clone_id` between replicates.
        - "random": does not require matching clone_ids.

    Returns
    -------
    correlations : dict
        Dictionary of Spearman correlation values keyed as "gene1-gene2".
        Each value corresponds to correlation of Δgene1 vs Δgene2.
    """
    rep_0 = rep_0.reset_index(drop=True)
    rep_1 = rep_1.reset_index(drop=True)

    if type_comparison == "twin":
        rep_0 = rep_0.sort_values("clone_id").reset_index(drop=True)
        rep_1 = rep_1.sort_values("clone_id").reset_index(drop=True)
        if not rep_0["clone_id"].equals(rep_1["clone_id"]):
            raise ValueError("After sorting, clone_ids in rep_0 and rep_1 do not match.")

    correlations = {}
    for gene_1 in gene_list:
        for gene_2 in gene_list:
            delta_1 = rep_0[f"{gene_1}_mRNA"] - rep_1[f"{gene_1}_mRNA"]
            delta_2 = rep_0[f"{gene_2}_mRNA"] - rep_1[f"{gene_2}_mRNA"]
            corr = spearmanr(delta_1, delta_2).correlation
            correlations[f"{gene_1}-{gene_2}"] = corr
    return correlations

def split_twin_pairs(simulation_single_time):
    """
    Splits a twin-eligible cell table into its two per-pair replicates.

    Parameters
    ----------
    simulation_single_time : pd.DataFrame
        One row per cell, must contain 'clone_id'. Each clone_id must have
        exactly 2 rows (a twin pair) -- clones with more than 2 rows raise
        rather than being silently dropped, since clone_id then no longer
        uniquely identifies a twin pair (e.g. barcoded data where a clone
        can have more than 2 cells).

    Returns
    -------
    rep_0 : pd.DataFrame
        The first cell of each pair.

    rep_1 : pd.DataFrame
        The second cell of each pair. Row i of rep_1 is the twin of row i
        of rep_0.

    Raises
    ------
    ValueError
        If any clone_id has more than 2 rows, or no valid twin pairs exist.
    """
    twins = simulation_single_time.groupby("clone_id")
    oversized_clones = {cid: len(group) for cid, group in twins if len(group) > 2}
    if oversized_clones:
        example_cid, example_n = next(iter(oversized_clones.items()))
        raise ValueError(
            f"{len(oversized_clones)} clone_id(s) have more than 2 cells at this "
            f"timepoint (e.g. clone_id={example_cid!r} has {example_n} cells), so "
            "clone_id does not uniquely identify a twin pair. Pass a table that is "
            "subset to clones with exactly 2 cells before calling this function."
        )

    twin_pairs = [group for cid, group in twins if len(group) == 2]
    if not twin_pairs:
        raise ValueError("No valid twin pairs (clone_id with exactly 2 cells) found!")

    rep_0 = pd.concat([g.iloc[[0]] for g in twin_pairs], ignore_index=True)
    rep_1 = pd.concat([g.iloc[[1]] for g in twin_pairs], ignore_index=True)
    return rep_0, rep_1

def calculate_twin_random_pair_correlations(simulation_random_pair, simulation_twin_pair, gene_list, n_random=None, seed=10100):
    """
    Computes twin and random pairwise gene-gene correlation matrices.

    Parameters
    ----------
    simulation_random_pair : pd.DataFrame
        Full dataset at the given time point(s), used for random pairing.
        Must contain 'clone_id' and '{gene}_mRNA' for each gene in gene_list.

    simulation_twin_pair : pd.DataFrame
        Subset at the same time point, used for true twin correlation.
        Must contain 'clone_id' and '{gene}_mRNA' for each gene in gene_list.

    gene_list : list of str
        List of gene names (without "_mRNA" suffix) to analyze.

    n_random : int, optional
        Number of random pairs to sample. If None, equals number of true twin pairs.

    seed : int, default=10100
        Random seed for reproducibility.

    Returns
    -------
    twin_corr_matrix : pd.DataFrame
        Gene–gene Spearman correlations between true twin pairs.

    random_corr_matrix : pd.DataFrame
        Gene–gene Spearman correlations between random pairs of cells.
    """
    rng = np.random.default_rng(seed)

    # --- Twin pairs: two cells with same clone_id ---
    rep_0, rep_1 = split_twin_pairs(simulation_twin_pair)

    twin_corr_dict = calculate_pair_correlation(rep_0, rep_1, gene_list, type_comparison="twin")
    twin_corr_matrix = dict_to_matrix(twin_corr_dict, gene_list)

    # --- Random pairs: random cells from different clones ---
    all_cells = simulation_random_pair.reset_index(drop=True)
    n_cells = len(all_cells)
    n_pairs = len(rep_0) if n_random is None else n_random

    # Draw random pairs without replacement in each position, ensuring different clone_ids
    rand_pairs = []
    attempts = 0
    max_attempts = n_pairs * 10
    while len(rand_pairs) < n_pairs and attempts < max_attempts:
        i, j = rng.choice(n_cells, size=2, replace=False)
        if all_cells.loc[i, "clone_id"] != all_cells.loc[j, "clone_id"]:
            rand_pairs.append((i, j))
        attempts += 1

    random_0 = all_cells.loc[[i for i, _ in rand_pairs]].reset_index(drop=True)
    random_1 = all_cells.loc[[j for _, j in rand_pairs]].reset_index(drop=True)

    random_corr_dict = calculate_pair_correlation(random_0, random_1, gene_list, type_comparison="random")
    random_corr_matrix = dict_to_matrix(random_corr_dict, gene_list)

    return twin_corr_matrix, random_corr_matrix

def differentiate_single_state_reg_and_multiple_states(measurements, potential_regulation, twin_correlation_matrix, random_correlation_matrix, gene_list, z_score_threshold=10, verbose = True):
    """
    Separates potential regulatory gene pairs into multiple-state vs single-state regulation.

    Parameters
    ----------
    measurements : pd.DataFrame
        The cell-gene dataframe containing sample information.
    potential_regulation : list of tuple
        List of gene pairs (gene_i, gene_j) with potential regulation.
    twin_correlation_matrix : pd.DataFrame
        Twin pair correlation matrix at time t1.
    random_correlation_matrix : pd.DataFrame
        Random pair correlation matrix at time t1.
    gene_list : list of str
        List of gene names (e.g., 'gene_1') in correct matrix order.
    z_score_threshold : float, optional
        Threshold for z-score between random pair correlations and twin pair correlation above which a pair is considered multi-state.
    Returns
    -------
    multiple_states_gene_pairs : list of tuple
       Gene pairs with z-score between random pair correlations and twin pair correlation greater than the z-score threshold.

    single_state_regulation : list of tuple
        Gene pairs with z-score between random pair correlations and twin pair correlation less than the z-score threshold.

    null_stats : dict[(str, str), tuple[float, float]]
        Per pair, `(random_pair_mean, random_pair_std)` of the random-pair
        correlation distribution -- lets a caller derive a threshold at any
        z-score cutoff (`random_pair_mean +/- z * random_pair_std`) without
        re-running `generate_random_shuffle`. `random_pair_std` is 0.0 for the
        zero-variance case below.

    z_score_calc : dict[(str, str), float or None]
        Per pair, the z-score of the twin correlation against the random-pair
        distribution; `None` for the zero-variance case below, where a z-score
        is not defined.
    """
    multiple_states_gene_pairs = []
    single_state_regulation = []
    null_stats = {}
    z_score_calc = {}

    random_pair_correlation_distribution = generate_random_shuffle(measurements, gene_list=gene_list)
    for gene_i, gene_j in potential_regulation:
        try:
            t_corr = twin_correlation_matrix.loc[gene_i, gene_j]
            r_corr = get_correlations(random_pair_correlation_distribution, gene_i, gene_j)
            r_corr_mean = np.mean(r_corr)
            r_corr_std = np.std(r_corr)
            if r_corr_std == 0:
                # All random correlations are identical
                print(f"Warning: Zero variance in random correlations for {gene_i}-{gene_j}")
                single_state_regulation.append((gene_i, gene_j))
                null_stats[(gene_i, gene_j)] = (float(r_corr_mean), 0.0)
                z_score_calc[(gene_i, gene_j)] = None
                continue
            z_score = (t_corr - r_corr_mean)/r_corr_std
            null_stats[(gene_i, gene_j)] = (float(r_corr_mean), float(r_corr_std))
            z_score_calc[(gene_i, gene_j)] = float(z_score)
            if verbose:
                plt.hist(r_corr)
                plt.axvline(t_corr, linestyle = "--", c = "red", label = r"twin difference correlation $\hat{\rho}_\Delta(t_1)$")
                plt.xlabel(r"random-pair difference correlation $\rho_\Delta(t_1)$")
                plt.ylabel("number of scrambles")
                plt.title(f"Random pair difference correlations vs twin difference correlation \
                    \n between {gene_i} and {gene_j} \
                    \n Z-score = {z_score}")
                plt.legend()
                plt.show()

            if abs(z_score) > abs(z_score_threshold):
                multiple_states_gene_pairs.append((gene_i, gene_j))
                print(f"gene 1: {gene_i}, gene 2: {gene_j}, z_score: {z_score} with threshold {z_score_threshold}")
            else:
                single_state_regulation.append((gene_i, gene_j))
                print(f"gene 1: {gene_i}, gene 2: {gene_j}, z_score: {z_score} with threshold {z_score_threshold}")
        except KeyError:
            raise ValueError(f"Missing gene pair ({gene_i}, {gene_j}) in correlation matrices.")
    return multiple_states_gene_pairs, single_state_regulation, null_stats, z_score_calc

def _bootstrap_spearman(delta_1, delta_2, n_boot, rng):
    """
    Bootstrap the Spearman correlation of two paired vectors by resampling pair
    INDICES with replacement (the same indices applied to both vectors, so pairing
    is preserved), n_boot times. Returns an array of n_boot correlation values.

    This is a different resampling operation from a permutation/shuffle null --
    it resamples the observed units with replacement to estimate the sampling
    variability of the correlation ESTIMATOR under the actual data, rather than
    permuting a fixed pairing to get the null variance under independence.
    Prototyped against the closed-form SE(d) on the figure_2 A_B/A_to_B simulation
    set: se_boot came out consistently larger (~0.038-0.047 vs 0.0365 fixed) --
    more conservative than, not a replacement for, the closed form.
    """
    n = len(delta_1)
    out = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        out[b] = spearmanr(delta_1[idx], delta_2[idx]).correlation
    return out


#TODO Change to the new definition soon!
def identify_reg_if_multiple_states(twin_correlation_matrix_t1, twin_correlation_matrix_t2, random_correlation_matrix_t1, random_correlation_matrix_t2, multiple_states_gene_pairs, gene_list, t1_twins, t2_twins, regulation_increase_threshold=0.024, alpha=0.05, compute_bootstrap_ci=False, n_bootstrap=2000, bootstrap_seed=101010, compute_shuffle_null=False, n_shuffles_null=10000, shuffle_null_seed=101010):
    """
    Among multiple-state gene pairs, identify which also show regulation
    (based on an increase in twin difference-correlation from t1 to t2).

    Replaces the old relative-increase test, relative_change = (corr_t2 -
    corr_t1) / abs(corr_t1): corr_t1 (twin correlation shortly after
    division) sits close to 0 for every pair regardless of regulation, so
    that ratio explodes numerically and, per calibration on the Figure-2
    simulations, flagged ~90% of a true no-regulation scenario as
    regulation (TwINFER_thresholds_and_CI_v2_1.pdf, Section 6). The
    replacement tests the absolute increase directly,

        d = corr_t2 - corr_t1,

    against regulation_increase_threshold. The classification (which list a
    pair goes into) is set by this point estimate alone, per the spec: "The
    decision is made on the point estimate; the interval annotates it and
    is not a second filter."

    The interval itself is now computed too, closed-form: t1_twins and
    t2_twins (the raw per-cell tables the caller already built to compute
    twin_correlation_matrix_t1/_t2) are split into their twin-pair
    replicates to recover N1, N2 -- the independent-unit counts (mother
    cells/clones, never cells or enumerated pairs) -- and

        SE(d) = sqrt(1/(N1-1) + 1/(N2-1))

    which is exact because t1_twins and t2_twins are built from disjoint
    clone sets (see infer_with_twinfer's t1_clones/t2_clones split), so the
    two variances add. This is the "pairs as units" closed form from the
    spec, valid here because split_twin_pairs already reduces each table to
    one row per independent twin pair; it is not the clone-stratified
    bootstrap the spec uses for barcoded data where one clone can supply
    several pairs across the two timepoints, since t1_twins/t2_twins as
    constructed here always satisfy N = pairs = independent units directly.
    Every gene pair shares the same N1, N2 and hence the same SE(d) -- only
    d itself varies pair to pair -- so it is computed once, not per pair.

    Each pair's 95% (by default) interval [d - z*SE, d + z*SE] is then read
    against regulation_increase_threshold three ways, matching the spec's
    reporting convention:
        interval entirely above threshold  -> "conclusive regulation"
        interval entirely below threshold  -> "conclusive no regulation"
        interval spans threshold           -> "inconclusive"
    This verdict is returned per pair (see stage3_details below) but does
    NOT change which list the pair is appended to -- the point estimate
    still decides that, so the classification stays exhaustive even when a
    call is statistically inconclusive.

    CAVEAT: the default 0.024 is calibrated on a single simulation run, a
    single gene pair, at a single measurement time t2 -- not a validated
    universal constant. The spec's own calibration table shows the
    "correct" threshold scaling roughly 8x across t2 = 2h to 36h, and even
    at the calibration point the regulation/no-regulation d ranges overlap
    over repeated draws. See REORG_CHECKLIST.md for the tracked TODO to
    validate this threshold across regulation strengths, multi-state
    separations, and measurement times before trusting it outside the
    calibration scenario. Relatedly, the spec finds a single experiment is
    structurally underpowered for a conclusive Stage III call at the
    calibration effect size (roughly 15 replicates needed) -- expect
    "inconclusive" often on a single run, and treat it as the honest
    answer, not a bug.

    Parameters
    ----------
    twin_correlation_matrix_t1 : pd.DataFrame
        Twin correlation matrix at earlier time t1.

    twin_correlation_matrix_t2 : pd.DataFrame
        Twin correlation matrix at later time t2.

    random_correlation_matrix_t1 : pd.DataFrame
        Random pair correlation matrix at t1 (unused in logic here, included for completeness).

    random_correlation_matrix_t2 : pd.DataFrame
        Random pair correlation matrix at t2 (unused in logic here, included for completeness).

    multiple_states_gene_pairs : list of tuple
        Gene pairs previously classified as showing multiple-state behavior.

    gene_list : list of str
        List of gene names (e.g., 'gene_1').

    t1_twins : pd.DataFrame
        The same per-cell table passed to calculate_twin_random_pair_correlations
        to build twin_correlation_matrix_t1 -- one row per cell, 'clone_id'
        shared by each twin pair. Used only to recover N1 (the twin-pair
        count at t1) for the standard error; not re-correlated.

    t2_twins : pd.DataFrame
        Same as t1_twins, for t2 (recovers N2).

    regulation_increase_threshold : float, optional
        Minimum increase d = corr_t2 - corr_t1 in twin correlation to call it
        regulation. Default 0.024, calibrated on the Figure-2 simulation set
        (see CAVEAT above -- not yet validated more broadly).

    alpha : float, optional
        Interval significance level; default 0.05 gives a 95% interval.

    compute_bootstrap_ci : bool, default=False
        If True, also computes a bootstrap CI on d per pair: resample twin-pair
        indices with replacement (independently at t1 and t2), recompute d each
        draw, take the empirical std (normal-approx CI) and the 2.5th/97.5th
        percentiles (percentile CI). This is diagnostic, not a replacement for
        the closed-form interval above -- prototyped on the figure_2 A_B/A_to_B
        simulation set, the bootstrap SE came out consistently *larger* than the
        closed form (more conservative), consistent with the closed form being
        the exact null-variance formula (verified in TwINFER_thresholds_and_CI_v2_1.pdf,
        the permutation-variance proof) rather than an approximation needing
        replacement. Adds compute cost: n_bootstrap Spearman calls x 2 timepoints
        x len(multiple_states_gene_pairs).

    n_bootstrap : int, default=2000
        Bootstrap draws per pair per timepoint, used only when compute_bootstrap_ci=True.

    bootstrap_seed : int, default=101010
        Seed for the bootstrap resampling, used only when compute_bootstrap_ci=True.

    compute_shuffle_null : bool, default=False
        If True, also computes a z-score against a fully-scrambled (non-twin,
        random-pairing) null per pair, via generate_random_shuffle at t1 and t2
        independently. CAVEAT: this reuses the tool built to estimate Stage II's
        ⟨ρ∆⟩ reference, not Stage III's d variance, so it tests a different (more
        lenient) null -- "is d different from zero real signal" -- rather than
        the calibrated "is d above regulation_increase_threshold" the closed-form
        interval tests. Treat as a diagnostic. Adds two generate_random_shuffle
        calls (each O(n_shuffles_null) over all gene pairs, run once per call,
        not per pair).

    n_shuffles_null : int, default=10000
        Shuffles per generate_random_shuffle call, used only when compute_shuffle_null=True.

    shuffle_null_seed : int, default=101010
        Base seed for the two generate_random_shuffle calls (t2 uses shuffle_null_seed+1),
        used only when compute_shuffle_null=True.

    Returns
    -------
    multiple_states_no_reg : list of tuple
        Gene pairs with multiple states but no significant increase in correlation (no regulation).

    multiple_states_and_reg : list of tuple
        Gene pairs with multiple states and increased correlation (suggesting regulation).

    stage3_details : dict
        Keyed by (gene_i, gene_j), one entry per pair in multiple_states_gene_pairs:
            - "d" : float -- corr_t2 - corr_t1, the point estimate.
            - "standard_err" : float -- sqrt(1/(N1-1) + 1/(N2-1)), shared across all pairs.
            - "n1", "n2" : int -- twin-pair counts at t1, t2.
            - "interval" : (float, float) -- the (1 - alpha) confidence interval on d.
            - "verdict" : str -- "conclusive regulation", "conclusive no regulation",
              or "inconclusive".
            - "call" : str -- "regulation" or "no regulation", the point-estimate
              classification (matches which of the two lists above the pair is in).
            - "se_boot", "interval_boot", "interval_boot_percentile" : only when
              compute_bootstrap_ci=True -- bootstrap std, normal-approx CI
              (d +/- z*se_boot), and percentile CI, respectively.
            - "null_mean_shuffle", "se_shuffle", "z_shuffle" : only when
              compute_shuffle_null=True -- the scrambled null's mean and std for
              this pair, and z_shuffle = (d - null_mean_shuffle) / se_shuffle.
    """
    multiple_states_no_reg = []
    multiple_states_and_reg = []
    stage3_details = {}

    rep_0_t1, rep_1_t1 = split_twin_pairs(t1_twins)
    rep_0_t2, rep_1_t2 = split_twin_pairs(t2_twins)
    n1 = len(rep_0_t1)
    n2 = len(rep_0_t2)
    standard_err = float(np.sqrt(1.0 / (n1 - 1) + 1.0 / (n2 - 1)))
    z_crit = float(stats.norm.ppf(1 - alpha / 2))

    if compute_bootstrap_ci:
        boot_rng = np.random.default_rng(bootstrap_seed)

    if compute_shuffle_null:
        # One shuffle run per timepoint, over every gene pair at once -- not per pair.
        null_dist_t1 = generate_random_shuffle(t1_twins, gene_list, n_shuffles=n_shuffles_null, random_state=shuffle_null_seed)
        null_dist_t2 = generate_random_shuffle(t2_twins, gene_list, n_shuffles=n_shuffles_null, random_state=shuffle_null_seed + 1)

    for gene_i, gene_j in multiple_states_gene_pairs:
        try:
            corr_t1 = twin_correlation_matrix_t1.loc[gene_i, gene_j]
            corr_t2 = twin_correlation_matrix_t2.loc[gene_i, gene_j]
            print(f"Testing for multiple states. Correlation at time t1 = {corr_t1} and at time t2 = {corr_t2}")

            d = corr_t2 - corr_t1
            lo, hi = d - z_crit * standard_err, d + z_crit * standard_err

            if lo > regulation_increase_threshold:
                verdict = "conclusive regulation"
            elif hi < regulation_increase_threshold:
                verdict = "conclusive no regulation"
            else:
                verdict = "inconclusive"

            call = "regulation" if d > regulation_increase_threshold else "no regulation"
            print(
                f"gene 1: {gene_i}, gene 2: {gene_j}, d: {d:.4f}, "
                f"{100 * (1 - alpha):.0f}% CI: [{lo:.4f}, {hi:.4f}], "
                f"call: {call}, verdict: {verdict}"
            )

            details = {
                "d": d, "standard_err": standard_err, "n1": n1, "n2": n2,
                "interval": (lo, hi), "verdict": verdict, "call": call,
            }

            if compute_bootstrap_ci:
                d1_t1 = (rep_0_t1[f"{gene_i}_mRNA"] - rep_1_t1[f"{gene_i}_mRNA"]).to_numpy()
                d2_t1 = (rep_0_t1[f"{gene_j}_mRNA"] - rep_1_t1[f"{gene_j}_mRNA"]).to_numpy()
                d1_t2 = (rep_0_t2[f"{gene_i}_mRNA"] - rep_1_t2[f"{gene_i}_mRNA"]).to_numpy()
                d2_t2 = (rep_0_t2[f"{gene_j}_mRNA"] - rep_1_t2[f"{gene_j}_mRNA"]).to_numpy()
                corr_t1_boot = _bootstrap_spearman(d1_t1, d2_t1, n_bootstrap, boot_rng)
                corr_t2_boot = _bootstrap_spearman(d1_t2, d2_t2, n_bootstrap, boot_rng)
                d_boot = corr_t2_boot - corr_t1_boot
                se_boot = float(np.std(d_boot, ddof=1))
                lo_boot, hi_boot = d - z_crit * se_boot, d + z_crit * se_boot
                lo_pct, hi_pct = np.percentile(d_boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
                details["se_boot"] = se_boot
                details["interval_boot"] = (float(lo_boot), float(hi_boot))
                details["interval_boot_percentile"] = (float(lo_pct), float(hi_pct))

            if compute_shuffle_null:
                key = tuple(sorted((gene_i, gene_j)))
                d_random = null_dist_t2[key] - null_dist_t1[key]
                null_mean_shuffle = float(np.mean(d_random))
                se_shuffle = float(np.std(d_random, ddof=1))
                z_shuffle = (d - null_mean_shuffle) / se_shuffle if se_shuffle > 0 else np.nan
                details["null_mean_shuffle"] = null_mean_shuffle
                details["se_shuffle"] = se_shuffle
                details["z_shuffle"] = float(z_shuffle)

            stage3_details[(gene_i, gene_j)] = details

            if call == "regulation":
                multiple_states_and_reg.append((gene_i, gene_j))
            else:
                multiple_states_no_reg.append((gene_i, gene_j))
        except KeyError:
            raise ValueError(f"Missing gene pair ({gene_i}, {gene_j}) in correlation matrices.")

    return multiple_states_no_reg, multiple_states_and_reg, stage3_details

def get_cross_correlations(rep_0_t1,
                                   rep_1_t2,
                                   gene_pairs,
                                   type_comparison="twin"):
    """
    Computes directional Spearman correlations between gene_1 (at t1) and gene_2 (at t2),
    and returns both raw and normalized directional matrices.

    Parameters
    ----------
    rep0_t1 : pd.DataFrame
        Simulation data at time t1 with one twin, with columns: 'replicate', 'clone_id', '{gene}_mRNA'.

    rep1_t2 : pd.DataFrame
        Simulation data at time t2 with the other twin, same structure as t1.

    gene_pairs : list of tuple
        List of (gene_1, gene_2) pairs to analyze directionally.

    type_comparison : str, optional
        If "twin", checks that clone_ids are aligned. If "random", no check is performed.

    Returns
    -------
    raw_matrix : pd.DataFrame
        Raw correlation matrix (gene_1 at t1 → gene_2 at t2).

    normalized_matrix : pd.DataFrame
        Normalized correlation matrix.
    """
    gene_pairs = gene_pairs.copy()
    gene_list = sorted(set(g for pair in gene_pairs for g in pair))

    # Separate replicates for t1 and t2
    rep_0_t1 = rep_0_t1.sort_values("clone_id").reset_index(drop=True)
    rep_1_t2 = rep_1_t2.sort_values("clone_id").reset_index(drop=True)


    all_genes = list(set(gene_1 for gene_1, _ in gene_pairs) | set(gene_2 for _, gene_2 in gene_pairs))
    self_pairs = [(gene, gene) for gene in all_genes if (gene, gene) not in gene_pairs]
    gene_pairs += self_pairs

    if type_comparison == "twin":
        if not rep_0_t1["clone_id"].equals(rep_1_t2["clone_id"]):
                print("Clone IDs do not match between replicates:")
                mismatched_ids = rep_1_t2[~rep_0_t1["clone_id"].isin(rep_1_t1["clone_id"])]
                print(mismatched_ids["clone_id"].unique())
                raise ValueError(f"Mismatch in clone_id")

    # Compute raw directional correlations
    raw_matrix = pd.DataFrame(index=gene_list, columns=gene_list, dtype=float)
    for gene_1, gene_2 in gene_pairs:
        x = rep_0_t1[f"{gene_1}_mRNA"]
        y = rep_1_t2[f"{gene_2}_mRNA"]
        corr = spearmanr(x, y).correlation
        raw_matrix.loc[gene_1, gene_2] = corr
    return raw_matrix

def identify_actual_directed_edges(rep_0_t1, rep_1_t2, direction_raw_matrix, gene_pairs, z_score_threshold=2.5,
                                          use_scramble=True, corr_threshold=0.049, n_shuffles=10000, n_cores_to_use = 4, verbose = False,
                                          base_seed = 101010, multiple_states_gene_pairs=None,
                                          return_z_scores = False, return_direction_diff = False):
    """
    Identify directed edges that cross significance thresholds using a shuffled null distribution,
    and (when both directions of a pair are individually significant) test whether the two
    directions are themselves distinguishable.

    Per-direction significance: two-sided z-test against the permutation null, matching
    check_gene_gene_correlation_threshold (Step 1) -- z = (actual_corr - null_mean) / null_std,
    is_significant = abs(z) > z_score_threshold.

    Direction-difference test (only for pairs where BOTH (A,B) and (B,A) are individually
    significant): the null for diff = corr(A->B) - corr(B->A) is built empirically as
    null_matrix[:, pos_AB] - null_matrix[:, pos_BA], not a closed-form sum of variances.
    This matters because, for a given shuffle, _rank_permutation_null_kernel computes every
    direction's null correlation from the SAME permutation draw -- the (A,B) and (B,A) null
    columns are paired row-by-row, not independent. Subtracting them directly carries over
    whatever covariance the two directional statistics share, with no separate derivation
    needed; a closed-form sqrt(Var(A->B)+Var(B->A)) would assume independence that isn't
    obviously true here (the two directions share the same twin pairs) and hasn't been
    validated the way the single-correlation variance was.

    CAVEAT scope: when the difference test finds one direction dominates, that call stands
    on its own -- the difference is itself a resolved point estimate against its own null,
    same as any other z-test in this module, so multi-state status doesn't cast doubt on it.
    The multi-state flag only matters for the OTHER outcome: "both directions individually
    significant, difference not significant" (a "bidirectional / mutual regulation" verdict).
    That specific signature is genuinely ambiguous between real mutual regulation and a
    multi-state confound inflating both directions -- a latent state variable need not couple
    to both genes' dynamics equally, so a bidirectional-looking pair could reflect population
    structure rather than symmetric regulation in both directions. There's no test here that
    tells those apart (that would need conditioning the cross-correlation on state assignment,
    which this function does not do), so `multiple_states_gene_pairs` is used only to caveat
    the bidirectional verdict, never to resolve it or to touch the dominant-direction verdict.

    Parameters
    ----------
    gene_matrix_1, gene_matrix_2 : np.ndarray
        Gene expression matrices (genes × cells)
    direction_raw_matrix : pd.DataFrame
        Actual correlation matrix between genes
    gene_pairs : list of tuples
        Gene pairs to analyze. Both directions of a pair, e.g. (A,B) and (B,A), must both be
        present for the direction-difference test to run on that pair; a pair with only one
        direction present is left as a plain directed-edge call, untouched by any of this.
    z_score_threshold : float, default=2.5
        Two-sided z-score cutoff for significance, used for BOTH the per-direction test and
        the direction-difference test (one shared threshold; see check_gene_gene_correlation_threshold
        for the same convention at Step 1).
    use_scramble : bool
        If True (default), significance is determined from a shuffled null distribution
        (abs(z) > z_score_threshold); skipped entirely when False, and the direction-difference
        test is unavailable (there's no permutation null to build it from).
    corr_threshold : float
        Absolute correlation cutoff used when `use_scramble=False`, bypassing the null
        distribution generation (default 0.049).
    n_shuffles : int
        Number of shuffle iterations
    multiple_states_gene_pairs : list of tuple, optional
        Unordered gene pairs already flagged multi-state (e.g. Step 2's
        differentiate_single_state_reg_and_multiple_states output). Used only when
        return_direction_diff=True, to caveat a "bidirectional / mutual regulation" verdict --
        see CAVEAT scope above; never applied to a dominant-direction verdict. Order within
        each tuple doesn't matter (checked as an unordered pair). Default None means no pair
        is caveated.
    return_z_scores : bool, default=False
        If True, also returns z_score_calc (per-direction z-scores), replacing the old
        return_p_values / p_value_calc.
    return_direction_diff : bool, default=False
        If True, also returns direction_diff_details (see Returns below). Requires
        use_scramble=True.

    Returns
    -------
    significant_edges : list of tuples
        Gene pairs that have significant directed correlations (unchanged by the
        direction-difference test -- that test only annotates, never removes an edge here).
    z_score_calc : dict, only if return_z_scores=True
        {(gene_1, gene_2): z} per-direction z-score (None when use_scramble=False).
    direction_diff_details : dict, only if return_direction_diff=True
        Keyed by frozenset({gene_a, gene_b}), one entry per unordered pair where BOTH
        directions were individually significant (pairs significant in only one direction,
        or neither, are not included here -- they're unambiguous, nothing to annotate):
            - "pair" : (gene_a, gene_b) as they appeared in gene_pairs' first-seen order.
            - "corr_ab", "corr_ba", "diff" : the two raw directional correlations and their difference.
            - "z_ab", "z_ba" : the two directions' own per-direction z-scores.
            - "diff_null_mean", "diff_null_std", "z_diff" : the empirical diff-null's mean/std
              and the resulting z-score of the observed difference against it.
            - "diff_significant" : bool, abs(z_diff) > z_score_threshold.
            - "dominant_direction" : (src, tgt) with the larger |z-score| of the two, if
              diff_significant, else None.
            - "verdict" : "directed: {src}->{tgt} dominates" (diff_significant) or
              "bidirectional / mutual regulation" (not diff_significant).
            - "multi_state_flagged" : bool, whether this pair is in multiple_states_gene_pairs.
              Only a meaningful caveat when verdict is "bidirectional / mutual regulation" --
              see CAVEAT scope above; ignore this field when verdict is a dominant direction.
    """

    gene_list = list(direction_raw_matrix.index)
    print(gene_list)

    if use_scramble:
        # Extract gene list from matrix
        gene_matrix_t1 = []
        gene_matrix_t2 = []

        for gene in gene_list:
            # Look for gene columns
            gene_col_t1 = f"{gene}_mRNA" if f"{gene}_mRNA" in rep_0_t1.columns else None
            gene_col_t2 = f"{gene}_mRNA" if f"{gene}_mRNA" in rep_1_t2.columns else None

            if not gene_col_t1:
                matching_cols = [col for col in rep_0_t1.columns if gene in col and 'mRNA' in col]
                gene_col_t1 = matching_cols[0] if matching_cols else None

            if not gene_col_t2:
                matching_cols = [col for col in rep_1_t2.columns if gene in col and 'mRNA' in col]
                gene_col_t2 = matching_cols[0] if matching_cols else None

            if gene_col_t1 and gene_col_t2:
                gene_matrix_t1.append(rep_0_t1[gene_col_t1].values)
                gene_matrix_t2.append(rep_1_t2[gene_col_t2].values)
            else:
                print(f"    Warning: Could not find {gene} data")
                return None

        gene_matrix_t1 = np.array(gene_matrix_t1)
        gene_matrix_t2 = np.array(gene_matrix_t2)

        _assert_no_nan(gene_matrix_t1, "identify_actual_directed_edges (rep_0_t1)")
        _assert_no_nan(gene_matrix_t2, "identify_actual_directed_edges (rep_1_t2)")

        numba.set_num_threads(max(1, n_cores_to_use))
        Rc1, s2_1 = _rank_center_and_sumsq(gene_matrix_t1)
        Rc2, s2_2 = _rank_center_and_sumsq(gene_matrix_t2)
        denom = np.sqrt(np.outer(s2_1, s2_2))
        denom[denom == 0] = np.nan

        gene_to_idx = {g: i for i, g in enumerate(gene_list)}
        pair_i = np.array([gene_to_idx[g1] for g1, g2 in gene_pairs], dtype=np.int64)
        pair_j = np.array([gene_to_idx[g2] for g1, g2 in gene_pairs], dtype=np.int64)

        seeds = _spawn_independent_seeds(base_seed, n_shuffles)

        null_matrix = _rank_permutation_null_kernel(Rc1, Rc2, denom, seeds, pair_i, pair_j)
        # null_matrix shape: (n_shuffles, n_pairs), columns aligned with gene_pairs order

    # Identify significant directed edges
    significant_edges = []
    z_score_calc = {}
    is_significant_map = {}
    is_relatively_normal = True
    print(f"number of gene pairs = {len(gene_pairs)}")
    for pos, (gene_1, gene_2) in enumerate(gene_pairs):
        # Get actual correlation
        actual_corr = direction_raw_matrix.loc[gene_1, gene_2]
        gene_pair_name = f"{gene_1} -> {gene_2}"

        if use_scramble:
            # Get shuffled correlations for this pair
            shuffled_vals = null_matrix[:, pos]
            null_mean = np.mean(shuffled_vals)
            null_std = np.std(shuffled_vals, ddof=1)
            z_val = (actual_corr - null_mean) / null_std if null_std > 0 else np.nan
            is_significant = not np.isnan(z_val) and abs(z_val) > z_score_threshold
            z_score_calc[(gene_1, gene_2)] = z_val
            print(f"Observed correlation for {gene_1} -> {gene_2}: {actual_corr:.4f}, z-score: {z_val:.4f}")
            print(f"Significant at |z|>{z_score_threshold}: {is_significant}")
            if is_significant:
                is_relatively_normal = plot_qq_distribution(shuffled_vals, actual_corr, gene_pair_name, verbose=verbose)
                print(f"{gene_pair_name}: normality of null: {is_relatively_normal}")
            if verbose:
                try:
                    print(f"{gene_pair_name}: actual = {actual_corr}, z-score = {z_val}")
                    plt.figure(figsize=(6, 4))

                    plt.hist(
                        shuffled_vals,
                        bins=40,
                        color="lightgray",
                        edgecolor="black"
                    )

                    # Actual correlation line
                    plt.axvline(
                        actual_corr,
                        color="blue",
                        linestyle="-",
                        linewidth=2,
                        label=f"actual = {actual_corr:.4g}"
                    )

                    plt.xlabel("Shuffled Spearman correlation")
                    plt.ylabel("number of scrambles")
                    plt.title(f"Null distribution: {gene_1} → {gene_2}")
                    plt.legend(frameon=False)

                    plt.tight_layout()
                    plt.show()
                except:
                        print(f"Error encountered when calculating correlation for Gene correlation: {gene_1} -> {gene_2}")
                        continue
        else:
            is_significant = abs(actual_corr) > corr_threshold
            z_score_calc[(gene_1, gene_2)] = None
            print(f"Observed correlation for {gene_1} -> {gene_2}: {actual_corr:.4f}, |corr| > {corr_threshold}: {is_significant}")

        is_significant_map[(gene_1, gene_2)] = is_significant
        # Check if actual correlation crosses threshold
        if is_significant and is_relatively_normal:
            significant_edges.append((gene_1, gene_2))

    # Direction-difference test: only for unordered pairs where BOTH directions are present
    # in gene_pairs and BOTH are individually significant. See the docstring's "CAVEAT scope"
    # for why multiple_states_gene_pairs caveats the bidirectional verdict only, never the
    # dominant-direction one.
    direction_diff_details = {}
    if return_direction_diff and use_scramble:
        pair_pos = {(g1, g2): pos for pos, (g1, g2) in enumerate(gene_pairs)}
        multi_state_pairs_set = set(frozenset(p) for p in (multiple_states_gene_pairs or []))
        seen = set()
        for (gene_1, gene_2) in gene_pairs:
            if gene_1 == gene_2:
                continue
            key = frozenset((gene_1, gene_2))
            if key in seen:
                continue
            if (gene_2, gene_1) not in pair_pos:
                continue  # only one direction present -- plain directed-edge call, nothing to diff
            seen.add(key)

            if not (is_significant_map.get((gene_1, gene_2)) and is_significant_map.get((gene_2, gene_1))):
                continue  # not the ambiguous case -- unambiguous, nothing to annotate

            pos_ab, pos_ba = pair_pos[(gene_1, gene_2)], pair_pos[(gene_2, gene_1)]
            corr_ab = direction_raw_matrix.loc[gene_1, gene_2]
            corr_ba = direction_raw_matrix.loc[gene_2, gene_1]
            diff_actual = corr_ab - corr_ba

            diff_null = null_matrix[:, pos_ab] - null_matrix[:, pos_ba]
            diff_null_mean = np.mean(diff_null)
            diff_null_std = np.std(diff_null, ddof=1)
            z_diff = (diff_actual - diff_null_mean) / diff_null_std if diff_null_std > 0 else np.nan
            diff_significant = not np.isnan(z_diff) and abs(z_diff) > z_score_threshold

            z_ab, z_ba = z_score_calc[(gene_1, gene_2)], z_score_calc[(gene_2, gene_1)]
            if diff_significant:
                dominant = (gene_1, gene_2) if abs(z_ab) > abs(z_ba) else (gene_2, gene_1)
                verdict = f"directed: {dominant[0]}->{dominant[1]} dominates"
            else:
                dominant = None
                verdict = "bidirectional / mutual regulation"

            direction_diff_details[key] = {
                "pair": (gene_1, gene_2),
                "corr_ab": float(corr_ab), "corr_ba": float(corr_ba), "diff": float(diff_actual),
                "z_ab": float(z_ab), "z_ba": float(z_ba),
                "diff_null_mean": float(diff_null_mean), "diff_null_std": float(diff_null_std),
                "z_diff": float(z_diff), "diff_significant": diff_significant,
                "dominant_direction": dominant, "verdict": verdict,
                "multi_state_flagged": key in multi_state_pairs_set,
            }
            print(f"Direction diff {gene_1}<->{gene_2}: z_diff={z_diff:.4f}, verdict={verdict}"
                  + (" [multi-state flagged]" if key in multi_state_pairs_set and dominant is None else ""))

    if return_z_scores and return_direction_diff:
        return significant_edges, z_score_calc, direction_diff_details
    if return_z_scores:
        return significant_edges, z_score_calc
    if return_direction_diff:
        return significant_edges, direction_diff_details
    return significant_edges


def separate_fan_outs_from_mutual_regulation(all_t1_measurements, twin_correlation_matrix_t1, gene_list,
                                              final_directed_edges, directed_z_scores, direction_matrix,
                                              z_score_threshold=8):
    """
    Distinguishes true mutual regulation (A<->B) from a fan-out artifact, for gene pairs
    that share a common upstream regulator C (C->A and C->B both present in
    final_directed_edges). Both A->B and B->A being directed edges could either reflect real
    reciprocal regulation, or simply be an artifact of A and B being co-driven by C with no
    direct A-B interaction. The two are told apart with the same twin-vs-random z-score
    already used in differentiate_single_state_reg_and_multiple_states (Step 3): a pair whose
    undirected correlation is far outside the random-pair null (|z| large) is behaving like it
    is driven by a shared hidden driver (i.e. C) rather than a direct link between A and B.

    Four cases per (A, B) pair with >=1 common regulator:
      1. Neither A->B nor B->A present: no cross-correlation, left untouched.
      2. Exactly one of A->B / B->A present: feed-forward loop, left untouched.
      3/4. Both present: |z| > z_score_threshold -> fan-out, both edges removed;
           |z| <= z_score_threshold -> mutual regulation, both edges kept.
           Exception: if direction_matrix shows BOTH A->B and B->A are negative (mutual
           repression / toggle-switch motif), the pair is always kept as mutual regulation,
           overriding the z-score call -- a genuine double-negative feedback loop can drive
           A and B into a large undirected z-score the same way a shared-regulator fan-out
           would, so the sign pattern takes precedence. This override does NOT apply to
           mutual activation (both positive): those still go through the z-score check as
           before.

    Parameters
    ----------
    all_t1_measurements : pd.DataFrame
        Single-timepoint (t1) cell-gene dataframe, same data used to build
        twin_correlation_matrix_t1, passed to generate_random_shuffle for the null.
    twin_correlation_matrix_t1 : pd.DataFrame
        Twin pair correlation matrix at t1 (same one used in Step 3).
    gene_list : list of str
        Gene names in matrix order.
    final_directed_edges : set of tuple
        Directed edges (gene_src, gene_tgt) inferred so far.
    directed_z_scores : dict
        Mapping {(gene_src, gene_tgt): z_score} for directed edges (from
        identify_actual_directed_edges' per-direction test).
    direction_matrix : pd.DataFrame
        Directional correlation matrix, thresholded to final_directed_edges.
    z_score_threshold : float, default=8
        |z| above this value on the undirected (A,B) pair marks it a fan-out.

    Returns
    -------
    final_directed_edges : set of tuple
        Updated in place: fan-out (A,B)/(B,A) edges removed.
    directed_z_scores : dict
        Updated in place: fan-out (A,B)/(B,A) entries removed.
    direction_matrix : pd.DataFrame
        Updated in place: fan-out (A,B)/(B,A) cells zeroed.
    fan_out_log : list of dict
        One entry per (A, B) pair that had a common regulator, recording the shared
        regulator(s), the z-score, and the decision made.
    """
    regulators_of = {gene: set() for gene in gene_list}
    for src, tgt in final_directed_edges:
        if src != tgt:
            regulators_of.setdefault(tgt, set()).add(src)

    random_pair_correlation_distribution = generate_random_shuffle(all_t1_measurements, gene_list=gene_list)

    fan_out_log = []
    for idx_a, gene_a in enumerate(gene_list):
        for gene_b in gene_list[idx_a + 1:]:
            common_regulators = (regulators_of.get(gene_a, set()) & regulators_of.get(gene_b, set())) - {gene_a, gene_b}
            if not common_regulators:
                continue

            a_to_b = (gene_a, gene_b) in final_directed_edges
            b_to_a = (gene_b, gene_a) in final_directed_edges

            if not a_to_b and not b_to_a:
                continue  # Case 1: no cross-correlation between A and B
            if a_to_b != b_to_a:
                continue  # Case 2: feed-forward loop, leave as is

            # Case 3 / 4: A->B and B->A both present -- check the undirected (A,B) z-score
            t_corr = twin_correlation_matrix_t1.loc[gene_a, gene_b]
            r_corr = get_correlations(random_pair_correlation_distribution, gene_a, gene_b)
            r_corr_std = np.std(r_corr)
            z_score = np.inf if r_corr_std == 0 else (t_corr - np.mean(r_corr)) / r_corr_std

            regulators_str = ", ".join(sorted(common_regulators))
            mutual_repression = (
                direction_matrix.loc[gene_a, gene_b] < 0 and direction_matrix.loc[gene_b, gene_a] < 0
            )
            if abs(z_score) > z_score_threshold and not mutual_repression:
                print(f"Inference: fan-out detected -- {regulators_str} regulate(s) both {gene_a} and {gene_b} "
                      f"(|z|={abs(z_score):.2f} > {z_score_threshold}); removing direct edge between {gene_a} and {gene_b}.")
                final_directed_edges.discard((gene_a, gene_b))
                final_directed_edges.discard((gene_b, gene_a))
                directed_z_scores.pop((gene_a, gene_b), None)
                directed_z_scores.pop((gene_b, gene_a), None)
                if gene_a in direction_matrix.index and gene_b in direction_matrix.columns:
                    direction_matrix.loc[gene_a, gene_b] = 0
                    direction_matrix.loc[gene_b, gene_a] = 0
                decision = "fan_out"
            elif mutual_repression and abs(z_score) > z_score_threshold:
                print(f"Inference: mutual regulation confirmed -- {gene_a} and {gene_b} share regulator(s) "
                      f"{regulators_str}, but {gene_a}->{gene_b} and {gene_b}->{gene_a} are both negative "
                      f"(mutual repression); overriding the fan-out z-score call "
                      f"(|z|={abs(z_score):.2f} > {z_score_threshold}); keeping both edges.")
                decision = "mutual_regulation"
            else:
                print(f"Inference: mutual regulation confirmed -- {gene_a} and {gene_b} share regulator(s) "
                      f"{regulators_str} but their direct correlation is not explained by it "
                      f"(|z|={abs(z_score):.2f} <= {z_score_threshold}); keeping both edges.")
                decision = "mutual_regulation"

            fan_out_log.append({
                "gene_a": gene_a,
                "gene_b": gene_b,
                "common_regulators": sorted(common_regulators),
                "z_score": z_score,
                "decision": decision,
                "mutual_repression_override": bool(mutual_repression and abs(z_score) > z_score_threshold),
            })

    return final_directed_edges, directed_z_scores, direction_matrix, fan_out_log

"""
correlation_analysis_helper_functions.py

Helper functions for analyzing directional gene-gene correlations and regulatory relationships
in simulated gene regulatory networks (GRNs), particularly based on twin-based inference.

This module includes utilities for:
- Extracting metadata from filenames
- Loading gene interaction matrices and simulation parameters
- Constructing correlation matrices from gene expression simulations
- Visualizing gene-gene correlations as heatmaps or directional network graphs
- Annotating relationships as regulatory or non-regulatory
- Printing categorized summaries for interpretability

Functions
---------
extract_param_index(filename: str) -> str
    Extracts the parameter index string (e.g., '0_1') from a simulation file path.

split_and_merge_simulations(simulation_paths: List[str]) -> pd.DataFrame
    Splits clone IDs as evenly as possible across multiple simulation CSV files,
    then extracts the assigned clones from each file and merges them into a single
    combined DataFrame.
    
read_input_matrix(path_to_matrix: str) -> Tuple[int, np.ndarray]
    Loads a gene interaction matrix from a file and returns its shape and contents.

get_param_data(param_df: pd.DataFrame, param_index: str) -> Dict[str, float]
    Retrieves a flat dictionary of kinetic and interaction parameters for a given simulation.

dict_to_matrix(correlation_dict: Dict[str, float], gene_list: List[str]) -> pd.DataFrame
    Converts a flat dictionary of gene-gene correlations to a square matrix DataFrame.

plot_matrix_as_heatmap(corr_matrix: pd.DataFrame, gene_list: List[str], ...)
    Plots a correlation matrix as a heatmap, highlighting regulated and unregulated gene pairs.

print_summary(no_regulation: List[Tuple[str, str]], 
              single_state_regulation: List[Tuple[str, str]], 
              multiple_states_no_reg: List[Tuple[str, str]], 
              multiple_states_and_reg: List[Tuple[str, str]])
    Prints a categorized summary of gene pair relationships.

plot_network(correlation_matrix: pd.DataFrame, gene_list: List[str], edges, title: Optional[str] = None)
    Visualizes gene-gene correlations as a directional network graph, using arrows or flat-headed lines
    to indicate inferred directionality or undetermined regulation.


Helper Functions
----------------
make_reds_blues_colormap() -> matplotlib.colors.Colormap
    Creates a custom red-blue colormap for correlation values.

shrink_arrow_endpoints(...) -> Tuple[Tuple[float, float], Tuple[float, float]]
    Computes arrow start and end coordinates that are offset from node centers.

flat_t_head_arrow(...) -> None
    Draws a repression-like arrow with a flat T-head.

polygon_layout(gene_list: List[str], radius: float = 1.0) -> Dict[str, Tuple[float, float]]
    Assigns circular coordinates to genes for network layout.
"""

#Import packages
def extract_param_index(filename: str) -> str:
    """
    Extracts the parameter row index (e.g., '0_1') from a simulation filename.

    Handles both 'df_row_' and 'df_rows_' prefixes. The extraction stops before
    an 8-digit date stamp (ddmmyyyy) if present.

    Args:
        filename (str): The simulation filename.

    Returns:
        str: The row identifier (e.g., '0_1'), or 'unknown' if the pattern is not found.
    """
    try:
        # Match df_row_ or df_rows_
        match = re.search(r"df_rows?_(\d+(?:_\d+)*)", filename)
        if not match:
            return "unknown"

        core = match.group(1)
        # Remove trailing date if mistakenly included
        parts = core.split("_")
        cleaned = []
        for part in parts:
            if part.isdigit() and len(part) == 8:  # ddmmyyyy date
                break
            cleaned.append(part)

        return "_".join(cleaned) if cleaned else "unknown"
    except Exception:
        return "unknown"

def get_param_data(param_df, param_index, n_genes):
    """
    Extracts and flattens parameters for a given simulation from a parameter DataFrame.

    This function:
      1. Selects specific rows from `param_df` based on `param_index`.
      2. Flattens gene-specific parameters so keys are labeled as `<term>_gene_<id>`.
      3. Calculates degradation rates (`k_deg_mRNA_gene_X`, `k_deg_protein_gene_X`)
         from the corresponding half-life values.
      4. Adds interaction parameters (non-gene-specific) from the first selected row.

    Args:
        param_df (pd.DataFrame):
            Parameter DataFrame containing both gene-level and interaction-level parameters.
            Must contain columns for each gene term in `gene_terms` and optionally extra metadata.
        param_index (str):
            Underscore-separated string of row indices in `param_df` to use.
            Example: "12_13" will select rows 12 and 13.
        n_genes (int):
            Number of genes expected in the simulation; must match the number of selected rows.

    Returns:
        dict:
            A flat dictionary mapping parameter names to values, e.g.:
            {
                "k_on_gene_1": ...,
                "k_off_gene_1": ...,
                "mrna_half_life_gene_1": ...,
                "k_deg_mRNA_gene_1": ...,
                ...
                "<interaction_param>": ...
            }

    Raises:
        AssertionError:
            If the number of selected rows does not match `n_genes`.
    """
    gene_terms = ["k_on", "k_off", "mrna_half_life", "protein_half_life", 
                  "k_prod_protein", "k_prod_mRNA"]
    extra_terms = ["pair_id", "gene_id"]

    rows = [int(i) for i in param_index.split("_")]
    assert len(rows) == n_genes, f"Mismatch in number of input genes and parameter rows. \n Rows = {rows} and n_genes = {n_genes}"

    selected_rows = param_df.iloc[rows]
    param_dict = {}

    # Gene-specific parameters
    for gene_id, row in enumerate(selected_rows.itertuples(index=False), start=1):
        for term in gene_terms:
            key = f"{term}_gene_{gene_id}"
            param_dict[key] = getattr(row, term)

        # Add degradation terms from half_life
        param_dict[f"k_deg_mRNA_gene_{gene_id}"] = np.log(2) / param_dict[f"mrna_half_life_gene_{gene_id}"]
        param_dict[f"k_deg_protein_gene_{gene_id}"] = np.log(2) / param_dict[f"protein_half_life_gene_{gene_id}"]

    # Interaction parameters (from first row)
    interaction_cols = [col for col in param_df.columns if col not in gene_terms + extra_terms]
    interaction_values = param_df.loc[rows[0], interaction_cols]
    for col in interaction_cols:
        param_dict[col] = interaction_values[col]

    return param_dict


def split_and_merge_simulations(simulation_paths):
    """
    Split clone IDs evenly across multiple simulations and merge selected clones.
    
    This function loads N simulation CSV files, determines the complete set of
    unique clone IDs in the *first* simulation, splits those clone IDs as evenly
    as possible across all N simulations, and merges together the corresponding
    subsets taken from each simulation file.

    Parameters
    ----------
    simulation_paths : list of str
        List of file paths to simulation CSV files.
        - All files must contain a "clone_id" column.
        - Assumes the same clone IDs appear in each simulation, but possibly 
          assigned to different states or having different observations.

    Returns
    -------
    pd.DataFrame
        A concatenated DataFrame containing:
        - The first 1/N of clone IDs from simulation 1,
        - The second 1/N of clone IDs from simulation 2,
        - ...
        - The last 1/N of clone IDs from simulation N.
        The order is determined by sorted clone IDs.

    Notes
    -----
    - Clone IDs are split *evenly* using integer division. 
      If the total number of clones is not exactly divisible by N, 
      the final chunk will contain the remainder.
    - The function returns a simulation file wherein each part of it contains data from a single simulation (not evenly mixed)
    """
    
    # Load all simulations
    sims = [pd.read_csv(path) for path in simulation_paths]
    num_sims = len(sims)

    # Extract clone IDs from first simulation (assumed consistent)
    clone_ids = sorted(sims[0]["clone_id"].unique())
    total_clones = len(clone_ids)
    
    # Compute chunk size
    chunk_size = total_clones // num_sims
    remainder = total_clones % num_sims

    # Determine clone ID chunks for each simulation
    clone_chunks = []
    start = 0
    for i in range(num_sims):
        # distribute the remainder one-by-one to early chunks
        extra = 1 if i < remainder else 0
        end = start + chunk_size + extra
        clone_chunks.append(clone_ids[start:end])
        start = end

    # Merge the subsets
    merged_df = pd.concat(
        [
            sims[i][sims[i]["clone_id"].isin(clone_chunks[i])]
            for i in range(num_sims)
        ],
        ignore_index=True
    )

    return merged_df


def read_input_matrix(path_to_matrix: str) -> tuple[int, np.ndarray]:
    """
    Reads an input matrix from a specified file path and returns its dimensions and content.

    Args:
        path_to_matrix (str): The file path to the matrix file. The file should contain
                              a comma-separated matrix of integers.

    Returns:
        tuple: A tuple containing:
            - int: The number of rows in the matrix.
            - np.ndarray: The matrix as a NumPy array. If the matrix is a single value,
                          it is reshaped into a 1x1 array.

    Raises:
        ValueError: If the file cannot be loaded.
    """
    try:
        matrix = np.loadtxt(path_to_matrix, dtype=int, delimiter=',')
        if matrix.ndim == 0:
            matrix = matrix.reshape((1,1))
        return matrix.shape[0], matrix
    except Exception as e:
        raise ValueError(f"Error loading matrix from {path_to_matrix}: {e}")

def dict_to_matrix(correlation_dict, gene_list):
    """
    Reshapes a "gene1-gene2" -> value dict into a gene x gene DataFrame.

    Parameters
    ----------
    correlation_dict : dict
        Keys formatted as "{gene_1}-{gene_2}" (e.g. from calculate_pairwise_gene_gene_correlation_matrix).
    gene_list : list of str
        Gene names; used as both the row and column index.

    Returns
    -------
    pd.DataFrame
        Gene x gene matrix; entries not present in correlation_dict are NaN.
    """
    matrix = pd.DataFrame(index=gene_list, columns=gene_list, dtype=float)
    for key, value in correlation_dict.items():
        g1, g2 = key.split("-")
        matrix.loc[g1, g2] = value
    return matrix


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm

import re

def sort_genes_numerically(gene_list):
    """
    Sort gene_# numerically if present.
    Otherwise, sort lexicographically.
    Returns (sorted_genes, sorted_indices).
    """
    gene_pattern = re.compile(r"gene_(\d+)$")

    has_numeric = any(gene_pattern.match(g) for g in gene_list)

    def extract_key(g):
        """Sort key for one gene name: numeric suffix if it matches 'gene_<N>', else the name itself."""
        m = gene_pattern.match(g)
        if m:
            return (0, int(m.group(1)))
        if has_numeric:
            return (1, g)      # non-numeric go after numeric genes
        return (0, g)          # pure lexicographic mode

    indexed = list(enumerate(gene_list))
    indexed_sorted = sorted(indexed, key=lambda x: extract_key(x[1]))

    sorted_indices = [i for i, _ in indexed_sorted]
    sorted_genes   = [g for _, g in indexed_sorted]

    return sorted_genes, sorted_indices

def plot_matrix_as_heatmap(corr_matrix, gene_list, no_regulation=None, potential_regulation=None, title=None, add_gene_labels=True,
                            add_time=False, time=None, gray_out_no_reg=False, vmin=None, vmax=None, cmap=None, 
                            return_plot=False, black_out_self=False,figsize_input= (8, 6), symmetric = True, draw_diagonal_multi_state_reg = False, multi_state_reg_edges = None):
    """
    Plot a gene-gene correlation matrix as a heatmap with regulatory overlays and dynamic formatting.
    """

    if add_time:
        if time is None or not isinstance(time, (list, tuple)) or len(time) == 0:
            raise ValueError("If add_time=True, you must provide a non-empty list of 1 or 2 time values in `time`.")
        if len(time) > 2:
            raise ValueError("Time can have at most two entries.")

    # Format gene names: gene_1 → g1
    # ---- Sort genes AND matrix together (single source of truth) ----
    sorted_genes, _ = sort_genes_numerically(gene_list)

    gene_list  = list(sorted_genes)
    base_names = gene_list

    # IMPORTANT: align by *names*, not by iloc positions
    plot_matrix = corr_matrix.reindex(index=gene_list, columns=gene_list)



    # Format axis labels
    if add_gene_labels:
        base_names = [i.replace("_", "-") for i in base_names]
        if add_time:
            if len(time) == 1:
                row_labels = [rf"$\text{{{i}}}_{{t{time[0]}}}$" for i in base_names]
                col_labels = row_labels
            else:
                row_labels = [rf"$\text{{{i}}}_{{t{time[0]}}}$" for i in base_names]
                col_labels = [rf"$\text{{{i}}}_{{t{time[1]}}}$" for i in base_names]
        else:
            row_labels = base_names
            col_labels = base_names
    else:
        row_labels = [""] * len(gene_list)
        col_labels = [""] * len(gene_list)

    # Prepare plot matrix
    # plot_matrix = corr_matrix.copy()
    if symmetric:
        # --- Symmetrize matrix by taking whichever side is non-zero ---
        A = plot_matrix.values
        sym_A = np.where(~np.isnan(A), A, A.T)
        plot_matrix = pd.DataFrame(sym_A, index=plot_matrix.index, columns=plot_matrix.columns)
    # --- Handle masking ---
    mask = np.zeros_like(plot_matrix.values, dtype=bool)
    if gray_out_no_reg and no_regulation:
        for g1, g2 in no_regulation:
            if g1 in gene_list and g2 in gene_list:
                i = gene_list.index(g1)
                j = gene_list.index(g2)
                plot_matrix.iloc[i, j] = 0
                mask[i, j] = True
                if symmetric:
                    plot_matrix.iloc[j, i] = 0
                    mask[j, i] = True   

    # --- Handle vmin/vmax auto-scaling ---
    temp_values = plot_matrix.values.copy()

    # Exclude diagonal values only for vmin/vmax estimation since self_values are being blacked out anyway
    if black_out_self:
        np.fill_diagonal(temp_values, np.nan)

    data_values = temp_values[~np.isnan(temp_values)]

    if len(data_values) == 0:
        vmin, vmax = -1.0, 1.0
    else:
        if vmin is None:
            vmin = np.nanmin(data_values)
        if vmax is None:
            vmax = np.nanmax(data_values)
        if vmin == vmax:
            vmin -= 1e-4
            vmax += 1e-4


    # --- Choose colormap adaptively ---
    if cmap is None and vmin < 0 and vmax > 0:
        cmap = make_reds_blues_colormap(vmin=vmin, vmax=vmax)
        center_span = max(abs(vmin), abs(vmax))
        norm = TwoSlopeNorm(vmin=-center_span, vcenter=0.0, vmax=center_span)
    else:
        norm = None
        if cmap is None:
            cmap = "Blues" if vmin >= 0 else "Reds_r"

    # --- Plot heatmap ---
    fig, ax = plt.subplots(figsize=figsize_input)
    if title:
      cbar_label = title 
    else:
      cbar_label = "Correlation"
    heatmap = sns.heatmap(
        plot_matrix,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        # center = 0,
        # norm=norm,
        xticklabels=col_labels,
        yticklabels=row_labels,
        square=True,
        cbar_kws={'label': cbar_label},
        linewidths=0.5,
        linecolor='lightgray',
        mask=mask
    )
    cbar = ax.collections[0].colorbar
    cbar.set_label(cbar_label, fontsize=10)
    # --- Add regulation boxes ---
    # --- Add regulation boxes (symmetric outlines) ---
    # --- Black out diagonal if requested ---
    if black_out_self:
        for k in range(len(gene_list)):
            rect = Rectangle((k, k), 1, 1, facecolor='#D9D9D9', edgecolor='none')
            ax.add_patch(rect)
    if potential_regulation:
        for g1, g2 in potential_regulation:
            if g1 in gene_list and g2 in gene_list:
                i = gene_list.index(g1)
                j = gene_list.index(g2)
                if symmetric:
                    # Outline (j, i)
                    rect1 = Rectangle((j, i), 1, 1, fill=False, edgecolor='black', linewidth=1)
                    ax.add_patch(rect1)

                    # Outline symmetric (i, j)
                    if i != j:  # avoid drawing twice on diagonal
                        rect2 = Rectangle((i, j), 1, 1, fill=False, edgecolor='black', linewidth=1)
                        ax.add_patch(rect2)
    #Adding diagonal lines for multi-state regulation
    if draw_diagonal_multi_state_reg and len(multi_state_reg_edges) > 0:
        for g1, g2 in multi_state_reg_edges:
            if g1 in gene_list and g2 in gene_list:
                i = gene_list.index(g1)
                j = gene_list.index(g2)

                # Draw a dashed black diagonal inside that cell
                ax.plot(
                    [j+1, j],      # x: left → right of the cell
                    [i+1, i],      # y: top → bottom of the cell
                    linestyle="--",
                    color="black",
                    linewidth=1.5,
                    clip_on=False
                )
                # Draw a dashed black diagonal inside that cell
                ax.plot(
                    [i+1, i],      # x: left → right of the cell
                    [j+1, j],      # y: top → bottom of the cell
                    linestyle="--",
                    color="black",
                    linewidth=1.5,
                    clip_on=False
                )

    # --- Title ---
    if title:
        if add_time:
            if len(time) == 1:
                title += f" @ time {time[0]}h"
            elif len(time) == 2:
                title += f" (rows: t{time[0]}, cols: t{time[1]})"
        ax.set_title(title, fontsize=12)

    plt.tight_layout()
    ax.set_clip_on(False)
    for artist in ax.get_children():
        try:
            artist.set_clip_on(False)
        except Exception:
            pass

    if return_plot:
        return fig, ax
    else:
        plt.show()

def print_summary(no_regulation, 
                  single_state_regulation, 
                  multiple_states_no_reg, 
                  multiple_states_and_reg):
    """
    Prints a structured summary of gene pair classifications.

    Parameters
    ----------
    no_regulation : list of tuple
        Gene pairs with no inferred regulation.

    single_state_regulation : list of tuple
        Gene pairs with single-state regulation.

    multiple_states_no_reg : list of tuple
        Gene pairs with multiple states but no additional regulation evidence.

    multiple_states_and_reg : list of tuple
        Gene pairs with multiple states and additional evidence of regulation.

    Returns
    -------
    None
    """
    def print_section(title, pairs):
        """Prints one titled, de-duplicated (unordered-pair) section of gene pairs, or '(none)'."""
        print(f"\n{'=' * len(title)}\n{title}\n{'=' * len(title)}")
        if not pairs:
            print("  (none)")
            return

        # Use a set to keep track of already-seen symmetric pairs
        seen = set()
        for g1, g2 in pairs:
            key = tuple(sorted((g1, g2)))  # unordered representation
            if key not in seen:
                print(f"  {g1} - {g2}")
                seen.add(key)


    print_section("1. No Regulation", no_regulation)
    print_section("2. Single-State Regulation", single_state_regulation)
    print_section("3. Multiple States (No Regulation)", multiple_states_no_reg)
    print_section("4. Multiple States with Regulation", multiple_states_and_reg)

# --- Helpers ---
def make_reds_blues_colormap(vmin=-0.05, vmax=0.18):
    """Custom red–white–blue colormap with pure white at 0, asymmetric."""
    # Calculate where 0 falls in the range [vmin, vmax]
    zero_position = (0 - vmin) / (vmax - vmin)
    
    # Number of colors for each segment (proportional to range)
    n_total = 256
    n_reds = int(zero_position * n_total)  # colors from vmin to 0
    n_blues = n_total - n_reds  # colors from 0 to vmax
    
    # Calculate intensity based on actual distance from zero
    # For reds: map from vmin to 0, so max intensity at vmin
    red_intensity = abs(vmin) / max(abs(vmin), abs(vmax))  # 0.05/0.18 ≈ 0.28
    # For blues: map from 0 to vmax, so max intensity at vmax  
    blue_intensity = abs(vmax) / max(abs(vmin), abs(vmax))  # 0.18/0.18 = 1.0
    
    # Create color arrays with scaled intensities
    reds = plt.cm.Reds(np.linspace(0.8 * red_intensity, 0, n_reds))  # scaled dark to light red
    whites = np.ones((1, 4))  # pure white at 0
    blues = plt.cm.Blues(np.linspace(0, 0.8 * blue_intensity, n_blues))  # light to scaled dark blue
    
    colors = np.vstack((reds, whites, blues))
    return LinearSegmentedColormap.from_list('RedsBlues', colors)