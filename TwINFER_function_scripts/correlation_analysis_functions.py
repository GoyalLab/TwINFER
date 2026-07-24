import numpy as np
import pandas as pd
from scipy.stats import spearmanr, linregress, pearsonr
from .correlation_analysis_helpers import dict_to_matrix
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from itertools import combinations, permutations
import os
from joblib import Parallel, delayed
from scipy import stats
import numba
import numpy as np
import pandas as pd

def steady_state_calc(param_dict, interaction_matrix, gene_list,
                                   sim_data, scale_k=None):
    """
    Calculates regulated steady-state protein levels using empirical Hill responses
    from simulation data. Assigns k values based on computed steady states.

    Args:
        param_dict (dict): Contains kinetic and interaction parameters.
        interaction_matrix (np.ndarray): Shape (n_genes, n_genes) — regulator → target.
        gene_list (list): Ordered list of gene names.
        sim_data (pd.DataFrame): Must contain 'gene_{i}_protein' for each gene.
        scale_k (np.ndarray): Optional scaling matrix for assigning k.

    Returns:
        protein_levels_sim_data (np.ndarray): Steady state protein levels estimated using simulation data.
    """
    def hill_fn(x, n, k):
        x = np.asarray(x)
        return x ** n / (x ** n + k ** n)

    n_genes = len(gene_list)
    if scale_k is None:
        scale_k = np.ones((n_genes, n_genes))

    protein_levels_sim_data = np.zeros(n_genes)

    for i, gene in enumerate(gene_list):
        p_on = param_dict[f'k_on_{gene}']
        p_off = param_dict[f'k_off_{gene}']
        p_prod_mRNA = param_dict[f'k_prod_mRNA_{gene}']
        p_deg_mRNA = param_dict[f'k_deg_mRNA_{gene}']
        p_prod_prot = param_dict[f'k_prod_protein_{gene}']
        p_deg_prot = param_dict[f'k_deg_protein_{gene}']

        reg_eff = 0.0
        regulators = np.where(interaction_matrix[:, i] != 0)[0]
        for r in regulators:
            src_gene = gene_list[r]
            edge = f"{src_gene}_to_{gene}"
            p_add = param_dict.get(f"k_add_{edge}", 0.0)
            n_val = param_dict.get(f"n_{edge}", 1.0)
            k_val = param_dict.get(f"k_{edge}", 1.0)
            sign = interaction_matrix[r, i]
            key = f"gene_{r+1}_protein"
            if key not in sim_data:
                raise ValueError(f"{key} not found in sim_data")

            x_vals = np.asarray(sim_data[key])
            hill_vals = hill_fn(x_vals, n_val, k_val)
            hill_response = np.mean(hill_vals)
            reg_eff += p_add * hill_response * sign

        p_on_eff = p_on + reg_eff
        burst_prob = p_on_eff / (p_on_eff + p_off)
        m = p_prod_mRNA * burst_prob / p_deg_mRNA
        protein = max(m * p_prod_prot / p_deg_prot, 0.1)
        protein_levels_sim_data[i] = protein

    return protein_levels_sim_data

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
            mean_val[i].append(sim_data_t[f'gene_{i + 1}_protein'].mean())

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
        "Relative Slope": relative_slopes,
        "Steady State?": steady_state_flags
    })

    return all(steady_state_flags), summary_df

def calculate_pairwise_gene_gene_correlation_matrix(simulation_at_t1, gene_list):
    correlations = {}
    for gene_1 in gene_list:
        for gene_2 in gene_list:
            gene_gene_corr = spearmanr(simulation_at_t1[f"{gene_1}_mRNA"], simulation_at_t1[f"{gene_2}_mRNA"]).correlation
            correlations[f"{gene_1}-{gene_2}"] = gene_gene_corr
    correlation_matrix = dict_to_matrix(correlations, gene_list)
    return correlation_matrix


def get_correlations(correlation_dict, gene_i, gene_j):
   return correlation_dict[tuple(sorted([gene_i, gene_j]))]

def generate_random_shuffle(simulation_data, gene_list, n_shuffles=10000, random_state=42):
    """
    Random-pair difference-correlation null distribution sized to N (half the cell pool,
    matching the number of true twin pairs), not the ~2N pairs generate_random_shuffle draws
    via two independent permutations of the full pool. Every shuffle draws a single
    permutation of the whole pool, splits it into two equal halves, and pairs them
    positionally -- so each shuffle yields floor(n_cells/2) random-pair deltas, matching the
    N true twin-pair deltas that twin_correlation_matrix (the statistic being compared
    against in differentiate_single_state_reg_and_multiple_states) was computed from. No
    assumption is made about which column/labels distinguish the two twins -- only that
    'clone_id' identifies which rows are each other's twin, so a position that would pair a
    cell with its own twin can be dropped (see _half_split_diff_null_kernel) instead of
    leaking real twin correlation into the "random" null.

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

    n_genes = expr.shape[1]
    triu_i, triu_j = np.triu_indices(n_genes, k=1)
    gene_pairs = [(gene_list[i], gene_list[j]) for i, j in zip(triu_i, triu_j)]

    seeds = _spawn_independent_seeds(random_state, n_shuffles)
    all_correlations = _half_split_diff_null_kernel(
        expr, clone_codes, seeds, triu_i.astype(np.int64), triu_j.astype(np.int64)
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
            if abs(idx_1[k] - idx_2[k]) > 1:
                keep[n_used] = k
                n_used += 1

        if n_used < 3:
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
def _half_split_diff_null_kernel(expr, clone_codes, seeds, triu_i, triu_j):
    """
    Random-pair difference-correlation null sized to N (half the cell pool), not the ~2N
    pairs _two_permutation_diff_null_kernel draws via two independent permutations of the
    full pool. Each shuffle draws a SINGLE permutation and splits it in half, pairing the
    first half against the second half positionally -- every cell is used exactly once, so
    there's no near-self-match to filter, and no assumption about which two labels
    distinguish the twins (unlike splitting by a 'replicate' column). Positions where the
    split happens to pair a cell with its own twin (same clone_codes value) are dropped
    instead, since that would leak the real twin correlation into the "random" null.
    """
    n_cells, n_genes = expr.shape
    half = n_cells // 2
    n_shuffles = seeds.shape[0]
    n_pairs = triu_i.shape[0]
    out = np.full((n_shuffles, n_pairs), np.nan, dtype=np.float64)

    for s in numba.prange(n_shuffles):
        np.random.seed(seeds[s])
        perm = np.random.permutation(n_cells)
        idx_a = perm[:half]
        idx_b = perm[half:2 * half]

        n_used = 0
        keep = np.empty(half, dtype=np.int64)
        for k in range(half):
            if clone_codes[idx_a[k]] != clone_codes[idx_b[k]]:
                keep[n_used] = k
                n_used += 1

        if n_used < 3:
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


def plot_qq_distribution(shuffled_full, obs_value, gene_pair_name):
    shuffled_full = np.asarray(shuffled_full)
    shuffled = shuffled_full[np.isfinite(shuffled_full)]
        
    # with 10k shuffles, anything under 9000 usable values is suspicious
    if len(shuffled) < 0.01*len(shuffled_full):
        return False
        
    # zero variance → degenerate distribution → not normal
    if np.std(shuffled) == 0:
        return False
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Histogram with fitted normal distribution
    ax1 = axes[0]
    n, bins, patches = ax1.hist(shuffled, bins=50, density=True, alpha=0.7, 
                                 edgecolor='black', label='Shuffled data')


    # Fit normal distribution
    mu, sigma = stats.norm.fit(shuffled)
    x = np.linspace(shuffled.min(), shuffled.max(), 100)
    fitted_normal = stats.norm.pdf(x, mu, sigma)
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

    (osm, osr), (slope, intercept, r) = stats.probplot(
        shuffled, dist="norm"
    )

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

def check_gene_gene_correlation_threshold(all_t1_t2_measurements,
                                          pairwise_gene_gene_correlation_matrix, 
                                          gene_list, 
                                          threshold=0.04,
                                          use_scramble=True,
                                          p_val_threshold=0.01,
                                          n_shuffles=10000,
                                          verbose=False,
                                          return_gene_corr_thresholds = True,
                                          n_cores_to_use = 4,
                                          base_seed = 101010):
    """
    Splits gene-gene pairs based on absolute correlation threshold.
    
    Returns: no_regulation, potential_regulation
    """
    
    # Extract gene matrices from DataFrame
    gene_matrix = []
    
    for gene in gene_list:
        # Look for gene columns (adapt this to your column naming)
        gene_col = f"{gene}_mRNA"  # Adjust this pattern as needed
        if gene_col in all_t1_t2_measurements.columns:
            # Split by time point or condition - adjust this logic for your data structure
            gene_data = all_t1_t2_measurements[gene_col].values
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

        percentile_threshold = (1 - p_val_threshold) * 100

    no_regulation, potential_regulation = [], []
    p_value_calc = {}
    threshold_p = {}
    is_significant = False
    corr_threshold = threshold
    is_relatively_normal = True
    for pos, (gi, gj) in enumerate(all_pairs):
        corr_val = pair_correlations[(gi, gj)]

        if use_scramble:
            shuffled_vals = null_matrix[:, pos]

            # Calculate p-value directly (no threshold needed)
            p_plus = np.mean(shuffled_vals >= corr_val)
            p_minus = np.mean(shuffled_vals <= corr_val)
            p_value = min(2 * p_plus, 2 * p_minus, 1.0)
            is_significant = p_value < p_val_threshold
            corr_threshold = np.nanpercentile(np.abs(shuffled_vals), 100 * (1 - p_val_threshold / 2))
            print(f"For gene {gi}, gene {gj}, observed correlation: {corr_val:.4f} with p-value: {p_value:.4f}")
            if verbose:
                try:
                    direction_str = "-"
                    plt.figure(figsize=(6, 4))
                    plt.hist(shuffled_vals, bins=50, color="skyblue", alpha=0.7, edgecolor="k")
                    plt.axvline(corr_val, color="black", linestyle="-", label=f"actual={(corr_val):.3f}")
                    plt.title(f"Gene correlation: {gi} {direction_str} {gj}, p-val = {p_value:.3f}")
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
                is_relatively_normal = plot_qq_distribution(shuffled_vals, corr_val, gene_pair_name)
                print(f"For gene {gi}, gene {gj}, null distribution is normal: {is_relatively_normal}")

        # Classify pairs
        threshold_p[(gi, gj)] = corr_threshold
        if use_scramble:
            p_value_calc[(gi, gj)] = p_value
        else:
            p_value_calc[(gi, gj)] = None  # No p-value calculated without scramble
        if is_significant and is_relatively_normal:
            potential_regulation.append((gi, gj))
        else:
            no_regulation.append((gi, gj))
        
    
    return no_regulation, potential_regulation, threshold_p, p_value_calc

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

def calculate_twin_random_pair_correlations(simulation_two_time, simulation_single_time, gene_list, n_random=None, seed=10100):
    """
    Computes twin and random pairwise gene-gene correlation matrices.

    Parameters
    ----------
    simulation_two_time : pd.DataFrame
        Full dataset at the given time point(s), used for random pairing.
        Must contain 'clone_id' and '{gene}_mRNA' for each gene in gene_list.

    simulation_single_time : pd.DataFrame
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
    twins = (
        simulation_single_time.groupby("clone_id")
        .filter(lambda g: len(g) == 2)  # only valid twin clones
        .groupby("clone_id")
    )

    twin_pairs = []
    for cid, group in twins:
        if len(group) == 2:
            twin_pairs.append(group)
    if not twin_pairs:
        raise ValueError("No valid twin pairs (clone_id with exactly 2 cells) found!")

    rep_0 = pd.concat([g.iloc[[0]] for g in twin_pairs], ignore_index=True)
    rep_1 = pd.concat([g.iloc[[1]] for g in twin_pairs], ignore_index=True)

    twin_corr_dict = calculate_pair_correlation(rep_0, rep_1, gene_list, type_comparison="twin")
    twin_corr_matrix = dict_to_matrix(twin_corr_dict, gene_list)

    # --- Random pairs: random cells from different clones ---
    all_cells = simulation_two_time.reset_index(drop=True)
    n_cells = len(all_cells)
    n_pairs = n_random or len(rep_0)

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

def differentiate_single_state_reg_and_multiple_states(all_t1_t2_measurements, potential_regulation, twin_correlation_matrix, random_correlation_matrix, gene_list, z_score_threshold=10, verbose = True):
    """
    Separates potential regulatory gene pairs into multiple-state vs single-state regulation.

    Parameters
    ----------
    all_t1_t2_measurements : pd.DataFrame
        The cell-gene dataframe containing sample information.
    potential_regulation : list of tuple
        List of gene pairs (gene_i, gene_j) with potential regulation.
    twin_correlation_matrix : pd.DataFrame
        Twin pair correlation matrix at time t2.
    random_correlation_matrix : pd.DataFrame
        Random pair correlation matrix at time t2.
    gene_list : list of str
        List of gene names (e.g., 'gene_1') in correct matrix order.
    z_score_threshold : float, optional
        Threshold for abs(random / twin) above which a pair is considered multi-state.
    Returns
    -------
    multiple_states_gene_pairs : list of tuple
        Gene pairs with abs(random / twin) >= threshold_ratio.

    single_state_regulation : list of tuple
        Gene pairs with z-score between random pair correlations and twin pair correlation greater than 10.
    """
    multiple_states_gene_pairs = []
    single_state_regulation = []

    random_pair_correlation_distribution = generate_random_shuffle(all_t1_t2_measurements, gene_list=gene_list)
    for gene_i, gene_j in potential_regulation:
        try:
            t_corr = twin_correlation_matrix.loc[gene_i, gene_j]
            r_corr = get_correlations(random_pair_correlation_distribution, gene_i, gene_j)
            r_corr_std = np.std(r_corr)
            if r_corr_std == 0:
                # All random correlations are identical (very rare)
                print(f"Warning: Zero variance in random correlations for {gene_i}-{gene_j}")
                single_state_regulation.append((gene_i, gene_j))
                continue
            z_score = (t_corr - np.mean(r_corr))/r_corr_std
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
        except ZeroDivisionError:
            # Handle case where twin correlation is 0
            raise ValueError(f"Division by zero for {gene_i} and {gene_j}")
        except KeyError:
            raise ValueError(f"Missing gene pair ({gene_i}, {gene_j}) in correlation matrices.")
    return multiple_states_gene_pairs, single_state_regulation

def identify_reg_if_multiple_states(twin_correlation_matrix_t1, twin_correlation_matrix_t2, random_correlation_matrix_t1, random_correlation_matrix_t2, multiple_states_gene_pairs, gene_list, threshold_relative_increase=0.1):
    """
    Among multiple-state gene pairs, identify which also show regulation
    (based on increased twin correlation from t1 to t2).

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

    threshold_relative_increase : float, optional
        Minimum relative increase in twin correlation from t1 to t2 to call it regulation.

    Returns
    -------
    multiple_states_no_reg : list of tuple
        Gene pairs with multiple states but no significant increase in correlation (no regulation).

    multiple_states_and_reg : list of tuple
        Gene pairs with multiple states and increased correlation (suggesting regulation).
    """
    multiple_states_no_reg = []
    multiple_states_and_reg = []

    for gene_i, gene_j in multiple_states_gene_pairs:
        try:
            corr_t1 = twin_correlation_matrix_t1.loc[gene_i, gene_j]
            corr_t2 = twin_correlation_matrix_t2.loc[gene_i, gene_j]
            print(f"Testing for multiple states. Correlation at time t1 = {corr_t1} and at time t2 = {corr_t2}")
            if corr_t1 == 0:
                relative_change = np.inf if corr_t2 != 0 else 0
            elif corr_t2 < 0:
                relative_change = abs(corr_t2 - corr_t1) / abs(corr_t1)
            else:
                relative_change = (corr_t2 - corr_t1) / abs(corr_t1)
            
            if relative_change > threshold_relative_increase:
                multiple_states_and_reg.append((gene_i, gene_j))
            else:
                multiple_states_no_reg.append((gene_i, gene_j))
        except KeyError:
            raise ValueError(f"Missing gene pair ({gene_i}, {gene_j}) in correlation matrices.")

    return multiple_states_no_reg, multiple_states_and_reg

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
                mismatched_ids = rep_1_t2[~rep_0_t1["clone_id"].isin(rep_0_t1["clone_id"])]
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

def identify_actual_directed_edges(rep_0_t1, rep_1_t2, direction_raw_matrix, gene_pairs, threshold=0.01, n_shuffles=10000, n_cores_to_use = 4, verbose = False,
                                          base_seed = 101010, return_p_values = False):
    """
    Identify directed edges that cross significance thresholds using shuffled null distribution.
    
    Parameters
    ----------
    gene_matrix_1, gene_matrix_2 : np.ndarray
        Gene expression matrices (genes × cells)
    direction_raw_matrix : pd.DataFrame
        Actual correlation matrix between genes
    gene_pairs : list of tuples
        Gene pairs to analyze
    threshold : float
        P-value threshold (default 0.01)
    n_shuffles : int
        Number of shuffle iterations
        
    Returns
    -------
    list of tuples
        Gene pairs that have significant directed correlations
    """
    
    # Extract gene list from matrix
    gene_matrix_t1 = []
    gene_matrix_t2 = []

    gene_list = list(direction_raw_matrix.index)
    print(gene_list)
    
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
    p_value_calc = {}
    percentile_threshold = (1 - threshold) * 100
    print(f"number of gene pairs = {len(gene_pairs)}")
    for pos, (gene_1, gene_2) in enumerate(gene_pairs):
        # Get actual correlation
        actual_corr = direction_raw_matrix.loc[gene_1, gene_2]

        # Get shuffled correlations for this pair
        shuffled_vals = null_matrix[:, pos]
        # Calculate threshold for this pair
        p_plus = np.mean(shuffled_vals >= actual_corr)
        p_minus = np.mean(shuffled_vals <= actual_corr)
        p_value = min(2 * p_plus, 2 * p_minus, 1.0)
        is_significant = p_value < threshold
        p_value_calc[(gene_1, gene_2)] = p_value
        print(f"Observed correlation for {gene_1} -> {gene_2}: {actual_corr:.4f}, p-value: {p_value:.4f}")
        print(f"Significant at α={threshold}: {is_significant}")
        gene_pair_name = f"{gene_1} -> {gene_2}"
        if is_significant:
            is_relatively_normal = plot_qq_distribution(shuffled_vals, actual_corr, gene_pair_name)
            print(f"{gene_pair_name}: normality of null: {is_relatively_normal}")
        if verbose:
            try:
                print(f"{gene_pair_name}: actual = {actual_corr}, p-value = {p_value}")
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
        # Check if actual correlation crosses threshold
        if is_significant and is_relatively_normal:
            significant_edges.append((gene_1, gene_2))
    if return_p_values:
        return significant_edges, p_value_calc
    return significant_edges


def separate_fan_outs_from_mutual_regulation(all_t1_measurements, twin_correlation_matrix_t1, gene_list,
                                              final_directed_edges, directed_p_values, direction_matrix,
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
    directed_p_values : dict
        Mapping {(gene_src, gene_tgt): p_value} for directed edges.
    direction_matrix : pd.DataFrame
        Directional correlation matrix, thresholded to final_directed_edges.
    z_score_threshold : float, default=8
        |z| above this value on the undirected (A,B) pair marks it a fan-out.

    Returns
    -------
    final_directed_edges : set of tuple
        Updated in place: fan-out (A,B)/(B,A) edges removed.
    directed_p_values : dict
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
            if abs(z_score) > z_score_threshold:
                print(f"Inference: fan-out detected -- {regulators_str} regulate(s) both {gene_a} and {gene_b} "
                      f"(|z|={abs(z_score):.2f} > {z_score_threshold}); removing direct edge between {gene_a} and {gene_b}.")
                final_directed_edges.discard((gene_a, gene_b))
                final_directed_edges.discard((gene_b, gene_a))
                directed_p_values.pop((gene_a, gene_b), None)
                directed_p_values.pop((gene_b, gene_a), None)
                if gene_a in direction_matrix.index and gene_b in direction_matrix.columns:
                    direction_matrix.loc[gene_a, gene_b] = 0
                    direction_matrix.loc[gene_b, gene_a] = 0
                decision = "fan_out"
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
            })

    return final_directed_edges, directed_p_values, direction_matrix, fan_out_log
