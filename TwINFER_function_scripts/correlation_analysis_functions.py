import numpy as np
import pandas as pd
from scipy.stats import spearmanr, linregress, pearsonr
from .correlation_analysis_helpers import dict_to_matrix
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from itertools import combinations, permutations
import os
from joblib import Parallel, delayed

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

# def get_scrambled_correlations(df, gene_i, gene_j, n=10_000, p_val=0.05, method="spearman", seed=101010):
#     """
#     Returns (null_corrs, abs_threshold) for the unordered pair {gene_i, gene_j}.
#     abs_threshold is the (1 - p_val) quantile of |null_corrs| (two-sided).
#     """
#     # Data prep
#     x = df[f"{gene_i}_mRNA"].to_numpy()
#     y = df[f"{gene_j}_mRNA"].to_numpy()
#     mask = np.isfinite(x) & np.isfinite(y)
#     x, y = x[mask], y[mask]
    
#     # Choose correlation function
#     if method == "spearman":
#         corr_func = lambda a, b: spearmanr(a, b)[0]  # returns (correlation, p-value)
#     elif method == "pearson":
#         corr_func = lambda a, b: pearsonr(a, b)[0]
#     else:
#         raise ValueError("method must be 'spearman' or 'pearson'")

#     rng = np.random.default_rng(seed)
#     n_obs = x.size
#     null_corrs = np.empty(n, dtype=float)
    
#     for k in range(n):
#         perm = rng.permutation(n_obs)
#         null_corrs[k] = corr_func(x, y[perm])

#     # threshold
#     thr = np.quantile(np.abs(null_corrs), 1.0 - p_val)
#     return null_corrs, thr


def get_correlations(correlation_dict, gene_i, gene_j):
   return correlation_dict[tuple(sorted([gene_i, gene_j]))]

# def generate_random_shuffle(simulation_data, gene_list, n_shuffles=10000, random_state=42):
#    np.random.seed(random_state)
   
#    rep_0 = simulation_data[simulation_data['replicate'] == 1].reset_index(drop=True)
#    rep_1 = simulation_data[simulation_data['replicate'] == 2].reset_index(drop=True)
#    gene_cols = [f"{gene}_mRNA" for gene in gene_list]
#    min_cells = min(len(rep_0), len(rep_1))
#    expr_0 = rep_0[gene_cols].iloc[:min_cells].values
#    expr_1 = rep_1[gene_cols].iloc[:min_cells].values

#    n_cells, n_genes = expr_0.shape
#    triu_indices = np.triu_indices(n_genes, k=1)
#    gene_pairs = [(gene_list[i], gene_list[j]) for i, j in zip(triu_indices[0], triu_indices[1])]
   
#    all_shuffle_indices = np.array([np.random.permutation(n_cells) for _ in range(n_shuffles)])
#    all_correlations = np.zeros((n_shuffles, len(gene_pairs)))
   
#    for batch_start in range(0, n_shuffles, 100):
#        batch_end = min(batch_start + 100, n_shuffles)
#        for i, shuffle_idx in enumerate(all_shuffle_indices[batch_start:batch_end]):
#            expr_1_shuffled = expr_1[shuffle_idx]
#            deltas = expr_0 - expr_1_shuffled
#            ranked_deltas = np.apply_along_axis(rankdata, 0, deltas)
#            corr_matrix = np.corrcoef(ranked_deltas.T)
#            all_correlations[batch_start + i] = corr_matrix[triu_indices]
   
#    # Store with sorted keys to avoid duplicates
#    correlation_dict = {}
#    for i, (gene_i, gene_j) in enumerate(gene_pairs):
#        key = tuple(sorted([gene_i, gene_j]))
#        correlation_dict[key] = all_correlations[:, i]
   
#    return correlation_dict

import numpy as np
import pandas as pd
from scipy.stats import rankdata

def generate_random_shuffle(simulation_data, gene_list, n_shuffles=10000, random_state=42):
    """
    Generate null distributions of difference correlations by random pairing of cells,
    independent of any 'replicate' column.

    Parameters
    ----------
    simulation_data : pd.DataFrame
        Simulation dataframe containing at least 'clone_id' and '{gene}_mRNA' columns.

    gene_list : list of str
        Gene base names (without '_mRNA' suffix).

    n_shuffles : int, default=10000
        Number of random shuffles to perform.

    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    correlation_dict : dict
        Mapping {(gene_i, gene_j): np.ndarray of shuffled correlations}.
    """
    rng = np.random.default_rng(random_state)

    gene_cols = [f"{gene}_mRNA" for gene in gene_list]
    expr = simulation_data[gene_cols].dropna().to_numpy()
    n_cells, n_genes = expr.shape

    triu_indices = np.triu_indices(n_genes, k=1)
    gene_pairs = [(gene_list[i], gene_list[j]) for i, j in zip(*triu_indices)]

    all_correlations = np.zeros((n_shuffles, len(gene_pairs)), dtype=np.float64)

    # --- Random pairing: pick two distinct sets of cells for difference computation
    for s in range(n_shuffles):
        idx_1 = rng.choice(n_cells, size=n_cells, replace=False)
        idx_2 = rng.choice(n_cells, size=n_cells, replace=False)
        # Optionally avoid same-cell pairings
        mask = idx_1 != idx_2
        expr_1 = expr[idx_1[mask]]
        expr_2 = expr[idx_2[mask]]

        n_used = min(len(expr_1), len(expr_2))
        if n_used < 3:
            continue

        deltas = expr_1[:n_used] - expr_2[:n_used]
        ranked_deltas = np.apply_along_axis(rankdata, 0, deltas)
        corr_matrix = np.corrcoef(ranked_deltas.T)
        all_correlations[s] = corr_matrix[triu_indices]

    # --- Package as dict for downstream comparison
    correlation_dict = {
        tuple(sorted((gi, gj))): all_correlations[:, k]
        for k, (gi, gj) in enumerate(gene_pairs)
    }

    return correlation_dict


# def check_gene_gene_correlation_threshold(all_t1_t2_measurements,
#                                           pairwise_gene_gene_correlation_matrix, 
#                                           gene_list, 
#                                           threshold=0.04,
#                                           use_scramble = True,
#                                           p_val_threshold = 0.01,
#                                           verbose=False):
#     """
#     Splits gene-gene pairs based on absolute correlation threshold.

#     Parameters
#     ----------
#     all_t1_t2_measurements : pd.DataFrame
#         The cell-gene dataframe containing sample information.
#     pairwise_gene_gene_correlation_matrix : pd.DataFrame
#         A square matrix of gene-gene correlations (gene × gene).
#     gene_list : list of str
#         List of gene names, must match the matrix row/column order.
#     threshold : float, optional
#         Correlation magnitude threshold to separate regulated vs unregulated.
#     use_scramble : bool, optional
#         If True, uses scrambled correlations for comparison.
#     p_val_threshold : float, optional
#         P-value threshold for significance testing.
#     verbose : bool, optional
#         If True, plots the null distribution and threshold for each gene pair.
    
#     Returns
#     -------
#     no_regulation : list of tuple
#         Gene pairs (i, j) where abs(correlation) ≤ threshold.

#     potential_regulation : list of tuple
#         Gene pairs (i, j) where abs(correlation) > threshold.
    
#     Note:
#     1. If both use_scrambled is True and threshold is provided, a new threshold is calculated 
#        based on the scrambled distribution and p_val_threshold.
#     """
#     no_regulation = []
#     potential_regulation = []
#     for i in range(len(gene_list)):
#         for j in range(len(gene_list)):
#             if i >= j:
#                 continue  # Skip diagonal
#             corr_val = pairwise_gene_gene_correlation_matrix.values[i, j]
#             pair = (gene_list[i], gene_list[j])
#             rev_pair = (gene_list[j], gene_list[i])
#             if use_scramble:
#                 # gene_i = 
#                 # gene_j = = 
#                 null_dist, threshold = get_scrambled_correlations(gene_i, gene_j, n=10_000, p_val=p_val_threshold)
#                 p_value = np.mean(np.abs(null_dist) >= np.abs(corr_val))
#                 if verbose:
#                     plt.figure(figsize=(6, 4))
#                     plt.hist(null_dist, bins=50, color="skyblue", alpha=0.7, edgecolor="k")
#                     plt.axvline(threshold, color="red", linestyle="--", label=f"+thr={threshold:.2e}")
#                     plt.axvline(-threshold, color="red", linestyle="--", label=f"-thr={threshold:.2e}")
#                     plt.axvline(corr_val, color="black", linestyle="-", label=f"actual={corr_val:.2e}")
#                     plt.title(f"Scrambled correlations: {gene_list[i]} vs {gene_list[j]}, p-val = {p_value:.2e}")
#                     plt.xlabel("Correlation")
#                     plt.ylabel("Count")
#                     plt.legend()
#                     plt.tight_layout()
#                     plt.show()
#             if abs(corr_val) > threshold:
#                     potential_regulation.append(pair)
#                     potential_regulation.append(rev_pair)
#             else:
#                 no_regulation.append(pair)
#                 no_regulation.append(rev_pair)
#     return no_regulation, potential_regulation, threshold

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
       raw_matrix[i, j] = corr if not np.isnan(corr) else 0.0
   
   return pd.DataFrame(raw_matrix, index=gene_list, columns=gene_list)

def single_cell_shuffle(gene_matrix_1, gene_matrix_2, gene_list, shuffle_pairs):
            n_cells = gene_matrix_1.shape[1]
            shuffled_indices = np.random.permutation(n_cells)
            return compute_correlation_matrix(gene_matrix_1, gene_matrix_2[:, shuffled_indices], 
                                                  gene_list, shuffle_pairs)

def check_gene_gene_correlation_threshold(all_t1_t2_measurements,
                                          pairwise_gene_gene_correlation_matrix, 
                                          gene_list, 
                                          threshold=0.04,
                                          use_scramble=True,
                                          p_val_threshold=0.01,
                                          n_shuffles=10000,
                                          verbose=False,
                                          return_gene_corr_thresholds = True,
                                          n_cores_to_use = 4):
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
        # Generate null distribution
        
        
        shuffled_results = Parallel(n_jobs=n_cores_to_use, verbose=0)(
            delayed(single_cell_shuffle)(gene_matrix, gene_matrix, gene_list, all_pairs) 
            for _ in range(n_shuffles)
        )
        
        percentile_threshold = (1 - p_val_threshold) * 100

    no_regulation, potential_regulation = [], []
    gene_corr_thresholds = {}
    for gi, gj in all_pairs:
        corr_val = pair_correlations[(gi, gj)]
        current_threshold = threshold
        
        if use_scramble:
            shuffled_vals = [r.loc[gi, gj]for r in shuffled_results]
            
            current_threshold = np.percentile(shuffled_vals, percentile_threshold)
            
            if verbose:
                p_value = np.mean(np.array(shuffled_vals) >= abs(corr_val))
                direction_str = "-"
                plt.figure(figsize=(6, 4))
                plt.hist(shuffled_vals, bins=50, color="skyblue", alpha=0.7, edgecolor="k")
                plt.axvline(current_threshold, color="red", linestyle="--", label=f"threshold={current_threshold:.3f}")
                plt.axvline(-1*current_threshold, color="red", linestyle="--")
                plt.axvline(corr_val, color="black", linestyle="-", label=f"actual={(corr_val):.3f}")
                plt.title(f"Gene correlation: {gi} {direction_str} {gj}, p-val = {p_value:.3f}")
                plt.xlabel(r"gene correlation $\rho$")
                plt.ylabel("number of scrambles")
                plt.legend()
                plt.tight_layout()
                plt.show()
        
        # Classify pairs
        gene_corr_thresholds[(gi, gj)] = current_threshold
        if abs(corr_val) > current_threshold:
            potential_regulation.append((gi, gj))
        else:
            no_regulation.append((gi, gj))
        
    
    return no_regulation, potential_regulation, gene_corr_thresholds

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

# def calculate_twin_random_pair_correlations(simulation_two_time, simulation_single_time, gene_list):
#     """
#     Computes twin and random pairwise correlation matrices at a given time point.

#     Parameters
#     ----------
#     simulation_single_time : pd.DataFrame
#         Subset of simulation data at a single time point. Must contain:
#         - 'replicate' column (1 for twin A, 2 for twin B)
#         - 'clone_id'
#         - '{gene}_mRNA' for each gene in gene_list

#     gene_list : list of str
#         List of gene names (without "_mRNA" suffix) to analyze.

#     Returns
#     -------
#     twin_correlation_matrix : pd.DataFrame
#         Matrix of gene-gene Spearman correlations computed between true twin pairs.

#     random_correlation_matrix : pd.DataFrame
#         Matrix of gene-gene Spearman correlations between randomly paired clones.
#     """
#     # Separate replicate 1 and replicate 2
#     rep_0 = simulation_single_time[simulation_single_time['replicate'] == 1]
#     rep_1 = simulation_single_time[simulation_single_time['replicate'] == 2]

#     # Calculate correlations using matched twin pairs
#     twin_correlation_dict = calculate_pair_correlation(rep_0, rep_1, gene_list, type_comparison="twin")
#     twin_correlation_matrix = dict_to_matrix(twin_correlation_dict, gene_list)

#     # Calculate correlations using randomly shuffled pairs
#     random_rep_0 = simulation_two_time[simulation_two_time['replicate'] == 1]
#     random_rep_1 = simulation_two_time[simulation_two_time['replicate'] == 2]
#     random_rep_1_shuffled = random_rep_1.sample(frac=1, random_state=10100).reset_index(drop=True)
#     random_correlation_dict = calculate_pair_correlation(random_rep_0, random_rep_1_shuffled, gene_list, type_comparison="random")
#     random_correlation_matrix = dict_to_matrix(random_correlation_dict, gene_list)
#     return twin_correlation_matrix, random_correlation_matrix

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
            else:
                single_state_regulation.append((gene_i, gene_j))
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

def identify_actual_directed_edges(rep_0_t1, rep_1_t2, direction_raw_matrix, gene_pairs, threshold=0.01, n_shuffles=10000, n_cores_to_use = 4, verbose = False):
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
    
    # Safe parallel processing
    
    # Get shuffled correlations using existing single_cell_shuffle function
    shuffled_results = Parallel(n_jobs=n_cores_to_use, verbose=0)(
        delayed(single_cell_shuffle)(gene_matrix_t1, gene_matrix_t2, gene_list, gene_pairs)
        for _ in range(n_shuffles)
    )
    
    # Identify significant directed edges
    significant_edges = []
    percentile_threshold = (1 - threshold) * 100
    
    for gene_1, gene_2 in gene_pairs:
        # Get actual correlation
        actual_corr = direction_raw_matrix.loc[gene_1, gene_2]
        
        # Get shuffled correlations for this pair
        shuffled_vals = []
        for result in shuffled_results:
            shuffled_matrix = result
            shuffled_vals.append(abs(shuffled_matrix.loc[gene_1, gene_2]))
        
        # Calculate threshold for this pair
        pair_threshold = np.percentile(shuffled_vals, percentile_threshold)
        if verbose:
            print(f"{gene_1} -> {gene_2}: threshold = {pair_threshold}, actual = {actual_corr}")
        # Check if actual correlation crosses threshold
        if abs(actual_corr) > pair_threshold:
            significant_edges.append((gene_1, gene_2))    
    return significant_edges
