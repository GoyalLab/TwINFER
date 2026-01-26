#This is the script to analyze the output of parameter scan simulations enmasse and output the 
# pairwise gene correlations, p-values associated with it 
# random-pair difference correlations, twin difference correlations and the z-score comparing them
# twin cross-correlations and the p-value compared to random.

import os, gc, warnings, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations, product
from joblib import Parallel, delayed
from scipy.stats import spearmanr, rankdata
from tqdm_joblib import tqdm_joblib
from numba import njit, prange, set_num_threads
from tqdm import tqdm
warnings.filterwarnings("ignore")

# -----------------------------
# Config knobs
# -----------------------------
N_JOBS = 1
BATCH_SIZE = 200
SAVE_INTERVAL = 200
SHUFFLES_GENE_GENE = 10000
SHUFFLES_RANDOM_DIFF = 10000
SHUFFLES_DIRECTED = 10000
set_num_threads(5)

# =============================
# Utility functions
# =============================

def find_csv_files_fast(folder_path):
    return [f.name for f in Path(folder_path).glob("df*.csv")]

def split_and_merge_simulations(path_to_simulation_files):
    simulation_1 = pd.read_csv(path_to_simulation_files[0])
    simulation_2 = pd.read_csv(path_to_simulation_files[1])
    clone_ids = sorted(simulation_1['clone_id'].unique())
    half_point = len(clone_ids) // 2
    clones_from_sim1 = clone_ids[:half_point]
    clones_from_sim2 = clone_ids[half_point:]
    sim1_subset = simulation_1[simulation_1['clone_id'].isin(clones_from_sim1)]
    sim2_subset = simulation_2[simulation_2['clone_id'].isin(clones_from_sim2)]
    return pd.concat([sim1_subset, sim2_subset], ignore_index=True)

def extract_param_index(filename):
    try:
        core = filename.split("df_row_")[1]
        parts = core.split("_")
        for part in parts:
            if part.isdigit() and len(part) == 8:
                return "_".join(parts[:parts.index(part)])
        return "unknown"
    except Exception:
        return "unknown"

def spearman_safe(x, y):
    if len(x) < 3 or len(y) < 3:
        return np.nan
    r = spearmanr(x, y).correlation
    return r if not np.isnan(r) else np.nan

# =============================
# Core correlation helpers
# =============================
def calculate_pairwise_gene_gene_correlation_matrix(df, gene_list):
    mat = pd.DataFrame(np.nan, index=gene_list, columns=gene_list)
    X = df[gene_list].values.T
    for i, gi in enumerate(gene_list):
        for j in range(i, len(gene_list)):
            gj = gene_list[j]
            r = spearman_safe(X[i], X[j])
            mat.loc[gi, gj] = mat.loc[gj, gi] = r
    return mat

def _spearman_matrix_from_ranked(R):
    """
    Compute Spearman correlation matrix from a rank-transformed matrix.

    Parameters
    ----------
    R : np.ndarray, shape (n_cells, n_genes)
        Rank matrix (midranks per column).

    Returns
    -------
    S : np.ndarray, shape (n_genes, n_genes)
        Spearman correlation matrix.
    """
    n = R.shape[0]
    m = (n + 1) / 2.0
    Rc = R - m
    s2 = np.sum(Rc**2, axis=0)
    denom = np.sqrt(np.outer(s2, s2))
    denom[denom == 0] = np.nan
    N = Rc.T @ Rc
    return N / denom


# =============================
# Subsampling for a time-pair (exactly your recipe)
# =============================
def subsample_for_timepair(simulation, t1, t2, rng):
    clone_ids = simulation["clone_id"].dropna().unique()
    rng.shuffle(clone_ids)
    if len(clone_ids) < 4:
        # not enough to split; return empties
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    n1 = n2 = len(clone_ids) // 4
    t1_clones = clone_ids[:n1]
    t2_clones = clone_ids[n1 : n1 + n2]
    across_t_clones = clone_ids[n1 + n2 :]

    t1_twins = simulation[
        (simulation["clone_id"].isin(t1_clones)) & (simulation["time_step"] == t1)
    ].reset_index(drop=True)
    t2_twins = simulation[
        (simulation["clone_id"].isin(t2_clones)) & (simulation["time_step"] == t2)
    ].reset_index(drop=True)

    across_t_twin1 = simulation[
        (simulation["clone_id"].isin(across_t_clones))
        & (simulation["time_step"] == t1)
        & (simulation["replicate"] == 1)
    ].reset_index(drop=True)

    across_t_twin2 = simulation[
        (simulation["clone_id"].isin(across_t_clones))
        & (simulation["time_step"] == t2)
        & (simulation["replicate"] == 2)
    ].reset_index(drop=True)

    all_t1_t2 = pd.concat(
        [t1_twins, t2_twins, across_t_twin1, across_t_twin2],
        ignore_index=True,
    )
    return t1_twins, t2_twins, across_t_twin1, across_t_twin2, all_t1_t2

# =============================
# STEP 1 — Gene correlation (pooled over (t1,t2)) + Null distribution for gene correlation
# =============================
def calculate_pairwise_gene_gene_correlation_matrix(df, gene_list):
    """Undirected full matrix over pooled cells from t1+t2."""
    mat = pd.DataFrame(np.nan, index=gene_list, columns=gene_list)
    X = df[gene_list].values.T  # genes x cells
    for i, gi in enumerate(gene_list):
        for j in range(i, len(gene_list)):
            gj = gene_list[j]
            r = spearman_safe(X[i], X[j])
            mat.loc[gi, gj] = mat.loc[gj, gi] = r
    return mat


@njit(parallel=True)
def _compute_gene_gene_null(Rc, denom, seeds, triu_i, triu_j):
    n_shuffles = len(seeds)
    n, p = Rc.shape
    n_pairs = len(triu_i)
    out = np.empty((n_shuffles, n_pairs), dtype=np.float64)

    for k in prange(n_shuffles):
        np.random.seed(seeds[k])
        idx = np.random.permutation(Rc.shape[0])
        Rc_perm = Rc[idx, :]
        N = Rc.T @ Rc_perm
        C = N / denom
        for pos in range(n_pairs):
            i, j = triu_i[pos], triu_j[pos]
            out[k, pos] = C[i, j]
    return out

def compute_gene_gene_null_distributions(all_t1_t2, gene_list, n_shuffles, n_jobs=None):
    """
    Null gene correlation distribution: Spearman null via shuffle of cells, optimized with Numba.
    Returns: dict[(gi, gj)] = np.ndarray(n_shuffles,)
    """
    X = all_t1_t2[gene_list].to_numpy()
    n, p = X.shape

    # Step 1: Pre-rank and center
    R = np.apply_along_axis(rankdata, 0, X).astype(np.float64)
    m = (n + 1) / 2.0
    Rc = R - m

    # Step 2: Precompute denominator matrix
    s2 = np.sum(Rc**2, axis=0)
    denom = np.sqrt(np.outer(s2, s2))
    denom[denom == 0] = np.nan

    # Step 3: Setup shuffle parameters
    triu_i, triu_j = np.triu_indices(p, k=0)
    seeds = np.random.randint(0, 2**31 - 1, size=n_shuffles)

    # Step 4: Compute nulls via Numba
    all_ut = _compute_gene_gene_null(Rc, denom, seeds, triu_i, triu_j)

    # Step 5: Format results
    null = {}
    for pos, (i, j) in enumerate(zip(triu_i, triu_j)):
        gi, gj = gene_list[i], gene_list[j]
        null[(min(gi, gj), max(gi, gj))] = all_ut[:, pos]
    return null


# =============================
# STEP 2 — Twin difference correlations at time t, per gene-pair + Distribution of random-pair difference correlations
# =============================

@njit
def rankdata_numba(a):
    n = a.size
    temp = np.argsort(a)
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        start = i
        val = a[temp[i]]
        while i + 1 < n and a[temp[i + 1]] == val:
            i += 1
        end = i
        avg_rank = 0.5 * (start + end) + 1
        for j in range(start, end + 1):
            ranks[temp[j]] = avg_rank
        i += 1
    return ranks


@njit
def spearman_matrix_from_diff(D):
    n, p = D.shape
    R = np.empty_like(D)
    for j in range(p):
        R[:, j] = rankdata_numba(D[:, j])

    m = (n + 1) / 2.0
    Rc = R - m
    s2 = np.sum(Rc**2, axis=0)
    denom = np.sqrt(np.outer(s2, s2))

    # Avoid boolean indexing: replace zeros manually
    n_rows, n_cols = denom.shape
    for i in range(n_rows):
        for j in range(n_cols):
            if denom[i, j] == 0.0:
                denom[i, j] = np.nan

    N = Rc.T @ Rc
    return N / denom

def twin_pair_correlation_matrix(df_twins, gene_list):
    """
    Spearman correlations between replicate differences at one timepoint.
    Uses scipy.stats.spearmanr instead of manual ranking.
    """
    mat = pd.DataFrame(np.nan, index=gene_list, columns=gene_list)
    if df_twins.empty:
        return mat

    rep_0 = df_twins[df_twins["replicate"] == 1]
    rep_1 = df_twins[df_twins["replicate"] == 2]

    # --- Must have matching replicates ---
    if rep_0.empty or rep_1.empty:
        raise ValueError("Both replicate 1 and replicate 2 must exist in the data.")

    # --- ALIGN THE REPLICATES BY clone_id (fix for Problem 2) ---
    # keep only clone_ids present in both replicates
    common_clones = np.intersect1d(rep_0["clone_id"].unique(), rep_1["clone_id"].unique())

    if len(common_clones) < 3:
        # not enough aligned pairs → return empty distribution
        return {
            (min(gi, gj), max(gi, gj)): np.array([])
            for gi in gene_list for gj in gene_list
        }

    # sort replicates by clone_id and filter to common clones
    rep_0 = rep_0[rep_0["clone_id"].isin(common_clones)].sort_values("clone_id").reset_index(drop=True)
    rep_1 = rep_1[rep_1["clone_id"].isin(common_clones)].sort_values("clone_id").reset_index(drop=True)

    n = min(len(rep_0), len(rep_1))
    if n < 3:
        return mat

    X1 = rep_0[gene_list].to_numpy()[:n]
    X2 = rep_1[gene_list].to_numpy()[:n]
    D = X1 - X2

    S, _ = spearmanr(D, axis=0)
    return pd.DataFrame(S, index=gene_list, columns=gene_list)

@njit(parallel=True)
def _generate_random_shuffle_fast(X1, X2, triu_i, triu_j, seeds):
    n_shuffles = len(seeds)
    n, p = X1.shape
    out = np.empty((n_shuffles, len(triu_i)), dtype=np.float64)

    for s in prange(n_shuffles):
        np.random.seed(seeds[s])
        idx = np.random.permutation(n)
        D = X1 - X2[idx, :]
        S = spearman_matrix_from_diff(D)
        for k in range(len(triu_i)):
            i, j = triu_i[k], triu_j[k]
            out[s, k] = S[i, j]
    return out

def generate_random_shuffle(simulation_data, gene_list, n_shuffles=10000, random_state=42):
    rng = np.random.default_rng(random_state)
    rep_0 = simulation_data[simulation_data["replicate"] == 1]
    rep_1 = simulation_data[simulation_data["replicate"] == 2]

    # --- Must have matching replicates ---
    if rep_0.empty or rep_1.empty:
        raise ValueError("Both replicate 1 and replicate 2 must exist in the data.")

    # --- ALIGN THE REPLICATES BY clone_id (fix for Problem 2) ---
    # keep only clone_ids present in both replicates
    common_clones = np.intersect1d(rep_0["clone_id"].unique(), rep_1["clone_id"].unique())

    if len(common_clones) < 3:
        # not enough aligned pairs → return empty distribution
        return {
            (min(gi, gj), max(gi, gj)): np.array([])
            for gi in gene_list for gj in gene_list
        }

    # sort replicates by clone_id and filter to common clones
    rep_0 = rep_0[rep_0["clone_id"].isin(common_clones)].sort_values("clone_id").reset_index(drop=True)
    rep_1 = rep_1[rep_1["clone_id"].isin(common_clones)].sort_values("clone_id").reset_index(drop=True)

    min_cells = min(len(rep_0), len(rep_1))
    p = len(gene_list)
    if min_cells < 3:
        return {
            (min(gi, gj), max(gi, gj)): np.array([])
            for i, gi in enumerate(gene_list)
            for j, gj in enumerate(gene_list)
            if j >= i
        }

    X1 = rep_0[gene_list].to_numpy()[:min_cells]
    X2 = rep_1[gene_list].to_numpy()[:min_cells]
    triu_i, triu_j = np.triu_indices(p, k=1)
    seeds = rng.integers(0, 2**31 - 1, size=n_shuffles)

    all_ut = _generate_random_shuffle_fast(X1, X2, triu_i, triu_j, seeds)
    correlation_dict = {}
    for pos, (i, j) in enumerate(zip(triu_i, triu_j)):
        gi, gj = gene_list[i], gene_list[j]
        correlation_dict[(min(gi, gj), max(gi, gj))] = all_ut[:, pos]

    # Add diagonals with empty arrays
    for g in gene_list:
        key = (g, g)
        if key not in correlation_dict:
            correlation_dict[key] = np.array([])
    return correlation_dict


# =============================
# STEP 3 — Twin cross-correlations + null distribution of random-pair cross-correlations
# =============================

@njit(parallel=True)
def _permutation_counts_cross_time(RXc, RYc, denom, seeds):
    n, p = RXc.shape
    counts = np.zeros((p, p), dtype=np.int32)
    n_shuffles = len(seeds)

    # Observed correlation
    N_obs = RXc.T @ RYc
    S_obs = N_obs / denom

    for k in prange(n_shuffles):
        np.random.seed(seeds[k])
        idx = np.random.permutation(n)
        RYc_perm = RYc[idx, :]
        N_perm = RXc.T @ RYc_perm
        S_perm = N_perm / denom
        for i in range(p):
            for j in range(p):
                if np.isnan(S_obs[i, j]) or np.isnan(S_perm[i, j]):
                    continue
                if abs(S_perm[i, j]) >= abs(S_obs[i, j]):
                    counts[i, j] += 1
    return S_obs, counts

def directed_cross_time_with_pvals(across_twin1, across_twin2, t1, t2, gene_list, n_shuffles):
    """
    Optimized: Compute cross-time Spearman correlations and p-values.
    Returns dict {(src_gene_time, tgt_gene_time): (corr, pval)}
    """
    out = {}
    if across_twin1.empty or across_twin2.empty:
        return out

    X = across_twin1[gene_list].to_numpy()
    Y = across_twin2[gene_list].to_numpy()
    n = min(len(X), len(Y))
    if n < 3:
        return out

    X = X[:n]
    Y = Y[:n]
    p = len(gene_list)

    # Step 1: Precompute ranks and center
    RX = np.apply_along_axis(rankdata, 0, X).astype(np.float64)
    RY = np.apply_along_axis(rankdata, 0, Y).astype(np.float64)
    m = (n + 1) / 2.0
    RXc = RX - m
    RYc = RY - m
    s2x = np.sum(RXc**2, axis=0)
    s2y = np.sum(RYc**2, axis=0)
    denom = np.sqrt(np.outer(s2x, s2y))
    denom[denom == 0] = np.nan

    # Step 2: Generate random seeds
    seeds = np.random.randint(0, 2**31 - 1, size=n_shuffles)

    # Step 3: Run fast permutation test
    S_obs, counts = _permutation_counts_cross_time(RXc, RYc, denom, seeds)

    # Step 4: Package results (excluding same-gene diagonals)
    for i, gi in enumerate(gene_list):
        for j, gj in enumerate(gene_list):
            if gi == gj:
                continue
            key = (f"{gi}_t{t1}", f"{gj}_t{t2}")
            corr = S_obs[i, j]
            pval = counts[i, j] / float(n_shuffles)
            out[key] = (corr, pval)
    return out

# =============================
# Simulation processor
# =============================

def process_simulation(
    sim_info,
    time_points,
    gene_list,
    n_shuffles_gene_gene=SHUFFLES_GENE_GENE,
    n_shuffles_random_diff=SHUFFLES_RANDOM_DIFF,
    n_shuffles_directed=SHUFFLES_DIRECTED,
    seed=2024,
    mode="single",
    remove_twin_structure = False
):
    """
    Process one simulation or a merged pair, depending on mode.
    mode = 'single' | 'pair'
    """
    rng = np.random.default_rng(seed)

    # -----------------------
    # Load simulation data
    # -----------------------
    #Analyzing single-state simulations
    if mode == "single":
        sim, folder = sim_info
        path = os.path.join(folder, sim)
        if not os.path.exists(path):
            print(f"[warn] missing {path}")
            return None
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[error] reading {path}: {e}")
            return None
        param_index = extract_param_index(sim)

    #Merging two simulations to create multi-state transcriptomic data
    elif mode == "pair":
        sims, folder = sim_info  # sims = (file1, file2)
        path_1 = os.path.join(folder, sims[0])
        path_2 = os.path.join(folder, sims[1])
        if not os.path.exists(path_1) or not os.path.exists(path_2):
            print(f"[warn] missing one of: {path_1}, {path_2}")
            return None
        try:
            df = split_and_merge_simulations([path_1, path_2])
        except Exception as e:
            print(f"[error] merging {path_1} and {path_2}: {e}")
            return None
        param_index = f"{extract_param_index(sims[0])}__{extract_param_index(sims[1])}"

    else:
        raise ValueError(f"Unknown mode: {mode}")
    unique_clones = df['clone_id'].unique()
    #############################
    # REMOVE TWIN STRUCTURE
    #############################
    if remove_twin_structure:
        # create a random permutation of clone IDs
        # --- Break twin structure but preserve within-cell continuity ---
        rng = np.random.default_rng(12345)

        unique_clones = np.array(df["clone_id"].unique())

        # --- Generate a derangement (no clone keeps its original ID) ---
        shuffled = unique_clones.copy()
        while np.any(shuffled == unique_clones):
            rng.shuffle(shuffled)

        shuffle_map = dict(zip(unique_clones, shuffled))

        # --- Apply mapping ONLY to replicate 2 ---
        mask_rep2 = df["replicate"] == 2
        df.loc[mask_rep2, "clone_id"] = df.loc[mask_rep2, "clone_id"].map(shuffle_map)
        df = df.sort_values(["replicate", "clone_id"]).reset_index(drop=True)


    row = {}
    row['param_index'] = param_index
    # -----------------------
    # Build time pairs (t1 < t2)
    # -----------------------
    time_pairs = [
        (time_points[i], time_points[j])
        for i in range(len(time_points))
        for j in range(i + 1, len(time_points))
    ]

    # -----------------------
    # Process each time pair
    # -----------------------
    for (t1, t2) in time_pairs:
        # Subsample
        t1_twins, t2_twins, twin1, twin2, all_t1_t2 = subsample_for_timepair(df, t1, t2, rng)

        # =====================================================
        # STEP 1: Pairwise gene correlation (pooled over t1+t2) + Null distribution of gene correlation
        # =====================================================
        if all_t1_t2.empty:
            # Fill NaNs if no data
            for i, gi in enumerate(gene_list):
                for j in range(i, len(gene_list)):
                    gj = gene_list[j]
                    row[f"corr_gene_gene_{gi}_{gj}_t{t1}_t{t2}"] = np.nan
                    row[f"pval_gene_gene_{gi}_{gj}_t{t1}_t{t2}"] = np.nan
        else:
            gg_mat = calculate_pairwise_gene_gene_correlation_matrix(all_t1_t2, gene_list)
            gg_null = compute_gene_gene_null_distributions(
                all_t1_t2,
                gene_list,
                n_shuffles=n_shuffles_gene_gene,
                n_jobs=min(4, os.cpu_count() - 2)
            )

            for i, gi in enumerate(gene_list):
                for j in range(i, len(gene_list)):
                    gj = gene_list[j]
                    obs = gg_mat.loc[gi, gj]
                    key = (min(gi, gj), max(gi, gj))
                    null_vals = gg_null.get(key, np.array([]))
                    pval = np.nan if null_vals.size == 0 else np.mean(np.abs(null_vals) >= abs(obs))
                    row[f"corr_gene_gene_{gi}_{gj}_t{t1}_t{t2}"] = obs
                    row[f"pval_gene_gene_{gi}_{gj}_t{t1}_t{t2}"] = pval

        # =====================================================
        # STEP 2: Twin difference correlations at t1, t2 vs Null distribution of random-pair difference correlation
        # =====================================================
        rd_null = generate_random_shuffle(
            all_t1_t2,
            gene_list,
            n_shuffles=n_shuffles_random_diff,
            random_state=seed
        )

        # --- Twin correlations at t1 ---
        t1_twin_mat = twin_pair_correlation_matrix(t1_twins, gene_list)
        for gi in gene_list:
            for gj in gene_list:
                if gi != gj:
                    obs = t1_twin_mat.loc[gi, gj]
                    key = (min(gi, gj), max(gi, gj))
                    null_vals = rd_null.get(key, np.array([]))

                    if null_vals.size == 0 or np.isnan(obs):
                        pval, zscore = np.nan, np.nan
                    else:
                        pval = np.mean(np.abs(null_vals) >= abs(obs))
                        null_mean = np.mean(null_vals)
                        null_std = np.std(null_vals, ddof=1)
                        zscore = (obs - null_mean) / null_std if null_std > 0 else np.nan

                    row[f"twin_corr_{gi}_{gj}_t{t1}"] = obs
                    row[f"pval_twin_vs_random_{gi}_{gj}_t{t1}"] = pval
                    row[f"zscore_twin_vs_random_{gi}_{gj}_t{t1}"] = zscore

        # --- Twin correlations at t2 ---
        t2_twin_mat = twin_pair_correlation_matrix(t2_twins, gene_list)
        for gi in gene_list:
            for gj in gene_list:
                if gi != gj:
                    obs = t2_twin_mat.loc[gi, gj]
                    key = (min(gi, gj), max(gi, gj))
                    null_vals = rd_null.get(key, np.array([]))

                    if null_vals.size == 0 or np.isnan(obs):
                        pval, zscore = np.nan, np.nan
                    else:
                        pval = np.mean(np.abs(null_vals) >= abs(obs))
                        null_mean = np.mean(null_vals)
                        null_std = np.std(null_vals, ddof=1)
                        zscore = (obs - null_mean) / null_std if null_std > 0 else np.nan

                    row[f"twin_corr_{gi}_{gj}_t{t2}"] = obs
                    row[f"pval_twin_vs_random_{gi}_{gj}_t{t2}"] = pval
                    row[f"zscore_twin_vs_random_{gi}_{gj}_t{t2}"] = zscore

        # =====================================================
        # STEP 3 — Twin cross-correlations + null distribution of random-pair cross-correlations
        # =====================================================
        dc = directed_cross_time_with_pvals(
            twin1, twin2, t1, t2, gene_list,
            n_shuffles=n_shuffles_directed
        )

        for (src, tgt), (corr, pval) in dc.items():
            row[f"directed_corr_{src}__{tgt}"] = corr
            row[f"directed_pval_{src}__{tgt}"] = pval

        # --- Self correlations ---
        for g in gene_list:
            x, y = twin1[g].values, twin2[g].values
            n = min(len(x), len(y))
            row[f"self_corr_{g}_t{t1}_t{t2}"] = np.nan if n < 3 else spearman_safe(x[:n], y[:n])

    del df
    gc.collect()
    return row

# =============================
# Batch runner with Brunner–Munzel logic
# =============================

def run_pipeline(path_to_simulations, output_folder, genes, time_points,
                 n_jobs=N_JOBS,
                 n_shuffles_gene_gene=SHUFFLES_GENE_GENE,
                 n_shuffles_random_diff=SHUFFLES_RANDOM_DIFF,
                 n_shuffles_directed=SHUFFLES_DIRECTED,
                 batch_size=BATCH_SIZE,
                 save_interval=SAVE_INTERVAL,
                 seed=2024,
                 start_index=0,
                 mode="single",
                 csv_path=None,
                 remove_twin_structure=0):

    from scipy.stats import brunnermunzel

    files = find_csv_files_fast(path_to_simulations)
    if len(files) == 0:
        raise ValueError("No simulation CSV files found!")

    rng = np.random.default_rng(seed)
    os.makedirs(output_folder, exist_ok=True)
    gene_list = list(genes)
    
    # === PAIR MODE: Two-state selection ===
    if mode == "pair":
        if csv_path:
            pairs_df = pd.read_csv(csv_path)
            work_items = list(zip(pairs_df["file1"], pairs_df["file2"]))
        else:
            n_target = 25000
            p_threshold = 0.01
            pairs = set()
            attempts = 0
            max_attempts = 200000
            cache = {}
            save_every = 1000
            temp_save_path = os.path.join(output_folder, "two_state_pairs_temp.csv")
            final_save_path = os.path.join(output_folder, "two_state_pairs_final.csv")

            print(f"Selecting up to {n_target} unique two-state pairs (Brunner–Munzel p<{p_threshold})...")

            def read_proteins_cached(fname, t=1):
                """Lazy cache reader: load only once and store minimal columns."""
                if fname in cache:
                    return cache[fname]
                path = os.path.join(path_to_simulations, fname)
                try:
                    usecols = ["time_step", "gene_1_protein", "gene_2_protein"]
                    df = pd.read_csv(path, usecols=usecols)
                    subset = df[df["time_step"] == t][["gene_1_protein", "gene_2_protein"]].dropna()
                    cache[fname] = subset
                    return subset
                except Exception:
                    cache[fname] = pd.DataFrame(columns=["gene_1_protein", "gene_2_protein"])
                    return cache[fname]

            while len(pairs) < n_target and attempts < max_attempts:
                a, b = rng.choice(files, 2, replace=True)
                attempts += 1
                if a == b:
                    continue

                key = tuple(sorted((a, b)))
                if key in pairs:
                    continue

                df_a = read_proteins_cached(a, t=1)
                df_b = read_proteins_cached(b, t=1)
                if df_a.empty or df_b.empty:
                    continue

                try:
                    bm1 = brunnermunzel(df_a["gene_1_protein"], df_b["gene_1_protein"], nan_policy="omit")
                    bm2 = brunnermunzel(df_a["gene_2_protein"], df_b["gene_2_protein"], nan_policy="omit")

                    if (bm1.pvalue < p_threshold) and (bm2.pvalue < p_threshold):
                        pairs.add(key)
                        if len(pairs) % save_every == 0:
                            print(f"  → {len(pairs)} two-state pairs accepted after {attempts} attempts")
                            pd.DataFrame(sorted(list(pairs)), columns=["file1", "file2"]).to_csv(
                                temp_save_path, index=False
                            )
                except Exception:
                    continue

            pairs = sorted(list(pairs))
            pd.DataFrame(pairs, columns=["file1", "file2"]).to_csv(final_save_path, index=False)
            print(f"✅ Formed {len(pairs)} unique two-state pairs after {attempts} attempts.")
            print(f"✅ Saved final pairs to {final_save_path}")
            work_items = pairs

    else:
        work_items = [(f,) for f in files]
        print(f"Found {len(files)} single simulations.")

    # === Batch processing ===

    all_rows, chunk_id = [], 0
    for i in range(0, len(work_items), batch_size):
        batch = work_items[i:min(i + batch_size, len(work_items))]
        print(f"[batch] {i}..{i + len(batch) - 1}")

        with tqdm_joblib(desc="Processing simulations", total=len(batch)):

            def safe_process(item, seed_offset):
                try:
                    return process_simulation(
                        (item if mode == "pair" else item[0], path_to_simulations),
                        time_points=time_points,
                        gene_list=gene_list,
                        n_shuffles_gene_gene=n_shuffles_gene_gene,
                        n_shuffles_random_diff=n_shuffles_random_diff,
                        n_shuffles_directed=n_shuffles_directed,
                        seed=seed + i + seed_offset,
                        mode=mode
                    )
                except Exception as e:
                    return {"file_or_pair": item, "error": str(e)}

            res = Parallel(n_jobs=n_jobs)(
                delayed(safe_process)(item, k) for k, item in enumerate(batch)
            )

        res = [r for r in res if r is not None]
        all_rows.extend(res)

        if len(all_rows) >= save_interval:
            pd.DataFrame(all_rows).to_csv(
                os.path.join(output_folder, f"results_chunk_{chunk_id:03d}.csv"),
                index=False
            )
            print(f"[saved] chunk {chunk_id} with {len(all_rows)} rows")
            all_rows.clear()
            chunk_id += 1

    if all_rows:
        pd.DataFrame(all_rows).to_csv(
            os.path.join(output_folder, f"results_chunk_{chunk_id:03d}.csv"),
            index=False
        )
        print(f"[saved] final chunk {chunk_id} with {len(all_rows)} rows")

#%% # ============================= # CLI # =============================
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Correlation pipeline with separate nulls for gene-gene and twin-random.") 
    parser.add_argument("--path_to_simulations", type=str, required=True)
    parser.add_argument("--output", type=str, required=True) 
    parser.add_argument("--genes", nargs="+", required=True, help="Exact column names for gene expressions (e.g., gene_1_mRNA gene_2_mRNA ...)") 
    parser.add_argument("--timepoints", nargs="+", type=int, required=True, help="List of time points (e.g., 1 5 10 20). All (t1<t2) pairs will be used.") 
    parser.add_argument("--shuffles_gene_gene", type=int, default=SHUFFLES_GENE_GENE) 
    parser.add_argument("--shuffles_random_diff", type=int, default=SHUFFLES_RANDOM_DIFF) 
    parser.add_argument("--shuffles_directed", type=int, default=SHUFFLES_DIRECTED) 
    parser.add_argument("--jobs", type=int, default=N_JOBS) 
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE) 
    parser.add_argument("--save_interval", type=int, default=SAVE_INTERVAL) 
    parser.add_argument("--seed", type=int, default=2024) 
    parser.add_argument("--start_index", type=int, default=0) 
    parser.add_argument("--mode", type=str, choices=["single", "pair"], default="single", help="Whether to process single simulations or random pairs (25k pairs).") 
    parser.add_argument("--csv", type=str, default=None, help="Path to csv file containing the pairwise combinations of parameters to combine to form two state simulations.") 
    parser.add_argument("--remove_twin_structure", type=int, default=0, help="If true, random cells will be paired and considered as twins, thereby losing all twin information.") 
    args = parser.parse_args()
    path_to_simulations=args.path_to_simulations 
    output_folder=args.output 
    genes=args.genes 
    time_points =args.timepoints 
    n_jobs=args.jobs 
    n_shuffles_gene_gene=args.shuffles_gene_gene 
    n_shuffles_random_diff=args.shuffles_random_diff 
    n_shuffles_directed=args.shuffles_directed 
    batch_size=args.batch_size 
    save_interval=args.save_interval 
    start_index=args.start_index 
    seed=args.seed 
    mode=args.mode 
    remove_twin_structure = bool(args.remove_twin_structure)
    csv_path=args.csv
    run_pipeline( 
        path_to_simulations=path_to_simulations, 
        output_folder=output_folder, 
        genes=genes,
        time_points=time_points, 
        n_jobs=n_jobs, 
        n_shuffles_gene_gene=n_shuffles_gene_gene, 
        n_shuffles_random_diff=n_shuffles_random_diff, 
        n_shuffles_directed=n_shuffles_directed, 
        batch_size=batch_size, 
        save_interval=save_interval, 
        seed=seed, 
        start_index = start_index,
        mode=mode,
        csv_path=csv_path,
        remove_twin_structure=remove_twin_structure)