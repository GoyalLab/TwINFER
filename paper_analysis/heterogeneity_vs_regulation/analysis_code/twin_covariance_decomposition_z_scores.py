"""Twin covariance decomposition z-scores for the six heterogeneity_vs_regulation scenarios.

Figure   : heterogeneity_vs_regulation (A,B vs A->B; single state, 2 states, drift).
Used for : z_rho / z_rho_s / z_rho_Delta box plots with 6,000 twin pairs, and the effect of the number of twin pairs
           (see plot_code/plot_z_scores_6000_twins.py and plot_code/plot_n_twins_effect.py).

Input    : the raw simulations (symlinked in final_figure_data/heterogeneity_vs_regulation/simulation_data/):
             single_and_two_state_simulations/<config>/df_rows_*_ncells_6000_<config>_rep_<n>_<id>.csv   (single state, low/high k_on)
             drift_lowhigh_Kramp/<network>/df_*_ncells_6000_<network>_drift_lowhigh_Kramp_rep_<n>_<id>.csv   (drift)
           Only time_step == 20 h is used (columns time_step, clone_id, replicate, gene_1_mRNA, gene_2_mRNA).
           clone_id = twin pair, replicate (1, 2) = the two sisters.
Output   : analysis_data/twin_covariance_decomposition_z_scores.csv, one row per scenario x replicate x number of twin pairs (N).

Scenarios (rep = 0..99 by default)
  A_B (single state)     figure_2 A_B                      A_to_B (single state)   figure_2 A_to_B
  A_B (two states)       3,000 twins from A_B_low_k_on + 3,000 from A_B_high_k_on (random clones, same replicate number)
  A_to_B (two states)    same with A_to_B_low_k_on / A_to_B_high_k_on
  A_B (drift)            drift_lowhigh_Kramp/A_B           A_to_B (drift)          drift_lowhigh_Kramp/A_to_B
For each scenario and replicate the 6,000 twin pairs are put in one fixed random order and the first N are used,
so the designs with fewer twin pairs are subsets of the larger ones.

Statistics (computed by twinfer.inference.covariance_decomposition_z_scores.twin_stats, where they are defined in detail)
  S            Spearman correlation between the two genes over all cells (both sisters)
  C            cross-sister part: correlation of gene X in sister 1 with gene Y in sister 2 (and vice versa)   -> shared part
  U            S - C                                                                                           -> unshared part
  h_x, h_y     sister correlation of each gene (how similar the two sisters are), clipped at 0 where used for the null
  sigma_rho    null standard deviation of S,  sqrt((0.5 + 0.5 h_x h_y) / N)
  z_rho        S / sigma_rho
  z_rho_s      sign(S) * C / sqrt((1 + S^2 + h_x h_y) / (2N))                 (shared-part z-score, z_rho_s)
  z_rho_Delta  r_delta * sqrt(N), r_delta = correlation of the sister differences of the two genes   (unshared-part z-score)
  A_s, A_u     C / sigma_rho and U / sigma_rho (z_rho = A_s + A_u)
  c            sqrt((1 - h_x)(1 - h_y)) * (1 / sqrt(N)) / sigma_rho  (A_u = c * z_rho_Delta)

Requires : numpy, pandas (the twinfer-code environment). Runtime: about 35 s for 100 replicates with 4 processes.
Usage    : python twin_covariance_decomposition_z_scores.py [--reps 0-99] [--n-twins 250 500 1000 3000 6000] [--jobs 8]
                                                          [--simulation-dir DIR] [--out CSV]
           (defaults: final_figure_data/heterogeneity_vs_regulation/{simulation_data, analysis_data}, N = 250 500 1000 1500 2000 2500 3000 6000)
"""
import argparse
import glob
import multiprocessing as mp
import os
import re

import numpy as np
import pandas as pd

from twinfer.inference.covariance_decomposition_z_scores import sister_arrays, twin_stats      # the statistics (shared by all figures)

N_CLONES = 6000                 # twin pairs per simulation
TIME_STEP = 20                  # hours after division used for the analysis
GENES = ["gene_1_mRNA", "gene_2_mRNA"]
DEFAULT_N_TWINS = [250, 500, 1000, 1500, 2000, 2500, 3000, 6000]

# scenario name -> (kind, source). Index in this list is part of the random seeds (drift scenarios use 7).
SCENARIOS = [
    ("A_B (single state)",   "single", ["A_B"]),
    ("A_to_B (single state)", "single", ["A_to_B"]),
    ("A_B (two states)",     "two_state", ["A_B_low_k_on", "A_B_high_k_on"]),
    ("A_to_B (two states)",  "two_state", ["A_to_B_low_k_on", "A_to_B_high_k_on"]),
    ("A_B (drift)",          "drift", ["A_B"]),
    ("A_to_B (drift)",       "drift", ["A_to_B"]),
]
DRIFT_SEED_INDEX = 7


# ---------------------------------------------------------------- reading the simulations
def read_sisters(path):
    """-> {gene: (N_CLONES, 2) array of mRNA counts at TIME_STEP}, rows ordered by clone_id, columns = sister 1, 2."""
    table = pd.read_csv(path, usecols=["time_step", "gene_1_mRNA", "gene_2_mRNA", "replicate", "clone_id"])
    return sister_arrays(table, GENES, TIME_STEP)[1]


def find_single_state_file(sim_dir, config, rep):
    """Newest file of this replicate (the folder can hold more than one run of the same replicate number)."""
    pattern = re.compile(rf"^df_rows_.*_ncells_{N_CLONES}_{config}_rep_{rep}_[0-9a-f]{{8}}\.csv$")
    folder = os.path.join(sim_dir, "single_and_two_state_simulations", config)
    files = [f for f in os.listdir(folder) if pattern.match(f)]
    if not files:
        raise FileNotFoundError(f"{config} replicate {rep} in {folder}")
    files.sort(key=lambda f: os.path.getmtime(os.path.join(folder, f)))
    return os.path.join(folder, files[-1])


def find_drift_file(sim_dir, network, rep):
    files = glob.glob(os.path.join(sim_dir, "drift_lowhigh_Kramp", network,
                                   f"df_*_ncells_{N_CLONES}_{network}_drift_lowhigh_Kramp_rep_{rep}_*.csv"))
    if len(files) != 1:
        raise FileNotFoundError(f"expected one drift file for {network} replicate {rep}, found {len(files)}")
    return files[0]


def scenario_data(sim_dir, kind, source, rep, seed_index):
    """The 6,000 twin pairs of one scenario and replicate, as {gene: (6000, 2)}."""
    if kind == "single":
        return read_sisters(find_single_state_file(sim_dir, source[0], rep))
    if kind == "drift":
        return read_sisters(find_drift_file(sim_dir, source[0], rep))
    # two states: half the twins from the low-k_on simulation, half from the high-k_on one
    low, high = [read_sisters(find_single_state_file(sim_dir, c, rep)) for c in source]
    rng = np.random.RandomState(rep * 10 + seed_index)
    pick_low = rng.choice(N_CLONES, N_CLONES // 2, replace=False)
    pick_high = rng.choice(N_CLONES, N_CLONES // 2, replace=False)
    return {g: np.vstack([low[g][pick_low], high[g][pick_high]]) for g in GENES}


# ---------------------------------------------------------------- one replicate
def analyse_replicate(args):
    sim_dir, rep, n_twins = args
    rows = []
    for index, (name, kind, source) in enumerate(SCENARIOS):
        seed_index = DRIFT_SEED_INDEX if kind == "drift" else index
        data = scenario_data(sim_dir, kind, source, rep, seed_index)
        order = np.random.RandomState(rep * 100 + seed_index + 12345).permutation(N_CLONES)
        for n in n_twins:
            chosen = order[:n]
            stats = twin_stats(data["gene_1_mRNA"][chosen], data["gene_2_mRNA"][chosen])
            rows.append(dict(scenario=name, rep=rep, N=n, **stats))
    return rows


if __name__ == "__main__":
    project = os.environ.get("TWINFER_PROJECT_ROOT", "/gpfs/projects/b1255/hzhang/TwINFER_KA")
    figure_data = os.path.join(project, "final_figure_data", "heterogeneity_vs_regulation")
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", default="0-99", help="replicate range, e.g. 0-99 or 5")
    ap.add_argument("--n-twins", type=int, nargs="+", default=DEFAULT_N_TWINS)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--simulation-dir", default=os.path.join(figure_data, "simulation_data"))
    ap.add_argument("--out", default=os.path.join(figure_data, "analysis_data", "twin_covariance_decomposition_z_scores.csv"))
    a = ap.parse_args()
    lo, hi = a.reps.split("-") if "-" in a.reps else (a.reps, a.reps)
    jobs = [(a.simulation_dir, rep, a.n_twins) for rep in range(int(lo), int(hi) + 1)]
    with mp.Pool(a.jobs) as pool:
        results = pool.map(analyse_replicate, jobs, chunksize=1)
    table = pd.DataFrame([row for rows in results for row in rows])
    table.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(table)} rows")
