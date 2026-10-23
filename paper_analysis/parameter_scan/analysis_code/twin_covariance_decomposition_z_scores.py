"""Twin covariance decomposition z-scores for the parameter scan (single-state simulations and two-state systems).

Figure   : parameter_scan (used by heterogeneity_vs_regulation and causal_direction_inference).
Used for : the ROC panels (analysis_code/roc_of_z_scores.py -> plot_code/plot_roc_panels.py) and the AUPRC plots (auprc_of_z_scores.py -> plot_auprc.py).

Input    : minimal simulation files, one per parameter set (final_figure_data/parameter_scan/simulation_data):
             parameter_scan_complete/<topology>/df_row_*.csv.gz   single-state simulations, one file per parameter set
             parameter_scan_minimal/<topology>/<file>             files named in the two-state pair lists
           input_folder/two_state_pairs_<topology>.csv           columns file1, file2: pairs of simulations (25,000 per topology)
                                                                  whose protein levels differ (Brunner-Munzel p < 0.01 for both genes)
           Topologies: A_B (no regulation), A_to_B (gene_1 -> gene_2), A_and_B (gene_1 <-> gene_2), A_rep_B (gene_1 -| gene_2).
           Each file has 6,000 twin pairs (clone_id; replicate 1, 2 = the two sisters) at time steps 1 and 20;
           only time_step == 20 and the columns gene_1_mRNA, gene_2_mRNA are used.
Systems  : single_state  the 6,000 twin pairs of one simulation
           two_state     3,000 twin pairs (the lower half of the sorted clone ids) from file1 + 3,000 twin pairs (the upper half) from file2
Output   : analysis_data/twin_covariance_decomposition_z_scores.csv, one row per system with columns
             topology, system, set_id (single_state) or file1/file2 (two_state), N (twin pairs, 6,000),
             z_rho, z_rho_s, z_rho_delta, h_x, h_y, S, C, U, sigma_rho, A_s, A_u, c, r_delta
           The statistics are computed by twinfer.inference.covariance_decomposition_z_scores (defined there; N = number of twin pairs).
           A system in which a gene has no variance gets empty statistics.
Random pairs (control): with --break-twins the twin structure is destroyed before the statistics are computed: the second sister of each
           clone is replaced by the second sister of another clone (a random permutation of the clones without fixed points, seed 12345, the same
           permutation for both genes). The output defaults to analysis_data/twin_covariance_decomposition_z_scores_random_pairs.csv.
Usage    : python twin_covariance_decomposition_z_scores.py [--topologies A_B A_to_B ...] [--jobs 8] [--limit 50] [--break-twins]
                  [--simulation-dir DIR] [--pairs-dir DIR] [--out CSV]
           --limit n only processes the first n single-state files and n pairs of each topology (for testing).
Requires : numpy, pandas (the twinfer-code environment).
Runtime  : about 0.15 s per file; all 4 topologies (200,000 systems) take about 8 CPU hours, i.e. minutes on a cluster node with many processes.
"""
import argparse
import multiprocessing as mp
import os
import re

import numpy as np
import pandas as pd

from twinfer.inference.covariance_decomposition_z_scores import sister_arrays, twin_stats_or_nan      # the statistics (shared by all figures)

TOPOLOGIES = ["A_B", "A_to_B", "A_and_B", "A_rep_B"]
TIME_STEP = 20
BREAK_TWINS = False                 # set from --break-twins in main(); inherited by the worker processes
SEED_BREAK = 12345
GENES = ["gene_1_mRNA", "gene_2_mRNA"]


# ---------------------------------------------------------------- reading the simulations
def read_sisters(path):
    """-> (clone ids, {gene: (n_twins, 2) array}) at TIME_STEP, rows ordered by clone_id, columns = sister 1, 2."""
    table = pd.read_csv(path, usecols=["clone_id", "replicate", "time_step"] + GENES)
    return sister_arrays(table, GENES, TIME_STEP)


def derangement(n, seed):
    """Random permutation of range(n) without fixed points."""
    rng = np.random.RandomState(seed)
    permutation = rng.permutation(n)
    while (permutation == np.arange(n)).any():
        permutation = rng.permutation(n)
    return permutation


def stats_of(data):
    """z-scores of the two genes of one system ({gene: (N, 2) sister arrays}); with BREAK_TWINS the twin pairing is destroyed first."""
    x, y = data[GENES[0]], data[GENES[1]]
    if BREAK_TWINS:                                             # random pairs: sister 2 comes from another clone (same for both genes)
        permutation = derangement(len(x), SEED_BREAK)
        x, y = np.column_stack([x[:, 0], x[permutation, 1]]), np.column_stack([y[:, 0], y[permutation, 1]])
    return twin_stats_or_nan(x, y)


def single_state(job):
    """Row of statistics for one single-state simulation file; job = (topology, path of the .csv.gz file)."""
    topology, path = job
    name = os.path.basename(path)
    row = dict(topology=topology, system="single_state", set_id=int(re.match(r"df_row_(\d+)_", name).group(1)) // 3, file1=name, file2="")
    clones, data = read_sisters(path)
    return {**row, "N": len(clones), **stats_of(data)}


def two_state(job):
    """Row of statistics for a two-state system made of two single-state files; job = (topology, path of file1, path of file2)."""
    topology, path1, path2 = job
    name1, name2 = os.path.basename(path1), os.path.basename(path2)
    clones1, data1 = read_sisters(path1)
    clones2, data2 = read_sisters(path2)
    half = len(clones1) // 2                                    # lower half of the clone ids from file1, upper half from file2
    merged = {g: np.vstack([data1[g][:half], data2[g][half:]]) for g in GENES}
    row = dict(topology=topology, system="two_state", set_id=-1, file1=name1, file2=name2)
    return {**row, "N": len(merged[GENES[0]]), **stats_of(merged)}


def safe(function, job):
    """Run function(job); a file that cannot be read (for example an empty file) gives an error row instead of stopping the whole run."""
    try:
        return function(job)
    except Exception as error:                                  # keep going; report at the end
        return {"topology": job[0], "system": function.__name__, "file1": str(job[1:]), "error": str(error)}


def run(job):
    """Worker entry point of the process pool: job = (function, its argument)."""
    function, payload = job
    return safe(function, payload)


if __name__ == "__main__":
    project = os.environ.get("TWINFER_PROJECT_ROOT", "/gpfs/projects/b1255/hzhang/TwINFER_KA")
    figure_data = os.path.join(project, "final_figure_data", "parameter_scan")
    ap = argparse.ArgumentParser()
    ap.add_argument("--topologies", nargs="+", default=TOPOLOGIES, choices=TOPOLOGIES)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="only the first n single-state files and n pairs per topology (testing)")
    ap.add_argument("--break-twins", action="store_true", help="random-pair control: destroy the twin structure (see header)")
    ap.add_argument("--simulation-dir", default=os.path.join(figure_data, "simulation_data"))
    ap.add_argument("--pairs-dir", default=os.path.join(figure_data, "input_folder"))
    ap.add_argument("--out", default=None, help="default analysis_data/twin_covariance_decomposition_z_scores[_random_pairs].csv")
    args = ap.parse_args()
    BREAK_TWINS = args.break_twins
    if args.out is None:
        name = "twin_covariance_decomposition_z_scores" + ("_random_pairs" if BREAK_TWINS else "") + ".csv"
        args.out = os.path.join(figure_data, "analysis_data", name)

    jobs = []                                                   # one job per single-state file and per two-state pair
    for topology in args.topologies:
        complete = os.path.join(args.simulation_dir, "parameter_scan_complete", topology)
        singles = sorted(f for f in os.listdir(complete) if f.endswith(".csv.gz"))[:args.limit]
        jobs += [(single_state, (topology, os.path.join(complete, f))) for f in singles]
        pairs = pd.read_csv(os.path.join(args.pairs_dir, f"two_state_pairs_{topology}.csv")).head(args.limit)
        minimal = os.path.join(args.simulation_dir, "parameter_scan_minimal", topology)
        jobs += [(two_state, (topology, os.path.join(minimal, a), os.path.join(minimal, b))) for a, b in zip(pairs.file1, pairs.file2)]
    print(f"{len(jobs)} systems, {args.jobs} processes", flush=True)

    with mp.Pool(args.jobs) as pool:
        rows = pool.map(run, jobs, chunksize=20)
    table = pd.DataFrame(rows)
    if "error" in table:
        failed = table[table.error.notna()]
        print(f"{len(failed)} systems failed, e.g.\n{failed.head(3).to_string()}")
        table = table[table.error.isna()].drop(columns="error")
    table.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(table)} rows")
