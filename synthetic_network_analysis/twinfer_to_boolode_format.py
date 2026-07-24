"""
Convert TwINFER's network_sweep raw simulation trajectories into BEELINE-format
inputs (ExpressionData.csv + PseudoTime.csv + GroundTruthNetwork.csv).

Raw simulation files (under path_to_simulation_data) are per-cell, per-timestep
stochastic trajectories with columns cell_id, time_step, gene_i_A, gene_i_I,
gene_i_mRNA, gene_i_protein (per gene), replicate, clone_id. Each simulation
has n_cells=6000 clones (twin pairs): cell_id = clone_id + 6000*(replicate-1),
so cell_ids 0..5999 are replicate-1 daughters and 6000..11999 are the matching
replicate-2 daughters of the same division event. time_step runs 0..48 for
every cell (see TwINFER_function_scripts/infer_with_twinfer.py for the twin
semantics this mirrors).

Two sampling schemes are produced per simulation file:

  twin_paired: the 6000 clone_ids are split 1500/1500/3000 into t1-only,
    t2-only, and across-time groups (matching infer_with_twinfer's own split
    ratio). Both daughters of a t1-only or t2-only clone are measured at
    their timepoint; an across-time clone's replicate-1 daughter is measured
    at t1 and its replicate-2 daughter at t2. Yields 12000 cells total (6000
    at t1, 6000 at t2).

  spread: all 12000 daughter cells are partitioned into disjoint random
    subsets, one per available time_step (0..48), each cell appearing
    exactly once at its assigned timepoint -- approximating a continuous
    pseudotime-ordered single-cell population.

Ground truth networks (input_data/network_sweep/grn_n6_*.txt, signed
adjacency matrices) are converted once per topology into BEELINE's
Gene1,Gene2,Type edge-list format.
"""

import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

T1 = 1
T2 = 20
SEED = 101010

# PATH_TO_SIMULATION_DATA = Path("/home/gzu5140/Keerthana_b1042/TwINFER/simulation_data/network_sweep_final/")
# PATH_TO_TOPOLOGY_DATA = Path("/home/gzu5140/Keerthana_b1042/TwINFER/input_data/network_sweep_final/")
PATH_TO_SIMULATION_DATA = Path("/home/gzu5140/Keerthana_b1042/TwINFER/simulation_data/cyclic_6_nodes/")
PATH_TO_TOPOLOGY_DATA = Path("/home/gzu5140/Keerthana_b1042/TwINFER/input_data/cycle_6_node.txt")
BEELINE_INPUT_ROOT = Path("/home/gzu5140/Keerthana_b1042/TwINFER/code/Beeline/inputs/network_sweep_final/")

# Matches e.g. "df_grn_n6_e5_pos50_density_rep0_rep1_12072026_013639_ncells_6000_grn_n6_e5_pos50_density_rep0_rep1_44f47434.csv"
# Captures topology name ("grn_n6_e5_pos50_density_rep0") and simulation replicate ("1").
# SIM_FILENAME_RE = re.compile(
#     r"^df_(grn_n6_.+?_rep\d+)_rep(\d+)_\d+_\d+_ncells_\d+_.*\.csv$"
# )
# Matches e.g. "df_rows_0_0_0_0_0_0_20072026_032458_ncells_6000_cycle_6_node_0_0_ba704c65.csv"
# Only one topology here (cycle_6_node), so group(1) is the literal topology name and
# group(2) is the trailing 8-char hex hash, used as the per-run replicate identifier.
SIM_FILENAME_RE = re.compile(
    r"^df_rows_.+_ncells_\d+_(cycle_6_node)_\d+_\d+_([0-9a-fA-F]{8})\.csv$"
)


def convert_ground_truth(topology_txt_path: Path, out_path: Path) -> int:
    """
    Convert a signed adjacency matrix (rows=regulators, cols=targets; values
    -1/0/1) into BEELINE's GroundTruthNetwork.csv (Gene1,Gene2,Type) format.

    Returns
    -------
    int
        Number of genes in the matrix (used to select gene_i_mRNA columns
        from the matching simulation file).
    """
    matrix = pd.read_csv(topology_txt_path, header=None).to_numpy()
    n_genes = matrix.shape[0]
    gene_names = [f"gene_{i + 1}" for i in range(n_genes)]

    edges = []
    for i in range(n_genes):
        for j in range(n_genes):
            if matrix[i, j] == 0:
                continue
            edges.append((gene_names[i], gene_names[j], "+" if matrix[i, j] > 0 else "-"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(edges, columns=["Gene1", "Gene2", "Type"]).to_csv(out_path, index=False)
    return n_genes


def load_simulation(sim_csv_path: Path, n_genes: int) -> pd.DataFrame:
    """Load only the columns needed: cell_id, time_step, each gene's mRNA count, replicate, clone_id."""
    mrna_cols = [f"gene_{i + 1}_mRNA" for i in range(n_genes)]
    usecols = ["cell_id", "time_step", "replicate", "clone_id"] + mrna_cols
    df = pd.read_csv(sim_csv_path, usecols=usecols)
    df = df.rename(columns={f"gene_{i + 1}_mRNA": f"gene_{i + 1}" for i in range(n_genes)})
    return df


def _write_beeline_run(rows: pd.DataFrame, gene_names: list, cell_labels: list,
                        pseudotime: np.ndarray, run_dir: Path) -> None:
    """Write ExpressionData.csv (genes x cells) and PseudoTime.csv for one run."""
    run_dir.mkdir(parents=True, exist_ok=True)

    expr = rows[gene_names].T
    expr.columns = cell_labels
    expr.index.name = None
    expr.to_csv(run_dir / "ExpressionData.csv")

    pt = pd.DataFrame({"PseudoTime1": pseudotime}, index=cell_labels)
    pt.to_csv(run_dir / "PseudoTime.csv")


def build_twin_paired(df: pd.DataFrame, gene_names: list, run_dir: Path,
                       t1: int = T1, t2: int = T2, seed: int = SEED) -> None:
    """
    Split clones 1500/1500/3000 into t1-only / t2-only / across-time groups
    (matching infer_with_twinfer's own split), extract the corresponding
    rows, and write as one BEELINE run.
    """
    rng = np.random.default_rng(seed)
    clone_ids = df["clone_id"].unique()
    assert len(clone_ids) == 6000, f"expected 6000 clones, got {len(clone_ids)}"
    shuffled = rng.permutation(clone_ids)
    t1_clones, t2_clones, across_clones = shuffled[:1500], shuffled[1500:3000], shuffled[3000:]

    t1_only = df[df["clone_id"].isin(t1_clones) & (df["time_step"] == t1)]
    t2_only = df[df["clone_id"].isin(t2_clones) & (df["time_step"] == t2)]
    across_t1 = df[df["clone_id"].isin(across_clones) & (df["replicate"] == 1) & (df["time_step"] == t1)]
    across_t2 = df[df["clone_id"].isin(across_clones) & (df["replicate"] == 2) & (df["time_step"] == t2)]

    rows = pd.concat([t1_only, t2_only, across_t1, across_t2], ignore_index=True)
    assert len(rows) == 12000, f"expected 12000 cells, got {len(rows)}"

    cell_labels = [f"cell{cid}_t{ts}" for cid, ts in zip(rows["cell_id"], rows["time_step"])]
    _write_beeline_run(rows, gene_names, cell_labels, rows["time_step"].to_numpy(), run_dir)


def build_spread(df: pd.DataFrame, gene_names: list, run_dir: Path, seed: int = SEED) -> None:
    """
    Partition all 12000 daughter cells into disjoint random subsets, one per
    available time_step, each cell appearing exactly once at its assigned
    timepoint.
    """
    rng = np.random.default_rng(seed)
    cell_ids = df["cell_id"].unique()
    time_steps = np.sort(df["time_step"].unique())
    assigned_ts = rng.permutation(np.tile(time_steps, int(np.ceil(len(cell_ids) / len(time_steps))))[:len(cell_ids)])
    shuffled_cells = rng.permutation(cell_ids)

    assignment = pd.DataFrame({"cell_id": shuffled_cells, "assigned_time_step": assigned_ts})
    rows = df.merge(assignment, on="cell_id").query("time_step == assigned_time_step")
    assert len(rows) == len(cell_ids), f"expected {len(cell_ids)} cells, got {len(rows)}"

    cell_labels = [f"cell{cid}_t{ts}" for cid, ts in zip(rows["cell_id"], rows["time_step"])]
    _write_beeline_run(rows, gene_names, cell_labels, rows["time_step"].to_numpy(), run_dir)


def main():
    # sim_files = sorted(PATH_TO_SIMULATION_DATA.glob("df_grn_n6_*_ncells_*.csv"))
    sim_files = sorted(PATH_TO_SIMULATION_DATA.glob("df_*_ncells_*.csv"))
    sim_files = [f for f in sim_files if not f.name.startswith("simulation_before_division")]

    print(f"Found {len(sim_files)} network_sweep simulation files.")
    ground_truth_cache = {}  # topology_name -> n_genes (also writes GroundTruthNetwork.csv once)

    for sim_path in sim_files:
        m = SIM_FILENAME_RE.match(sim_path.name)
        if not m:
            print(f"[skip] filename didn't match expected pattern: {sim_path.name}")
            continue
        topology_name, sim_rep = m.group(1), m.group(2)

        dataset_dir = BEELINE_INPUT_ROOT / topology_name
        if topology_name not in ground_truth_cache:
            # topology_txt = PATH_TO_TOPOLOGY_DATA / f"{topology_name}.txt"
            topology_txt = PATH_TO_TOPOLOGY_DATA  # single topology file now, not a directory
            n_genes = convert_ground_truth(topology_txt, dataset_dir / "GroundTruthNetwork.csv")
            ground_truth_cache[topology_name] = n_genes
        n_genes = ground_truth_cache[topology_name]
        gene_names = [f"gene_{i + 1}" for i in range(n_genes)]

        print(f"Processing {sim_path.name} (topology={topology_name}, sim_rep={sim_rep}, n_genes={n_genes})...")
        df = load_simulation(sim_path, n_genes)

        build_twin_paired(df, gene_names, dataset_dir / f"simrep{sim_rep}_twin_paired")
        build_spread(df, gene_names, dataset_dir / f"simrep{sim_rep}_spread")

        del df

    print("Done.")


if __name__ == "__main__":
    main()
