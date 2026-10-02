"""
Sibling of twinfer_to_boolode_format.py: converts the new multi-state
real-network reruns (GSD/HSC/EMT IC-unset "multistate" + all-5 seeded-IC
"seeded" variants -- see real_network_multistate_sim.py /
real_network_seeded_sim.py) into BEELINE-format inputs (ExpressionData.csv +
PseudoTime.csv + GroundTruthNetwork.csv), same twin_paired/spread sampling
schemes as the original script.

Kept separate rather than added to that script's DATASETS dict because these
runs live under per-variant run_tag subfolders (not "simulate/latest") and
use dataset_key names distinct from the existing real-network dataset_ids
(GSD_multistate, GSD_seeded, ... ) so they land in their own
inputs/real_networks/<dataset_key>/ folders without colliding with the
existing GSD/HSC/EMT/VSC/mCAD legacy-track inputs used elsewhere.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from twinfer.utils.paths import get_data_root, get_external_repo_path

T1 = 1
T2 = 20
SEED = 101010

DATA_ROOT = get_data_root()
PROJECT_ROOT = DATA_ROOT.parent
BEELINE_INPUT_ROOT = get_external_repo_path("beeline") / "inputs" / "real_networks"

DATASETS = {
    "GSD_multistate": dict(
        sim_dir=DATA_ROOT / "paper_analysis" / "GSD" / "simulate" / "multistate_2000steps",
        type_token="GSD_multistate",
        topology=PROJECT_ROOT / "input_data" / "real_world_networks" / "GSD.txt",
    ),
    "GSD_seeded": dict(
        sim_dir=DATA_ROOT / "paper_analysis" / "GSD" / "simulate" / "multistate_2000steps_seeded",
        type_token="GSD_seeded",
        topology=PROJECT_ROOT / "input_data" / "real_world_networks" / "GSD.txt",
    ),
    "HSC_multistate": dict(
        sim_dir=DATA_ROOT / "paper_analysis" / "HSC" / "simulate" / "multistate_2000steps",
        type_token="HSC_multistate",
        topology=PROJECT_ROOT / "input_data" / "real_world_networks" / "HSC.txt",
    ),
    "HSC_seeded": dict(
        sim_dir=DATA_ROOT / "paper_analysis" / "HSC" / "simulate" / "multistate_2000steps_seeded",
        type_token="HSC_seeded",
        topology=PROJECT_ROOT / "input_data" / "real_world_networks" / "HSC.txt",
    ),
    "EMT_multistate": dict(
        sim_dir=DATA_ROOT / "paper_analysis" / "EMT" / "simulate" / "multistate_2000steps",
        type_token="EMT_multistate",
        topology=PROJECT_ROOT / "input_data" / "real_world_networks" / "EMT.txt",
    ),
    "EMT_seeded": dict(
        sim_dir=DATA_ROOT / "paper_analysis" / "EMT" / "simulate" / "multistate_2000steps_seeded",
        type_token="EMT_seeded",
        topology=PROJECT_ROOT / "input_data" / "real_world_networks" / "EMT.txt",
    ),
    "VSC_seeded": dict(
        sim_dir=DATA_ROOT / "paper_analysis" / "VSC" / "simulate" / "multistate_6000steps_seeded",
        type_token="VSC_seeded",
        topology=PROJECT_ROOT / "input_data" / "real_world_networks" / "VSC.txt",
    ),
    "mCAD_seeded": dict(
        sim_dir=DATA_ROOT / "paper_analysis" / "mCAD" / "simulate" / "multistate_2000steps_seeded",
        type_token="mCAD_seeded",
        topology=PROJECT_ROOT / "input_data" / "real_world_networks" / "mCAD.txt",
    ),
}


def sim_filename_re(type_token: str) -> re.Pattern:
    return re.compile(
        rf"^df_rows_.+_ncells_\d+_{re.escape(type_token)}_rep_(\d+)_[0-9a-fA-F]{{8}}\.csv$"
    )


def convert_ground_truth(topology_txt_path: Path, out_path: Path) -> int:
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
    mrna_cols = [f"gene_{i + 1}_mRNA" for i in range(n_genes)]
    usecols = ["cell_id", "time_step", "replicate", "clone_id"] + mrna_cols
    df = pd.read_csv(sim_csv_path, usecols=usecols)
    df = df.rename(columns={f"gene_{i + 1}_mRNA": f"gene_{i + 1}" for i in range(n_genes)})
    return df


def _write_beeline_run(rows: pd.DataFrame, gene_names: list, cell_labels: list,
                        pseudotime: np.ndarray, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    expr = rows[gene_names].T
    expr.columns = cell_labels
    expr.index.name = None
    expr.to_csv(run_dir / "ExpressionData.csv")
    pt = pd.DataFrame({"PseudoTime1": pseudotime}, index=cell_labels)
    pt.to_csv(run_dir / "PseudoTime.csv")


def build_twin_paired(df: pd.DataFrame, gene_names: list, run_dir: Path,
                       t1: int = T1, t2: int = T2, seed: int = SEED) -> None:
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
    for dataset_key, cfg in DATASETS.items():
        sim_files = sorted(cfg["sim_dir"].glob("df_*_ncells_*.csv"))
        sim_files = [f for f in sim_files if not f.name.startswith("simulation_before_division")]
        print(f"[{dataset_key}] found {len(sim_files)} simulation files under {cfg['sim_dir']}")

        dataset_dir = BEELINE_INPUT_ROOT / dataset_key
        n_genes = convert_ground_truth(cfg["topology"], dataset_dir / "GroundTruthNetwork.csv")
        gene_names = [f"gene_{i + 1}" for i in range(n_genes)]

        filename_re = sim_filename_re(cfg["type_token"])
        for sim_path in sim_files:
            m = filename_re.match(sim_path.name)
            if not m:
                print(f"[skip] filename didn't match expected pattern: {sim_path.name}")
                continue
            sim_rep = m.group(1)

            print(f"Processing {sim_path.name} (dataset={dataset_key}, sim_rep={sim_rep}, n_genes={n_genes})...")
            df = load_simulation(sim_path, n_genes)

            build_twin_paired(df, gene_names, dataset_dir / f"simrep{sim_rep}_twin_paired")
            build_spread(df, gene_names, dataset_dir / f"simrep{sim_rep}_spread")

            del df

    print("Done.")


if __name__ == "__main__":
    main()
