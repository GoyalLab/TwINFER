"""
[RECONSTRUCTED 2026-09-30] Convert TwINFER's network_sweep raw simulation trajectories into BEELINE-format
inputs (ExpressionData.csv + PseudoTime.csv + GroundTruthNetwork.csv).

The original script was lost (only convert_network_sweep_to_beeline.cpython-312.pyc, 2026-07-17, survived).
main(), the constants and the filename regex are rebuilt from that bytecode (see
_recovered_from_pycache/convert_network_sweep_to_beeline.dis.txt). The four worker functions
(convert_ground_truth, load_simulation, build_twin_paired, build_spread) were later generalised in
twinfer_to_boolode_format.py (same docstring, same T1/T2/SEED) and are imported from there instead of being
copied. Verified by re-running one simulation and comparing with the stored inputs (see REVIEW_LOG.md).

Raw simulation files are per-cell, per-timestep trajectories (cell_id, time_step, gene_i_* columns, replicate,
clone_id), 6000 clones (twin pairs) each. Two sampling schemes are written per simulation:
  twin_paired: clone_ids split 1500/1500/3000 into t1-only, t2-only and across-time groups (12000 cells).
  spread: all 12000 daughter cells partitioned into disjoint random subsets, one per time_step.
Ground truth (input_data/network_sweep/grn_n6_*.txt signed adjacency matrices) -> Gene1,Gene2,Type edge list.
"""
import re
from pathlib import Path

from twinfer.utils.paths import get_data_root, get_external_repo_path
from benchmarks.network_benchmarks.to_beeline.twinfer_to_boolode_format import build_spread, build_twin_paired, convert_ground_truth, load_simulation

ROOT = get_data_root().parent  # project root (parent of analysis_data)
PATH_TO_SIMULATION_DATA = ROOT / "simulation_data" / "network_sweep"
PATH_TO_TOPOLOGY_DATA = ROOT / "input_data" / "network_sweep"
BEELINE_INPUT_ROOT = get_external_repo_path("beeline") / "inputs" / "network_sweep"

SIM_FILENAME_RE = re.compile(r"^df_(grn_n6_.+?_rep\d+)_rep(\d+)_\d+_\d+_ncells_\d+_.*\.csv$")


def main():
    sim_files = sorted(f for f in PATH_TO_SIMULATION_DATA.glob("df_grn_n6_*_ncells_*.csv")
                       if not f.name.startswith("simulation_before_division"))
    print(f"Found {len(sim_files)} network_sweep simulation files.")

    ground_truth_cache = {}  # topology_name -> n_genes (convert each topology once)
    for sim_path in sim_files:
        m = SIM_FILENAME_RE.match(sim_path.name)
        if not m:
            print(f"[skip] filename didn't match expected pattern: {sim_path.name}")
            continue
        topology_name, sim_rep = m.group(1), m.group(2)
        dataset_dir = BEELINE_INPUT_ROOT / topology_name

        if topology_name not in ground_truth_cache:
            topology_txt = PATH_TO_TOPOLOGY_DATA / f"{topology_name}.txt"
            n_genes = convert_ground_truth(topology_txt, dataset_dir / "GroundTruthNetwork.csv")
            ground_truth_cache[topology_name] = n_genes
        n_genes = ground_truth_cache[topology_name]
        gene_names = [f"gene_{i + 1}" for i in range(n_genes)]

        print(f"Processing {sim_path.name} (topology={topology_name}, sim_rep={sim_rep}, n_genes={n_genes})...")
        df = load_simulation(sim_path, n_genes)

        build_twin_paired(df, gene_names, dataset_dir / f"simrep{sim_rep}_twin_paired")
        build_spread(df, gene_names, dataset_dir / f"simrep{sim_rep}_spread")

    print("Done.")


if __name__ == "__main__":
    main()
