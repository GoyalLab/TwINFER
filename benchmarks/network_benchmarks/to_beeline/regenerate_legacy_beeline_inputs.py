"""
Regenerate BEELINE inputs (ExpressionData.csv + PseudoTime.csv + GroundTruthNetwork.csv)
for GSD/HSC/mCAD/VSC, the 4 legacy TwINFER-Gillespie-simulated real networks.

The original copy of this data lived at /scratch/gzu5140/twinfer_real/inputs/ --
Quest's scratch storage, purged after ~30 days of inactivity. That purge already
took out every file under it (see infer_network_simulation_real_network.py's
module docstring); only the directory names survived, which is how
allowed_legacy_hashes() below still knows which raw replicate hashes BEELINE was
originally scoped to.

This regenerates the exact same 9/10/10/10 replicate set (GSD/HSC/mCAD/VSC) from
the still-intact raw simulation CSVs in simulation_data/real_data/, using the same
conversion logic as twinfer_to_boolode_format.py (which only covers the 4 newer
real networks + cyclic ones), and writes the result to shared, non-purged storage
under code/Beeline/inputs/real_networks_legacy/ instead of scratch.
"""
from pathlib import Path

# [2026-09-30 commented out: filtered driver archived (user: keep the no-filter variants); the legacy-task helpers are identical in the no-filter module]
# from benchmarks.network_benchmarks.infer import infer_network_simulation_real_network as legacy_infer
from benchmarks.network_benchmarks.infer import infer_network_simulation_real_network_nofilter as legacy_infer
from benchmarks.network_benchmarks.to_beeline.twinfer_to_boolode_format import (
    build_spread,
    build_twin_paired,
    convert_ground_truth,
    load_simulation,
)

OUT_ROOT = legacy_infer.PROJECT_ROOT / "code" / "Beeline" / "inputs" / "real_networks_legacy"

LEGACY_NETWORKS = ["GSD", "HSC", "mCAD", "VSC"]


def main():
    for net in LEGACY_NETWORKS:
        cfg = legacy_infer.NETWORKS[net]
        base_config = legacy_infer.make_base_config(net, cfg)
        tasks = legacy_infer.build_legacy_tasks(net, cfg, base_config)
        print(f"[{net}] {len(tasks)} replicates to convert")

        dataset_dir = OUT_ROOT / net
        n_genes = convert_ground_truth(
            legacy_infer.INPUT_DATA / "real_world_networks" / cfg["topology"],
            dataset_dir / "GroundTruthNetwork.csv",
        )
        gene_names = [f"gene_{i + 1}" for i in range(n_genes)]

        for sim_path, _net, rep_id, _base_config in tasks:
            file_hash = rep_id.rsplit("_", 1)[-1]

            twin_paired_run = dataset_dir / f"simrep{file_hash}_twin_paired"
            spread_run = dataset_dir / f"simrep{file_hash}_spread"
            if (twin_paired_run / "ExpressionData.csv").exists() and (spread_run / "ExpressionData.csv").exists():
                print(f"  [skip, already present] {net} {file_hash}")
                continue

            print(f"  Processing {net} {file_hash} ({Path(sim_path).name})...")
            df = load_simulation(Path(sim_path), n_genes)
            build_twin_paired(df, gene_names, twin_paired_run)
            build_spread(df, gene_names, spread_run)
            del df

    print("Done.")


if __name__ == "__main__":
    main()
