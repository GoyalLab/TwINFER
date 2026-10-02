"""
Extend BEELINE inputs for EMT/Pluripotent to cover the additional simulation
replicates now available on disk beyond what BEELINE was originally run on
(EMT: reps 0-7 done, 8-9 missing; Pluripotent: reps 0-1 done, 2-5 missing).

twin_paired only -- spread is out of scope (matches the legacy-network
backfill's scope). Reuses twinfer_to_boolode_format.py's conversion
functions directly rather than its main(), which has no skip-if-exists
check and would redundantly reprocess every already-done dataset/rep
(including B_cell_activation/Circadian_cycle, untouched here).
"""
from pathlib import Path

import pandas as pd

from benchmarks.network_benchmarks.to_beeline.twinfer_to_boolode_format import (
    BEELINE_INPUT_ROOT,
    DATASETS,
    build_twin_paired,
    load_simulation,
    sim_filename_re,
)

NETWORKS_TO_EXTEND = ["EMT", "Pluripotent"]


def main():
    for dataset_key in NETWORKS_TO_EXTEND:
        cfg = DATASETS[dataset_key]
        dataset_dir = BEELINE_INPUT_ROOT / dataset_key

        gt_path = dataset_dir / "GroundTruthNetwork.csv"
        assert gt_path.exists(), f"missing ground truth for {dataset_key}: {gt_path}"
        # Same n_genes derivation as convert_ground_truth: the topology
        # matrix's row/column count, not the GT edge list (which can omit
        # isolated genes with no edges at all).
        n_genes = pd.read_csv(cfg["topology"], header=None).to_numpy().shape[0]

        sim_files = sorted(cfg["sim_dir"].glob("df_*_ncells_*.csv"))
        sim_files = [f for f in sim_files if not f.name.startswith("simulation_before_division")]
        filename_re = sim_filename_re(cfg["type_token"])

        gene_names = [f"gene_{i + 1}" for i in range(n_genes)]

        for sim_path in sim_files:
            m = filename_re.match(sim_path.name)
            if not m:
                continue
            sim_rep = m.group(1)

            run_dir = dataset_dir / f"simrep{sim_rep}_twin_paired"
            if (run_dir / "ExpressionData.csv").exists():
                continue  # already have this rep

            print(f"[{dataset_key}] generating missing rep {sim_rep} from {sim_path.name}...")
            df = load_simulation(sim_path, n_genes)
            build_twin_paired(df, gene_names, run_dir)
            del df

    print("Done.")


if __name__ == "__main__":
    main()
