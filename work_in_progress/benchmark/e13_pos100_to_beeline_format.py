"""
Convert the 3 x 10 e13_pos100 simulations into BEELINE-format inputs, same
layout/conventions as mixed_to_beeline_format.py:

    code/Beeline/inputs/e13_pos100/<label>/
        GroundTruthNetwork.csv
        simrep<k>_twin_paired/{ExpressionData.csv, PseudoTime.csv}
        simrep<k>_spread/{ExpressionData.csv, PseudoTime.csv}

See infer_e13_pos100.py's docstring for the label -> ground-truth-topology
mapping (rep1->center_rep1, rep2->center_rep2, rep3->center_rep0; verified by
reconstructing each sim's edges from its logged K_/k_add_ parameters).

    python e13_pos100_to_beeline_format.py
"""
import glob
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from twinfer_to_boolode_format import build_spread, build_twin_paired, convert_ground_truth, load_simulation

PROJECT_ROOT = Path("/home/gzu5140/TwINFER_KA")
SIM_DIR = PROJECT_ROOT / "simulation_data" / "network_sweep_final" / "e13_pos100"
GT_DIR = PROJECT_ROOT / "input_data" / "network_sweep_final"
BEELINE_INPUT_ROOT = PROJECT_ROOT / "code" / "Beeline" / "inputs" / "e13_pos100"

N_GENES = 6
LABEL_TO_GT = {
    "n6_e13_pos100_rep1": GT_DIR / "grn_n6_e13_pos100_center_rep1.txt",
    "n6_e13_pos100_rep2": GT_DIR / "grn_n6_e13_pos100_center_rep2.txt",
    "n6_e13_pos100_rep3": GT_DIR / "grn_n6_e13_pos100_center_rep0.txt",
}

SIM_RE = re.compile(r"^df_rows_.+?_ncells_6000_(n6_e13_pos100_rep[123])_rep_(\d+)_[0-9a-fA-F]+\.csv$")


def discover():
    out = []
    for p in sorted(glob.glob(str(SIM_DIR / "df_*.csv"))):
        b = os.path.basename(p)
        if b.startswith("simulation_before_division"):
            continue
        m = SIM_RE.match(b)
        if not m:
            print(f"[skip] filename didn't match: {b}")
            continue
        out.append((m.group(1), int(m.group(2)), Path(p)))
    return out


def main():
    tasks = discover()
    print(f"{len(tasks)} simulation(s) found under {SIM_DIR}")

    gene_names = [f"gene_{i+1}" for i in range(N_GENES)]
    gt_done = set()

    for label, k, sim_path in tasks:
        dataset_dir = BEELINE_INPUT_ROOT / label
        if label not in gt_done:
            convert_ground_truth(LABEL_TO_GT[label], dataset_dir / "GroundTruthNetwork.csv")
            gt_done.add(label)

        need = [s for s in ("twin_paired", "spread")
                if not (dataset_dir / f"simrep{k}_{s}" / "ExpressionData.csv").exists()]
        if not need:
            continue

        print(f"[{label}] simrep{k}: building {need} from {sim_path.name}")
        df = load_simulation(sim_path, N_GENES)
        if "twin_paired" in need:
            build_twin_paired(df, gene_names, dataset_dir / f"simrep{k}_twin_paired")
        if "spread" in need:
            build_spread(df, gene_names, dataset_dir / f"simrep{k}_spread")
        del df

    print("Done.")


if __name__ == "__main__":
    main()
