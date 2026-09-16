"""
Convert finished mixed_network_sweep simulations into BEELINE-format inputs,
mirroring the network_sweep_final_20260824 layout:

    code/Beeline/inputs/mixed_network_sweep/<net>/
        GroundTruthNetwork.csv
        simrep<k>_twin_paired/{ExpressionData.csv, PseudoTime.csv}
        simrep<k>_spread/{ExpressionData.csv, PseudoTime.csv}

<net> is e.g. grn_n6_e6_c3_a3_pos50_rep0 (48 networks); <k> is the simulation
replicate. The two sampling schemes (twin_paired, spread) and the conversion
maths are reused verbatim from twinfer_to_boolode_format.py.

Idempotent: skips any (net, k, scheme) whose ExpressionData.csv already
exists, so it is safe to rerun as more simulations finish.

    python mixed_to_beeline_format.py
    python mixed_to_beeline_format.py --schemes twin_paired
"""
import argparse
import glob
import os
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from twinfer_to_boolode_format import (
    build_spread,
    build_twin_paired,
    convert_ground_truth,
    load_simulation,
)

PROJECT_ROOT = Path("/home/gzu5140/TwINFER_KA")
SIM_DIR = PROJECT_ROOT / "simulation_data" / "mixed_network_sweep"
TOPOLOGY_DIR = PROJECT_ROOT / "input_data" / "mixed_network_sweep"
BEELINE_INPUT_ROOT = PROJECT_ROOT / "code" / "Beeline" / "inputs" / "mixed_network_sweep"

SIM_RE = re.compile(
    r"^df_(grn_.+?_rep\d+)_rep(\d+)_\d{8}_\d{6}_ncells_\d+_\1_rep\2_[0-9a-fA-F]+\.csv$"
)


def discover():
    """[(net, sim_rep, path), ...] for every finished simulation on disk."""
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", nargs="+", default=["twin_paired", "spread"],
                    choices=["twin_paired", "spread"])
    args = ap.parse_args()

    tasks = discover()
    print(f"{len(tasks)} finished simulation(s) found under {SIM_DIR}")

    gt_done = set()
    for net, k, sim_path in tasks:
        dataset_dir = BEELINE_INPUT_ROOT / net

        if net not in gt_done:
            n_genes = convert_ground_truth(TOPOLOGY_DIR / f"{net}.txt",
                                           dataset_dir / "GroundTruthNetwork.csv")
            gt_done.add(net)
        else:
            n_genes = pd.read_csv(TOPOLOGY_DIR / f"{net}.txt", header=None).shape[0]
        gene_names = [f"gene_{i + 1}" for i in range(n_genes)]

        need = [s for s in args.schemes
                if not (dataset_dir / f"simrep{k}_{s}" / "ExpressionData.csv").exists()]
        if not need:
            continue

        print(f"[{net}] simrep{k}: building {need} from {sim_path.name}")
        df = load_simulation(sim_path, n_genes)
        if "twin_paired" in need:
            build_twin_paired(df, gene_names, dataset_dir / f"simrep{k}_twin_paired")
        if "spread" in need:
            build_spread(df, gene_names, dataset_dir / f"simrep{k}_spread")
        del df

    print("Done.")


if __name__ == "__main__":
    main()
