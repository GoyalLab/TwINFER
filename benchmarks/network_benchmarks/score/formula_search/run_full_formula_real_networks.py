from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

# HERE = os.path.dirname(os.path.abspath(__file__))   [2026-09-30 replaced: these scripts used to sit in analysis_data/network_sweep_final and read/write their CSVs next to themselves; that data dir is now referenced explicitly]
HERE = f'{TWINFER_PROJECT_ROOT}/analysis_data/network_sweep_final'
ROOT = f'{TWINFER_PROJECT_ROOT}'
SIM_DIR = f"{ROOT}/simulation_data/real_data"
EXTRA_SIM_DIRS = {
    "EMT": f"{ROOT}/analysis_data/paper_analysis/EMT/simulate/20260825_224653",
    "B_cell_activation": f"{ROOT}/analysis_data/paper_analysis/B_cell_activation/simulate/20260812_153622",
    "Pluripotent": f"{ROOT}/analysis_data/paper_analysis/Pluripotent/simulate/20260825_224511",
}
TOPO_DIR = f"{ROOT}/input_data/real_world_networks"

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, HERE)
from twinfer.scoring.analytic_core import full_table_disjoint, compute_scores, score_topk

T2 = 20
NETS = {"GSD": "GSD.txt", "HSC_balanced": "HSC.txt", "mCAD": "mCAD.txt", "VSC": "VSC.txt",
        "EMT": "EMT.txt", "B_cell_activation": "B_cell.txt", "Pluripotent": "Pluripotent.txt"}


def true_edges(topo_path):
    M = np.loadtxt(topo_path, delimiter=",", dtype=int)
    n = M.shape[0]
    genes = [f"gene_{i+1}" for i in range(n)]
    true = {(genes[i], genes[j]) for i in range(n) for j in range(n) if i != j and M[i, j] != 0}
    poss = [(genes[i], genes[j]) for i in range(n) for j in range(n) if i != j]
    return true, poss, genes


def main():
    rows = []
    t0 = time.time()
    for tok, topo_file in NETS.items():
        tf = f"{TOPO_DIR}/{topo_file}"
        true, poss, GENES = true_edges(tf)
        sdir = EXTRA_SIM_DIRS.get(tok, SIM_DIR)
        sims = sorted(f for f in glob.glob(f"{sdir}/*_{tok}_*.csv")
                      if "simulation_before_division" not in os.path.basename(f))
        for sim in sims:
            for T1 in (1, 10):
                tsi = full_table_disjoint(sim, T1, T2, GENES)
                if tsi is None:
                    continue
                sc_all = compute_scores(tsi, GENES)
                for method in ("existdir", "full"):
                    mag = {p: v[method] for p, v in sc_all.items()}
                    sc = score_topk(mag, true, poss)
                    rows.append(dict(net=tok, sim=os.path.basename(sim), T1=T1, n_genes=len(GENES), method=method, **sc))
        print(f"[{time.time()-t0:.0f}s] {tok} ({len(sims)} sims, n={len(GENES)})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{HERE}/full_formula_real_networks_results.csv", index=False)
    summ = df.groupby(["net", "T1", "method"])[["auprc", "f1", "precision", "recall"]].mean()
    summ.to_csv(f"{HERE}/full_formula_real_networks_summary.csv")
    print("\n=== real_data networks: existdir vs full (LARRY-tuned weights), per network ===")
    print(summ.round(3).to_string())
    print("\n=== pooled across all 4 networks ===")
    print(df.groupby(["T1", "method"])[["auprc", "f1", "precision", "recall"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
