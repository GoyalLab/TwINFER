"""full_gated (fixed penalty=-2.0) on the three newly-discovered real_data networks
(EMT/B_cell_activation/Pluripotent), using their paper_analysis sim-dir override --
same process_one/true_edges logic as full_gated_fanout.py, just pointed at the right
simulation directory and topo file per network.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob
import os
import sys
import time

import pandas as pd

ROOT = f'{TWINFER_PROJECT_ROOT}'
# HERE = os.path.dirname(os.path.abspath(__file__))   [2026-09-30 replaced: this script used to sit in the data dir analysis_data/paper_analysis/real_networks and read/write next to itself; that dir is now referenced explicitly]
HERE = f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/real_networks'
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, HERE)
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, f"{ROOT}/analysis_data/network_sweep_final")
from paper_analysis.real_networks.full_gated_fanout import process_one, PENALTIES
from benchmarks.network_benchmarks.score.formula_search.zhet_everywhere import true_edges

EXTRA_SIM_DIRS = {
    "EMT": (f"{ROOT}/analysis_data/paper_analysis/EMT/simulate/20260825_224653", "EMT.txt", "EMT", 6),
    "B_cell_activation": (f"{ROOT}/analysis_data/paper_analysis/B_cell_activation/simulate/20260812_153622", "B_cell.txt", "B_cell_activation", 6),
    "Pluripotent": (f"{ROOT}/analysis_data/paper_analysis/Pluripotent/simulate/20260825_224511", "Pluripotent.txt", "Pluripotent", 3),
}


def run_one_network(tok):
    sdir, topo_file, sim_tag, max_reps = EXTRA_SIM_DIRS[tok]
    tf = f"{ROOT}/input_data/real_world_networks/{topo_file}"
    true_e, poss = true_edges(tf)
    n = len(set(g for p in poss for g in p))
    GENES = [f"gene_{i+1}" for i in range(n)]
    sims = sorted(f for f in glob.glob(f"{sdir}/*_{sim_tag}_*.csv")
                  if "simulation_before_division" not in os.path.basename(f))[:max_reps]
    rows = []
    t0 = time.time()
    for sim in sims:
        try:
            r = process_one(sim, tf, true_e, poss, GENES)
        except Exception as e:
            print("  skip", os.path.basename(sim), e, flush=True)
            continue
        if r:
            r["dataset"] = f"real_data:{tok}"; r["net"] = tok
            rows.append(r)
        print(f"  [{time.time()-t0:.0f}s] {os.path.basename(sim)} done", flush=True)
    print(f"[{time.time()-t0:.0f}s] real_data/{tok} ({len(sims)} sims, n={n})", flush=True)
    return rows


def main():
    rows = []
    for tok in ["EMT", "B_cell_activation", "Pluripotent"]:
        rows += run_one_network(tok)

    df = pd.DataFrame(rows)
    df.to_csv(f"{HERE}/full_gated_fanout_new_networks_results.csv", index=False)
    print(f"\nwrote {len(df)} rows")
    print(df.groupby("dataset").mean(numeric_only=True).round(4).to_string())


if __name__ == "__main__":
    main()
