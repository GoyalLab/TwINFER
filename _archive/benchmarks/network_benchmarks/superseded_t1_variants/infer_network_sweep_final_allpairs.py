"""ALL_PAIRS unrestricted rerun for the REST of network_sweep_final's OFAT topology sweep --
e5_pos100_density, e9_pos0_sign_ratio, e9_pos50_sign_ratio, e9_pos100_center, e17_pos100_density
(150 simulations total; e13_pos100_center's 30 were already done separately by
infer_e13_pos100_allpairs.py, same ground truth/filename convention, kept in its own directory).
Same pattern as infer_mixed_network_sweep_allpairs.py: alpha_gene_gene_corr/alpha_stage3->0.9999,
z_score_threshold_two_states/z_score_threshold_cross_correlation->0, minimum_null_draws=5000. The
existing network_inference_final/ output for this set predates the current package's
z_het/gated_regulation additions (legacy schema, no twin_score_inputs at all) -- this reruns with
the current package to get the modern schema TODO4v2 needs.

Output: analysis_data/network_sweep_final/twinfer_inference_allpairs/<net>_rep<k>_all_results.json

    python infer_network_sweep_final_allpairs.py --n-cores 4 [--start N --count M]
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/home/gzu5140/TwINFER_KA/code/TwINFER/package")
from twinfer.inference.infer import infer_with_twinfer

PROJECT_ROOT = f'{TWINFER_PROJECT_ROOT}'
SIM_DIR = f"{PROJECT_ROOT}/simulation_data/network_sweep_final"
TOPOLOGY_DIR = f"{PROJECT_ROOT}/input_data/network_sweep_final"
OUTPUT_DIR = f"{PROJECT_ROOT}/analysis_data/network_sweep_final/twinfer_inference_allpairs"
PARAM_CSV = f"{PROJECT_ROOT}/input_data/network_sweep/parameters.csv"

T1, T2 = 1, 20
ALPHA = 0.9999
N_GENES = 6

SIM_RE = re.compile(
    r"^df_(grn_.+?_rep\d+)_rep(\d+)_\d{8}_\d{6}_ncells_\d+_\1_rep\2_[0-9a-fA-F]+\.csv$"
)

BASE_CONFIG = {
    "n_cells": 6000,
    "simulation_time_before_division": 6000,
    "twin_simulation_time_after_division": 48,
    "twin_measurement_resolution": 1,
    "path_to_connectivity_matrix": None,
    "param_csv": PARAM_CSV,
    "rows_to_use": [[0] * N_GENES],
    "output_folder": f"{OUTPUT_DIR}/_scratch/",
    "log_file": f"{OUTPUT_DIR}/_scratch/logs/network_sweep_final_allpairs.jsonl",
    "type": "network_sweep_final",
    "combinatorial_interaction_type": "additive",
    "number_of_parallel_parameters": 1,
    "number_of_cores_per_parameter": 56,
    "log_pi_on": False,
    "ranked_list": True,
}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def make_json_safe(obj):
    if isinstance(obj, pd.DataFrame):
        return {"__type__": "DataFrame", "index": [str(i) for i in obj.index.tolist()],
                "columns": [str(c) for c in obj.columns.tolist()], "data": obj.values.tolist()}
    if isinstance(obj, pd.Series):
        return {"__type__": "Series", "index": [str(i) for i in obj.index.tolist()], "data": obj.values.tolist()}
    if isinstance(obj, dict):
        return {("__".join(map(str, k)) if isinstance(k, tuple) else str(k)): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (set, frozenset)):
        return [make_json_safe(x) for x in obj]
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(x) for x in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def discover_tasks():
    tasks = []
    for path in sorted(glob.glob(os.path.join(SIM_DIR, "df_*.csv"))):
        b = os.path.basename(path)
        if b.startswith("simulation_before_division"):
            continue
        m = SIM_RE.match(b)
        if not m:
            continue
        net, sim_rep = m.group(1), int(m.group(2))
        gt_path = os.path.join(TOPOLOGY_DIR, f"{net}.txt")
        if not os.path.exists(gt_path):
            continue
        tasks.append((net, sim_rep, path))
    return tasks


def run_one(net, sim_rep, source_path, n_cores):
    out_path = os.path.join(OUTPUT_DIR, f"{net}_rep{sim_rep}_all_results.json")
    if os.path.exists(out_path):
        return out_path, "skipped (already exists)"

    matrix_path = os.path.join(TOPOLOGY_DIR, f"{net}.txt")
    base_config = dict(BASE_CONFIG)
    base_config["path_to_connectivity_matrix"] = matrix_path

    results = infer_with_twinfer(
        source_path,
        merge_to_multiple_states=False,
        base_config=base_config,
        t1=T1, t2=T2,
        check_for_steady_state=False,
        match_sim_details=False,
        seed=101010,
        n_cores=n_cores,
        alpha_gene_gene_corr=ALPHA,
        alpha_stage3=ALPHA,
        z_score_threshold_two_states=0,
        z_score_threshold_cross_correlation=0,
        minimum_null_draws=5000,
        ranked_list=True,
        separate_fan_outs_from_mutual_regulation_flag=True,
        fan_out_z_score_threshold=3.2,
        use_scramble_cross_correlation=False,
    )

    gene_names = [f"gene_{i+1}" for i in range(N_GENES)]
    record = {
        "dataset_id": net, "label": sim_rep, "source_file": source_path,
        "gene_names": gene_names, "n_genes": N_GENES,
        "ground_truth_matrix": matrix_path,
        **make_json_safe(results),
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(record, f, cls=NumpyEncoder, indent=2)
    return out_path, "computed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cores", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--count", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(f"{OUTPUT_DIR}/_scratch/logs", exist_ok=True)
    tasks = discover_tasks()
    if args.start is not None:
        tasks = tasks[args.start:args.start + (args.count or len(tasks))]
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"{len(tasks)} simulation(s) discovered", flush=True)
    if args.dry_run:
        for net, k, _ in tasks:
            done = os.path.exists(os.path.join(OUTPUT_DIR, f"{net}_rep{k}_all_results.json"))
            print(f"  {'done' if done else 'TODO'}  {net}_rep{k}")
        sys.exit(0)

    for i, (net, k, src) in enumerate(tasks):
        t0 = time.time()
        path, status = run_one(net, k, src, args.n_cores)
        print(f"[{i+1}/{len(tasks)}] {net} rep {k}: {status} ({time.time()-t0:.1f}s)", flush=True)
