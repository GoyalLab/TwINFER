"""ALL_PAIRS unrestricted rerun of infer_mixed_network_sweep.py: same simulations, same
everything, except alpha_gene_gene_corr/alpha_stage3 -> 0.9999 and z_score_threshold_two_states/
z_score_threshold_cross_correlation -> 0 (matching LARRY's run_infer_correlation_high.py
ALL_PAIRS=1 convention), plus minimum_null_draws=5000 (same reason LARRY pins it -- ALL_PAIRS's
alpha=0.9999 would otherwise coarsen every null to the 2000-draw floor). The original
strict-threshold run (z_score_threshold_two_states=4.501) left twin_score_inputs at ~34% mean
coverage of the full pair universe -- TODO4v2 needs the full universe to be scored fairly.

Output: analysis_data/mixed_network_sweep/twinfer_inference_allpairs/
        <net>_rep<k>_all_results.json (never overwrites the strict-threshold run).

    python infer_mixed_network_sweep_allpairs.py --n-cores 4
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
SIM_DIR = f"{PROJECT_ROOT}/simulation_data/mixed_network_sweep"
TOPOLOGY_DIR = f"{PROJECT_ROOT}/input_data/mixed_network_sweep"
OUTPUT_DIR = f"{PROJECT_ROOT}/analysis_data/mixed_network_sweep/twinfer_inference_allpairs_t1_10"
PARAM_CSV = f"{PROJECT_ROOT}/input_data/network_sweep/parameters.csv"

T1, T2 = 10, 20
ALPHA = 0.9999

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
    "rows_to_use": None,
    "output_folder": f"{OUTPUT_DIR}/_scratch/",
    "log_file": f"{OUTPUT_DIR}/_scratch/logs/mixed_allpairs.jsonl",
    "type": "mixed_network_sweep",
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
            print(f"[skip] filename didn't match: {b}")
            continue
        tasks.append((m.group(1), int(m.group(2)), path))
    return tasks


def run_one(net, sim_rep, source_path, n_cores):
    out_path = os.path.join(OUTPUT_DIR, f"{net}_rep{sim_rep}_all_results.json")
    if os.path.exists(out_path):
        return out_path, "skipped (already exists)"

    matrix_path = os.path.join(TOPOLOGY_DIR, f"{net}.txt")
    n_genes = pd.read_csv(matrix_path, header=None).shape[0]
    base_config = dict(BASE_CONFIG)
    base_config["path_to_connectivity_matrix"] = matrix_path
    base_config["rows_to_use"] = [[0] * n_genes]

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

    gene_names = [f"gene_{i+1}" for i in range(n_genes)]

    record = {
        "dataset_id": net, "label": sim_rep, "source_file": source_path,
        "gene_names": gene_names, "n_genes": n_genes,
        **make_json_safe(results),
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(record, f, cls=NumpyEncoder, indent=2)
    return out_path, "computed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cores", type=int, default=4)
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=None, help="slice tasks[start:start+count] (for array jobs)")
    parser.add_argument("--count", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(f"{OUTPUT_DIR}/_scratch/logs", exist_ok=True)

    tasks = discover_tasks()
    if args.only:
        wanted = set(args.only.split(","))
        tasks = [t for t in tasks if f"{t[0]}:{t[1]}" in wanted]
    if args.start is not None:
        tasks = tasks[args.start:args.start + (args.count or len(tasks))]
    if args.limit:
        tasks = tasks[:args.limit]

    todo = [t for t in tasks
            if not os.path.exists(os.path.join(OUTPUT_DIR, f"{t[0]}_rep{t[1]}_all_results.json"))]
    print(f"{len(tasks)} discovered simulation(s), {len(todo)} without inference output yet", flush=True)
    if args.dry_run:
        for net, k, _ in tasks:
            done = os.path.exists(os.path.join(OUTPUT_DIR, f"{net}_rep{k}_all_results.json"))
            print(f"  {'done' if done else 'TODO'}  {net}_rep{k}")
        sys.exit(0)

    for i, (net, k, src) in enumerate(tasks):
        t0 = time.time()
        path, status = run_one(net, k, src, args.n_cores)
        print(f"[{i+1}/{len(tasks)}] {net} rep {k}: {status} ({time.time()-t0:.1f}s)", flush=True)
