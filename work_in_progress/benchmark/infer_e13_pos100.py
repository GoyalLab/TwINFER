"""
TwINFER inference over the 3 x 10 e13_pos100 simulations at
simulation_data/network_sweep_final/e13_pos100/ -- these are 3 of the
network_sweep_final OFAT-sweep "center" topologies (density=13, sign_ratio=1.0)
that were simulated separately from the main 15-network/150-run benchmark and
never went through TwINFER/BEELINE inference.

Ground truth mapping (topology reconstructed from each sim's logged K_/k_add_
parameter keys and matched byte-for-byte against the 3 known
input_data/network_sweep_final/grn_n6_e13_pos100_center_rep{0,1,2}.txt
matrices -- confirmed exact):

    sim label "rep1" -> grn_n6_e13_pos100_center_rep1.txt
    sim label "rep2" -> grn_n6_e13_pos100_center_rep2.txt
    sim label "rep3" -> grn_n6_e13_pos100_center_rep0.txt   (renamed from the
                         original "rep20" sim-file label, which was a mangled
                         topology-rep-0 identifier -- kept as its own "rep3"
                         dataset id per instruction, mapped to its correct
                         (rep0) ground truth for scoring)

Uses the same infer_with_twinfer settings as infer_mixed_network_sweep.py
(current package signature -- see memory: reference_twinfer_infer_kwargs_stale).

Output: one <label>_rep<k>_all_results.json per simulation under OUTPUT_DIR.
Skips any entry whose output already exists.

    python infer_e13_pos100.py
"""
import glob
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/gzu5140/TwINFER_KA/code/TwINFER/package")
from twinfer.inference.infer import infer_with_twinfer

PROJECT_ROOT = "/home/gzu5140/TwINFER_KA"
SIM_DIR = f"{PROJECT_ROOT}/simulation_data/network_sweep_final/e13_pos100"
GT_DIR = f"{PROJECT_ROOT}/input_data/network_sweep_final"
OUTPUT_DIR = f"{PROJECT_ROOT}/analysis_data/network_sweep_final/e13_pos100/twinfer_inference"
PARAM_CSV = f"{PROJECT_ROOT}/input_data/network_sweep/parameters.csv"

T1, T2 = 1, 20
N_GENES = 6

LABEL_TO_GT = {
    "n6_e13_pos100_rep1": f"{GT_DIR}/grn_n6_e13_pos100_center_rep1.txt",
    "n6_e13_pos100_rep2": f"{GT_DIR}/grn_n6_e13_pos100_center_rep2.txt",
    "n6_e13_pos100_rep3": f"{GT_DIR}/grn_n6_e13_pos100_center_rep0.txt",
}

SIM_RE = re.compile(r"^df_rows_.+?_ncells_6000_(n6_e13_pos100_rep[123])_rep_(\d+)_[0-9a-fA-F]+\.csv$")

BASE_CONFIG = {
    "n_cells": 6000,
    "simulation_time_before_division": 6000,
    "twin_simulation_time_after_division": 48,
    "twin_measurement_resolution": 1,
    "param_csv": PARAM_CSV,
    "rows_to_use": [[0] * N_GENES],
    "output_folder": f"{OUTPUT_DIR}/_scratch/",
    "log_file": f"{OUTPUT_DIR}/_scratch/logs/e13_pos100.jsonl",
    "type": "e13_pos100",
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


def run_one(label, sim_rep, source_path, n_cores):
    out_path = os.path.join(OUTPUT_DIR, f"{label}_rep{sim_rep}_all_results.json")
    if os.path.exists(out_path):
        return out_path, "skipped (already exists)"

    matrix_path = LABEL_TO_GT[label]
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
        z_score_threshold_two_states=4.501,
        ranked_list=True,
        separate_fan_outs_from_mutual_regulation_flag=True,
        fan_out_z_score_threshold=3.2,
        use_scramble_cross_correlation=False,
    )

    gene_names = [f"gene_{i+1}" for i in range(N_GENES)]
    record = {
        "dataset_id": label, "label": sim_rep, "source_file": source_path,
        "gene_names": gene_names, "n_genes": N_GENES,
        "ground_truth_matrix": matrix_path,
        **make_json_safe(results),
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(record, f, cls=NumpyEncoder, indent=2)
    return out_path, "computed"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cores", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.makedirs(f"{OUTPUT_DIR}/_scratch/logs", exist_ok=True)
    tasks = discover_tasks()
    print(f"{len(tasks)} simulation(s) discovered", flush=True)
    if args.dry_run:
        for label, k, _ in tasks:
            done = os.path.exists(os.path.join(OUTPUT_DIR, f"{label}_rep{k}_all_results.json"))
            print(f"  {'done' if done else 'TODO'}  {label}_rep{k} -> {LABEL_TO_GT[label]}")
        sys.exit(0)

    for i, (label, k, src) in enumerate(tasks):
        t0 = time.time()
        path, status = run_one(label, k, src, args.n_cores)
        print(f"[{i+1}/{len(tasks)}] {label} rep {k}: {status} ({time.time()-t0:.1f}s)", flush=True)
