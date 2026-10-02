# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""
Run the current twinfer.inference.infer.infer_with_twinfer pipeline (twinfer-code
conda env, editable install of code/TwINFER/package) on 3 replicates each of:
  - BoolODE-simulated HSC/mCAD (simulation_data/twinfer_format/<net>/replicate_{0,1,2}_simulation.csv)
  - TwINFER's own Gillespie-simulated HSC/mCAD, post-division twin data only
    (converted by convert_gillespie_to_twinfer_input.py into the same schema)

Scores inferred directed edges against the known ground-truth connectivity
matrix (simulation_data/twinfer_format/<net>/interaction_matrix.csv) and
reports precision/recall/F1 per (method, network, replicate).
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/home/gzu5140/TwINFER_KA/code/TwINFER/package")
from twinfer.inference.infer import infer_with_twinfer

BOOLODE_DIR = Path(f'{TWINFER_PROJECT_ROOT}/simulation_data/twinfer_format')
GILLESPIE_DIR = Path(f'{TWINFER_PROJECT_ROOT}/clean_data/benchmarks/boolode/gillespie_comparison/gillespie_twinfer_input')
SEEDED_DIR = GILLESPIE_DIR / "HSC_seeded"
OUT_DIR = Path(f'{TWINFER_PROJECT_ROOT}/clean_data/benchmarks/boolode/gillespie_comparison/inference_results')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# BoolODE t1/t2 convention from infer_network_simulation_boolode_sims.py
BOOLODE_TIMEPOINTS = {"HSC": dict(t1=500, t2=799), "mCAD": dict(t1=300, t2=499)}
# Gillespie post-division window is time_step 0..48; t=0 is the branch point
# itself (twins still identical there -> zero-variance/NaN), so t1 must be >0.
GILLESPIE_TIMEPOINTS = {"HSC": dict(t1=1, t2=48), "mCAD": dict(t1=1, t2=48)}

N_CORES = 8


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


def run_one(net, method, rep_idx, source_path, t1, t2, matrix_path, n_genes):
    tag = f"{net}_{method}_rep{rep_idx}"
    out_path = OUT_DIR / f"{tag}_all_results.json"
    if out_path.exists():
        print(f"[skip] {tag} already exists")
        return out_path

    base_config = {
        "n_cells": 6000,
        "twin_simulation_time_after_division": t2,
        "twin_measurement_resolution": 100,
        "path_to_connectivity_matrix": str(matrix_path),
        "param_csv": str(BOOLODE_DIR / net / "param_placeholder.csv"),
        "rows_to_use": [[0]],
        "output_folder": str(OUT_DIR / "_scratch"),
        "log_file": str(OUT_DIR / "_scratch" / "logs" / f"{tag}.jsonl"),
        "type": f"{net}_{method}",
        "combinatorial_interaction_type": "additive",
        "number_of_parallel_parameters": 1,
        "number_of_cores_per_parameter": N_CORES,
        "log_pi_on": False,
    }
    (OUT_DIR / "_scratch" / "logs").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    results = infer_with_twinfer(
        str(source_path),
        merge_to_multiple_states=False,
        base_config=base_config,
        t1=t1, t2=t2,
        check_for_steady_state=False,
        match_sim_details=False,
        seed=101010,
        n_cores=N_CORES,
        z_score_threshold_two_states=4.501,
        ranked_list=True,
        separate_fan_outs_from_mutual_regulation_flag=True,
        fan_out_z_score_threshold=3.2,
        use_scramble_cross_correlation=False,
    )
    dt = time.time() - t0

    gene_names = [f"gene_{i+1}" for i in range(n_genes)]
    record = {
        "dataset_id": net, "method": method, "rep": rep_idx,
        "source_file": str(source_path), "gene_names": gene_names, "n_genes": n_genes,
        "runtime_sec": dt,
        **make_json_safe(results),
    }
    with open(out_path, "w") as f:
        json.dump(record, f, cls=NumpyEncoder, indent=2)
    print(f"[done] {tag}: {dt:.1f}s -> {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None,
                        help="comma-separated <net>_<method>_rep<k>, e.g. mCAD_boolode_rep0")
    args = parser.parse_args()

    tasks = []
    for net, gene_order_file_genes in [("HSC", 11), ("mCAD", 5)]:
        matrix_path = BOOLODE_DIR / net / "interaction_matrix.csv"
        for rep_idx in range(3):
            tasks.append((net, "boolode", rep_idx,
                          BOOLODE_DIR / net / f"replicate_{rep_idx}_simulation.csv",
                          BOOLODE_TIMEPOINTS[net]["t1"], BOOLODE_TIMEPOINTS[net]["t2"],
                          matrix_path, gene_order_file_genes))
            tasks.append((net, "gillespie", rep_idx,
                          GILLESPIE_DIR / net / f"replicate_{rep_idx}_simulation.csv",
                          GILLESPIE_TIMEPOINTS[net]["t1"], GILLESPIE_TIMEPOINTS[net]["t2"],
                          matrix_path, gene_order_file_genes))
        if net == "HSC":
            # Seeded (multi-state) HSC Gillespie data -- this is the intended
            # Gillespie-side comparison partner for BoolODE HSC (plain/empty-IC
            # HSC collapses to a single state + has 2 permanently-off genes,
            # making it a poor match for BoolODE's multi-branch dynamics).
            for rep_idx in range(3):
                tasks.append(("HSC", "gillespie_seeded", rep_idx,
                              SEEDED_DIR / f"replicate_{rep_idx}_simulation.csv",
                              GILLESPIE_TIMEPOINTS["HSC"]["t1"], GILLESPIE_TIMEPOINTS["HSC"]["t2"],
                              matrix_path, gene_order_file_genes))

    if args.only:
        wanted = set(args.only.split(","))
        tasks = [t for t in tasks if f"{t[0]}_{t[1]}_rep{t[2]}" in wanted]

    for net, method, rep_idx, src, t1, t2, matrix_path, n_genes in tasks:
        run_one(net, method, rep_idx, src, t1, t2, matrix_path, n_genes)
