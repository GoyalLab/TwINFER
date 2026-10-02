"""ALL_PAIRS unrestricted rerun of infer_network_simulation_real_network.py: same 65 replicates
across 7 networks (GSD/HSC/VSC/mCAD/B_cell_activation/Circadian_cycle/Pluripotent -- EMT has no
discoverable raw files, same as the original script), same everything, except
alpha_gene_gene_corr/alpha_stage3 -> 0.9999 and z_score_threshold_two_states/
z_score_threshold_cross_correlation -> 0 (matching LARRY's ALL_PAIRS=1 convention), plus
minimum_null_draws=5000. The original run (z_score_threshold_two_states=12, and predating the
current package's z_het/gated_regulation additions -- see score_real_networks_analytic_tuned.py's
docstring) has NO permutation z-scores at all (no gated_regulation key, only raw u_abs_rho_*
columns in twin_score_inputs) -- TODO4v2 needs z_reg_gated/z_het/z_gamma, so this reruns with the
current package to get them, unrestricted so the full pair universe is scored.

Output: analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs/
        <sim_type>_rep_<rep_id>_all_results.json (never overwrites the original run).

    python infer_real_network_allpairs.py [--n-jobs N] [--n-cores-per-task N] [--only NET] [--dry-run]
"""
import argparse
import glob
import json
import os
import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from twinfer.inference.infer import infer_with_twinfer
from twinfer.utils.paths import get_data_root

warnings.filterwarnings("ignore")

DATA_ROOT = get_data_root()
PROJECT_ROOT = DATA_ROOT.parent
INPUT_DATA = PROJECT_ROOT / "input_data"

LEGACY_SIM_DATA = PROJECT_ROOT / "simulation_data" / "real_data"
NEW_SIM_ROOT = DATA_ROOT / "paper_analysis"
BEELINE_LEGACY_INPUTS = Path("/scratch/gzu5140/twinfer_real/inputs")

OUTPUT_DIR = DATA_ROOT / "paper_analysis" / "real_networks" / "twinfer_inference_allpairs_t1_10"
os.makedirs(OUTPUT_DIR, exist_ok=True)

T1, T2 = 10, 20
ALPHA = 0.9999

NETWORKS = {
    "GSD": dict(n_genes=19, topology="GSD.txt", sim_time_before_division=6000, filename_token="GSD", source="legacy"),
    "HSC": dict(n_genes=11, topology="HSC.txt", sim_time_before_division=6000, filename_token="HSC_balanced", source="legacy"),
    "VSC": dict(n_genes=8, topology="VSC.txt", sim_time_before_division=6000, filename_token="VSC", source="legacy"),
    "mCAD": dict(n_genes=5, topology="mCAD.txt", sim_time_before_division=6000, filename_token="mCAD", source="legacy"),
    "B_cell_activation": dict(n_genes=10, topology="B_cell.txt", sim_time_before_division=1000, filename_token="B_cell_activation", source="new"),
    "Circadian_cycle": dict(n_genes=4, topology="circadian.txt", sim_time_before_division=1000, filename_token="Circadian", source="new"),
    "EMT": dict(n_genes=17, topology="EMT.txt", sim_time_before_division=1000, filename_token="EMT", source="new"),
    "Pluripotent": dict(n_genes=36, topology="Pluripotent.txt", sim_time_before_division=1000, filename_token="Pluripotent", source="new"),
}


def make_base_config(net, cfg):
    return {
        "n_cells": 6000,
        "simulation_time_before_division": cfg["sim_time_before_division"],
        "twin_simulation_time_after_division": 48,
        "twin_measurement_resolution": 1,
        "path_to_connectivity_matrix": str(INPUT_DATA / "real_world_networks" / cfg["topology"]),
        "param_csv": str(INPUT_DATA / "network_sweep" / "parameters.csv"),
        "rows_to_use": [[0] * cfg["n_genes"]],
        "type": net,
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


def build_simulation_record(path_to_simulation_file, sim_type, rep_id, base_config, t1, t2, output_path, n_cores):
    analysis_key = f"{sim_type}_rep_{rep_id}"
    out_path = os.path.join(output_path, f"{analysis_key}_all_results.json")
    if os.path.exists(out_path):
        return f"{analysis_key}: SKIPPED (exists)"
    try:
        results = infer_with_twinfer(
            path_to_simulation_file,
            merge_to_multiple_states=False,
            base_config=base_config,
            t1=t1, t2=t2,
            match_sim_details=False,
            check_for_steady_state=False,
            seed=101010,
            n_cores=n_cores,
            alpha_gene_gene_corr=ALPHA,
            alpha_stage3=ALPHA,
            z_score_threshold_two_states=0,
            z_score_threshold_cross_correlation=0,
            minimum_null_draws=5000,
            ranked_list=True,
        )
    except Exception as e:
        print(f"Error on {analysis_key}: {e}")
        return f"{analysis_key}: FAILED ({e})"

    n_genes = len(base_config["rows_to_use"][0])
    gene_names = [f"gene_{i + 1}" for i in range(n_genes)]
    record = {
        "sim_type": sim_type, "rep_id": rep_id, "analysis_key": analysis_key,
        "gene_names": gene_names, "n_genes": n_genes,
        **make_json_safe(results),
    }
    with open(out_path, "w") as f:
        json.dump(record, f, cls=NumpyEncoder, indent=2)
    print(f"Saved record to {out_path}")
    return f"{analysis_key}: OK"


def allowed_legacy_hashes(net):
    hash_re = re.compile(r"^simrep([0-9a-fA-F]{8})_spread$")
    hashes = set()
    for d in (BEELINE_LEGACY_INPUTS / net).glob("simrep*_spread"):
        m = hash_re.match(d.name)
        if m:
            hashes.add(m.group(1))
    if not hashes:
        raise ValueError(f"No BEELINE simrep*_spread folders found for {net}")
    return hashes


def build_legacy_tasks(net, cfg, base_config):
    token = cfg["filename_token"]
    file_re = re.compile(rf"^df_rows_.*_ncells_\d+_{re.escape(token)}_(\d+)_(\d+)_([0-9a-fA-F]{{8}})\.csv$")
    pattern = str(LEGACY_SIM_DATA / f"df_rows_*_ncells_*_{token}_*.csv")
    allowed_hashes = allowed_legacy_hashes(net)
    matched_hashes = set()
    tasks = []
    for f in sorted(glob.glob(pattern)):
        name = os.path.basename(f)
        if name.startswith("simulation_before_division"):
            continue
        m = file_re.match(name)
        if not m:
            continue
        config_idx, rep, file_hash = m.groups()
        if file_hash not in allowed_hashes:
            continue
        matched_hashes.add(file_hash)
        rep_id = f"{config_idx}_{rep}_{file_hash}"
        tasks.append((f, net, rep_id, base_config))
    return tasks


def build_new_tasks(net, cfg, base_config):
    token = cfg["filename_token"]
    sim_dir = NEW_SIM_ROOT / net / "simulate" / "latest"
    file_re = re.compile(rf"^df_rows_.*_ncells_\d+_{re.escape(token)}_rep_(\d+)_([0-9a-fA-F]{{8}})\.csv$")
    tasks = []
    for f in sorted(sim_dir.glob("df_*_ncells_*.csv")):
        if f.name.startswith("simulation_before_division"):
            continue
        m = file_re.match(f.name)
        if not m:
            continue
        rep, file_hash = m.groups()
        rep_id = f"{rep}_{file_hash}"
        tasks.append((str(f), net, rep_id, base_config))
    return tasks


def build_tasks(only=None):
    tasks = []
    for net, cfg in NETWORKS.items():
        if only and net != only:
            continue
        base_config = make_base_config(net, cfg)
        try:
            if cfg["source"] == "legacy":
                tasks.extend(build_legacy_tasks(net, cfg, base_config))
            else:
                tasks.extend(build_new_tasks(net, cfg, base_config))
        except ValueError as e:
            print(f"[skip {net}] {e}")
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--n-cores-per-task", type=int, default=8)
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    tasks = build_tasks(only=args.only)
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"Total replicate tasks: {len(tasks)}")
    print(Counter(t[1] for t in tasks))
    if args.dry_run:
        return

    results = Parallel(n_jobs=args.n_jobs, backend="loky")(
        delayed(build_simulation_record)(path, sim_type, rep_id, base_config, T1, T2, OUTPUT_DIR, args.n_cores_per_task)
        for path, sim_type, rep_id, base_config in tasks
    )
    n_ok = sum(1 for r in results if r.endswith("OK") or "SKIPPED" in r)
    n_fail = len(results) - n_ok
    print(f"Done: {n_ok} ok/skipped, {n_fail} failed")
    for r in results:
        if not (r.endswith("OK") or "SKIPPED" in r):
            print(r)


if __name__ == "__main__":
    main()
