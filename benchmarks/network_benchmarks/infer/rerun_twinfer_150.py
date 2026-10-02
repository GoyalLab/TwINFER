# [2026-09-30 PORTED to current twinfer.inference.infer.infer_with_twinfer (user instruction). Removed keyword args are kept as dated comments at the call.
#  BEHAVIOUR CHANGES vs the pre-reorg runs: (1) Step 1 now uses alpha_gene_gene_corr (default 0.01) with a clone/analytic null, not use_scramble/threshold_gene_gene_corr;
#  (2) merge_time_points and infer_direction_for_which_edges have no equivalent; (3) the result dict is NESTED (settings/classification/correlations/direction/...), not the old flat schema,
#  so old-schema scorers (twinfer_scoring_extract, score_mixed_network_sweep) cannot read the new JSON. NOT yet executed; smoke test pending. See REVIEW_LOG.md.]
"""
Full, clean TwINFER rerun against the finalized, SCODE-anchor-verified
150-file manifest (source_manifest_150.json) -- the exact same source
simulation files used to build the BEELINE inputs under
inputs/network_sweep_final_20260824/. Every parameter below is copied
verbatim from benchmark_network_sweep.ipynb's own Part 1 cell
(run_twinfer_with_fanout / base_configs), so results are directly comparable
to -- and supersede -- the earlier, mixed-provenance twinfer_analysis_output.csv.

Output: one {dataset_id}_label{N}_all_results.json per manifest entry, under
OUTPUT_DIR. Skips (does not recompute) any entry whose output file already
exists, so this script is safe to resume after a partial run.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import sys
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/home/gzu5140/TwINFER_KA/code/TwINFER/package")

import json
import os
import re
import argparse
import numpy as np
import pandas as pd
from twinfer.inference.infer import infer_with_twinfer

DEFAULT_MANIFEST_PATH = f'{TWINFER_PROJECT_ROOT}/analysis_data/synthetic_network_benchmark_20260824/source_manifest_150.json'
DEFAULT_OUTPUT_DIR = f'{TWINFER_PROJECT_ROOT}/analysis_data/synthetic_network_benchmark_20260824/twinfer_inference'
MANIFEST_PATH = DEFAULT_MANIFEST_PATH
OUTPUT_DIR = DEFAULT_OUTPUT_DIR

T1, T2 = 1, 20

# n_genes is read unconditionally from base_config["path_to_connectivity_matrix"]
# (infer.py L318), regardless of match_sim_details -- so the placeholder matrix's
# gene count MUST match the actual dataset's gene count, even though its topology
# content is otherwise unused. Every network_sweep_final/cyclic dataset here is
# 6-gene, matching the default placeholder below; autoregulation datasets are
# 2- or 3-gene and need their own connectivity matrix override (see
# CONNECTIVITY_MATRIX_OVERRIDE).
CONNECTIVITY_MATRIX_OVERRIDE = {
    "autoreg_g2_pos": (f'{TWINFER_PROJECT_ROOT}/input_data/autoreg_g2_pos.txt', 2),
    "autoreg_g2_neg": (f'{TWINFER_PROJECT_ROOT}/input_data/autoreg_g2_neg.txt', 2),
    "autoreg_g3_pos": (f'{TWINFER_PROJECT_ROOT}/input_data/autoreg_g3_pos.txt', 3),
    "autoreg_g3_neg": (f'{TWINFER_PROJECT_ROOT}/input_data/autoreg_g3_neg.txt', 3),
}

BASE_CONFIG = {
    'n_cells': 6000,
    'simulation_time_before_division': 6000,
    'twin_simulation_time_after_division': 48,
    'twin_measurement_resolution': 1,
    "path_to_connectivity_matrix": f'{TWINFER_PROJECT_ROOT}/input_data/network_sweep/grn_n6_e5_pos50_density_rep0.txt',
    "param_csv": f'{TWINFER_PROJECT_ROOT}/input_data/network_sweep/parameters.csv',
    "rows_to_use": [[0] * 6],
    "output_folder": f"{OUTPUT_DIR}/_scratch/",
    "log_file": f"{OUTPUT_DIR}/_scratch/logs/HSC.jsonl",
    "type": "HSC_balanced",
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


def run_one(dataset_id, label, source_path, n_cores):
    out_path = os.path.join(OUTPUT_DIR, f"{dataset_id}_label{label}_all_results.json")
    if os.path.exists(out_path):
        return out_path, "skipped (already exists)"

    base_config = BASE_CONFIG
    if dataset_id in CONNECTIVITY_MATRIX_OVERRIDE:
        matrix_path, n_genes_expected = CONNECTIVITY_MATRIX_OVERRIDE[dataset_id]
        base_config = dict(BASE_CONFIG)
        base_config["path_to_connectivity_matrix"] = matrix_path
        base_config["rows_to_use"] = [[0] * n_genes_expected]

    results = infer_with_twinfer(
        source_path,
        merge_to_multiple_states=False,
        base_config=base_config,
        t1=T1, t2=T2,
        check_for_steady_state=False,
        # show_scrambled_distribution_gene_correlation=False,   [2026-09-30 ported to current twinfer.inference.infer: keyword no longer exists]
        # plot_correlation_matrices_as_heatmap=False,   [2026-09-30 ported to current twinfer.inference.infer: keyword no longer exists]
        # return_gene_corr_thresholds=False,   [2026-09-30 ported to current twinfer.inference.infer: keyword no longer exists]
        match_sim_details=False,
        seed=101010,
        n_cores=n_cores,
        # use_scramble=False,   [2026-09-30 ported to current twinfer.inference.infer: keyword no longer exists]
        # merge_time_points=False,   [2026-09-30 ported to current twinfer.inference.infer: keyword no longer exists]
        z_score_threshold_two_states=4.501,
        # infer_direction_for_which_edges="all-edges",   [2026-09-30 ported to current twinfer.inference.infer: keyword no longer exists]
        ranked_list=True,
        separate_fan_outs_from_mutual_regulation_flag=True,
        fan_out_z_score_threshold=3.2,
        use_scramble_cross_correlation=False,
    )

    # n_genes = None   [2026-09-30 replaced: the current result dict is nested (no top-level DataFrames); gene list is in results["settings"]["genes"]]
    n_genes = len(results["settings"]["genes"])
    gene_names = [f"g{i+1}" for i in range(n_genes)] if n_genes is not None else None

    record = {
        "dataset_id": dataset_id, "label": label, "source_file": source_path,
        "gene_names": gene_names, "n_genes": n_genes,
        **make_json_safe(results),
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(record, f, cls=NumpyEncoder, indent=2)
    return out_path, "computed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cores", type=int, default=8)
    parser.add_argument("--only", type=str, default=None,
                         help="comma-separated dataset_id:label to run (for a quick test); default = all manifest entries")
    parser.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    MANIFEST_PATH = args.manifest
    OUTPUT_DIR = args.output_dir
    BASE_CONFIG["output_folder"] = f"{OUTPUT_DIR}/_scratch/"
    BASE_CONFIG["log_file"] = f"{OUTPUT_DIR}/_scratch/logs/HSC.jsonl"

    manifest = json.load(open(MANIFEST_PATH))
    manifest.pop("logs", None)

    tasks = []
    for ds, labels in manifest.items():
        for label_str, info in labels.items():
            if "source" not in info:
                continue
            tasks.append((ds, int(label_str), info["source"]))

    if args.only:
        wanted = set(args.only.split(","))
        tasks = [t for t in tasks if f"{t[0]}:{t[1]}" in wanted]

    print(f"Running {len(tasks)} TwINFER task(s)...", flush=True)
    for i, (ds, label, src) in enumerate(tasks):
        import time
        t0 = time.time()
        path, status = run_one(ds, label, src, args.n_cores)
        print(f"[{i+1}/{len(tasks)}] {ds} label {label}: {status} ({time.time()-t0:.1f}s) -> {path}", flush=True)
