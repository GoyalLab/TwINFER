# [2026-09-30 PORTED to current twinfer.inference.infer.infer_with_twinfer (user instruction). Removed keyword args are kept as dated comments at the call.
#  BEHAVIOUR CHANGES vs the pre-reorg runs: (1) Step 1 now uses alpha_gene_gene_corr (default 0.01) with a clone/analytic null, not use_scramble/threshold_gene_gene_corr;
#  (2) merge_time_points and infer_direction_for_which_edges have no equivalent; (3) the result dict is NESTED (settings/classification/correlations/direction/...), not the old flat schema,
#  so old-schema scorers (twinfer_scoring_extract, score_mixed_network_sweep) cannot read the new JSON. NOT yet executed; smoke test pending. See REVIEW_LOG.md.]
"""
Reruns TwINFER inference (with fan-out separation) on the cyclic_g3/g4/g5/g6 raw
simulations, mirroring benchmark_network_sweep.ipynb's Part 1 rerun
(run_twinfer_with_fanout / infer_with_twinfer call with the same kwargs) so the
resulting *_all_results.json files are structurally identical to that notebook's
network_sweep_final output and can be scored with its existing
score_twinfer_dataset / score_twinfer_sweep_folder machinery unmodified.

cyclic_g3/g4/g5 raw sims live under simulation_data/synthetic_network/cyclic_g{n}/
(newer driver script, "..._rep_<n>_<hash>.csv" naming). cyclic_g6 reuses an older run
under simulation_data/cyclic_6_nodes/ with a different filename convention -- see
SIM_OVERRIDES below.

40 total runs (4 topologies x 10 reps) -- small enough (3-6 genes each) to run
directly here rather than via SLURM.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]

import glob
import json
import os
import re

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

import twinfer
from twinfer.inference.infer import infer_with_twinfer

PROJECT_ROOT = f'{TWINFER_PROJECT_ROOT}'
SIM_ROOT = f"{PROJECT_ROOT}/simulation_data/synthetic_network"
OUTPUT_DIR = f"{PROJECT_ROOT}/analysis_data/paper_analysis/cyclic_g3456/twinfer_inference"
PARAM_CSV = f"{PROJECT_ROOT}/input_data/network_sweep/parameters.csv"

T1, T2 = 1, 20
TOPOLOGIES = [3, 4, 5, 6]


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy scalar/array types infer_with_twinfer's result dict contains."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def make_json_safe(obj):
    """Recursively converts infer_with_twinfer's result dict (DataFrames, sets, tuple-keyed
    dicts) into a JSON-serializable structure. Verbatim copy of the helper in
    benchmark_network_sweep.ipynb so output JSONs are structurally identical."""
    if isinstance(obj, pd.DataFrame):
        return {
            "__type__": "DataFrame",
            "index": [str(i) for i in obj.index.tolist()],
            "columns": [str(c) for c in obj.columns.tolist()],
            "data": obj.values.tolist(),
        }
    if isinstance(obj, pd.Series):
        return {"__type__": "Series", "index": [str(i) for i in obj.index.tolist()], "data": obj.values.tolist()}
    if isinstance(obj, dict):
        return {
            ("__".join(map(str, k)) if isinstance(k, tuple) else str(k)): make_json_safe(v)
            for k, v in obj.items()
        }
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


def sim_filename_re(type_token: str) -> re.Pattern:
    return re.compile(
        rf"^df_rows_.+_ncells_\d+_{re.escape(type_token)}_rep_(\d+)_[0-9a-fA-F]{{8}}\.csv$"
    )


def run_twinfer_with_fanout(path_to_simulation_file, sim_type, rep_id, base_config, t1, t2, output_path):
    """Same kwargs as benchmark_network_sweep.ipynb's run_twinfer_with_fanout."""
    analysis_key = f"{sim_type}_rep_{rep_id}"
    f_result_path = os.path.join(output_path, f"{analysis_key}_all_results.json")
    if os.path.exists(f_result_path):
        return f_result_path  # skip -- already computed

    results = infer_with_twinfer(
        path_to_simulation_file,
        merge_to_multiple_states=False,
        base_config=base_config,
        t1=t1,
        t2=t2,
        check_for_steady_state=False,
        # show_scrambled_distribution_gene_correlation=False,   [2026-09-30 ported to current twinfer.inference.infer: keyword no longer exists]
        # plot_correlation_matrices_as_heatmap=False,   [2026-09-30 ported to current twinfer.inference.infer: keyword no longer exists]
        # return_gene_corr_thresholds=False,   [2026-09-30 ported to current twinfer.inference.infer: keyword no longer exists]
        match_sim_details=False,
        seed=101010,
        n_cores=15,
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
    gene_names = [f"gene_{i+1}" for i in range(n_genes)] if n_genes is not None else None

    record = {
        "sim_type": sim_type, "rep_id": rep_id, "analysis_key": analysis_key,
        "gene_names": gene_names, "n_genes": n_genes,
        **make_json_safe(results),
    }
    os.makedirs(output_path, exist_ok=True)
    with open(f_result_path, "w") as f:
        json.dump(record, f, cls=NumpyEncoder, indent=2)
    print(f"Saved {f_result_path}")
    return f_result_path


# cyclic_g6 reuses an older run (simulation_data/cyclic_6_nodes/) that pre-dates the
# cyclic_g3/g4/g5 driver script and uses the older "{config_index}_{replicate}" filename
# convention (e.g. "..._cycle_6_node_0_5_<hash>.csv") instead of "..._rep_<n>_<hash>.csv"
# -- same override pattern as twinfer_to_boolode_format.py's DATASETS["cyclic_g6"].
SIM_OVERRIDES = {
    6: dict(
        sim_dir=f"{PROJECT_ROOT}/simulation_data/cyclic_6_nodes",
        glob_pattern="df_rows_*_ncells_*_cycle_6_node_*.csv",
        rep_regex=re.compile(r"^df_rows_.+_ncells_\d+_cycle_6_node_\d+_(\d+)_[0-9a-fA-F]{8}\.csv$"),
    ),
}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tasks = []
    for n_genes in TOPOLOGIES:
        sim_type = f"cyclic_g{n_genes}"
        base_config = {
            "n_cells": 6000,
            "twin_simulation_time_after_division": 48,
            "twin_measurement_resolution": 1,
            "path_to_connectivity_matrix": f"{PROJECT_ROOT}/input_data/cycle_g{n_genes}.txt",
            "param_csv": PARAM_CSV,
            "rows_to_use": [[0] * n_genes],
        }
        override = SIM_OVERRIDES.get(n_genes)
        if override:
            sim_dir = override["sim_dir"]
            pattern = os.path.join(sim_dir, override["glob_pattern"])
            regex = override["rep_regex"]
        else:
            sim_dir = os.path.join(SIM_ROOT, sim_type)
            pattern = os.path.join(sim_dir, f"df_rows_*_ncells_*_{sim_type}_rep_*.csv")
            regex = sim_filename_re(sim_type)
        for f in sorted(glob.glob(pattern)):
            m = regex.match(os.path.basename(f))
            if not m:
                print(f"[skip] filename didn't match: {f}")
                continue
            rep_id = m.group(1)
            tasks.append((f, sim_type, rep_id, base_config))

    print(f"Collected {len(tasks)} TwINFER rerun tasks.")

    result_paths = Parallel(n_jobs=1, backend="loky")(
        delayed(run_twinfer_with_fanout)(path, sim_type, rep_id, base_config, T1, T2, OUTPUT_DIR)
        for path, sim_type, rep_id, base_config in tqdm(tasks)
    )
    print(f"Done. {len(result_paths)} result file(s) available in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
