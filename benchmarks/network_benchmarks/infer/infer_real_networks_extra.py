# [2026-09-30 PORTED to current twinfer.inference.infer.infer_with_twinfer (user instruction). Removed keyword args are kept as dated comments at the call.
#  BEHAVIOUR CHANGES vs the pre-reorg runs: (1) Step 1 now uses alpha_gene_gene_corr (default 0.01) with a clone/analytic null, not use_scramble/threshold_gene_gene_corr;
#  (2) merge_time_points and infer_direction_for_which_edges have no equivalent; (3) the result dict is NESTED (settings/classification/correlations/direction/...), not the old flat schema,
#  so old-schema scorers (twinfer_scoring_extract, score_mixed_network_sweep) cannot read the new JSON. NOT yet executed; smoke test pending. See REVIEW_LOG.md.]
"""
Modernized TwINFER-inference JSON generator for benchmark_twinfer_vs_boolode.ipynb's
TwINFER-simulation track.

Regenerates the `fanout_on`/`fanout_off` all_results.json pairs the notebook reads
(one per (network, rep), separate_fan_outs_from_mutual_regulation_flag True/False)
using the current `twinfer` package and its current parameter convention, for:
  - the original 4 networks (GSD, HSC, VSC, mCAD) -- overwritten in place, matching
    the exact (network, rep) set the existing JSONs cover (resolved by the 8-char hex
    hash embedded in both the JSON filename and the raw simulation CSV's filename),
    since their existing JSONs were built by the deprecated
    infer_network_simulation_real_network.py (legacy TwINFER_function_scripts import,
    stale z_score_threshold_two_states=12, scramble left at default-True).
  - the 4 new real-network datasets (B_cell_activation, Circadian_cycle, EMT,
    Pluripotent) -- freshly generated, one JSON pair per raw simulation replicate.

Does NOT touch the BoolODE track's separate JSON sources
(analysis_data/boolode_sims/twinfer_inference/, /scratch/gzu5140/ka_mi/z8_plain/).
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

from twinfer.inference.infer import infer_with_twinfer
from twinfer.utils.paths import get_data_root

DATA_ROOT = get_data_root()  # <TwINFER_KA>/analysis_data
PROJECT_ROOT = DATA_ROOT.parent  # <TwINFER_KA>
INPUT_DATA = PROJECT_ROOT / "input_data"

# [2026-10-01 commented out: /scratch/gzu5140 is purged scratch; see REPOINT_LOG.tsv] FANOUT_ON = "/scratch/gzu5140/twinfer_nb/fanout_on"
FANOUT_ON = f"{TWINFER_PROJECT_ROOT}/benchmarking_analysis/twinfer_nb/fanout_on"
# [2026-10-01 commented out: /scratch/gzu5140 is purged scratch; see REPOINT_LOG.tsv] FANOUT_OFF = "/scratch/gzu5140/twinfer_nb/fanout_off"
FANOUT_OFF = f"{TWINFER_PROJECT_ROOT}/benchmarking_analysis/twinfer_nb/fanout_off"
os.makedirs(FANOUT_ON, exist_ok=True)
os.makedirs(FANOUT_OFF, exist_ok=True)

LEGACY_RAW_DATA = f'{TWINFER_PROJECT_ROOT}/simulation_data/real_data'

T1, T2 = 1, 20

# --------------------------------------------------------------------------
# Per-network config
# --------------------------------------------------------------------------
NETWORKS = {
    "GSD": dict(n_genes=19, topology="GSD.txt", sim_time_before_division=6000, type_token="GSD", source="legacy"),
    "HSC": dict(n_genes=11, topology="HSC.txt", sim_time_before_division=6000, type_token="HSC", source="legacy"),
    "VSC": dict(n_genes=8, topology="VSC.txt", sim_time_before_division=6000, type_token="VSC", source="legacy"),
    "mCAD": dict(n_genes=5, topology="mCAD.txt", sim_time_before_division=6000, type_token="mCAD", source="legacy"),
    "B_cell_activation": dict(n_genes=10, topology="B_cell.txt", sim_time_before_division=1000, type_token="B_cell_activation", source="new"),
    "Circadian_cycle": dict(n_genes=4, topology="circadian.txt", sim_time_before_division=1000, type_token="Circadian", source="new"),
    "EMT": dict(n_genes=17, topology="EMT.txt", sim_time_before_division=1000, type_token="EMT", source="new"),
    "Pluripotent": dict(n_genes=36, topology="Pluripotent.txt", sim_time_before_division=1000, type_token="Pluripotent", source="new"),
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
        "type": cfg["type_token"],
    }


# --------------------------------------------------------------------------
# JSON serialization (unchanged from infer_network_simulation_real_network.py --
# package-independent)
# --------------------------------------------------------------------------
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
        return {
            "__type__": "DataFrame",
            "index": [str(i) for i in obj.index.tolist()],
            "columns": [str(c) for c in obj.columns.tolist()],
            "data": obj.values.tolist(),
        }
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


# --------------------------------------------------------------------------
# Task list: (net, analysis_key, raw_csv_path, base_config)
# --------------------------------------------------------------------------
def build_tasks():
    tasks = []

    for net, cfg in NETWORKS.items():
        base_config = make_base_config(net, cfg)

        if cfg["source"] == "legacy":
            # Reprocess exactly the (network, rep) set the existing JSONs cover --
            # resolved via the 8-char hex hash shared by the JSON filename and the
            # raw simulation CSV's filename.
            existing = sorted(glob.glob(f"{FANOUT_ON}/{net}_rep_*_all_results.json"))
            for j in existing:
                analysis_key = os.path.basename(j)[: -len("_all_results.json")]
                m = re.search(r"([0-9a-fA-F]{8})$", analysis_key)
                if not m:
                    print(f"[skip] couldn't extract hash from {j}")
                    continue
                h = m.group(1)
                matches = glob.glob(f"{LEGACY_RAW_DATA}/*_{net}_*{h}.csv")
                matches = [f for f in matches if not os.path.basename(f).startswith("simulation_before_division")]
                if len(matches) != 1:
                    print(f"[skip] expected 1 raw file for {net} hash {h}, found {len(matches)}: {matches}")
                    continue
                tasks.append((net, analysis_key, matches[0], base_config))

        else:
            sim_dir = DATA_ROOT / "paper_analysis" / net / "simulate" / "latest"
            sim_files = sorted(sim_dir.glob("df_*_ncells_*.csv"))
            sim_files = [f for f in sim_files if not f.name.startswith("simulation_before_division")]
            filename_re = re.compile(rf"^df_rows_.+_ncells_\d+_{re.escape(cfg['type_token'])}_rep_(\d+)_[0-9a-fA-F]{{8}}\.csv$")
            for f in sim_files:
                m = filename_re.match(f.name)
                if not m:
                    print(f"[skip] filename didn't match expected pattern: {f.name}")
                    continue
                rep = m.group(1)
                analysis_key = f"{net}_rep_{rep}"
                tasks.append((net, analysis_key, str(f), base_config))

    return tasks


# --------------------------------------------------------------------------
# Inference (current package, current parameter convention)
# --------------------------------------------------------------------------
def run_one(net, analysis_key, raw_csv_path, base_config):
    for fanout_flag, out_dir in [(True, FANOUT_ON), (False, FANOUT_OFF)]:
        try:
            results = infer_with_twinfer(
                raw_csv_path,
                merge_to_multiple_states=True,
                base_config=base_config,
                t1=T1, t2=T2,
                # use_scramble=False,   [2026-09-30 ported to current twinfer.inference.infer: keyword no longer exists]
                # threshold_gene_gene_corr=0.0126,   [2026-09-30 ported to current twinfer.inference.infer: keyword no longer exists]
                corr_threshold_cross_correlation=0.0421,
                z_score_threshold_two_states=4.51,
                use_scramble_cross_correlation=False,
                # merge_time_points=False,   [2026-09-30 ported to current twinfer.inference.infer: keyword no longer exists]
                check_for_steady_state=False,
                # show_scrambled_distribution_gene_correlation=True,   [2026-09-30 ported to current twinfer.inference.infer: keyword no longer exists]
                # plot_correlation_matrices_as_heatmap=False,   [2026-09-30 ported to current twinfer.inference.infer: keyword no longer exists]
                # return_gene_corr_thresholds=True,   [2026-09-30 ported to current twinfer.inference.infer: keyword no longer exists]
                match_sim_details=False,
                seed=101010,
                # infer_direction_for_which_edges="all-edges",   [2026-09-30 ported to current twinfer.inference.infer: keyword no longer exists]
                separate_fan_outs_from_mutual_regulation_flag=fanout_flag,
            )
        except Exception as e:
            print(f"Error on {net} {analysis_key} fanout={fanout_flag}: {e}")
            return f"{net}/{analysis_key}: FAILED ({e})"

        n_genes = len(base_config["rows_to_use"][0])
        record = {
            "sim_type": net,
            "analysis_key": analysis_key,
            "gene_names": [f"gene_{i+1}" for i in range(n_genes)],
            "fanout": fanout_flag,
            **make_json_safe(results),
        }
        out_path = os.path.join(out_dir, f"{analysis_key}_all_results.json")
        with open(out_path, "w") as f:
            json.dump(record, f, cls=NumpyEncoder, indent=2)

    return f"{net}/{analysis_key}: OK"


def main():
    tasks = build_tasks()
    print(f"Total (network, rep) tasks: {len(tasks)}")
    from collections import Counter
    print(Counter(t[0] for t in tasks))

    results = Parallel(n_jobs=8, backend="loky")(
        delayed(run_one)(net, analysis_key, raw_csv_path, base_config)
        for net, analysis_key, raw_csv_path, base_config in tasks
    )
    n_ok = sum(1 for r in results if r.endswith("OK"))
    n_fail = len(results) - n_ok
    print(f"Done: {n_ok} ok, {n_fail} failed")
    for r in results:
        if not r.endswith("OK"):
            print(r)


if __name__ == "__main__":
    main()
