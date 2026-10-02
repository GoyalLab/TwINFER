"""
Full TwINFER inference over every real-network simulation replicate, with every
COVERAGE-controlling significance/threshold gate in the pipeline disabled so
~every directed pair reaches ranked_edges/twinScore instead of only the pairs
that clear Step 1's correlation-significance screen (see infer.py:
check_gene_gene_correlation_threshold uses z_critical = norm.ppf(1 -
alpha_gene_gene_corr/2), so alpha=1.0 -> z_critical=0, not alpha=0 which would
make z_critical infinite and let NOTHING through -- the opposite of "no
filtering"; alpha must also be strictly <1, hence 0.999999). Concretely,
relative to the filtered driver (infer_network_simulation_real_network.py):
  alpha_gene_gene_corr=0.999999       (was 0.01; Step 1 undirected screen)
  alpha_stage3=0.999999               (was 0.01; Stage 3 gate)
  z_score_threshold_two_states=0      (was 12 override / 5 default)
z_score_threshold_cross_correlation, corr_threshold_cross_correlation, and
fan_out_z_score_threshold are left at their DEFAULTS (2.5 / 0.0421 / 8): those
control Stage 4/5's actual is_final_directed_edge edge-call, not candidate-
panel coverage. Zeroing them (tried first, on HSC_multistate/seeded) made
TwINFER call literally every pair an edge (precision 0.209, recall 1.0) --
harmless to our magnitude-ranked AUPRC/F1 scoring (which never reads that
column) but pointless, since it destroys that column's own meaning for zero
scoring benefit. fan_out_z_score_threshold is moot regardless: it only feeds
Stage 5, which never runs here (separate_fan_outs_from_mutual_regulation_flag
defaults to False and this driver never overrides it).
Output goes to a separate twinfer_inference_nofilter/ directory so the original
(filtered) results are preserved for comparison, not overwritten.

Modernized to use the current `twinfer` package (twinfer.inference.infer.
infer_with_twinfer) instead of the deprecated `TwINFER_function_scripts`
import and hardcoded Keerthana_b1042 paths. Input/output now follow the
twinfer.utils.paths convention (get_data_root()), and every raw replicate
found on disk is processed -- not a fixed subset.

Networks covered, and where their raw simulation CSVs live:
  - GSD, HSC, VSC, mCAD: simulation_data/real_data/, restricted to exactly
    the replicates BEELINE was run on, so TwINFER and BEELINE are compared
    on identical input. BEELINE's run for these 4 networks lives at
    /scratch/gzu5140/twinfer_real/inputs/<net>/simrep<hash>_*, one folder
    per raw file's 8-hex-char simulator hash -- NOT
    code/Beeline/inputs/<net>, which is a separate, unrelated BoolODE
    simulation track (different simulator, real gene-symbol schema,
    explicitly marked "unused" for TwINFER comparison in
    score_real_networks_summary.py, and not even in TwINFER's canonical
    clone_id/cell_id/time_step/gene_N_mRNA schema).
  - B_cell_activation, Circadian_cycle, EMT, Pluripotent:
    analysis_data/paper_analysis/<network>/simulate/latest/, every
    replicate on disk (BEELINE's code/Beeline/inputs/real_networks/<net>
    run covers fewer reps for EMT/Pluripotent, but the full on-disk set is
    used here regardless).

infer_with_twinfer defaults to ranked_list=True, which computes TwinScore
for every directed candidate pair (result["twin_score_inputs"] and
result["ranked_edges"], the latter carrying the "twinScore" column). Both
are captured verbatim -- along with every other key infer_with_twinfer
returns -- in each output JSON via make_json_safe(results); nothing is
filtered or renamed.

Each replicate's rep_id is derived from its raw filename's config/rep index
AND its 8-hex-char simulator hash, not just the config/rep index alone --
multiple raw files can share the same config/rep index with different
hashes, and using only the index would silently overwrite one replicate's
output JSON with another's. Verified empirically: no two raw files for the
same network share a hash.

Output: one <analysis_key>_all_results.json per replicate, written to
analysis_data/paper_analysis/real_networks/twinfer_inference/.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-10-01 added: project root for benchmarking_analysis paths]
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

# BEELINE's TwINFER-vs-BEELINE comparison track for GSD/HSC/mCAD/VSC -- one
# simrep<hash>_{spread,twin_paired} folder per raw file's 8-hex-char
# simulator hash. Legacy task discovery is restricted to these hashes so
# TwINFER is run on exactly the replicates BEELINE was run on.
# [2026-10-01 commented out: /scratch/gzu5140 is purged scratch; see REPOINT_LOG.tsv] BEELINE_LEGACY_INPUTS = Path("/scratch/gzu5140/twinfer_real/inputs")
BEELINE_LEGACY_INPUTS = Path(f"{TWINFER_PROJECT_ROOT}/benchmarking_analysis/twinfer_real/inputs")

OUTPUT_DIR = DATA_ROOT / "paper_analysis" / "real_networks" / "twinfer_inference_nofilter"
os.makedirs(OUTPUT_DIR, exist_ok=True)

T1, T2 = 1, 20

N_JOBS_OUTER = 4
N_CORES_PER_TASK = 15

# --------------------------------------------------------------------------
# Per-network config.
#
# filename_token is the literal network label embedded in each raw CSV's
# filename (legacy: "..._ncells_<N>_<token>_<config>_<rep>_<hash>.csv"; new:
# "..._ncells_<N>_<token>_rep_<rep>_<hash>.csv"). n_genes must match the
# connectivity matrix's row/column count.
# --------------------------------------------------------------------------
NETWORKS = {
    "GSD": dict(n_genes=19, topology="GSD.txt", sim_time_before_division=6000, filename_token="GSD", source="legacy"),
    "HSC": dict(n_genes=11, topology="HSC.txt", sim_time_before_division=6000, filename_token="HSC_balanced", source="legacy"),
    "VSC": dict(n_genes=8, topology="VSC.txt", sim_time_before_division=6000, filename_token="VSC", source="legacy"),
    "mCAD": dict(n_genes=5, topology="mCAD.txt", sim_time_before_division=6000, filename_token="mCAD", source="legacy"),
    "B_cell_activation": dict(n_genes=10, topology="B_cell.txt", sim_time_before_division=1000, filename_token="B_cell_activation", source="new"),
    "Circadian_cycle": dict(n_genes=4, topology="circadian.txt", sim_time_before_division=1000, filename_token="Circadian", source="new"),
    # "latest" for EMT has since been repointed to multistate_2000steps by the
    # later multi-state simulation work in this session (its raw filenames also
    # carry "_multistate_" so wouldn't match this token's regex anyway) -- the
    # original 10-replicate single-state run this driver needs still lives at
    # this specific run-tag directory, so point there explicitly rather than
    # via "latest" (and rather than repointing the symlink, which the
    # multistate driver/scripts may still rely on).
    "EMT": dict(n_genes=17, topology="EMT.txt", sim_time_before_division=1000, filename_token="EMT", source="new", sim_dir_override="20260825_224653"),
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


# --------------------------------------------------------------------------
# JSON serialization (package-independent).
# --------------------------------------------------------------------------
class NumpyEncoder(json.JSONEncoder):
    """
    JSON encoder that handles numpy scalar and array types, which the
    standard json module cannot serialize natively (np.int64, np.float64,
    np.ndarray all raise TypeError otherwise).
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def make_json_safe(obj):
    """
    Recursively converts an arbitrary nested object into a JSON-serializable
    structure. Must be recursive: results dicts from infer_with_twinfer
    contain DataFrames (including ranked_edges/twin_score_inputs), sets, and
    dicts with tuple keys, sometimes nested inside other dicts/lists, so a
    single top-level type check is not sufficient.
    """
    if isinstance(obj, pd.DataFrame):
        return {
            "__type__": "DataFrame",
            "index": [str(i) for i in obj.index.tolist()],
            "columns": [str(c) for c in obj.columns.tolist()],
            "data": obj.values.tolist(),
        }
    if isinstance(obj, pd.Series):
        return {
            "__type__": "Series",
            "index": [str(i) for i in obj.index.tolist()],
            "data": obj.values.tolist(),
        }
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


def build_simulation_record(path_to_simulation_file, sim_type, rep_id, base_config, t1, t2, output_path):
    """
    Runs one replicate through TwINFER and saves the entire results dict
    infer_with_twinfer returns -- settings, classification, directed_edges,
    correlations, direction, stage3, fan_out, twin_score_inputs, and
    ranked_edges -- to a single JSON file, unfiltered.

    Returns a short status string instead of raising, so one bad replicate
    does not abort the rest of a full-scale parallel run.
    """
    analysis_key = f"{sim_type}_rep_{rep_id}"
    try:
        results = infer_with_twinfer(
            path_to_simulation_file,
            merge_to_multiple_states=False,
            base_config=base_config,
            t1=t1,
            t2=t2,
            match_sim_details=False,
            check_for_steady_state=False,
            seed=101010,
            n_cores=N_CORES_PER_TASK,
            # Coverage gates disabled -- see module docstring. Stage 4's actual
            # edge-call thresholds (z_score_threshold_cross_correlation,
            # corr_threshold_cross_correlation) are left at their defaults.
            alpha_gene_gene_corr=0.999999,
            alpha_stage3=0.999999,
            z_score_threshold_two_states=0,
            ranked_list=True,
        )
    except Exception as e:
        print(f"Error on {analysis_key}: {e}")
        return f"{analysis_key}: FAILED ({e})"

    n_genes = len(base_config["rows_to_use"][0])
    gene_names = [f"gene_{i + 1}" for i in range(n_genes)]

    record = {
        "sim_type": sim_type,
        "rep_id": rep_id,
        "analysis_key": analysis_key,
        "gene_names": gene_names,
        "n_genes": n_genes,
        **make_json_safe(results),
    }

    f_result_path = os.path.join(output_path, f"{analysis_key}_all_results.json")
    with open(f_result_path, "w") as f:
        json.dump(record, f, cls=NumpyEncoder, indent=2)

    print(f"Saved record to {f_result_path}")
    return f"{analysis_key}: OK"


# --------------------------------------------------------------------------
# Task discovery: every raw replicate CSV on disk, per network. rep_id
# always includes the raw filename's simulator hash so two files that
# happen to share the same config/rep index never collide on output name.
# --------------------------------------------------------------------------
def allowed_legacy_hashes(net):
    """
    8-hex-char simulator hashes of the raw files BEELINE was actually run on
    for this legacy network, read from BEELINE_LEGACY_INPUTS/<net>/
    simrep<hash>_spread. Both _spread and _twin_paired folders exist per
    hash; _spread alone already gives the full hash set.
    """
    hash_re = re.compile(r"^simrep([0-9a-fA-F]{8})_spread$")
    hashes = set()
    for d in (BEELINE_LEGACY_INPUTS / net).glob("simrep*_spread"):
        m = hash_re.match(d.name)
        if m:
            hashes.add(m.group(1))
    if not hashes:
        raise ValueError(
            f"No BEELINE simrep*_spread folders found for {net} under "
            f"{BEELINE_LEGACY_INPUTS / net}; cannot restrict to BEELINE's "
            "replicate set."
        )
    return hashes


def build_legacy_tasks(net, cfg, base_config):
    token = cfg["filename_token"]
    file_re = re.compile(
        rf"^df_rows_.*_ncells_\d+_{re.escape(token)}_(\d+)_(\d+)_([0-9a-fA-F]{{8}})\.csv$"
    )
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

    missing = allowed_hashes - matched_hashes
    if missing:
        raise ValueError(
            f"{net}: BEELINE simrep hashes {sorted(missing)} have no "
            f"matching raw file under {LEGACY_SIM_DATA}."
        )
    return tasks


def build_new_tasks(net, cfg, base_config):
    token = cfg["filename_token"]
    sim_dir = NEW_SIM_ROOT / net / "simulate" / cfg.get("sim_dir_override", "latest")
    file_re = re.compile(
        rf"^df_rows_.*_ncells_\d+_{re.escape(token)}_rep_(\d+)_([0-9a-fA-F]{{8}})\.csv$"
    )

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


def build_tasks():
    tasks = []
    for net, cfg in NETWORKS.items():
        base_config = make_base_config(net, cfg)
        if cfg["source"] == "legacy":
            tasks.extend(build_legacy_tasks(net, cfg, base_config))
        else:
            tasks.extend(build_new_tasks(net, cfg, base_config))

    rep_ids_seen = Counter((net, rep_id) for _, net, rep_id, _ in tasks)
    duplicates = [key for key, count in rep_ids_seen.items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate (network, rep_id) task keys found: {duplicates}")

    return tasks


def main():
    tasks = build_tasks()
    print(f"Total replicate tasks: {len(tasks)}")
    print(Counter(t[1] for t in tasks))

    print("Starting parallel processing...")
    results = Parallel(n_jobs=N_JOBS_OUTER, backend="loky")(
        delayed(build_simulation_record)(path, sim_type, rep_id, base_config, T1, T2, OUTPUT_DIR)
        for path, sim_type, rep_id, base_config in tasks
    )

    n_ok = sum(1 for r in results if r.endswith("OK"))
    n_fail = len(results) - n_ok
    print(f"Done: {n_ok} ok, {n_fail} failed")
    for r in results:
        if not r.endswith("OK"):
            print(r)


if __name__ == "__main__":
    main()
