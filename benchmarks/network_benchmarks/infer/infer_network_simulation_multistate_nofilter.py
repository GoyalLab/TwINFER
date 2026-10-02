"""
TwINFER inference (infer_with_twinfer) over the new multi-state real-network
simulations: the IC-unset ("multistate") and seeded-IC ("seeded") reruns of
GSD/HSC/EMT/VSC/mCAD at per-network tuned (Hill n, k_add) -- see
code/TwINFER/paper_analysis/heterogeneity_vs_regulation/real_network_multistate_sim.py
and real_network_seeded_sim.py for how these were generated.

Sibling of infer_network_simulation_real_network.py (same infer_with_twinfer
call, same output JSON schema); kept separate rather than folded into that
script's NETWORKS dict because these runs use different sim_dirs (per-variant
run_tag subfolders, not "latest") and different sim_time_before_division
per variant (2000/800 unset vs seeded, 6000/1500 for VSC).

No-filter variant: every significance/threshold gate that controls candidate-
panel COVERAGE is disabled, same rationale and same parameter values as
infer_network_simulation_real_network_nofilter.py (see that file's docstring):
  alpha_gene_gene_corr=0.999999   (Step 1 undirected screen; alpha must be <1)
  alpha_stage3=0.999999           (Stage 3 gate; alpha must be <1)
  z_score_threshold_two_states=0
Confirmed on HSC_multistate/HSC_seeded rep 0: ranked_edges/twinScore coverage
went from 47%/33% to 100%/100%, AUPRC(twinScore) 0.129->0.302 / 0.251->0.338,
F1(topk) 0.043->0.298 / 0.213->0.340 -- a substantial real improvement, not
noise, strongly suggesting the original "TwINFER ranks last on every new
multistate/seeded dataset" finding (score_multistate_benchmark.py, scored via
direction.unfiltered_matrix) was largely a coverage artifact: unfiltered_matrix
is NOT full-coverage-always as previously assumed -- tracing infer.py shows it
is built over direction_gene_pairs, which comes from the same Step 1-3
direction_candidates pool as ranked_edges, so it inherits the exact same gate.
z_score_threshold_cross_correlation/corr_threshold_cross_correlation are left
at their DEFAULTS here (2.5 / 0.0421), unlike the real_network_nofilter driver
-- those control Stage 4's actual is_final_directed_edge edge-call, not
candidate-panel coverage; zeroing them (as first tried) made TwINFER call
every single pair an edge (precision 0.209, recall 1.0), which doesn't affect
our magnitude-ranked AUPRC/F1 scoring but destroys that column's own meaning
for no benefit, so there's no reason to zero them here.
fan_out_z_score_threshold is irrelevant either way: separate_fan_outs_from_
mutual_regulation_flag defaults to False and is never turned on by this
driver, so Stage 5 (which is the only thing that parameter feeds) never runs.

Output: one <net>_<variant>_rep_<rep_id>_all_results.json per replicate,
written to analysis_data/paper_analysis/real_networks/twinfer_inference_multistate_nofilter/.
"""
import glob
import json
import os
import re
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from twinfer.inference.infer import infer_with_twinfer
from twinfer.utils.paths import get_data_root

warnings.filterwarnings("ignore")

DATA_ROOT = get_data_root()
PROJECT_ROOT = DATA_ROOT.parent
INPUT_DATA = PROJECT_ROOT / "input_data"
NEW_SIM_ROOT = DATA_ROOT / "paper_analysis"

OUTPUT_DIR = DATA_ROOT / "paper_analysis" / "real_networks" / "twinfer_inference_multistate_nofilter"
os.makedirs(OUTPUT_DIR, exist_ok=True)

T1, T2 = 1, 20
N_JOBS_OUTER = 4
N_CORES_PER_TASK = 12  # 4 x 12 = 48, matching run_infer_multistate.sh's --cpus-per-task on the p32655/short partition

# (net, variant) -> sim_dir run_tag, n_genes, topology file, before-division
# sim time actually used for that run (see the two driver scripts above).
RUNS = {
    ("GSD", "multistate"): dict(run_tag="multistate_2000steps", n_genes=19, topology="GSD.txt", sim_time=2000),
    ("GSD", "seeded"):      dict(run_tag="multistate_2000steps_seeded", n_genes=19, topology="GSD.txt", sim_time=800),
    ("HSC", "multistate"):  dict(run_tag="multistate_2000steps", n_genes=11, topology="HSC.txt", sim_time=2000),
    ("HSC", "seeded"):      dict(run_tag="multistate_2000steps_seeded", n_genes=11, topology="HSC.txt", sim_time=800),
    ("EMT", "multistate"):  dict(run_tag="multistate_2000steps", n_genes=17, topology="EMT.txt", sim_time=2000),
    ("EMT", "seeded"):      dict(run_tag="multistate_2000steps_seeded", n_genes=17, topology="EMT.txt", sim_time=800),
    ("VSC", "seeded"):      dict(run_tag="multistate_6000steps_seeded", n_genes=8, topology="VSC.txt", sim_time=1500),
    ("mCAD", "seeded"):     dict(run_tag="multistate_2000steps_seeded", n_genes=5, topology="mCAD.txt", sim_time=800),
}


def make_base_config(net, cfg):
    return {
        "n_cells": 6000,
        "simulation_time_before_division": cfg["sim_time"],
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
        return {("__".join(map(str, k)) if isinstance(k, tuple) else str(k)): make_json_safe(v)
                for k, v in obj.items()}
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
        "sim_type": sim_type, "rep_id": rep_id, "analysis_key": analysis_key,
        "gene_names": gene_names, "n_genes": n_genes, **make_json_safe(results),
    }
    f_result_path = os.path.join(output_path, f"{analysis_key}_all_results.json")
    with open(f_result_path, "w") as f:
        json.dump(record, f, cls=NumpyEncoder, indent=2)
    print(f"Saved record to {f_result_path}")
    return f"{analysis_key}: OK"


def build_tasks():
    tasks = []
    for (net, variant), cfg in RUNS.items():
        base_config = make_base_config(net, cfg)
        token = f"{net}_{variant}"
        sim_dir = NEW_SIM_ROOT / net / "simulate" / cfg["run_tag"]
        file_re = re.compile(rf"^df_rows_.*_ncells_\d+_{re.escape(token)}_rep_(\d+)_([0-9a-fA-F]{{8}})\.csv$")
        for f in sorted(glob.glob(str(sim_dir / "df_*_ncells_*.csv"))):
            name = os.path.basename(f)
            if name.startswith("simulation_before_division"):
                continue
            m = file_re.match(name)
            if not m:
                continue
            rep, file_hash = m.groups()
            sim_type = token  # e.g. "GSD_multistate", "GSD_seeded"
            rep_id = f"{rep}_{file_hash}"
            tasks.append((f, sim_type, rep_id, base_config))

    rep_ids_seen = Counter((t, r) for _, t, r, _ in tasks)
    dup = [k for k, c in rep_ids_seen.items() if c > 1]
    if dup:
        raise ValueError(f"Duplicate (sim_type, rep_id) task keys found: {dup}")
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
    print(f"Done: {n_ok} ok, {len(results) - n_ok} failed")
    for r in results:
        if not r.endswith("OK"):
            print(r)


if __name__ == "__main__":
    main()
