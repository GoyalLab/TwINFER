"""
Standalone SLURM-batch driver for the causal_direction_inference Figure-3
5-gene linear cascade (Part B): two infer_with_twinfer calls per replicate
-- the real network, and a remove_twin_structure=True random control
(matching the old notebook's twinfer_kwargs / twinfer_kwargs_random) --
with rescue_opposite_sign_pairs=True (Part 0).

Only one cascade replicate file exists at the current data location; this
script is built to handle N files from the start (glob-based task
collection), so more replicates can be added later without code changes.

Output: one {rep_id}.json per task (real + random results together), under
--output-dir. Skips (does not recompute) any task whose output file already
exists.
"""
import argparse
import glob
import json
import os
import re
import time
from itertools import product

import numpy as np
import pandas as pd
import numba
from threadpoolctl import threadpool_limits

# [2026-09-30 commented out: infer_with_twinfer_updated was merged into twinfer.inference.infer]
# from twinfer.inference.infer_with_twinfer_updated import infer_with_twinfer
from twinfer.inference.infer import infer_with_twinfer

T1, T2 = 1, 20
SEED = 101010


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def result_to_json_safe(results):
    """
    Flatten the new nested infer_with_twinfer result dict to a JSON-safe
    dict, keeping the fields the old notebook's cell 24 scoring cell (Part
    C) and plot.ipynb's heatmap cells need: directed_edges,
    unfiltered_direction_matrix, classification (esp.
    multiple_states_and_reg), genes, plus the new rescued_from_no_regulation.
    There is no "all_gene_pairs" key anymore -- reconstruct it here so
    downstream consumers don't need to.
    """
    genes = results["settings"]["genes"]
    return {
        "genes": genes,
        "all_gene_pairs": list(product(genes, repeat=2)),
        "directed_edges": [list(e) for e in results["directed_edges"]],
        "unfiltered_direction_matrix": results["direction"]["unfiltered_matrix"].to_dict(),
        "pairwise_gene_gene_correlation_matrix": results["correlations"]["gene_t1"].to_dict(),
        "classification": {
            "no_regulation": [list(p) for p in results["classification"]["no_regulation"]],
            "single_state_regulation": [list(p) for p in results["classification"]["single_state_regulation"]],
            "multiple_states_no_reg": [list(p) for p in results["classification"]["multiple_states_no_reg"]],
            "multiple_states_and_reg": [list(p) for p in results["classification"]["multiple_states_and_reg"]],
        },
        "rescued_from_no_regulation": [list(p) for p in results["direction"]["rescued_from_no_regulation"]],
    }


def run_cascade_replicate(path_to_simulation_file, rep_id, n_cores=1, n_shuffles=None):
    numba.set_num_threads(1)
    shuffle_kwargs = {}
    if n_shuffles is not None:
        shuffle_kwargs = dict(
            n_shuffles_step1=n_shuffles,
            n_shuffles_step2=n_shuffles,
            n_shuffles_stage3=n_shuffles,
            n_shuffles_direction=n_shuffles,
            n_shuffles_fanout=n_shuffles,
        )
    with threadpool_limits(limits=1):
        simulation = pd.read_csv(path_to_simulation_file)

        real_results = infer_with_twinfer(
            None,
            data=simulation,
            is_simulation_data=True,
            t1=T1,
            t2=T2,
            check_for_steady_state=False,
            match_sim_details=False,
            seed=SEED,
            rescue_opposite_sign_pairs=True,
            plot=False,
            verbose=False,
            ranked_list=False,
            n_cores=n_cores,
            **shuffle_kwargs,
        )

        random_results = infer_with_twinfer(
            None,
            data=simulation.copy(),
            is_simulation_data=True,
            t1=T1,
            t2=T2,
            check_for_steady_state=False,
            match_sim_details=False,
            seed=SEED,
            rescue_opposite_sign_pairs=True,
            remove_twin_structure=True,
            plot=False,
            verbose=False,
            ranked_list=False,
            n_cores=n_cores,
            **shuffle_kwargs,
        )

    return {
        "rep_id": rep_id,
        "real": result_to_json_safe(real_results),
        "random": result_to_json_safe(random_results),
    }


def collect_tasks(sim_folder, name_contains):
    # A numeric "df_rows_0_0_0_0_0" prefix match is NOT safe here: the
    # unrelated 14-gene Figure1Network file's row-index prefix has more
    # zeros ("df_rows_0_0_0_0_0_0_0_0_...") and still matches as a string
    # prefix. Match on a substring that's actually specific to the cascade
    # files instead.
    tasks = []
    for f in sorted(glob.glob(os.path.join(sim_folder, "df_*.csv"))):
        if name_contains not in os.path.basename(f):
            continue
        m = re.search(r"rep_(\d+)", os.path.basename(f))
        rep_id = m.group(1) if m else "0"
        tasks.append((f, rep_id))
    return tasks


def run_one(path, rep_id, output_dir, n_cores, n_shuffles=None):
    out_path = os.path.join(output_dir, f"cascade_rep_{rep_id}.json")
    if os.path.exists(out_path):
        return out_path, "skipped (already exists)"

    record = run_cascade_replicate(path, rep_id, n_cores=n_cores, n_shuffles=n_shuffles)
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(record, f, cls=NumpyEncoder, indent=2)
    return out_path, "computed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cores", type=int, default=1)
    parser.add_argument("--n-shuffles", type=int, default=None,
                         help="override n_shuffles for all null-generation steps (default: automatic)")
    parser.add_argument("--sim-folder", type=str,
                         default="/projects/b1255/yscher/Transcriptomic Distance/simulation_data/figure_3_simulations")
    parser.add_argument("--file-prefix", type=str, default="five_gene_cascade",
                         help="substring identifying 5-gene cascade files (matched anywhere in the filename)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--only", type=str, default=None, help="comma-separated rep_id to run")
    parser.add_argument("--list-tasks", action="store_true")
    args = parser.parse_args()

    if not args.list_tasks and not args.output_dir:
        parser.error("--output-dir is required unless --list-tasks is given")

    tasks = collect_tasks(args.sim_folder, args.file_prefix)

    if args.list_tasks:
        print("\n".join(rep_id for _, rep_id in tasks))
        raise SystemExit(0)

    if args.only:
        wanted = set(args.only.split(","))
        tasks = [t for t in tasks if t[1] in wanted]

    print(f"Running {len(tasks)} cascade task(s), output -> {args.output_dir}", flush=True)
    for i, (path, rep_id) in enumerate(tasks):
        t0 = time.time()
        out_path, status = run_one(path, rep_id, args.output_dir, args.n_cores, n_shuffles=args.n_shuffles)
        print(f"[{i+1}/{len(tasks)}] cascade rep {rep_id}: {status} ({time.time()-t0:.1f}s) -> {out_path}", flush=True)
