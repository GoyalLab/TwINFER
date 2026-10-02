#!/usr/bin/env python
"""ALL_PAIRS TwINFER inference on the BoolODE real-network twin sims (B_cell_activation, EMT_real,
Pluripotent_real), converted by boolode_real_convert.py. Same settings as infer_real_network_allpairs.py
(the Gillespie real-network run): alpha_gene_gene_corr/alpha_stage3=0.9999, both z thresholds 0,
minimum_null_draws=5000, check_for_steady_state=False, match_sim_details=False.

t1 = first saved step after the branch point (twins are identical AT the branch), t2 = last step:
    B_cell_activation (T=8,  branch 400): t1=500,  t2=799
    EMT_real / Pluripotent_real (T=16, branch 800): t1=900, t2=1599

Output: analysis_data/boolode_sims_real_networks/twinfer_inference_allpairs/<net>_rep_<r>_all_results.json

    python boolode_real_infer_allpairs.py [--only NET] [--reps 0 1 ..] [--n-jobs N] [--n-cores-per-task N] [--dry-run]
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import argparse
import json
import os
import warnings

import numpy as np
from joblib import Parallel, delayed

from twinfer.inference.infer import infer_with_twinfer
from benchmarks.network_benchmarks.infer.infer_real_network_allpairs import make_json_safe, NumpyEncoder  # reuse serializers (no side effects beyond mkdir)

warnings.filterwarnings("ignore")

ROOT = f'{TWINFER_PROJECT_ROOT}'
FMT = f"{ROOT}/analysis_data/boolode_sims_real_networks/twinfer_format"
OUT = f"{ROOT}/analysis_data/boolode_sims_real_networks/twinfer_inference_allpairs"
ALPHA = 0.9999

RW = f"{ROOT}/input_data/real_world_networks"
OLD = f"{ROOT}/simulation_data/twinfer_format"  # rule-derived matrices for the curated BoolODE nets
# t1 = branch step + 100, t2 = last step (GSD/HSC: T=8 branch 400; mCAD/VSC: T=5 branch 250)
NETS = {
    "B_cell_activation": dict(n_genes=10, topology=f"{RW}/B_cell.txt", t1=500, t2=799),
    "EMT_real": dict(n_genes=17, topology=f"{RW}/EMT.txt", t1=900, t2=1599),
    "Pluripotent_real": dict(n_genes=36, topology=f"{RW}/Pluripotent.txt", t1=900, t2=1599),
    "GSD": dict(n_genes=19, topology=f"{OLD}/GSD/interaction_matrix.txt", t1=500, t2=799),
    "HSC": dict(n_genes=11, topology=f"{OLD}/HSC/interaction_matrix.txt", t1=500, t2=799),
    "mCAD": dict(n_genes=5, topology=f"{OLD}/mCAD/interaction_matrix.txt", t1=300, t2=499),
    "VSC": dict(n_genes=8, topology=f"{OLD}/VSC/interaction_matrix.txt", t1=300, t2=499),
}


def run_one(net, rep, n_cores):
    cfg = NETS[net]
    out_path = f"{OUT}/{net}_rep_{rep}_all_results.json"
    if os.path.exists(out_path):
        return f"{net} rep {rep}: SKIPPED (exists)"
    base_config = {
        "n_cells": 6000,
        "simulation_time_before_division": 0,
        "twin_simulation_time_after_division": cfg["t2"],
        "twin_measurement_resolution": 100,
        "path_to_connectivity_matrix": cfg["topology"],
        "param_csv": f"{ROOT}/input_data/network_sweep/parameters.csv",
        "rows_to_use": [[0] * cfg["n_genes"]],
        "type": net,
    }
    try:
        results = infer_with_twinfer(
            f"{FMT}/{net}/replicate_{rep}_simulation.csv",
            merge_to_multiple_states=False, base_config=base_config,
            t1=cfg["t1"], t2=cfg["t2"], match_sim_details=False, check_for_steady_state=False,
            seed=101010, n_cores=n_cores, alpha_gene_gene_corr=ALPHA, alpha_stage3=ALPHA,
            z_score_threshold_two_states=0, z_score_threshold_cross_correlation=0,
            minimum_null_draws=5000, ranked_list=True,
        )
    except Exception as e:
        return f"{net} rep {rep}: FAILED ({e!r})"
    record = {"sim_type": net, "rep_id": rep, "analysis_key": f"{net}_rep_{rep}",
              "gene_names": [f"gene_{i + 1}" for i in range(cfg["n_genes"])], "n_genes": cfg["n_genes"],
              **make_json_safe(results)}
    os.makedirs(OUT, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(record, f, cls=NumpyEncoder, indent=2)
    return f"{net} rep {rep}: OK"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--reps", type=int, nargs="+", default=None)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--n-cores-per-task", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    def reps_of(n):
        import glob
        return sorted(int(os.path.basename(f).split("_")[1]) for f in glob.glob(f"{FMT}/{n}/replicate_*_simulation.csv"))
    tasks = [(n, r) for n in NETS if not args.only or n == args.only for r in (args.reps or reps_of(n))]
    print(f"{len(tasks)} tasks", flush=True)
    if args.dry_run:
        for n, r in tasks:
            print(" ", n, r, "done" if os.path.exists(f"{OUT}/{n}_rep_{r}_all_results.json") else "TODO")
        return
    res = Parallel(n_jobs=args.n_jobs, backend="loky")(delayed(run_one)(n, r, args.n_cores_per_task) for n, r in tasks)
    for r in res:
        print(r)


if __name__ == "__main__":
    main()
