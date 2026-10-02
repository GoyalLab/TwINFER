#!/usr/bin/env python3
"""2026-09-22 user request: AUPRC should be computed over ALL gene pairs in a gene set, not
just the subset that happened to pass infer_with_twinfer's internal Stage 1/3 significance
gates (which is what ranked_edges_*.csv contains) -- and missing pairs should get an actual
computed twinScore, not a nan_to_num placeholder.

infer_with_twinfer's gates (alpha_gene_gene_corr for Stage 1's no_regulation/potential_regulation
split, alpha_stage3 for Stage 3's multiple_states_no_reg cutoff) are real, deliberate significance
filters -- pairs that fail them never reach Step 4 (direction) and never get a twinScore row at
all in the normal pipeline. Relaxing both to their loosest legal value (alpha=0.999, i.e. z_critical
~0) lets (almost) every candidate pair fall through to potential_regulation / multiple_states_and_reg
and proceed through the exact same Step 1-4 permutation machinery as any other pair -- so every
pair gets a REAL computed twinScore from real permutation nulls, not an imputed one. This is the
same statistical pipeline, just without discarding low-significance pairs before scoring them.

Supports both the pooled (t1=t2) and real A/B-split (replicate A=t1, B=t2) constructions via
--absplit, reusing build_twinfer_input / build_twinfer_input_ab from the sibling scripts.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/home/gzu5140/TwINFER_KA/code/TwINFER/package")
from twinfer.inference.infer import infer_with_twinfer

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_analysis.fatemap_pipeline.run_infer_fatemap import build_twinfer_input as build_pooled_input, TIME_STEP
from paper_analysis.fatemap_pipeline.run_infer_fatemap_ab_split import build_twinfer_input_ab

ALPHA_PERMISSIVE = 0.999  # z_critical = norm.ppf(1 - alpha/2) ~ 0: let (almost) every pair through
# 2026-09-23: --alpha overrides this (default unchanged). alpha=1.0 makes the Step 1/2/Stage 3 cutoffs 0 / -inf,
# i.e. really every pair reaches Step 4 (0.999 still dropped pairs with z_d_het < -3.09 or ~zero correlation).


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["FM01", "FM06", "FM08", "Watermelon_naive", "Watermelon_lag", "Watermelon_late"])
    ap.add_argument("--gene-set", default="correlation_high")
    ap.add_argument("--n-shuffles", type=int, default=500)
    ap.add_argument("--n-cores", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=ALPHA_PERMISSIVE, help="alpha for alpha_gene_gene_corr and alpha_stage3 (1.0 = no filtering); when != 0.999 outputs get an _alpha<value> suffix")
    ap.add_argument("--absplit", action="store_true", help="replicate A=t1, B=t2 instead of pooled t1=t2")
    args = ap.parse_args()

    label = args.dataset.lower()
    gene_sets_path = f"{TWINFER_PROJECT_ROOT}/analysis_data/{label}/data/gene_sets_{label}.json"
    gene_set = json.load(open(gene_sets_path))[args.gene_set]
    print(f"[{args.dataset}] gene set '{args.gene_set}': {len(gene_set)} genes", flush=True)

    if args.absplit:
        df = build_twinfer_input_ab(args.dataset, gene_set)
        t1, t2 = 0, 1
        suffix = "_absplit_allpairs"
    else:
        df = build_pooled_input(args.dataset, gene_set)
        t1, t2 = TIME_STEP, TIME_STEP
        suffix = "_allpairs"
    if args.alpha != ALPHA_PERMISSIVE:
        suffix += f"_alpha{args.alpha:g}"  # never overwrite the alpha=0.999 outputs

    out_dir = f"{TWINFER_PROJECT_ROOT}/analysis_data/{label}/data"
    os.makedirs(out_dir, exist_ok=True)

    result = infer_with_twinfer(
        data=df,
        is_simulation_data=False,
        t1=t1, t2=t2,
        check_for_steady_state=False,
        use_clone=True,
        unit="clone",
        alpha_gene_gene_corr=args.alpha,
        alpha_stage3=args.alpha,
        rescue_opposite_sign_pairs=True,
        bypass_stage3_gate=True,
        n_shuffles_step1=args.n_shuffles,
        n_shuffles_step2=args.n_shuffles,
        n_shuffles_stage3=args.n_shuffles,
        n_shuffles_direction=args.n_shuffles,
        n_shuffles_fanout=args.n_shuffles,
        n_cores=args.n_cores,
        seed=101010,
        verbose=True,
        plot=False,
    )

    ranked_edges = result["ranked_edges"]
    out_csv = os.path.join(out_dir, f"ranked_edges_{label}_{args.gene_set}{suffix}.csv")
    ranked_edges.to_csv(out_csv, index=False)
    print(f"[{args.dataset}] wrote {out_csv} ({len(ranked_edges)} directed edge candidates "
          f"out of {len(gene_set)*(len(gene_set)-1)} possible ordered pairs)", flush=True)

    # needed by apply_twinscore_supplement_fatemap*.py's direction_term/gamma computation --
    # with bypass_stage3_gate=True every pair reaches Step 4, so this json now has full
    # (non-gated) coverage too, unlike the original gated run's z_dagger_{label}_{gs}.json.
    z_dagger = {
        f"{a}__{b}": float(details["z_rho_cross"])
        for (a, b), details in result["direction"]["rho_cross_null"].items()
        if np.isfinite(details["z_rho_cross"])
    }
    z_dagger_path = os.path.join(out_dir, f"z_dagger_{label}_{args.gene_set}{suffix}.json")
    json.dump(z_dagger, open(z_dagger_path, "w"))
    print(f"[{args.dataset}] wrote {z_dagger_path} ({len(z_dagger)} entries)", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
