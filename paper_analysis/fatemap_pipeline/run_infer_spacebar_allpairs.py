#!/usr/bin/env python3
"""SpaceBar analog of run_infer_fatemap_allpairs.py: relaxed Stage 1 alpha + bypass_stage3_gate
so every candidate pair gets a real, non-imputed twinScore -- no ranked_edges gating.

Note: unlike FM06/FM08's replicate A/B (which DO share clone lineages across the two wells --
verified, 2,101/5,393 FM06 clones span both), SpaceBar's clone_id is prefixed per-section
("S2_x", "S4_x" -- cluster IDs are local to each section, per run_infer_spacebar.py's own
docstring), so no clone spans both Section2 and Section4. A real A/B-style t1/t2 split isn't
possible here; this only reruns the existing pooled (t1=t2=0, Section2+Section4 combined)
construction with all gates removed.
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
from paper_analysis.fatemap_pipeline.run_infer_spacebar import build_twinfer_input, TIME_STEP, MAX_CLONE_SIZE, SEED

ALPHA_PERMISSIVE = 0.999


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene-set", choices=["correlation_high", "variability_high"], default="correlation_high")
    ap.add_argument("--n-shuffles", type=int, default=500)
    ap.add_argument("--n-cores", type=int, default=8)
    args = ap.parse_args()

    out_dir = f'{TWINFER_PROJECT_ROOT}/analysis_data/spacebar/data'
    os.makedirs(out_dir, exist_ok=True)

    df = build_twinfer_input(args.gene_set, MAX_CLONE_SIZE)

    result = infer_with_twinfer(
        data=df,
        is_simulation_data=False,
        t1=TIME_STEP, t2=TIME_STEP,
        check_for_steady_state=False,
        use_clone=True,
        unit="clone",
        alpha_gene_gene_corr=ALPHA_PERMISSIVE,
        alpha_stage3=ALPHA_PERMISSIVE,
        rescue_opposite_sign_pairs=True,
        bypass_stage3_gate=True,
        n_shuffles_step1=args.n_shuffles,
        n_shuffles_step2=args.n_shuffles,
        n_shuffles_stage3=args.n_shuffles,
        n_shuffles_direction=args.n_shuffles,
        n_shuffles_fanout=args.n_shuffles,
        n_cores=args.n_cores,
        seed=SEED,
        verbose=True,
        plot=False,
    )

    ranked_edges = result["ranked_edges"]
    out_csv = os.path.join(out_dir, f"ranked_edges_spacebar_{args.gene_set}_allpairs.csv")
    ranked_edges.to_csv(out_csv, index=False)
    print(f"[SpaceBar] wrote {out_csv} ({len(ranked_edges)} directed edge candidates)", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
