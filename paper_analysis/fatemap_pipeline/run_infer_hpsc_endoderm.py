#!/usr/bin/env python3
"""Runs infer_with_twinfer on hPSC_20260927 (endo_T0/endo_T1) as a genuine two-timepoint
direction-stage run: t1=0 (endo_T0), t2=1 (endo_T1) -- NOT the t1=t2 single-timepoint
shortcut HS054/FM06/FM08 use (see infer.py's docstring: real Step 4 cross-time twins are
all t1 x t2 cell combinations within a clone). Input is
twinfer_input_hpsc_20260927_chipatlas_perturbseq_panel.csv (build_hpsc_endoderm_twinfer_input.py),
a 489-gene panel built from ChIP-Atlas pluripotent/endoderm candidate pairs (144 TFs, top-4
most-correlated targets each) with NANOG and POU5F1 expanded to their top-20 ChIP-bound
targets that are ALSO Perturb-seq DEGs (GSE283614) -- the only 2 TFs in the panel with both
ChIP-seq and perturbation evidence.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import argparse
import json
import os

import numpy as np
import pandas as pd

import sys
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/package")
from twinfer.inference.infer import infer_with_twinfer

LABEL = "hpsc_20260927"
GENE_SET = "chipatlas_perturbseq_panel"
DATA_DIR = f"{TWINFER_PROJECT_ROOT}/analysis_data/hPSC_20260927/data"
T1, T2 = 0, 1  # endo_T0, endo_T1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-shuffles", type=int, default=2000)
    ap.add_argument("--n-cores", type=int, default=8)
    args = ap.parse_args()

    input_csv = f"{DATA_DIR}/twinfer_input_{LABEL}_{GENE_SET}.csv"
    df = pd.read_csv(input_csv)
    n_genes = sum(c.endswith("_mRNA") for c in df.columns)
    print(f"[hPSC_20260927] TwINFER input: {len(df):,} cells, {n_genes} genes, "
          f"{df['clone_id'].nunique():,} clones, time_step counts {df['time_step'].value_counts().to_dict()}",
          flush=True)

    result = infer_with_twinfer(
        data=df,
        is_simulation_data=False,
        t1=T1,
        t2=T2,
        check_for_steady_state=False,
        use_clone=True,
        unit="clone",
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

    out_dir = DATA_DIR
    os.makedirs(out_dir, exist_ok=True)

    ranked_edges = result["ranked_edges"]
    out_csv = os.path.join(out_dir, f"ranked_edges_{LABEL}_{GENE_SET}.csv")
    ranked_edges.to_csv(out_csv, index=False)
    print(f"[hPSC_20260927] wrote {out_csv} ({len(ranked_edges)} directed edge candidates)", flush=True)

    z_reg_gated = {
        f"{a}__{b}": (None if v is None or not np.isfinite(v) else float(v))
        for (a, b), v in result["gated_regulation"]["z_reg_gated"].items()
    }
    z_reg_path = os.path.join(out_dir, f"z_reg_gated_{LABEL}_{GENE_SET}.json")
    json.dump(z_reg_gated, open(z_reg_path, "w"))
    print(f"[hPSC_20260927] wrote {z_reg_path}", flush=True)

    z_dagger = {
        f"{a}__{b}": float(details["z_rho_cross"])
        for (a, b), details in result["direction"]["rho_cross_null"].items()
        if np.isfinite(details["z_rho_cross"])
    }
    z_dagger_path = os.path.join(out_dir, f"z_dagger_{LABEL}_{GENE_SET}.json")
    json.dump(z_dagger, open(z_dagger_path, "w"))
    print(f"[hPSC_20260927] wrote {z_dagger_path}", flush=True)

    print(f"[hPSC_20260927] classification: {result['classification']}", flush=True)
    if len(ranked_edges):
        print(ranked_edges[["gene_1", "gene_2", "twinScore"]]
              .sort_values("twinScore", ascending=False).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
