#!/usr/bin/env python3
"""Builds the canonical TwINFER input (clone_id, cell_id, time_step, {gene}_mRNA) from
HS054_Sot48h's un-integrated, log1p(CP10k) expression (HS054_Sot48h_integrated.h5ad's
.raw -- see fatemap_integration_utils.normalize_and_pca), and runs infer_with_twinfer
with t1=t2 (single timepoint -- see infer.py's single_timepoint gated-regulation
bypass) on one gene set at a time. Mirrors run_infer_fatemap.py, single-sample version
(no oversized-clone-collision caveat known yet -- HS054's genuine max clone size is 54,
just above MAX_CLONE_SIZE=50, so at most one clone gets dropped, unlike FM08's 64%-of-
dataset barcode-collision clone).
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import argparse
import json
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/home/gzu5140/TwINFER_KA/code/TwINFER/package")
from twinfer.inference.infer import infer_with_twinfer

DATASET = "HS054_Sot48h"
LABEL = "hs054"
TIME_STEP = 0  # single timepoint
MAX_CLONE_SIZE = 50


def build_twinfer_input(gene_set):
    h5ad_path = f"{TWINFER_PROJECT_ROOT}/finalized_data/{DATASET}_data/{DATASET}_integrated.h5ad"
    A = ad.read_h5ad(h5ad_path)
    missing = [g for g in gene_set if g not in A.raw.var_names]
    if missing:
        raise ValueError(f"{DATASET}: gene set has genes not in matrix: {missing}")

    expr = A.raw[:, gene_set].X
    expr = np.asarray(expr.todense()) if hasattr(expr, "todense") else np.asarray(expr)

    df = pd.DataFrame(expr, columns=[f"{g}_mRNA" for g in gene_set], index=A.obs_names)
    df.insert(0, "time_step", TIME_STEP)
    df.insert(0, "cell_id", A.obs_names)
    clone = A.obs["fatemap_clone_singletcode"].astype(str)
    has_clone = clone.str.len() > 0
    df.insert(0, "clone_id", clone)
    df = df[has_clone.to_numpy()].reset_index(drop=True)

    counts = df["clone_id"].value_counts()
    oversized = counts[counts > MAX_CLONE_SIZE]
    if len(oversized):
        print(f"[{DATASET}] dropping {len(oversized)} oversized clone(s) (>{MAX_CLONE_SIZE} cells): "
              f"{oversized.head(5).to_dict()}", flush=True)
    keep_clones = counts[(counts >= 2) & (counts <= MAX_CLONE_SIZE)].index
    n_before = len(df)
    df = df[df["clone_id"].isin(keep_clones)].reset_index(drop=True)
    print(f"[{DATASET}] clone-size filter: {n_before:,} -> {len(df):,} cells "
          f"({len(keep_clones):,} clones, 2-{MAX_CLONE_SIZE} cells each)", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene-set", default="correlation_high")
    ap.add_argument("--n-shuffles", type=int, default=2000)
    ap.add_argument("--n-cores", type=int, default=4)
    args = ap.parse_args()

    gene_sets_path = f"{TWINFER_PROJECT_ROOT}/analysis_data/{LABEL}/data/gene_sets_{LABEL}.json"
    gene_sets = json.load(open(gene_sets_path))
    gene_set = gene_sets[args.gene_set]
    print(f"[{DATASET}] gene set '{args.gene_set}': {len(gene_set)} genes: {gene_set}", flush=True)

    df = build_twinfer_input(gene_set)
    print(f"[{DATASET}] TwINFER input: {len(df):,} cells, "
          f"{df['clone_id'].nunique():,} clones with >=2 cells", flush=True)

    out_dir = f"{TWINFER_PROJECT_ROOT}/analysis_data/{LABEL}/data"
    os.makedirs(out_dir, exist_ok=True)
    input_csv = os.path.join(out_dir, f"twinfer_input_{LABEL}_{args.gene_set}.csv")
    df.to_csv(input_csv, index=False)
    print(f"[{DATASET}] wrote {input_csv}", flush=True)

    result = infer_with_twinfer(
        data=df,
        is_simulation_data=False,
        t1=TIME_STEP,
        t2=TIME_STEP,
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

    ranked_edges = result["ranked_edges"]
    out_csv = os.path.join(out_dir, f"ranked_edges_{LABEL}_{args.gene_set}.csv")
    ranked_edges.to_csv(out_csv, index=False)
    print(f"[{DATASET}] wrote {out_csv} ({len(ranked_edges)} directed edge candidates)", flush=True)

    z_reg_gated = {
        f"{a}__{b}": (None if v is None or not np.isfinite(v) else float(v))
        for (a, b), v in result["gated_regulation"]["z_reg_gated"].items()
    }
    z_reg_path = os.path.join(out_dir, f"z_reg_gated_{LABEL}_{args.gene_set}.json")
    json.dump(z_reg_gated, open(z_reg_path, "w"))
    print(f"[{DATASET}] wrote {z_reg_path}", flush=True)

    z_dagger = {
        f"{a}__{b}": float(details["z_rho_cross"])
        for (a, b), details in result["direction"]["rho_cross_null"].items()
        if np.isfinite(details["z_rho_cross"])
    }
    z_dagger_path = os.path.join(out_dir, f"z_dagger_{LABEL}_{args.gene_set}.json")
    json.dump(z_dagger, open(z_dagger_path, "w"))
    print(f"[{DATASET}] wrote {z_dagger_path}", flush=True)

    print(f"[{DATASET}] classification: {result['classification']}", flush=True)
    if len(ranked_edges):
        print(ranked_edges[["gene_1", "gene_2", "twinScore"]]
              .sort_values("twinScore", ascending=False).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
