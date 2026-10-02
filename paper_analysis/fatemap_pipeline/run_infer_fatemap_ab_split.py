#!/usr/bin/env python3
"""Exploratory check (2026-09-22, user request): FM06/FM08 replicate A and B are two wells of
the SAME experiment, not a real time-course -- but unlike run_infer_fatemap.py's t1=t2 pooled
mode (which randomly splits each clone's cells into two twin halves, making rho_cross_xy and
rho_cross_yx algebraically identical -- verified: diffs ~1e-16, pure float noise -- which is why
gamma/direction_term are structurally dead there), treating replicate A as t1 and B as t2 uses a
FIXED, non-arbitrary split (which well a cell happened to land in). That breaks the guaranteed
symmetry: rho_cross_xy = corr(gene_x in A-half, gene_y in B-half) and rho_cross_yx = corr(gene_y
in A-half, gene_x in B-half) are no longer forced to be equal by construction, so this checks
whether TwINFER's direction/gamma machinery produces non-degenerate output given a real (if
still not temporally meaningful -- both wells are the same timepoint) fixed split, and whether
AUPRC-x against CollecTRI changes at all vs the pooled t1=t2 result already on disk.

Only clones present in BOTH replicates are usable (FM06: 2,101/5,393 clones; FM08: 56/207,
dominated by the already-documented barcode-collision artifact -- apply the same MAX_CLONE_SIZE
cap used elsewhere in this pipeline).
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

MAX_CLONE_SIZE = 50
# Per-dataset override of the per-side cap in the A/B construction (None = keep every clone).
# FM01: the >50-cell sides are graded (51-262), not one barcode-collision outlier, and the 5
# A/B-spanning clones they affect hold half of all spanning cells -- keep them (user decision 2026-09-23).
AB_MAX_CLONE_SIZE_OVERRIDE = {"FM01": None}


# Watermelon (T47D) stages, run one stage at a time as their own A/B dataset (replicate 1 = A, 2 = B): the clones are
# huge (sides up to ~950 cells -> millions of twin pairs), so each (clone, side) is SUBSAMPLED to at most this many cells
# instead of dropping the clone. Selection is by a hash of the cell id (order-independent), so the inference and scoring
# loaders, which read cells in different orders, pick exactly the same cells.
AB_SUBSAMPLE_PER_SIDE = {f"Watermelon_{st}": 100 for st in ("naive", "lag", "late")}
AB_MAX_CLONE_SIZE_OVERRIDE.update({k: None for k in AB_SUBSAMPLE_PER_SIDE})  # subsample, never drop
SUBSAMPLE_SEED = 101010


def ab_max_clone_size(dataset):
    return AB_MAX_CLONE_SIZE_OVERRIDE.get(dataset, MAX_CLONE_SIZE)


def subsample_per_side(df, cap, seed=SUBSAMPLE_SEED):
    """Keep at most `cap` cells per (clone_id, time_step): the `cap` with the smallest hash of (seed, cell_id).
    Deterministic and independent of row order."""
    key = pd.util.hash_pandas_object(df["cell_id"].astype(str) + f"|{seed}", index=False)
    rank = key.groupby([df["clone_id"], df["time_step"]]).rank(method="first")
    return df[(rank <= cap).to_numpy()].reset_index(drop=True)


def build_twinfer_input_ab(dataset, gene_set):
    h5ad_path = f"{TWINFER_PROJECT_ROOT}/finalized_data/{dataset}_data/{dataset}_integrated.h5ad"
    A = ad.read_h5ad(h5ad_path)
    missing = [g for g in gene_set if g not in A.raw.var_names]
    if missing:
        raise ValueError(f"{dataset}: gene set has genes not in matrix: {missing}")

    expr = A.raw[:, gene_set].X
    expr = np.asarray(expr.todense()) if hasattr(expr, "todense") else np.asarray(expr)

    df = pd.DataFrame(expr, columns=[f"{g}_mRNA" for g in gene_set], index=A.obs_names)
    df.insert(0, "time_step", (A.obs["replicate"].to_numpy() == "B").astype(int))  # A=0, B=1
    df.insert(0, "cell_id", A.obs_names)
    clone = A.obs["fatemap_clone_singletcode"].astype(str)
    has_clone = clone.str.len() > 0
    df.insert(0, "clone_id", clone)
    df = df[has_clone.to_numpy()].reset_index(drop=True)

    # only clones present in BOTH replicates are usable for a t1=A/t2=B cross-correlation
    span = df.groupby("clone_id")["time_step"].nunique()
    spanning_clones = span[span == 2].index
    n_before = len(df)
    df = df[df["clone_id"].isin(spanning_clones)].reset_index(drop=True)
    print(f"[{dataset}] A/B-spanning-clone filter: {n_before:,} -> {len(df):,} cells "
          f"({len(spanning_clones):,} clones present in both replicates)", flush=True)

    # same oversized-clone guard as run_infer_fatemap.py, applied per (clone, timepoint) side
    sub_cap = AB_SUBSAMPLE_PER_SIDE.get(dataset)
    if sub_cap:
        n_sub = len(df)
        df = subsample_per_side(df, sub_cap)
        print(f"[{dataset}] subsampled each (clone, side) to <= {sub_cap} cells: {n_sub:,} -> {len(df):,} cells", flush=True)
    cap = ab_max_clone_size(dataset)
    if cap is None:
        print(f"[{dataset}] no per-side clone-size cap (largest side: "
              f"{int(df.groupby(['clone_id', 'time_step']).size().max())} cells)", flush=True)
    else:
        per_side = df.groupby(["clone_id", "time_step"]).size()
        oversized_keys = per_side[per_side > cap].index.get_level_values("clone_id").unique()
        if len(oversized_keys):
            print(f"[{dataset}] dropping {len(oversized_keys)} clone(s) with an oversized side "
                  f"(>{cap} cells on A or B, likely barcode-collision artifact)", flush=True)
        df = df[~df["clone_id"].isin(oversized_keys)].reset_index(drop=True)
    print(f"[{dataset}] final: {len(df):,} cells, {df['clone_id'].nunique():,} clones "
          f"(t1=A: {(df.time_step==0).sum():,} cells, t2=B: {(df.time_step==1).sum():,} cells)", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["FM06", "FM08"])
    ap.add_argument("--gene-set", default="correlation_high")
    ap.add_argument("--n-shuffles", type=int, default=500)
    ap.add_argument("--n-cores", type=int, default=8)
    args = ap.parse_args()

    label = args.dataset.lower()
    gene_sets_path = f"{TWINFER_PROJECT_ROOT}/analysis_data/{label}/data/gene_sets_{label}.json"
    gene_sets = json.load(open(gene_sets_path))
    gene_set = gene_sets[args.gene_set]
    print(f"[{args.dataset}] gene set '{args.gene_set}': {len(gene_set)} genes", flush=True)

    df = build_twinfer_input_ab(args.dataset, gene_set)

    out_dir = f"{TWINFER_PROJECT_ROOT}/analysis_data/{label}/data"
    os.makedirs(out_dir, exist_ok=True)
    input_csv = os.path.join(out_dir, f"twinfer_input_{label}_{args.gene_set}_absplit.csv")
    df.to_csv(input_csv, index=False)
    print(f"[{args.dataset}] wrote {input_csv}", flush=True)

    result = infer_with_twinfer(
        data=df,
        is_simulation_data=False,
        t1=0, t2=1,
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
    out_csv = os.path.join(out_dir, f"ranked_edges_{label}_{args.gene_set}_absplit.csv")
    ranked_edges.to_csv(out_csv, index=False)
    print(f"[{args.dataset}] wrote {out_csv} ({len(ranked_edges)} directed edge candidates)", flush=True)

    z_reg_gated = {
        f"{a}__{b}": (None if v is None or not np.isfinite(v) else float(v))
        for (a, b), v in result["gated_regulation"]["z_reg_gated"].items()
    }
    json.dump(z_reg_gated, open(os.path.join(out_dir, f"z_reg_gated_{label}_{args.gene_set}_absplit.json"), "w"))

    z_dagger = {
        f"{a}__{b}": float(details["z_rho_cross"])
        for (a, b), details in result["direction"]["rho_cross_null"].items()
        if np.isfinite(details["z_rho_cross"])
    }
    json.dump(z_dagger, open(os.path.join(out_dir, f"z_dagger_{label}_{args.gene_set}_absplit.json"), "w"))

    print(f"[{args.dataset}] classification: {result['classification']}", flush=True)
    if len(ranked_edges):
        print(ranked_edges[["gene_1", "gene_2", "rho_cross_xy", "rho_cross_yx", "gamma", "z_gamma", "twinScore"]]
              .sort_values("twinScore", ascending=False).to_string(index=False), flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
