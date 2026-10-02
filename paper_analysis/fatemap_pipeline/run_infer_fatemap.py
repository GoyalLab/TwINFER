#!/usr/bin/env python3
"""Builds the canonical TwINFER input (clone_id, cell_id, time_step, {gene}_mRNA)
from FM06/FM08's un-integrated, batch-aware-normalized log1p(CP10k) expression
(FM0X_integrated.h5ad's .raw -- see fatemap_integration_utils.normalize_and_pca;
integration embeddings (scanorama/harmony) are not used here at all, only for
visualization diagnostics -- since FM06/FM08 showed no real batch effect, the
plain per-cell-normalized expression is used directly), and runs infer_with_twinfer
with t1=t2 (single timepoint -- see infer.py's single_timepoint gated-regulation
bypass) on one gene set at a time (default: correlation_high, "yscher"-style
CollecTRI panel from pick_gene_sets_fatemap.py).
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
from twinfer.inference.correlation_functions import calculate_twin_score

TIME_STEP = 0  # single timepoint for both FM06 and FM08
MAX_CLONE_SIZE = 50  # cells/clone; see build_twinfer_input for why


def build_twinfer_input(dataset, gene_set):
    h5ad_path = f"{TWINFER_PROJECT_ROOT}/finalized_data/{dataset}_data/{dataset}_integrated.h5ad"
    A = ad.read_h5ad(h5ad_path)
    missing = [g for g in gene_set if g not in A.raw.var_names]
    if missing:
        raise ValueError(f"{dataset}: gene set has genes not in matrix: {missing}")

    expr = A.raw[:, gene_set].X
    expr = np.asarray(expr.todense()) if hasattr(expr, "todense") else np.asarray(expr)

    df = pd.DataFrame(expr, columns=[f"{g}_mRNA" for g in gene_set], index=A.obs_names)
    df.insert(0, "time_step", TIME_STEP)
    df.insert(0, "cell_id", A.obs_names)
    clone = A.obs["fatemap_clone_singletcode"].astype(str)
    has_clone = clone.str.len() > 0
    df.insert(0, "clone_id", clone)
    df = df[has_clone.to_numpy()].reset_index(drop=True)

    # TwINFER twin construction needs >=2 cells per clone at this timepoint, but real-data
    # twins are ALL unordered within-clone pairs -- a clone above MAX_CLONE_SIZE is not a
    # plausible single lineage (FM06's genuine max is 15 cells/clone; FM08 has a barcode-
    # collision artifact where one "clone" is 2,498/3,897 cells (64%) and another is 501 --
    # almost certainly degenerate/generic barcode reads, not real clones) and its C(n,2)
    # pairs would both corrupt the twin-pair statistics and blow up memory (measured: one
    # such FM08 clone alone produced ~3.1M pairs, OOM-killing the SLURM job at 32GB).
    counts = df["clone_id"].value_counts()
    oversized = counts[counts > MAX_CLONE_SIZE]
    if len(oversized):
        print(f"[{dataset}] dropping {len(oversized)} oversized clone(s) (>{MAX_CLONE_SIZE} cells, "
              f"likely barcode-collision artifacts): "
              f"{oversized.head(5).to_dict()}", flush=True)
    keep_clones = counts[(counts >= 2) & (counts <= MAX_CLONE_SIZE)].index
    n_before = len(df)
    df = df[df["clone_id"].isin(keep_clones)].reset_index(drop=True)
    print(f"[{dataset}] clone-size filter: {n_before:,} -> {len(df):,} cells "
          f"({len(keep_clones):,} clones, 2-{MAX_CLONE_SIZE} cells each)", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["FM06", "FM08"])
    ap.add_argument("--gene-set", default="correlation_high")
    ap.add_argument("--n-shuffles", type=int, default=2000)
    ap.add_argument("--n-cores", type=int, default=4)
    args = ap.parse_args()

    label = args.dataset.lower()
    gene_sets_path = f"{TWINFER_PROJECT_ROOT}/analysis_data/{label}/data/gene_sets_{label}.json"
    gene_sets = json.load(open(gene_sets_path))
    gene_set = gene_sets[args.gene_set]
    print(f"[{args.dataset}] gene set '{args.gene_set}': {len(gene_set)} genes: {gene_set}", flush=True)

    df = build_twinfer_input(args.dataset, gene_set)
    print(f"[{args.dataset}] TwINFER input: {len(df):,} cells, "
          f"{df['clone_id'].nunique():,} clones with >=2 cells", flush=True)

    out_dir = f"{TWINFER_PROJECT_ROOT}/analysis_data/{label}/data"
    os.makedirs(out_dir, exist_ok=True)
    input_csv = os.path.join(out_dir, f"twinfer_input_{label}_{args.gene_set}.csv")
    df.to_csv(input_csv, index=False)
    print(f"[{args.dataset}] wrote {input_csv}", flush=True)

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
    out_csv = os.path.join(out_dir, f"ranked_edges_{label}_{args.gene_set}.csv")
    ranked_edges.to_csv(out_csv, index=False)
    print(f"[{args.dataset}] wrote {out_csv} ({len(ranked_edges)} directed edge candidates)", flush=True)

    # TODO4v2 (see apply_todo4v2_vs_collectri.py) needs z_reg_gated per undirected gene pair
    # and a per-directed-pair z-score for rho_cross ("z_dagger"). The reference LARRY script
    # gets the latter from an analytic formula calibrated to LARRY's own null SDs
    # (analytic_zscores.DEFAULT_SD["cross"]) -- wrong to reuse for FM06/FM08's different
    # sample sizes/clone structure, so this saves the actual empirical null z-score
    # infer_with_twinfer already computed for this dataset instead.
    z_reg_gated = {
        f"{a}__{b}": (None if v is None or not np.isfinite(v) else float(v))
        for (a, b), v in result["gated_regulation"]["z_reg_gated"].items()
    }
    z_reg_path = os.path.join(out_dir, f"z_reg_gated_{label}_{args.gene_set}.json")
    json.dump(z_reg_gated, open(z_reg_path, "w"))
    print(f"[{args.dataset}] wrote {z_reg_path}", flush=True)

    z_dagger = {
        f"{a}__{b}": float(details["z_rho_cross"])
        for (a, b), details in result["direction"]["rho_cross_null"].items()
        if np.isfinite(details["z_rho_cross"])
    }
    z_dagger_path = os.path.join(out_dir, f"z_dagger_{label}_{args.gene_set}.json")
    json.dump(z_dagger, open(z_dagger_path, "w"))
    print(f"[{args.dataset}] wrote {z_dagger_path}", flush=True)

    print(f"[{args.dataset}] classification: {result['classification']}", flush=True)
    if len(ranked_edges):
        print(ranked_edges[["gene_1", "gene_2", "twinScore"]]
              .sort_values("twinScore", ascending=False).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
