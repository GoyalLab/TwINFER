#!/usr/bin/env python3
"""2026-09-23: real (non-degenerate) A/B split for SpaceBar, built from within-clone SPATIAL
STRUCTURE rather than a second replicate (SpaceBar has none -- clone_id is section-local, see
run_infer_spacebar_allpairs.py's docstring).

Investigated in conversation: single-cell "twin vs cousin" splits (farthest-anchor pair,
per-clone) don't separate cleanly -- within-clone pairwise transcriptional distance is nearly
flat (CV~0.09-0.10 in every section) and a single nearest/farthest PAIR per clone is too noisy
(only ~54% of clones order correctly even after correcting for the generic spatial-autocorrelation
background). What DOES carry a clean, non-degenerate signal is each cell's distance to its own
clone's SPATIAL CENTROID, aggregated over the whole clone (not a single pair): central vs.
peripheral halves differ from the spatial background at p=6e-77 (vs p=3.9e-15 for the pair
version), and critically, a direct rho_cross_xy vs rho_cross_yx check on this split gave a mean
|diff| of 0.023 (max 0.075, real numbers, not the exact-zero degeneracy of the old pooled/
random-split construction).

Construction: per clone (n>=3, both sections 2+4), median-split cells by distance to the
clone's spatial centroid -> time_step=0 ("central", A) / time_step=1 ("peripheral", B). Keep
only clones spanning both sides; cap each side at MAX_SIDE cells (same oversized-clone guard
as run_infer_fatemap_ab_split.py, protects against a handful of huge clones dominating pairs).
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/home/gzu5140/TwINFER_KA/code/TwINFER/package")
from twinfer.inference.infer import infer_with_twinfer

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_analysis.fatemap_pipeline.run_infer_spacebar import GENE_SETS as _OLD_GENE_SETS, RAW_DIR, SECTIONS, SEED

# 2026-09-23: SpaceBar-native correlation_{high,mid,low} (pick_gene_sets_spacebar.py, FM06 recipe)
# replace the FM06-intersection sets ("correlation_high" now means the native 29-gene set).
GENE_SETS = dict(_OLD_GENE_SETS)
GENE_SETS.update(json.load(open(f'{TWINFER_PROJECT_ROOT}/analysis_data/spacebar/data/gene_sets_spacebar.json')))

ALPHA_PERMISSIVE = 0.999
MAX_SIDE = 30

_META = {"object_id", "cell_id", "n_called_barcodes", "called_barcodes", "barcode_names",
         "nucleus_centroid", "center_x", "center_y", "bc_cluster", "bc_cluster_n_bcs",
         "bc_cluster_found_bcs", "bc_cluster_bc_names", "Unnamed: 0", "nucleus",
         "nucleus_dilated", "area"}


def _full_gene_panel():
    """SpaceBar's raw per-section files ARE the QC'd data (2026-09-23 user correction --
    Section{i}_cell_by_gene_clustered.csv is the raw-count, already-QC'd matrix, same role
    FM06/FM08's qc_filtered .mtx plays). Section2 and Section4 share an IDENTICAL 114-gene
    panel (verified), so no intersection logic is needed."""
    hdr = pd.read_csv(f"{RAW_DIR}/Section{SECTIONS[0]}_cell_by_gene_clustered.csv", nrows=0).columns.tolist()
    bc_cols = [c for c in hdr if c.startswith("bc_0")]
    return sorted(c for c in hdr if c not in _META and c not in bc_cols)


def build_ab_split_with_full_panel(gene_set_name):
    """Builds the centroid-split A/B twinfer input (target gene_set, log1p(CP10k)) AND, aligned
    to the exact same final retained cells/order, the RAW integer-count matrix over the FULL
    114-gene panel (needed for Wz's split-half binomial-thinning reliability estimate, which
    requires true raw counts + whole-panel library size, not just the target gene subset)."""
    genes = GENE_SETS[gene_set_name]
    genes_full = _full_gene_panel()
    frames = []
    raw_count_frames = []
    for i in SECTIONS:
        cols = ["cell_id", "n_called_barcodes", "bc_cluster", "center_x", "center_y"] + genes_full
        path = f"{RAW_DIR}/Section{i}_cell_by_gene_clustered.csv"
        df = pd.read_csv(path, usecols=cols)
        df = df[df["bc_cluster"].notna()].copy()
        df["clone_id"] = f"S{i}_" + df["bc_cluster"].astype(int).astype(str)
        df["cell_id"] = f"S{i}_" + df["cell_id"].astype(str)

        raw_counts_full = df[genes_full].to_numpy(dtype=float)
        total_counts = df[genes].sum(axis=1).to_numpy().astype(float)  # target-panel-only, matches run_infer_spacebar.py
        total_counts[total_counts == 0] = 1.0
        mat = np.log1p(df[genes].to_numpy(dtype=float) / total_counts[:, None] * 1e4)
        for j, g in enumerate(genes):
            df[f"{g}_mRNA"] = mat[:, j]
        xy = df[["center_x", "center_y"]].to_numpy()

        # 2026-09-23: subsample (not drop) oversized clones BEFORE the centroid split, per the
        # SpaceBar convention (oversized clones are real expansion, see run_infer_spacebar.py).
        # 2*MAX_SIDE total -> at most MAX_SIDE per side after the median split.
        rng = np.random.default_rng(SEED)
        keep_rows = np.ones(len(df), dtype=bool)
        for cid, idx in df.groupby("clone_id").indices.items():
            if len(idx) > 2 * MAX_SIDE:
                drop = np.setdiff1d(idx, rng.choice(idx, 2 * MAX_SIDE, replace=False))
                keep_rows[drop] = False
        raw_counts_full = raw_counts_full[keep_rows]
        df = df[keep_rows].reset_index(drop=True)
        xy = df[["center_x", "center_y"]].to_numpy()

        df["time_step"] = -1
        for cid, idx in df.groupby("clone_id").indices.items():
            if len(idx) < 3:
                continue
            xy_c = xy[idx]
            centroid = xy_c.mean(axis=0)
            d = np.linalg.norm(xy_c - centroid, axis=1)
            med = np.median(d)
            ts = (d > med).astype(int)  # 0 = central (A), 1 = peripheral (B)
            df.iloc[idx, df.columns.get_loc("time_step")] = ts
        keep = (df["time_step"] >= 0).to_numpy()
        df = df[keep].reset_index(drop=True)
        raw_count_frames.append(raw_counts_full[keep])
        frames.append(df)
        print(f"[SpaceBar/{gene_set_name}] Section{i}: {len(df):,} cells (clone n>=3, "
              f"centroid split assigned)", flush=True)

    all_df = pd.concat(frames, ignore_index=True)
    raw_counts_full = np.concatenate(raw_count_frames, axis=0)

    span = all_df.groupby("clone_id")["time_step"].nunique()
    spanning = span[span == 2].index
    n_before = len(all_df)
    keep = all_df["clone_id"].isin(spanning).to_numpy()
    all_df = all_df[keep].reset_index(drop=True)
    raw_counts_full = raw_counts_full[keep]
    print(f"[SpaceBar/{gene_set_name}] A/B-spanning filter: {n_before:,} -> {len(all_df):,} cells "
          f"({len(spanning):,} clones)", flush=True)

    per_side = all_df.groupby(["clone_id", "time_step"]).size()
    oversized = per_side[per_side > MAX_SIDE].index.get_level_values("clone_id").unique()
    keep = (~all_df["clone_id"].isin(oversized)).to_numpy()
    all_df = all_df[keep].reset_index(drop=True)
    raw_counts_full = raw_counts_full[keep]
    print(f"[SpaceBar/{gene_set_name}] after oversize-side cap({MAX_SIDE}): {len(all_df):,} cells, "
          f"{all_df['clone_id'].nunique():,} clones (A: {(all_df.time_step==0).sum():,}, "
          f"B: {(all_df.time_step==1).sum():,})", flush=True)

    out = all_df[["clone_id", "cell_id", "time_step"] + [f"{g}_mRNA" for g in genes]].reset_index(drop=True)
    X_full = csr_matrix(raw_counts_full)
    return out, X_full, genes_full


def build_twinfer_input_centroid_ab(gene_set_name):
    out, _, _ = build_ab_split_with_full_panel(gene_set_name)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene-set", choices=list(GENE_SETS), default="correlation_high")
    ap.add_argument("--n-shuffles", type=int, default=500)
    ap.add_argument("--n-cores", type=int, default=8)
    args = ap.parse_args()

    out_dir = f'{TWINFER_PROJECT_ROOT}/analysis_data/spacebar/data'
    os.makedirs(out_dir, exist_ok=True)

    df = build_twinfer_input_centroid_ab(args.gene_set)
    input_csv = os.path.join(out_dir, f"twinfer_input_spacebar_{args.gene_set}_centroid_ab.csv")
    df.to_csv(input_csv, index=False)
    print(f"[SpaceBar] wrote {input_csv}", flush=True)

    result = infer_with_twinfer(
        data=df,
        is_simulation_data=False,
        t1=0, t2=1,
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
    out_csv = os.path.join(out_dir, f"ranked_edges_spacebar_{args.gene_set}_centroid_ab_allpairs.csv")
    ranked_edges.to_csv(out_csv, index=False)
    print(f"[SpaceBar] wrote {out_csv} ({len(ranked_edges)} directed edge candidates)", flush=True)

    z_reg_gated = {
        f"{a}__{b}": (None if v is None or not np.isfinite(v) else float(v))
        for (a, b), v in result["gated_regulation"]["z_reg_gated"].items()
    }
    json.dump(z_reg_gated, open(os.path.join(
        out_dir, f"z_reg_gated_spacebar_{args.gene_set}_centroid_ab_allpairs.json"), "w"))

    z_dagger = {
        f"{a}__{b}": float(d["z_rho_cross"])
        for (a, b), d in result["direction"]["rho_cross_null"].items()
        if np.isfinite(d["z_rho_cross"])
    }
    json.dump(z_dagger, open(os.path.join(
        out_dir, f"z_dagger_spacebar_{args.gene_set}_centroid_ab_allpairs.json"), "w"))

    print(f"[SpaceBar] classification: {result['classification']}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
