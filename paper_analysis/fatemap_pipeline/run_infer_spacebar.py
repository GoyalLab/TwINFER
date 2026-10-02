#!/usr/bin/env python3
"""SpaceBar (spatial, lineage-barcoded melanoma tumor sections) TwINFER inference, on the
overlap between SpaceBar's fixed 119-gene targeted panel and FM06's yscher-style gene sets
(correlation_high: 12 shared genes; variability_high: 15 shared genes -- see conversation).

Data: real_data/SpaceBar_data/Section{1..5}_cell_by_gene_clustered.csv. clone_id = bc_cluster,
prefixed per-section ("S{i}_{cluster}") since cluster IDs are local to each section, not global
lineages. Unlike FM06/FM08 (barcode-collision artifacts -> hard exclusion), SpaceBar's oversized
clones are REAL clonal expansion (untreated tumor -- TODO.md: "Fatemap 06 - untreated tumor -
matched with SpaceBar"): median clone size is 1, but the tail runs out to 4,105 cells (a few
founder subclones sweeping the tumor). Per 2026-09-22 user instruction: subsample (not exclude)
clones above MAX_CLONE_SIZE=30 cells, keeping every clone represented while capping worst-case
per-clone pairs at C(30,2)=435 (measured: ~500K total pairs across all clones at this cap, same
scale as the FM06/FM08 runs).

Single timepoint (all sections pooled, t1=t2=0), same gated-regulation Stage-3 bypass as
run_infer_fatemap.py.
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

RAW_DIR = f'{TWINFER_PROJECT_ROOT}/real_data/SpaceBar_data'
# 2026-09-22: restricted to Section2+Section4 for the preliminary analysis -- pseudobulk
# gene-expression correlation across all 5 sections showed these two are by far the most
# similar pair (Pearson r=0.998, Spearman rho=0.970 on log1pCP10k profiles over the 115
# genes common to all sections), and they also share an identical gene panel (both missing
# the same 6 genes vs. the union panel).
SECTIONS = [2, 4]
TIME_STEP = 0
MAX_CLONE_SIZE = 30
SEED = 101010

GENE_SETS = {
    # FM06 correlation_high (34 genes) intersected with SpaceBar's 119-gene panel
    "correlation_high": sorted([
        "CCND1", "CDK1", "FN1", "FOS", "FOSB", "FOSL1", "JUN", "MITF",
        "MLANA", "MYBL2", "MYC", "SOX10",
    ]),
    # FM06 variability_high (48 genes) intersected with SpaceBar's 119-gene panel.
    # IGFBP7 dropped: not present in Section2/Section4's gene panel (present only in
    # Section1/Section3).
    "variability_high": sorted([
        "CCND1", "CDK1", "CDKN1B", "FN1", "FOS", "FOSB", "FOSL1",
        "JUN", "MITF", "MLANA", "MMP1", "MYC", "SERPINE1", "TFAP2A",
    ]),
}


def load_all_sections(gene_set):
    frames = []
    for i in SECTIONS:
        cols = ["cell_id", "n_called_barcodes", "bc_cluster"] + gene_set
        path = f"{RAW_DIR}/Section{i}_cell_by_gene_clustered.csv"
        df = pd.read_csv(path, usecols=cols)
        df = df[df["bc_cluster"].notna()].copy()
        df["clone_id"] = f"S{i}_" + df["bc_cluster"].astype(int).astype(str)
        df["section"] = f"S{i}"
        df["cell_id"] = f"S{i}_" + df["cell_id"].astype(str)
        frames.append(df)
        print(f"[SpaceBar] Section{i}: {len(df):,} cells with a called clone", flush=True)
    all_df = pd.concat(frames, ignore_index=True)
    print(f"[SpaceBar] all sections: {len(all_df):,} cells, "
          f"{all_df['clone_id'].nunique():,} distinct clones", flush=True)
    return all_df


def subsample_oversized_clones(df, max_size, seed=SEED):
    rng = np.random.default_rng(seed)
    keep_idx = []
    n_subsampled = 0
    for clone_id, idx in df.groupby("clone_id").indices.items():
        if len(idx) > max_size:
            idx = rng.choice(idx, size=max_size, replace=False)
            n_subsampled += 1
        keep_idx.append(idx)
    keep_idx = np.concatenate(keep_idx)
    out = df.iloc[np.sort(keep_idx)].reset_index(drop=True)
    print(f"[SpaceBar] subsampled {n_subsampled:,} clones down to {max_size} cells each "
          f"({len(df):,} -> {len(out):,} cells)", flush=True)
    return out


def build_twinfer_input(gene_set_name, max_clone_size):
    genes = GENE_SETS[gene_set_name]
    df = load_all_sections(genes)
    df = subsample_oversized_clones(df, max_clone_size)

    counts = df["clone_id"].value_counts()
    keep_clones = counts[counts >= 2].index
    n_before = len(df)
    df = df[df["clone_id"].isin(keep_clones)].reset_index(drop=True)
    print(f"[SpaceBar] clone-size>=2 filter: {n_before:,} -> {len(df):,} cells "
          f"({len(keep_clones):,} clones)", flush=True)

    total_counts = df[genes].sum(axis=1).to_numpy().astype(float)
    total_counts[total_counts == 0] = 1.0
    mat = np.log1p(df[genes].to_numpy(dtype=float) / total_counts[:, None] * 1e4)
    out = pd.DataFrame(mat, columns=[f"{g}_mRNA" for g in genes])
    out.insert(0, "time_step", TIME_STEP)
    out.insert(0, "cell_id", df["cell_id"].to_numpy())
    out.insert(0, "clone_id", df["clone_id"].to_numpy())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene-set", choices=list(GENE_SETS), default="correlation_high")
    ap.add_argument("--n-shuffles", type=int, default=2000)
    ap.add_argument("--n-cores", type=int, default=8)
    ap.add_argument("--max-clone-size", type=int, default=MAX_CLONE_SIZE)
    args = ap.parse_args()

    out_dir = f'{TWINFER_PROJECT_ROOT}/analysis_data/spacebar/data'
    os.makedirs(out_dir, exist_ok=True)
    # cap-specific suffix so a non-default --max-clone-size run never clobbers the MAX_CLONE_SIZE=30
    # outputs already analyzed (empty suffix for the default cap keeps existing filenames/paths).
    suffix = "" if args.max_clone_size == MAX_CLONE_SIZE else f"_cap{args.max_clone_size}"

    df = build_twinfer_input(args.gene_set, args.max_clone_size)
    input_csv = os.path.join(out_dir, f"twinfer_input_spacebar_{args.gene_set}{suffix}.csv")
    df.to_csv(input_csv, index=False)
    print(f"[SpaceBar] wrote {input_csv}", flush=True)

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
        seed=SEED,
        verbose=True,
        plot=False,
    )

    ranked_edges = result["ranked_edges"]
    out_csv = os.path.join(out_dir, f"ranked_edges_spacebar_{args.gene_set}{suffix}.csv")
    ranked_edges.to_csv(out_csv, index=False)
    print(f"[SpaceBar] wrote {out_csv} ({len(ranked_edges)} directed edge candidates)", flush=True)

    z_reg_gated = {
        f"{a}__{b}": (None if v is None or not np.isfinite(v) else float(v))
        for (a, b), v in result["gated_regulation"]["z_reg_gated"].items()
    }
    json.dump(z_reg_gated, open(os.path.join(out_dir, f"z_reg_gated_spacebar_{args.gene_set}{suffix}.json"), "w"))

    z_dagger = {
        f"{a}__{b}": float(d["z_rho_cross"])
        for (a, b), d in result["direction"]["rho_cross_null"].items()
        if np.isfinite(d["z_rho_cross"])
    }
    json.dump(z_dagger, open(os.path.join(out_dir, f"z_dagger_spacebar_{args.gene_set}{suffix}.json"), "w"))

    print(f"[SpaceBar] classification: {result['classification']}", flush=True)
    if len(ranked_edges):
        print(ranked_edges[["gene_1", "gene_2", "twinScore"]]
              .sort_values("twinScore", ascending=False).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
