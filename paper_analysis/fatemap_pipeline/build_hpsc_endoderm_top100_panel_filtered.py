#!/usr/bin/env python3
"""Filtered variant of the top100 panel: 40 most-measured TFs (Lambert catalogue, by total
expression, unfiltered as before) + 60 most-expressed TARGET genes restricted to the
target gene universe (hpsc_endoderm_target_gene_universe.json -- detected in >=5% of QC
cells, minus the GO cell-cycle/Tirosh S-G2M/histone/MALAT1-NEAT1-XIST-FIRRE/mito exclusion
list built earlier this pipeline). Still no ChIP-Atlas/Perturb-seq restriction -- this only
removes cell-cycle/translation-machinery/mito/histone housekeeping genes from the TARGET
pool so expression-magnitude ranking doesn't just rediscover the ribosome/mitochondrion.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-10-01 added: replaces hardcoded project-root paths]
import io
import json

import numpy as np
import pandas as pd
import requests
import scipy.io as sio

# QC_DIR = "/gpfs/projects/b1255/hzhang/TwINFER_KA/finalized_data/hPSC_20260927_data/qc_filtered"   # [2026-10-01 replaced by TWINFER_PROJECT_ROOT]
QC_DIR = f'{TWINFER_PROJECT_ROOT}/finalized_data/hPSC_20260927_data/qc_filtered'
# OUT_DIR = "/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/hPSC_20260927/data"   # [2026-10-01 replaced by TWINFER_PROJECT_ROOT]
OUT_DIR = f'{TWINFER_PROJECT_ROOT}/analysis_data/hPSC_20260927/data'
TARGET_UNIVERSE_PATH = f"{OUT_DIR}/hpsc_endoderm_target_gene_universe.json"
LAMBERT_URL = "https://humantfs.ccbr.utoronto.ca/download/v_1.01/DatabaseExtract_v_1.01.csv"
N_TFS = 40
N_TARGETS = 60
MAX_CLONE_SIZE = 50


def main():
    X = sio.mmread(f"{QC_DIR}/hPSC_20260927_qc_counts.mtx").tocsr()
    genes = np.array(open(f"{QC_DIR}/genes.txt").read().split())
    cells = np.array(open(f"{QC_DIR}/cells.txt").read().split())
    obs = pd.read_csv(f"{QC_DIR}/obs_metadata.csv", index_col=0)

    total_expr = np.asarray(X.sum(axis=1)).ravel()
    expr_series = pd.Series(total_expr, index=genes).groupby(level=0).sum()  # dedupe any repeated symbols

    r = requests.get(LAMBERT_URL, timeout=60)
    r.raise_for_status()
    tf_df = pd.read_csv(io.StringIO(r.text))
    lambert_tfs = set(tf_df.loc[tf_df["Is TF?"] == "Yes", "HGNC symbol"].dropna().astype(str))

    tf_candidates = expr_series[expr_series.index.isin(lambert_tfs)].sort_values(ascending=False)
    top_tfs = tf_candidates.head(N_TFS).index.tolist()
    print(f"top {N_TFS} most-measured TFs (by total expression): {top_tfs}", flush=True)

    target_universe = set(json.load(open(TARGET_UNIVERSE_PATH)))
    non_tf_candidates = expr_series[
        expr_series.index.isin(target_universe) & ~expr_series.index.isin(set(top_tfs))
    ].sort_values(ascending=False)
    top_targets = non_tf_candidates.head(N_TARGETS).index.tolist()
    print(f"top {N_TARGETS} most-expressed non-TF genes (restricted to the {len(target_universe)}-gene "
          f"exclusion-filtered universe): {top_targets}", flush=True)

    panel = sorted(set(top_tfs) | set(top_targets))
    print(f"\nfinal panel: {len(panel)} genes ({len(top_tfs)} TFs + {len(top_targets)} targets)", flush=True)
    json.dump({"tfs": top_tfs, "targets": top_targets, "panel": panel},
              open(f"{OUT_DIR}/hpsc_endoderm_top100_panel.json", "w"))

    gidx = {g: i for i, g in enumerate(genes)}
    col = [gidx[g] for g in panel]
    cell_total = np.asarray(X.sum(axis=0)).ravel().astype(float)
    cell_total[cell_total == 0] = 1.0
    expr = np.log1p(X[col, :].toarray().astype(float) / cell_total[None, :] * 1e4).T  # cells x genes

    df = pd.DataFrame(expr, columns=[f"{g}_mRNA" for g in panel], index=cells)
    df.insert(0, "time_step", obs["time_step"].to_numpy())
    df.insert(0, "cell_id", obs["cell_barcode"].to_numpy())
    df.insert(0, "clone_id", obs["fatemap_clone_singletcode"].astype(str).to_numpy())
    df = df.reset_index(drop=True)

    counts = df["clone_id"].value_counts()
    oversized = counts[counts > MAX_CLONE_SIZE]
    if len(oversized):
        print(f"dropping {len(oversized)} oversized clone(s) (>{MAX_CLONE_SIZE} cells)", flush=True)
    keep_clones = counts[(counts >= 2) & (counts <= MAX_CLONE_SIZE)].index
    n_before = len(df)
    df = df[df["clone_id"].isin(keep_clones)].reset_index(drop=True)
    print(f"clone-size filter: {n_before:,} -> {len(df):,} cells ({len(keep_clones):,} clones)", flush=True)
    print(df["time_step"].value_counts().to_dict(), flush=True)

    out_csv = f"{OUT_DIR}/twinfer_input_hpsc_20260927_top100.csv"
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv} ({len(df):,} cells x {len(panel)} genes)", flush=True)


if __name__ == "__main__":
    main()
