#!/usr/bin/env python3
"""Builds the canonical TwINFER input (clone_id, cell_id, time_step, {gene}_mRNA) for
hPSC_20260927 (endo_T0/endo_T1) from the QC+singletCode-filtered raw-count matrix
(build_hpsc_endoderm_qc_matrix.py's output), log1p(CP10k) normalized per cell (no
Harmony/scanorama batch integration needed -- only one library per timepoint, see that
script's docstring), restricted to the final ChIP-Atlas+Perturb-seq-informed gene panel
(pick_hpsc_endoderm_gene_panel.py + expand_nanog_oct4_targets.py, 489 genes). Real two
-timepoint data: time_step is 0 for endo_T0 cells, 1 for endo_T1 cells (t1=0, t2=1 passed
to infer_with_twinfer in run_infer_hpsc_endoderm.py) -- NOT the t1=t2 single-timepoint
shortcut HS054/FM06/FM08 use.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
import os

import numpy as np
import pandas as pd
import scipy.io as sio

QC_DIR = f'{TWINFER_PROJECT_ROOT}/finalized_data/hPSC_20260927_data/qc_filtered'
GENE_PANEL_PATH = f'{TWINFER_PROJECT_ROOT}/analysis_data/hPSC_20260927/data/hpsc_endoderm_final_gene_panel.json'
OUT_DIR = f'{TWINFER_PROJECT_ROOT}/analysis_data/hPSC_20260927/data'
LABEL = "hpsc_20260927"
MAX_CLONE_SIZE = 50


def build_twinfer_input():
    X = sio.mmread(f"{QC_DIR}/hPSC_20260927_qc_counts.mtx").tocsc()
    genes = np.array(open(f"{QC_DIR}/genes.txt").read().split())
    cells = np.array(open(f"{QC_DIR}/cells.txt").read().split())
    obs = pd.read_csv(f"{QC_DIR}/obs_metadata.csv", index_col=0)
    assert list(obs.index) == list(cells)

    gene_panel = json.load(open(GENE_PANEL_PATH))
    gidx = {g: i for i, g in enumerate(genes)}
    missing = [g for g in gene_panel if g not in gidx]
    if missing:
        raise ValueError(f"gene panel has genes not in matrix: {missing}")
    col = [gidx[g] for g in gene_panel]

    total = np.asarray(X.sum(axis=0)).ravel().astype(float)
    total[total == 0] = 1.0
    expr = np.log1p(X[col, :].toarray().astype(float) / total[None, :] * 1e4).T  # cells x genes

    df = pd.DataFrame(expr, columns=[f"{g}_mRNA" for g in gene_panel], index=cells)
    df.insert(0, "time_step", obs["time_step"].to_numpy())
    df.insert(0, "cell_id", obs["cell_barcode"].to_numpy())
    df.insert(0, "clone_id", obs["fatemap_clone_singletcode"].astype(str).to_numpy())
    df = df.reset_index(drop=True)

    counts = df["clone_id"].value_counts()
    oversized = counts[counts > MAX_CLONE_SIZE]
    if len(oversized):
        print(f"[hPSC_20260927] dropping {len(oversized)} oversized clone(s) (>{MAX_CLONE_SIZE} cells): "
              f"{oversized.head(5).to_dict()}", flush=True)
    keep_clones = counts[(counts >= 2) & (counts <= MAX_CLONE_SIZE)].index
    n_before = len(df)
    df = df[df["clone_id"].isin(keep_clones)].reset_index(drop=True)
    print(f"[hPSC_20260927] clone-size filter: {n_before:,} -> {len(df):,} cells "
          f"({len(keep_clones):,} clones, 2-{MAX_CLONE_SIZE} cells each)", flush=True)
    print(df["time_step"].value_counts().to_dict(), flush=True)
    return df, gene_panel


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df, gene_panel = build_twinfer_input()
    out_csv = f"{OUT_DIR}/twinfer_input_{LABEL}_chipatlas_perturbseq_panel.csv"
    df.to_csv(out_csv, index=False)
    print(f"[hPSC_20260927] wrote {out_csv} ({len(df):,} cells x {len(gene_panel)} genes)", flush=True)


if __name__ == "__main__":
    main()
