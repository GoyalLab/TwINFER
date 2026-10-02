#!/usr/bin/env python3
"""Watermelon (T47D naive/lag/late x 2 replicates) QC: singletCode on ALL six samples in one call
(treated as one experiment; auto UMI cutoff -- no min_umi_cutoff passed) + cross-sample criterion-4
rescue + per-sample human MT- mito filter and gene-count floor/ceiling.

RNA barcodes carry a cellranger-aggr suffix (-1..-6 = naive-1, naive-2, lag-1, lag-2, late-1, late-2);
it is stripped so cell barcodes match the lineage table's 16bp cellID.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io as sio
import scipy.sparse as sp

from paper_analysis.fatemap_pipeline.fatemap_qc_utils import apply_qc_filters, run_singlet_calling

RAW_DIR = f'{TWINFER_PROJECT_ROOT}/real_data/watermelon_data'
BARCODE_CSV = f"{RAW_DIR}/Watermelon_barcode_cellID_UMIcounts.csv"
OUT_DIR = f'{TWINFER_PROJECT_ROOT}/finalized_data/Watermelon_data/qc_filtered'
DATASET_NAME = "Watermelon"
MIN_UMI_CUTOFF = None  # not set: singletCode auto-chooses

SAMPLES = ["T47D-naive-1", "T47D-naive-2", "T47D-lag-1", "T47D-lag-2", "T47D-late-1", "T47D-late-2"]
# per-sample QC thresholds (user-specified)
MITO_PCT = {"T47D-naive-1": 20.0, "T47D-naive-2": 20.0, "T47D-lag-1": 25.0, "T47D-lag-2": 25.0,
            "T47D-late-1": 25.0, "T47D-late-2": 25.0}
MIN_GENES = {s: 2500 for s in SAMPLES}
MAX_GENES = {"T47D-naive-1": 7500, "T47D-naive-2": 7500, "T47D-lag-1": 8000, "T47D-lag-2": 8000,
             "T47D-late-1": 8000, "T47D-late-2": 8000}


def load_sample(sample):
    """One Cell Ranger triplet -> AnnData (cells x genes), var_names = gene symbol,
    obs_names = 16bp barcode with the -N aggr suffix stripped."""
    d = f"{RAW_DIR}/{sample}"
    adata = sc.read_mtx(f"{d}/matrix.mtx.gz").T
    features = pd.read_csv(f"{d}/features.tsv.gz", sep="\t", header=None)
    barcodes = pd.read_csv(f"{d}/barcodes.tsv.gz", sep="\t", header=None)
    assert adata.shape == (len(barcodes), len(features))
    adata.var_names = features[1].to_numpy()
    adata.var_names_make_unique()
    adata.obs_names = barcodes[0].str.replace(r"-\d+$", "", regex=True).to_numpy()
    assert adata.obs_names.is_unique
    adata.obs["sample"] = sample
    return adata


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("[Watermelon] loading 6 samples", flush=True)
    adatas = [load_sample(s) for s in SAMPLES]
    var_names = adatas[0].var_names
    for a in adatas[1:]:
        assert (a.var_names == var_names).all(), "gene order differs between samples"
    X = sp.vstack([a.X for a in adatas]).tocsr()
    cell_ids = np.concatenate([a.obs_names.to_numpy() for a in adatas])
    sample = np.concatenate([a.obs["sample"].to_numpy() for a in adatas])
    n_raw = X.shape[0]
    print(f"[Watermelon] raw cells: {n_raw}", flush=True)

    # the lineage table is already a singletCode sample sheet (cellID, barcode, sample)
    sample_sheet = pd.read_csv(BARCODE_CSV)[["cellID", "barcode", "sample"]]
    print(f"[Watermelon] sample sheet rows: {len(sample_sheet)}", flush=True)
    singlets, singlet_clone, rescued_cellids = run_singlet_calling(
        sample_sheet, dataset_name=DATASET_NAME, min_umi_cutoff=MIN_UMI_CUTOFF
    )
    singlet_keys = set(zip(singlets["sample"], singlets["cellID"]))
    is_singlet = np.array([(s, c) in singlet_keys for s, c in zip(sample, cell_ids)])
    print(f"[Watermelon] singletCode: {len(singlets)} singlet rows, "
          f"{len(set(rescued_cellids))} cellIDs rescued by criterion 4, {int(is_singlet.sum())} raw cells matched", flush=True)

    # per-sample mito + gene-count filters
    n = X.shape[0]
    pct_mt, n_genes, total_counts = np.zeros(n), np.zeros(n, int), np.zeros(n)
    pass_mito, pass_genes = np.zeros(n, bool), np.zeros(n, bool)
    for s in SAMPLES:
        idx = np.flatnonzero(sample == s)
        qc = apply_qc_filters(X[idx], var_names, MITO_PCT[s], MIN_GENES[s], MAX_GENES[s])
        pct_mt[idx], n_genes[idx], total_counts[idx] = qc["pct_counts_mt"], qc["n_genes"], qc["total_counts"]
        pass_mito[idx], pass_genes[idx] = qc["pass_mito"], qc["pass_genes"]

    keep = pass_mito & is_singlet & pass_genes
    per_sample = pd.DataFrame({
        "raw": pd.Series(sample).value_counts(),
        "pass_mito": pd.Series(sample[pass_mito]).value_counts(),
        "pass_mito+singlet": pd.Series(sample[pass_mito & is_singlet]).value_counts(),
        "final": pd.Series(sample[keep]).value_counts(),
    }).loc[SAMPLES].fillna(0).astype(int)
    print(per_sample.to_string(), flush=True)

    keep_idx = np.flatnonzero(keep)
    kept_cells = pd.Series([f"{s}:{c}" for s, c in zip(sample[keep], cell_ids[keep])])
    obs_qc = pd.DataFrame({
        "sample": sample[keep],
        "stage": [s.split("-")[1] for s in sample[keep]],
        "replicate": [s.split("-")[2] for s in sample[keep]],
        "cell_barcode": cell_ids[keep],
        "pct_counts_mt": pct_mt[keep], "n_genes": n_genes[keep], "total_counts": total_counts[keep],
    }, index=kept_cells)
    clone_lookup = {f"{s}:{c}": bc for (s, c), bc in singlet_clone.items()}
    obs_qc["fatemap_clone_singletcode"] = kept_cells.map(clone_lookup).fillna("").to_numpy()
    obs_qc["criterion4_rescued"] = pd.Series(cell_ids[keep]).isin(set(rescued_cellids)).to_numpy()
    assert obs_qc.index.is_unique

    sio.mmwrite(f"{OUT_DIR}/watermelon_qc_counts.mtx", X[keep_idx])
    open(f"{OUT_DIR}/genes.txt", "w").write("\n".join(var_names) + "\n")
    open(f"{OUT_DIR}/cells.txt", "w").write("\n".join(kept_cells) + "\n")
    obs_qc.to_csv(f"{OUT_DIR}/obs_metadata.csv")
    per_sample.to_csv(f"{OUT_DIR}/per_sample_counts.csv")
    summary = {"dataset": DATASET_NAME, "n_raw_cells": int(n_raw), "n_final": int(keep.sum()),
               "mito_pct": MITO_PCT, "min_genes": MIN_GENES, "max_genes": MAX_GENES,
               "singletcode_min_umi_cutoff": "auto (singletCode default)",
               "n_distinct_cellids_rescued_by_criterion4": len(set(rescued_cellids))}
    open(f"{OUT_DIR}/preprocessing_summary.json", "w").write(json.dumps(summary, indent=2))
    print(f"[Watermelon] wrote {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
