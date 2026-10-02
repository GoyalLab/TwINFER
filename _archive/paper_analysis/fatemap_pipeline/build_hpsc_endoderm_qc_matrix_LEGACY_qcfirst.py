#!/usr/bin/env python3
"""hPSC_20260927 (endo_T0/endo_T1, two-timepoint hESC->definitive-endoderm lineage-tracing
data) QC pipeline: RNA-QC threshold (n_genes>=1000, mito<=10%, chosen from the observed
per-cell QC distributions -- see analysis_data/hPSC_20260927/plots/hPSC_20260927_qc_distributions.png)
+ singletCode doublet removal (criteria 1-4, no manual UMI pre-filter, auto ratio-based
cutoff) on the POOLED stepThree table (BC50StarcodeD7 collapsed jointly across both
timepoints, so a clone shared across endo_T0/endo_T1 gets an identical barcode string in
both samples -- required for cross-sample twin construction in infer_with_twinfer's t1/t2
direction stage).

Unlike HS054 (single timepoint, no batch/replicate structure needing PCA/Harmony
integration), endo_T0/endo_T1 are two REAL timepoints of one lineage, not two replicates of
one timepoint -- normalization is a direct per-cell log1p(CP10k), computed later per gene set
in build_twinfer_input (mirrors larry_hematopoiesis_validation/apply_twinscore_supplement_larry.py's
load_raw()), not via fatemap_integration_utils's batch-correction recipe (no HVG/PCA/Harmony
step is needed here: infer_with_twinfer consumes raw log1p(CP10k) expression per gene
directly, and there is only one library per timepoint -- nothing to batch-integrate).

Writes the full-gene (38,606 x N_qc_cells) raw-count matrix + obs metadata (sample,
time_step, QC values, singletCode clone_id) for both timepoints combined, mirroring
build_hs054_qc_matrix.py's output layout so downstream gene-set-specific
build_twinfer_input functions can reuse the same pattern.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import gc
import json
import os
import sys

import h5py
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline")
from paper_analysis.fatemap_pipeline.fatemap_qc_utils import run_singlet_calling

DATASET = "hPSC_20260927"
SAMPLES = ["endo_T0", "endo_T1"]
TIME_STEP = {"endo_T0": 0, "endo_T1": 1}
H5_PATHS = {
    "endo_T0": "/projects/b1042/GoyalLab/Hanxiao/twinfer_diff/cellranger_out/endo_T0/outs/filtered_feature_bc_matrix.h5",
    "endo_T1": "/projects/b1042/GoyalLab/Hanxiao/twinfer_diff/cellranger_out/endo_T1/outs/filtered_feature_bc_matrix.h5",
}
POOLED_STARCODE_PATH = (
    "/projects/b1042/GoyalLab/Hanxiao/twinfer_diff/sidereaction/stepThreed7/"
    "pooled/stepThreeStarcodeShavedReads_POOLED.txt"
)
BARCODE_COL = "BC50StarcodeD7"
MIN_GENES = 1000
MAX_MITO = 10.0
OUT_DIR = f"{TWINFER_PROJECT_ROOT}/finalized_data/{DATASET}_data/qc_filtered"


def load_h5_csc(path):
    with h5py.File(path, "r") as f:
        grp = f["matrix"]
        shape = grp["shape"][:]
        data = grp["data"][:]
        indices = grp["indices"][:]
        indptr = grp["indptr"][:]
        barcodes = grp["barcodes"][:].astype(str)
        gene_names = grp["features"]["name"][:].astype(str)
    mat = sp.csc_matrix((data, indices, indptr), shape=tuple(shape))
    cell_barcodes = np.array([b.replace("-1", "") for b in barcodes])
    return mat, gene_names, cell_barcodes


def build_lineage_sample_sheet(path, barcode_col=BARCODE_COL):
    df = pd.read_csv(path, sep="\t", usecols=["sample", "cellID", "UMI", barcode_col])
    df = df.drop_duplicates(["sample", "cellID", "UMI", barcode_col]).rename(columns={barcode_col: "barcode"})
    return df[["cellID", "barcode", "sample"]].reset_index(drop=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- 1. per-cell QC metrics + RNA-QC pass mask, one sample at a time (avoid OOM) ----
    qc_frames = []
    gene_names_ref = None
    per_sample_mat = {}
    for sample in SAMPLES:
        print(f"[{DATASET}] {sample}: loading {H5_PATHS[sample]}", flush=True)
        mat, gene_names, cell_barcodes = load_h5_csc(H5_PATHS[sample])
        if gene_names_ref is None:
            gene_names_ref = gene_names
        else:
            assert np.array_equal(gene_names_ref, gene_names), f"{sample}: gene list mismatch"

        mito_mask = np.array([g.upper().startswith("MT-") for g in gene_names])
        total_counts = np.asarray(mat.sum(axis=0)).ravel().astype(float)
        n_genes_per_cell = np.asarray((mat > 0).sum(axis=0)).ravel().astype(float)
        mito_counts = np.asarray(mat[mito_mask, :].sum(axis=0)).ravel().astype(float)
        pct_mt = np.divide(100.0 * mito_counts, total_counts, out=np.zeros_like(total_counts), where=total_counts > 0)

        qc_frames.append(pd.DataFrame({
            "sample": sample, "cell_barcode": cell_barcodes,
            "total_counts": total_counts, "n_genes": n_genes_per_cell, "pct_counts_mt": pct_mt,
        }))
        per_sample_mat[sample] = (mat, cell_barcodes)
        print(f"[{DATASET}] {sample}: {mat.shape[1]} raw cells", flush=True)
        gc.collect()

    qc = pd.concat(qc_frames, ignore_index=True)
    qc["pass_qc"] = (qc["n_genes"] >= MIN_GENES) & (qc["pct_counts_mt"] <= MAX_MITO)
    n_raw = qc.groupby("sample").size()
    n_pass = qc[qc["pass_qc"]].groupby("sample").size()
    print(f"[{DATASET}] RNA-QC threshold (n_genes>={MIN_GENES}, mito<={MAX_MITO}%): "
          f"{dict(n_raw)} -> {dict(n_pass)}", flush=True)

    # ---- 2. singletCode on the lineage-barcode sample sheet, restricted to RNA-QC-pass cellIDs ----
    print(f"[{DATASET}] building pooled lineage-barcode sample sheet", flush=True)
    sheet = build_lineage_sample_sheet(POOLED_STARCODE_PATH)
    pass_keys = set(zip(qc.loc[qc["pass_qc"], "sample"], qc.loc[qc["pass_qc"], "cell_barcode"]))
    sheet = sheet[[(s, c) in pass_keys for c, s in zip(sheet["cellID"], sheet["sample"])]].reset_index(drop=True)
    print(f"[{DATASET}] sample sheet rows after RNA-QC restriction: {len(sheet)}", flush=True)

    singlets, singlet_clone, rescued_cellids = run_singlet_calling(
        sheet, dataset_name=DATASET, min_umi_cutoff=None
    )
    n_singlets = singlets.groupby("sample").size()
    print(f"[{DATASET}] singletCode + rescue singlets (post RNA-QC): {dict(n_singlets)} "
          f"({len(set(rescued_cellids))} distinct cellIDs rescued by criterion 4)", flush=True)
    clone_lookup = {(s, c): bc for (s, c), bc in singlet_clone.items()}

    # ---- 3. assemble final QC+singlet-filtered matrix, both samples concatenated ----
    keep_mats, keep_cells, obs_rows = [], [], []
    for sample in SAMPLES:
        mat, cell_barcodes = per_sample_mat[sample]
        qc_s = qc[qc["sample"] == sample].set_index("cell_barcode")
        keep_idx, kept_ids, kept_clone, kept_rescued = [], [], [], []
        for i, cb in enumerate(cell_barcodes):
            key = (sample, cb)
            if key in clone_lookup:
                keep_idx.append(i)
                kept_ids.append(cb)
                kept_clone.append(clone_lookup[key])
                kept_rescued.append(cb in set(rescued_cellids))
        mat_keep = mat[:, keep_idx]
        keep_mats.append(mat_keep)
        cell_labels = [f"{sample}:{cb}" for cb in kept_ids]
        keep_cells.extend(cell_labels)
        qc_keep = qc_s.loc[kept_ids]
        obs_rows.append(pd.DataFrame({
            "sample": sample,
            "time_step": TIME_STEP[sample],
            "cell_barcode": kept_ids,
            "pct_counts_mt": qc_keep["pct_counts_mt"].to_numpy(),
            "n_genes": qc_keep["n_genes"].to_numpy(),
            "total_counts": qc_keep["total_counts"].to_numpy(),
            "fatemap_clone_singletcode": kept_clone,
            "criterion4_rescued": kept_rescued,
        }, index=cell_labels))
        print(f"[{DATASET}] {sample}: {mat_keep.shape[1]} cells kept (RNA-QC + singlet)", flush=True)
        del mat
        gc.collect()

    X_qc = sp.hstack(keep_mats).tocsr()
    obs_qc = pd.concat(obs_rows)
    assert X_qc.shape[1] == len(obs_qc) == len(keep_cells)
    assert obs_qc.index.is_unique

    sio.mmwrite(f"{OUT_DIR}/{DATASET}_qc_counts.mtx", X_qc)
    open(f"{OUT_DIR}/genes.txt", "w").write("\n".join(gene_names_ref) + "\n")
    open(f"{OUT_DIR}/cells.txt", "w").write("\n".join(keep_cells) + "\n")
    obs_qc.to_csv(f"{OUT_DIR}/obs_metadata.csv")

    summary = {
        "dataset": DATASET,
        "samples": SAMPLES,
        "n_raw_cells": {s: int(n_raw[s]) for s in SAMPLES},
        "n_after_rna_qc": {s: int(n_pass.get(s, 0)) for s in SAMPLES},
        "n_after_singletcode_plus_rescue": {s: int(n_singlets.get(s, 0)) for s in SAMPLES},
        "min_genes": MIN_GENES,
        "max_mito_pct": MAX_MITO,
        "singletcode_min_umi_cutoff": "auto (singletCode ratio-based default; observed cutoff=2 for both samples)",
        "n_distinct_cellids_rescued_by_criterion4": len(set(rescued_cellids)),
        "final_n_cells": int(X_qc.shape[1]),
        "starcode_source": POOLED_STARCODE_PATH,
        "rna_matrix_source": H5_PATHS,
    }
    open(f"{OUT_DIR}/preprocessing_summary.json", "w").write(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[{DATASET}] wrote {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
