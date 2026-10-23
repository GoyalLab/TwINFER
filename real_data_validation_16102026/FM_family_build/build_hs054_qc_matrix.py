#!/usr/bin/env python3
"""HS054_Sot48h (PDAC Sotorasib 48h) QC pipeline: singletCode doublet removal
(criteria 1-4, no manual UMI pre-filter) + human MT- mito filter (<=25%) + gene-count
floor/ceiling (200-7000). Single-sample dataset (no RNA replicates, no SampleNum column
in the raw barcode table -- unlike FM01/FM06/FM08), so the singletCode sample sheet has
exactly one "sample" value and criterion-4 cross-sample rescue can only rescue via
barcode-combination recurrence within this one sample (still run for parity with the
rest of the fatemap_pipeline, but expect little/no effect with a single sample).

RNA matrix is Cell Ranger's *filtered* (called-cell) output, re-synced from the
canonical run directory -- see preprocessing_summary.json's "matrix_source" field for
provenance (the copy that shipped in real_data/pancreatic_cancer/ initially did not
match the singletCode output sitting next to it: only 12/7342 raw-barcode-table cellIDs
overlapped; the canonical matrix below has 100% overlap, 7342/7342).
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
import os

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io as sio
import scipy.sparse as sp

from paper_analysis.fatemap_pipeline.fatemap_qc_utils import apply_qc_filters, run_singlet_calling

RNA_DIR = (
    f'{TWINFER_PROJECT_ROOT}/real_data/pancreatic_cancer/HS054_Sot48h/outs/filtered_feature_bc_matrix'
)
STARCODE_PATH = (
    "/projects/b1042/GoyalLab/Hanxiao/twinfer/barcodes/output/stepThree/"
    "HS054_Sot48h/stepThreeStarcodeShavedReads.txt"
)
OUT_DIR = f'{TWINFER_PROJECT_ROOT}/finalized_data/HS054_Sot48h_data/qc_filtered'
DATASET_NAME = "HS054_Sot48h"
SAMPLE_LABEL = "HS054_Sot48h"
BARCODE_COL = "BC50StarcodeD8"
MITO_PCT_CUTOFF = 25.0
MIN_GENES = 200
MAX_GENES = 7000
MIN_UMI_CUTOFF = None  # not set: singletCode auto-chooses its UMI cutoff (package default floor + ratio method).
# Note: a fresh run here (3040 singlets) does not exactly reproduce the singlets_all.txt
# files already sitting in real_data/pancreatic_cancer/ (3264 singlets, from an external
# singletCode run whose exact umi_cutoff_method/thresholds could not be reverse-engineered
# from a sweep over min_umi_cutoff in {None,1,2,3} x barcode column in {BC50,40,30StarcodeD8,BC30StarcodeD6}
# -- none reproduced the original's 2740-low-umi-removed/4602-classified split). Per user
# instruction, using this fresh auto-cutoff run as the singlet definition going forward;
# the old txt files are treated as superseded.


def load_10x(rna_dir, sample_label):
    adata = sc.read_mtx(f"{rna_dir}/matrix.mtx.gz").T
    features = pd.read_csv(f"{rna_dir}/features.tsv.gz", sep="\t", header=None)
    barcodes = pd.read_csv(f"{rna_dir}/barcodes.tsv.gz", sep="\t", header=None)
    assert adata.shape == (len(barcodes), len(features))
    adata.var_names = features[1].to_numpy()
    adata.var["ensembl_id"] = features[0].to_numpy()
    adata.var_names_make_unique()
    adata.obs_names = barcodes[0].str.replace(r"-1$", "", regex=True).to_numpy()
    adata.obs_names_make_unique()
    adata.obs["sample"] = sample_label
    return adata


def build_single_sample_sheet(starcode_path, sample_label, barcode_col=BARCODE_COL):
    """No SampleNum/sampleNum column in this table (unlike FM01/06/08) -- the whole
    file is one sample. One row per (cellID, UMI, barcode) read-support record, exact
    duplicates collapsed; no manual UMI pre-filter, per repo convention."""
    df = pd.read_csv(starcode_path, sep="\t", usecols=["cellID", "UMI", barcode_col])
    df = df.drop_duplicates(["cellID", "UMI", barcode_col]).rename(columns={barcode_col: "barcode"})
    df["sample"] = sample_label
    return df[["cellID", "barcode", "sample"]].reset_index(drop=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[{DATASET_NAME}] loading Cell Ranger filtered matrix", flush=True)
    adata = load_10x(RNA_DIR, SAMPLE_LABEL)
    var_names = adata.var_names
    X = adata.X.tocsr()
    cell_ids = adata.obs_names.to_numpy()
    sample = adata.obs["sample"].to_numpy()
    n_raw = X.shape[0]
    print(f"[{DATASET_NAME}] raw cells: {n_raw}", flush=True)

    print(f"[{DATASET_NAME}] building singletCode sample sheet (single sample)", flush=True)
    sample_sheet = build_single_sample_sheet(STARCODE_PATH, SAMPLE_LABEL)
    print(f"[{DATASET_NAME}] sample sheet rows: {len(sample_sheet)}", flush=True)

    print(f"[{DATASET_NAME}] running singletCode + cross-sample rescue", flush=True)
    singlets, singlet_clone, rescued_cellids = run_singlet_calling(
        sample_sheet, dataset_name=DATASET_NAME, min_umi_cutoff=MIN_UMI_CUTOFF
    )
    singlet_keys = set(zip(singlets["sample"], singlets["cellID"]))
    is_singlet = np.array([(s, c) in singlet_keys for s, c in zip(sample, cell_ids)])
    n_singlet = int(is_singlet.sum())
    print(
        f"[{DATASET_NAME}] singletCode: {len(singlets)} singlet rows, "
        f"{len(set(rescued_cellids))} distinct cellIDs rescued by criterion 4, "
        f"{n_singlet} raw cells matched to a singlet cellID",
        flush=True,
    )

    print(f"[{DATASET_NAME}] applying mito + gene-count filters", flush=True)
    qc = apply_qc_filters(X, var_names, MITO_PCT_CUTOFF, MIN_GENES, MAX_GENES)

    after_mito = qc["pass_mito"]
    after_singlet = after_mito & is_singlet
    after_genes = after_singlet & qc["pass_genes"]

    n1, n2, n3 = int(after_mito.sum()), int(after_singlet.sum()), int(after_genes.sum())
    print(
        f"[{DATASET_NAME}] after raw={n_raw}  after mito={n1}  after singletCode(+rescue)={n2}  "
        f"after gene floor/ceiling={n3}",
        flush=True,
    )

    keep = after_genes
    keep_idx = np.flatnonzero(keep)
    X_qc = X[keep_idx]
    kept_cells = pd.Series([f"{s}:{c}" for s, c in zip(sample[keep], cell_ids[keep])])

    obs_qc = pd.DataFrame(
        {
            "sample": sample[keep],
            "cell_barcode": cell_ids[keep],
            "pct_counts_mt": qc["pct_counts_mt"][keep],
            "n_genes": qc["n_genes"][keep],
            "total_counts": qc["total_counts"][keep],
        },
        index=kept_cells,
    )
    clone_lookup = {f"{s}:{c}": bc for (s, c), bc in singlet_clone.items()}
    obs_qc["fatemap_clone_singletcode"] = kept_cells.map(clone_lookup).fillna("").to_numpy()
    rescued_set = set(rescued_cellids)
    obs_qc["criterion4_rescued"] = pd.Series(cell_ids[keep]).isin(rescued_set).to_numpy()

    assert X_qc.shape[0] == len(obs_qc) == len(kept_cells)
    assert obs_qc.index.is_unique

    sio.mmwrite(f"{OUT_DIR}/hs054_sot48h_qc_counts.mtx", X_qc)
    open(f"{OUT_DIR}/genes.txt", "w").write("\n".join(var_names) + "\n")
    open(f"{OUT_DIR}/cells.txt", "w").write("\n".join(kept_cells) + "\n")
    obs_qc.to_csv(f"{OUT_DIR}/obs_metadata.csv")

    summary = {
        "dataset": DATASET_NAME,
        "n_raw_cells": n_raw,
        "n_after_mito": n1,
        "n_after_singletcode_plus_rescue": n2,
        "n_after_gene_floor_ceiling": n3,
        "mito_pct_cutoff": MITO_PCT_CUTOFF,
        "min_genes": MIN_GENES,
        "max_genes": MAX_GENES,
        "singletcode_min_umi_cutoff": "auto (singletCode default)",
        "n_distinct_cellids_rescued_by_criterion4": len(rescued_set),
        "rna_matrix_source": (
            "/projects/b1042/GoyalLab/Hanxiao/twinfer/10x/HS054_Sot48h/outs/filtered_feature_bc_matrix "
            "(re-synced into real_data/pancreatic_cancer/HS054_Sot48h/outs/ -- the copy originally there "
            "did not match the singletCode output next to it)"
        ),
        "starcode_source": STARCODE_PATH,
    }
    open(f"{OUT_DIR}/preprocessing_summary.json", "w").write(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[{DATASET_NAME}] wrote {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
