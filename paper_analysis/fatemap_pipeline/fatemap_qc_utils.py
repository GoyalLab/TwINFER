#!/usr/bin/env python3
"""Shared helpers for the FM06/FM08 QC pipeline, mirroring the filter/rescue logic
in larry_pipeline/filtering/build_final_matrix_umi3.py, adapted for datasets whose
RNA matrices are already Cell Ranger *filtered* (called-cell) output -- no
dip-test/Otsu whitelisting step is needed here, unlike LARRY's raw inDrops matrix.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import singletCode

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(
# 0,
# "/home/gzu5140/TwINFER_KA/code/TwINFER/paper_analysis/"
# "larry_hematopoiesis_validation/preprocessing",
# )
from paper_analysis.larry_hematopoiesis_validation.preprocessing.singletcode_cross_sample_rescue import apply_cross_sample_rescue  # noqa: E402

RAW_DIR = f'{TWINFER_PROJECT_ROOT}/real_data/Fatemap_barcodes'


def load_10x_replicate(prefix, replicate_label):
    """Load one Cell Ranger triplet (barcodes/features/matrix.tsv/mtx.gz) as AnnData,
    var_names = gene symbol, obs_names = 16bp barcode (Cell Ranger '-1' suffix
    stripped so it matches the lineage-barcode table's cellID format)."""
    adata = sc.read_mtx(f"{RAW_DIR}/{prefix}_matrix.mtx.gz").T
    features = pd.read_csv(f"{RAW_DIR}/{prefix}_features.tsv.gz", sep="\t", header=None)
    barcodes = pd.read_csv(f"{RAW_DIR}/{prefix}_barcodes.tsv.gz", sep="\t", header=None)
    assert adata.shape == (len(barcodes), len(features))
    adata.var_names = features[1].to_numpy()
    adata.var["ensembl_id"] = features[0].to_numpy()
    adata.var_names_make_unique()
    adata.obs_names = barcodes[0].str.replace(r"-1$", "", regex=True).to_numpy()
    adata.obs_names_make_unique()
    adata.obs["replicate"] = replicate_label
    return adata


def build_sample_sheet(starcode_path, sample_num_to_replicate, barcode_col="BC50StarcodeD8"):
    """Build the singletCode sample sheet (cellID, barcode, sample) directly from a
    stepThreeStarcodeShavedReads.txt table. No manual UMI-count pre-filter -- one row
    per (cellID, UMI, barcode) read-support record is kept (collapsing only exact
    duplicate rows), letting singletCode.get_singlets()'s own min_umi_cutoff handle
    thresholding, per the user's explicit instruction not to pre-filter on UMI count.

    sample_num_to_replicate: dict mapping the table's SampleNum values that
    correspond to an actual RNA replicate to that replicate's label (e.g. {1: "A",
    2: "B"}). SampleNum values not present in this dict are dropped -- e.g. FM08 has
    a SampleNum 3 in the barcode table with no matching RNA replicate.
    """
    # FM01's stepFour table has no UMI column (each row is one UMI record, already deduplicated) and calls the sample column "sampleNum".
    header = pd.read_csv(starcode_path, sep="\t", nrows=0).columns
    sample_col = "SampleNum" if "SampleNum" in header else "sampleNum"
    umi_col = ["UMI"] if "UMI" in header else []
    df = pd.read_csv(starcode_path, sep="\t", usecols=["cellID", *umi_col, barcode_col, sample_col])
    df = df.rename(columns={sample_col: "SampleNum"})
    if not umi_col:
        df["UMI"] = range(len(df))  # each row is already one UMI record: give each a unique id so drop_duplicates keeps them all
    df = df[df["SampleNum"].isin(sample_num_to_replicate)].copy()
    dropped_n = 0
    df["sample"] = df["SampleNum"].map(sample_num_to_replicate)
    sheet = (
        df.drop_duplicates(["cellID", "UMI", barcode_col, "sample"])
        .rename(columns={barcode_col: "barcode"})[["cellID", "barcode", "sample"]]
        .reset_index(drop=True)
    )
    return sheet


def run_singlet_calling(sample_sheet, dataset_name, min_umi_cutoff=3):
    # min_umi_cutoff=None: do not pass it; singletCode then uses its own default floor and ratio-based auto cutoff.
    """singletCode criteria 1-3 + this repo's own cross-sample criterion-4 rescue,
    same call pattern as larry_pipeline/filtering/build_final_matrix_umi3.py."""
    singletCode.check_sample_sheet(sample_sheet)
    kwargs = {} if min_umi_cutoff is None else {"min_umi_cutoff": min_umi_cutoff}
    good_data, singlet_stats = singletCode.get_singlets(
        sample_sheet, dataset_name=dataset_name, **kwargs
    )
    good_data, rescued_cellids = apply_cross_sample_rescue(good_data)
    singlets = good_data[good_data["label"] == "Singlet"]
    singlet_clone = (
        singlets.groupby(["sample", "cellID"])["barcode"]
        .apply(lambda s: "".join(sorted(set(s))))
    )
    return singlets, singlet_clone, rescued_cellids


def apply_qc_filters(X, var_names, mito_pct_cutoff, min_genes, max_genes):
    """Mito% (human 'MT-' prefix, uppercase) + gene-count floor/ceiling. Returns a
    dict of the underlying per-cell QC arrays and boolean pass masks for reporting.
    X: cells x genes sparse matrix. var_names: gene symbols aligned to X's columns."""
    genes = pd.Series(var_names)
    mito_mask = genes.str.upper().str.startswith("MT-").to_numpy()
    assert mito_mask.sum() > 0, "no MT- genes found -- check gene symbol column/species"

    X = X.tocsr()
    total_counts = np.asarray(X.sum(axis=1)).ravel().astype(float)
    mito_counts = np.asarray(X[:, mito_mask].sum(axis=1)).ravel().astype(float)
    pct_counts_mt = np.divide(
        100.0 * mito_counts, total_counts,
        out=np.zeros_like(total_counts), where=total_counts > 0,
    )
    n_genes_per_cell = np.asarray((X > 0).sum(axis=1)).ravel()

    pass_mito = pct_counts_mt <= mito_pct_cutoff
    pass_genes = (n_genes_per_cell >= min_genes) & (n_genes_per_cell <= max_genes)

    qc = {
        "pct_counts_mt": pct_counts_mt,
        "n_genes": n_genes_per_cell,
        "total_counts": total_counts,
        "pass_mito": pass_mito,
        "pass_genes": pass_genes,
    }
    return qc
