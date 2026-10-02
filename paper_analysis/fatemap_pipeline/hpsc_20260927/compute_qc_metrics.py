#!/usr/bin/env python3
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Per-cell QC metrics (total UMI counts, detected genes, mito%) for endo_T0 and
endo_T1 Cell Ranger filtered_feature_bc_matrix.h5, computed via direct h5py + scipy
CSC construction (no scanpy/AnnData) and processed one sample at a time so the
matrix is freed before the next sample loads -- avoids the OOM (exit 137) hit
earlier this session when loading both full AnnData objects in one process.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths]
import gc

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

H5_PATHS = {
    "endo_T0": "/projects/b1042/GoyalLab/Hanxiao/twinfer_diff/cellranger_out/endo_T0/outs/filtered_feature_bc_matrix.h5",
    "endo_T1": "/projects/b1042/GoyalLab/Hanxiao/twinfer_diff/cellranger_out/endo_T1/outs/filtered_feature_bc_matrix.h5",
}
OUT_CSV = f'{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/fatemap_pipeline/hpsc_20260927/qc_metrics.csv'


def compute_one(sample, path):
    with h5py.File(path, "r") as f:
        grp = f["matrix"]
        shape = grp["shape"][:]  # [n_genes, n_barcodes]
        data = grp["data"][:]
        indices = grp["indices"][:]
        indptr = grp["indptr"][:]
        barcodes = grp["barcodes"][:].astype(str)
        gene_names = grp["features"]["name"][:].astype(str)

    mat = sp.csc_matrix((data, indices, indptr), shape=tuple(shape))
    del data, indices, indptr
    gc.collect()

    mito_mask = np.char.upper(gene_names.astype(str)).astype(str)
    mito_mask = np.array([g.startswith("MT-") for g in mito_mask])
    print(f"[{sample}] n_genes={shape[0]} n_cells={shape[1]} n_mito_genes={mito_mask.sum()}", flush=True)

    total_counts = np.asarray(mat.sum(axis=0)).ravel().astype(float)
    n_genes_per_cell = np.asarray((mat > 0).sum(axis=0)).ravel().astype(float)
    mito_counts = np.asarray(mat[mito_mask, :].sum(axis=0)).ravel().astype(float)
    pct_mt = np.divide(100.0 * mito_counts, total_counts, out=np.zeros_like(total_counts), where=total_counts > 0)

    del mat
    gc.collect()

    df = pd.DataFrame(
        {
            "sample": sample,
            "cell_barcode": [b.replace("-1", "") for b in barcodes],
            "total_counts": total_counts,
            "n_genes": n_genes_per_cell,
            "pct_counts_mt": pct_mt,
        }
    )
    return df


def main():
    frames = []
    for sample, path in H5_PATHS.items():
        print(f"[{sample}] loading {path}", flush=True)
        frames.append(compute_one(sample, path))
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV} ({len(out)} rows)", flush=True)

    for sample, g in out.groupby("sample"):
        print(f"\n[{sample}] n={len(g)}")
        print(g[["total_counts", "n_genes", "pct_counts_mt"]].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))


if __name__ == "__main__":
    main()
