#!/usr/bin/env python3
"""HS054_Sot48h downstream: log1pCP10k normalization -> PCA(50). Single sample (no
replicate/batch split, unlike FM06/FM08), so no batch-aware HVG, no scanorama/harmony --
there is nothing to integrate across. Reads the QC'd matrix from
build_hs054_qc_matrix.py's output.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import os

import anndata as ad
import pandas as pd
import scanpy as sc
import scipy.io as sio

from paper_analysis.fatemap_pipeline.fatemap_integration_utils import normalize_and_pca

QC_DIR = f'{TWINFER_PROJECT_ROOT}/finalized_data/HS054_Sot48h_data/qc_filtered'
OUT_H5AD = f'{TWINFER_PROJECT_ROOT}/finalized_data/HS054_Sot48h_data/HS054_Sot48h_integrated.h5ad'
PLOT_DIR = f'{TWINFER_PROJECT_ROOT}/analysis_data/hs054/plots'
DATA_DIR = f'{TWINFER_PROJECT_ROOT}/analysis_data/hs054/data'


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    print("[HS054_Sot48h] loading QC'd matrix", flush=True)
    X = sio.mmread(f"{QC_DIR}/hs054_sot48h_qc_counts.mtx").tocsr()
    genes = pd.Index(open(f"{QC_DIR}/genes.txt").read().split())
    cells = pd.Index(open(f"{QC_DIR}/cells.txt").read().split())
    obs = pd.read_csv(f"{QC_DIR}/obs_metadata.csv", index_col=0)
    assert X.shape == (len(cells), len(genes)) == (len(obs), len(genes))
    A = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))
    A.obs_names = cells

    print("[HS054_Sot48h] normalization + PCA (no batch_key: single sample)", flush=True)
    A, n_hvg = normalize_and_pca(A, batch_key=None)
    print(f"[HS054_Sot48h] {n_hvg} HVGs, PCA shape {A.obsm['X_pca'].shape}", flush=True)

    sc.pp.neighbors(A, use_rep="X_pca")
    sc.tl.umap(A)
    sc.tl.leiden(A, key_added="leiden")

    _plot(A, PLOT_DIR)

    A.write_h5ad(OUT_H5AD)
    print(f"[HS054_Sot48h] wrote {OUT_H5AD}", flush=True)

    n_clone = int((A.obs["fatemap_clone_singletcode"] != "").sum())
    clone_sizes = A.obs.loc[A.obs["fatemap_clone_singletcode"] != "", "fatemap_clone_singletcode"].value_counts()
    report = pd.DataFrame({
        "metric": ["n_cells", "n_hvg", "n_cells_with_clone_id", "n_distinct_clones",
                   "median_clone_size", "max_clone_size"],
        "value": [A.n_obs, n_hvg, n_clone, clone_sizes.nunique() and len(clone_sizes),
                  clone_sizes.median() if len(clone_sizes) else 0,
                  clone_sizes.max() if len(clone_sizes) else 0],
    })
    report.to_csv(f"{DATA_DIR}/hs054_integration_report.csv", index=False)
    print(report.to_string(index=False), flush=True)


def _plot(A, plot_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4.5))
    emb = A.obsm["X_umap"]
    ax.scatter(emb[:, 0], emb[:, 1], s=3, alpha=0.6, c=A.obs["leiden"].astype("category").cat.codes, cmap="tab20")
    ax.set_title("HS054_Sot48h: PCA-space UMAP (leiden clusters)")
    fig.tight_layout()
    fig.savefig(f"{plot_dir}/hs054_umap_leiden.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
