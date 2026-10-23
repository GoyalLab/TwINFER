#!/usr/bin/env python3
"""FM06 downstream: batch-aware normalization (log1pCP10k, all cells together,
batch_key='replicate' for HVG) -> PCA(50) -> scanorama + Harmony integration
(compared) -> join GoyalEtAl_FM06.csv cluster annotation. Reads the QC'd matrix
from build_fm06_qc_matrix.py's output.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import os

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io as sio

from paper_analysis.fatemap_pipeline.fatemap_integration_utils import (
    integrate_harmony,
    integrate_scanorama,
    load_goyal_annotation,
    normalize_and_pca,
)

QC_DIR = f'{TWINFER_PROJECT_ROOT}/finalized_data/FM06_data/qc_filtered'
OUT_H5AD = f'{TWINFER_PROJECT_ROOT}/finalized_data/FM06_data/FM06_integrated.h5ad'
ANNOT_CSV = f'{TWINFER_PROJECT_ROOT}/real_data/Fatemap_barcodes/GoyalEtAl_FM06.csv'
ANNOT_PREFIX_MAP = {"FM06-WM989Naive-1": "A", "FM06-WM989Naive-2": "B"}
PLOT_DIR = f'{TWINFER_PROJECT_ROOT}/analysis_data/fatemap06/plots'
DATA_DIR = f'{TWINFER_PROJECT_ROOT}/analysis_data/fatemap06/data'


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    print("[FM06] loading QC'd matrix", flush=True)
    X = sio.mmread(f"{QC_DIR}/fm06_qc_counts.mtx").tocsr()
    genes = pd.Index(open(f"{QC_DIR}/genes.txt").read().split())
    cells = pd.Index(open(f"{QC_DIR}/cells.txt").read().split())
    obs = pd.read_csv(f"{QC_DIR}/obs_metadata.csv", index_col=0)
    assert X.shape == (len(cells), len(genes)) == (len(obs), len(genes))
    A = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))
    A.obs_names = cells

    print("[FM06] batch-aware normalization + PCA", flush=True)
    A, n_hvg = normalize_and_pca(A, batch_key="replicate")
    print(f"[FM06] {n_hvg} HVGs (batch_key='replicate'), "
          f"PCA shape {A.obsm['X_pca'].shape}", flush=True)

    print("[FM06] scanorama integration", flush=True)
    X_scan, scan_order = integrate_scanorama(A, batch_key="replicate")
    A = A[scan_order].copy()
    A.obsm["X_scanorama"] = X_scan

    print("[FM06] harmony integration", flush=True)
    A.obsm["X_pca_harmony"] = integrate_harmony(A, batch_key="replicate")

    print("[FM06] joining GoyalEtAl_FM06.csv cluster annotation", flush=True)
    annot = load_goyal_annotation(ANNOT_CSV, ANNOT_PREFIX_MAP)
    key = pd.Series(
        [f"{r}:{b}" for r, b in zip(A.obs["replicate"], A.obs["cell_barcode"])],
        index=A.obs_names,
    )
    A.obs["cluster"] = key.map(annot).to_numpy()
    n_matched = int(A.obs["cluster"].notna().sum())
    print(f"[FM06] annotation matched {n_matched}/{A.n_obs} cells "
          f"({100 * n_matched / A.n_obs:.1f}%)", flush=True)

    for basis, name in [("X_pca", "pca"), ("X_scanorama", "scanorama"), ("X_pca_harmony", "harmony")]:
        sc.pp.neighbors(A, use_rep=basis, key_added=name)
        sc.tl.umap(A, neighbors_key=name)
        A.obsm[f"X_umap_{name}"] = A.obsm["X_umap"]

    _plot_comparison(A, PLOT_DIR, "FM06")

    A.write_h5ad(OUT_H5AD)
    print(f"[FM06] wrote {OUT_H5AD}", flush=True)

    report = pd.DataFrame({
        "metric": ["n_cells", "n_hvg", "n_annotation_matched", "pct_annotation_matched"],
        "value": [A.n_obs, n_hvg, n_matched, round(100 * n_matched / A.n_obs, 2)],
    })
    report.to_csv(f"{DATA_DIR}/fm06_integration_report.csv", index=False)


def _plot_comparison(A, plot_dir, dataset_label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, name, title in zip(
        axes, ["pca", "scanorama", "harmony"],
        ["No integration (PCA only)", "Scanorama", "Harmony"],
    ):
        emb = A.obsm[f"X_umap_{name}"]
        for rep, marker in zip(sorted(A.obs["replicate"].unique()), ["o", "^"]):
            m = (A.obs["replicate"] == rep).to_numpy()
            ax.scatter(emb[m, 0], emb[m, 1], s=2, alpha=0.5, label=rep, marker=marker)
        ax.set_title(f"{dataset_label}: {title}")
        ax.legend(markerscale=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{plot_dir}/{dataset_label.lower()}_integration_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
