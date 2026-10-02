#!/usr/bin/env python3
"""Watermelon (T47D) per-stage integration + UMAP + replicate-separation metrics.

Each stage (naive / lag / late) is its own dataset with two replicates: replicate 1 -> "A", 2 -> "B" (the A/B
split used for inference). Per stage: batch-aware log1pCP10k + PCA(50) (fatemap_integration_utils.normalize_and_pca,
batch_key="replicate"), scanorama + harmony (compared, as for FM01/FM06/FM08), UMAP on each of the three embeddings.
Separation metrics (per embedding): silhouette by replicate, and the kNN same-replicate fraction vs its expectation
under perfect mixing.  Also one global embedding of all 33k cells to show how well the three STAGES separate.

Run as: build_watermelon_stage_integrated.py naive|lag|late|global  (one SLURM array task each).
Writes per stage  finalized_data/Watermelon_{stage}_data/Watermelon_{stage}_integrated.h5ad  (.raw = log1pCP10k, obs has
replicate A/B + fatemap_clone_singletcode), so pick_gene_sets_fatemap.py Watermelon_{stage} works unchanged.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io as sio
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from paper_analysis.fatemap_pipeline.fatemap_integration_utils import integrate_harmony, integrate_scanorama, normalize_and_pca

QC_DIR = f'{TWINFER_PROJECT_ROOT}/finalized_data/Watermelon_data/qc_filtered'
DATA_DIR = f'{TWINFER_PROJECT_ROOT}/analysis_data/watermelon/data'
PLOT_DIR = f'{TWINFER_PROJECT_ROOT}/analysis_data/watermelon/plots'
STAGES = ["naive", "lag", "late"]
K = 30
SEED = 0


def knn_mixing(emb, labels, k=K):
    """Mean fraction of each cell's k nearest neighbours (in emb) that share its label, vs the value expected under
    perfect mixing (a cell's own label frequency). ratio ~1 = well mixed, >1 = separated by label."""
    labels = np.asarray(labels)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(emb)
    idx = nn.kneighbors(emb, return_distance=False)[:, 1:]
    same = (labels[idx] == labels[:, None]).mean(axis=1)
    freq = pd.Series(labels).value_counts(normalize=True)
    expected = pd.Series(labels).map(freq).to_numpy()
    return float(same.mean()), float(expected.mean())


def separation(emb, labels, name):
    sil = float(silhouette_score(emb, labels, sample_size=min(5000, len(labels)), random_state=SEED))
    same, exp = knn_mixing(emb, labels)
    return dict(embedding=name, silhouette=round(sil, 4), knn_same=round(same, 4), knn_expected=round(exp, 4),
                knn_ratio=round(same / exp, 3))


def umap_on(A, rep_key, name):
    sc.pp.neighbors(A, use_rep=rep_key, key_added=name, random_state=SEED)
    sc.tl.umap(A, neighbors_key=name, random_state=SEED)
    A.obsm[f"X_umap_{name}"] = A.obsm["X_umap"].copy()


def scatter(ax, emb, labels, order, colors, title, marker_by=None):
    labels = np.asarray(labels)
    for lab in order:
        m = labels == lab
        ax.scatter(emb[m, 0], emb[m, 1], s=1.5, alpha=0.5, label=f"{lab} (n={m.sum():,})", color=colors[lab], rasterized=True)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(markerscale=6, fontsize=8, frameon=False)


def load_all():
    print("[Watermelon] loading QC'd matrix (176M nnz, slow)", flush=True)
    X = sio.mmread(f"{QC_DIR}/watermelon_qc_counts.mtx").tocsr()
    genes = pd.Index(open(f"{QC_DIR}/genes.txt").read().split())
    cells = pd.Index(open(f"{QC_DIR}/cells.txt").read().split())
    obs = pd.read_csv(f"{QC_DIR}/obs_metadata.csv", index_col=0, dtype={"replicate": str})
    assert X.shape == (len(cells), len(genes)) == (len(obs), len(genes))
    assert (obs.index == cells).all()
    obs["replicate"] = obs["replicate"].map({"1": "A", "2": "B"})
    assert obs["replicate"].notna().all()
    print(f"[Watermelon] {X.shape[0]:,} cells x {X.shape[1]:,} genes", flush=True)
    return X, genes, cells, obs


REP_COLORS = {"A": "#2a78b6", "B": "#e08a1e"}


def process_stage(stage, X, genes, cells, obs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = []
    sel = np.flatnonzero((obs["stage"] == stage).to_numpy())
    A = ad.AnnData(X=X[sel].copy(), obs=obs.iloc[sel].copy(), var=pd.DataFrame(index=genes))
    A.obs_names = cells[sel]
    print(f"[{stage}] {A.n_obs:,} cells; replicates {A.obs['replicate'].value_counts().to_dict()}", flush=True)

    A, n_hvg = normalize_and_pca(A, batch_key="replicate")
    print(f"[{stage}] {n_hvg} HVGs, PCA {A.obsm['X_pca'].shape}", flush=True)
    X_scan, scan_order = integrate_scanorama(A, batch_key="replicate")
    A = A[scan_order].copy()
    A.obsm["X_scanorama"] = X_scan
    A.obsm["X_pca_harmony"] = integrate_harmony(A, batch_key="replicate")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (rep_key, name, title) in zip(
        axes, [("X_pca", "pca", "No integration (PCA)"), ("X_scanorama", "scanorama", "Scanorama"),
               ("X_pca_harmony", "harmony", "Harmony")]):
        umap_on(A, rep_key, name)
        lab = A.obs["replicate"].to_numpy()
        metrics.append(dict(stage=stage, **separation(A.obsm[rep_key], lab, name)))
        m = metrics[-1]
        scatter(ax, A.obsm[f"X_umap_{name}"], lab, ["A", "B"], REP_COLORS,
                f"{stage}: {title}\nsilhouette {m['silhouette']:.3f} | kNN same-rep {m['knn_same']:.2f} (mixed = {m['knn_expected']:.2f})")
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/watermelon_{stage}_replicate_umap.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(metrics).to_csv(f"{DATA_DIR}/watermelon_replicate_separation_{stage}.csv", index=False)

    out_dir = f"{TWINFER_PROJECT_ROOT}/finalized_data/Watermelon_{stage}_data"
    os.makedirs(out_dir, exist_ok=True)
    A.write_h5ad(f"{out_dir}/Watermelon_{stage}_integrated.h5ad")
    print(f"[{stage}] wrote {out_dir}/Watermelon_{stage}_integrated.h5ad", flush=True)


def process_global(X, genes, cells, obs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # global embedding: how well do the three STAGES separate (all cells together, batch = sample)
    print("[global] all-stage PCA/UMAP", flush=True)
    G = ad.AnnData(X=X, obs=obs.copy(), var=pd.DataFrame(index=genes))
    G.obs_names = cells
    G, n_hvg = normalize_and_pca(G, batch_key="sample")
    umap_on(G, "X_pca", "pca")
    stage_colors = {"naive": "#2a78b6", "lag": "#e08a1e", "late": "#4aa66b"}
    gm = [dict(scope="global", labels="stage", **separation(G.obsm["X_pca"], G.obs["stage"].to_numpy(), "pca")),
          dict(scope="global", labels="sample", **separation(G.obsm["X_pca"], G.obs["sample"].to_numpy(), "pca"))]
    pd.DataFrame(gm).to_csv(f"{DATA_DIR}/watermelon_global_separation.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    scatter(axes[0], G.obsm["X_umap_pca"], G.obs["stage"].to_numpy(), STAGES, stage_colors,
            f"All stages (PCA UMAP): by stage\nsilhouette {gm[0]['silhouette']:.3f} | kNN same-stage {gm[0]['knn_same']:.2f} (mixed = {gm[0]['knn_expected']:.2f})")
    scatter(axes[1], G.obsm["X_umap_pca"], G.obs["replicate"].to_numpy(), ["A", "B"], REP_COLORS, "All stages: by replicate (A=1, B=2)")
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/watermelon_all_stages_umap.png", dpi=150)
    pd.DataFrame(G.obsm["X_umap_pca"], index=G.obs_names, columns=["umap1", "umap2"]).join(
        G.obs[["sample", "stage", "replicate"]]).to_csv(f"{DATA_DIR}/watermelon_all_stages_umap_coords.csv")
    print("[global]", gm, flush=True)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "global"
    assert which in STAGES + ["global"], f"usage: {sys.argv[0]} naive|lag|late|global"
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    X, genes, cells, obs = load_all()
    if which == "global":
        process_global(X, genes, cells, obs)
    else:
        process_stage(which, X, genes, cells, obs)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
