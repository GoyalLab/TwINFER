#!/usr/bin/env python3
"""Normalization + PCA + integration + marker-annotation helpers, shared by
build_fm06_integrated.py / build_fm08_integrated.py. Mirrors larry_raw_annotate.py's
classifier-space recipe (log1pCP10k -> batch-aware HVG -> scale -> PCA), except the
result IS the output here (not a throwaway classifier space), and normalization is
done once across all cells together with batch_key="replicate" -- not per-replicate
separately -- per the user's explicit correction.
"""
import numpy as np
import pandas as pd
import scanpy as sc

N_HVG = 2000
N_PC = 50
SEED = 0


def normalize_and_pca(adata, batch_key="replicate"):
    """log1pCP10k on all cells together, batch-aware HVG selection (batch_key,
    so no gene is chosen because one replicate varies), scale, PCA(n_comps=50).
    Mutates a copy; returns it. Batch-aware, not batch-separate."""
    A = adata.copy()
    A.X = A.X.astype(np.float32)
    A.layers["counts"] = A.X.copy()
    sc.pp.normalize_total(A, target_sum=1e4)
    sc.pp.log1p(A)
    A.layers["lognorm"] = A.X.copy()
    sc.pp.highly_variable_genes(A, n_top_genes=N_HVG, batch_key=batch_key)
    n_hvg = int(A.var["highly_variable"].sum())
    assert n_hvg > 0, "no highly variable genes selected"
    A.raw = A
    Ahvg = A[:, A.var["highly_variable"]].copy()
    sc.pp.scale(Ahvg, max_value=10)
    sc.tl.pca(
        Ahvg,
        n_comps=min(N_PC, Ahvg.n_vars - 1, Ahvg.n_obs - 1),
        svd_solver="arpack",
        random_state=SEED,
    )
    A.obsm["X_pca"] = Ahvg.obsm["X_pca"]
    A.uns["pca"] = Ahvg.uns["pca"]
    return A, n_hvg


def integrate_scanorama(adata, batch_key="replicate"):
    """scanorama.correct_scanpy across batch_key groups, run on the log-normalized
    HVG-restricted matrix (scanorama's own recommended input). `adata` is the object
    returned by normalize_and_pca (full gene set, .raw holds lognorm, .var has
    'highly_variable'). Returns (X_scanorama array, obs_names in that array's row order)."""
    import anndata as ad
    import scanorama

    hvg_names = adata.var_names[adata.var["highly_variable"]]
    A_hvg = adata.raw[:, hvg_names].to_adata()
    A_hvg.obs = adata.obs.copy()

    batches = A_hvg.obs[batch_key].astype(str).unique().tolist()
    adatas = [A_hvg[A_hvg.obs[batch_key] == b].copy() for b in batches]
    corrected = scanorama.correct_scanpy(adatas, return_dimred=True)
    merged = ad.concat(corrected, join="outer") if len(corrected) > 1 else corrected[0]
    return merged.obsm["X_scanorama"], merged.obs_names


def integrate_harmony(adata, batch_key="replicate"):
    """Harmony on the same PCA embedding produced by normalize_and_pca. Calls
    harmonypy directly rather than sc.external.pp.harmony_integrate -- the installed
    harmonypy==2.0.0's Z_corr output shape isn't handled correctly by scanpy's
    wrapper (raises an obsm shape-validation error), so this avoids that entirely."""
    import harmonypy

    ho = harmonypy.run_harmony(
        adata.obsm["X_pca"], adata.obs, [batch_key], random_state=SEED
    )
    Z = np.asarray(ho.Z_corr)
    if Z.shape[0] != adata.n_obs:
        Z = Z.T
    assert Z.shape == (adata.n_obs, adata.obsm["X_pca"].shape[1]), \
        f"unexpected harmony output shape {Z.shape}"
    return Z


def load_goyal_annotation(csv_path, sample_prefix_map):
    """GoyalEtAl_FM0X.csv columns: '', umap_1, umap_2, cluster, cellID.
    sample_prefix_map: {csv cellID prefix: replicate label}, e.g.
    {"FM06-WM989Naive-1": "A", "FM06-WM989Naive-2": "B"} for FM06, or
    {"run1_sample3": "A", "run1_sample4": "B"} for FM08.
    Returns a Series keyed by "{replicate}:{16bp barcode}" -> cluster label."""
    df = pd.read_csv(csv_path)
    df["cellID"] = df["cellID"].astype(str)
    prefixes = sorted(sample_prefix_map, key=len, reverse=True)
    keys, clusters = [], []
    unmatched_prefix = 0
    for cid, cluster in zip(df["cellID"], df["cluster"]):
        matched = None
        for p in prefixes:
            if cid.startswith(p + "_"):
                matched = p
                break
        if matched is None:
            unmatched_prefix += 1
            continue
        rep = sample_prefix_map[matched]
        barcode = cid[len(matched) + 1:].split("-")[0]
        keys.append(f"{rep}:{barcode}")
        clusters.append(cluster)
    if unmatched_prefix:
        print(f"[annotation] {unmatched_prefix} rows had no matching prefix, dropped", flush=True)
    return pd.Series(clusters, index=keys)
