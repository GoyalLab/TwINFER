"""
cnmf_programs.py

Gene programs (modules) for a single-cell dataset with cNMF (Kotliar et al., eLife 2019).
Models raw counts, so no metacells are needed; stability comes from a consensus over many
NMF replicates per K.

Steps: prepare (HVG selection + variance-normalised counts) -> factorize (n_iter NMF runs per K)
-> combine -> k_selection_plot -> consensus per K -> export programs.

Input is the 'counts' layer of an h5ad (cNMF needs counts, not log-normalised values).
Outputs go under --out/<name>/ ; exported tables are written to --out/<name>/programs_K<k>/.
"""
from __future__ import annotations

import argparse
import os

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def make_counts_h5ad(h5ad, layer, out_path, n_cells=None, seed=0):
    a = ad.read_h5ad(h5ad)
    X = a.layers[layer] if layer else a.X
    X = sp.csr_matrix(X)
    b = ad.AnnData(X=X, obs=pd.DataFrame(index=a.obs_names), var=pd.DataFrame(index=a.var_names))
    b.var_names_make_unique()
    if n_cells and n_cells < b.n_obs:
        rng = np.random.default_rng(seed)
        b = b[np.sort(rng.choice(b.n_obs, n_cells, replace=False))].copy()
    sc_keep = np.asarray((b.X > 0).sum(0)).ravel() > 0           # cNMF drops zero genes itself; be explicit
    b = b[:, sc_keep].copy()
    b.write_h5ad(out_path)
    return b.n_obs, b.n_vars


def export(cn, k, dt, out_dir, n_top):
    usage, spectra_score, spectra_tpm, top_genes = cn.load_results(K=k, density_threshold=dt, n_top_genes=n_top)
    os.makedirs(out_dir, exist_ok=True)
    usage.to_csv(f"{out_dir}/usage.csv")                                   # cells x programs
    spectra_score.to_csv(f"{out_dir}/gene_spectra_score.csv")              # genes x programs
    spectra_tpm.to_csv(f"{out_dir}/gene_spectra_tpm.csv")
    top_genes.to_csv(f"{out_dir}/top_genes.csv")                           # rank x program
    # hard module assignment: each gene -> program with the highest score
    ss = spectra_score                                                     # load_results returns genes x programs
    pd.DataFrame({"module": ss.values.argmax(1) + 1, "max_score": ss.max(1).values},
                 index=ss.index).to_csv(f"{out_dir}/modules.csv")


def main():
    ap = argparse.ArgumentParser(description="cNMF gene programs")
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--layer", default="counts", help="layer holding raw counts ('' = .X)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="cnmf")
    ap.add_argument("--k", type=int, nargs="+", default=[5, 10, 15, 20, 25, 30])
    ap.add_argument("--n-iter", type=int, default=100, help="NMF replicates per K")
    ap.add_argument("--n-hvg", type=int, default=2000)
    ap.add_argument("--density-threshold", type=float, default=0.2,
                    help="outlier filter for consensus (0.2 is the cNMF-recommended starting point; 2.0 = no filter)")
    ap.add_argument("--n-top-genes", type=int, default=100)
    ap.add_argument("--n-workers", type=int, default=1, help="processes for factorize")
    ap.add_argument("--n-cells", type=int, default=None, help="subsample cells (testing)")
    ap.add_argument("--seed", type=int, default=14)
    a = ap.parse_args()

    from cnmf import cNMF
    os.makedirs(a.out, exist_ok=True)
    counts = f"{a.out}/{a.name}_counts.h5ad"
    print("cells, genes:", make_counts_h5ad(a.h5ad, a.layer, counts, a.n_cells, a.seed), flush=True)

    cn = cNMF(output_dir=a.out, name=a.name)
    cn.prepare(counts_fn=counts, components=a.k, n_iter=a.n_iter, seed=a.seed, num_highvar_genes=a.n_hvg)

    if a.n_workers > 1:                                    # cNMF splits the replicate list by worker index
        from multiprocessing import get_context
        with get_context("fork").Pool(a.n_workers) as pool:
            pool.starmap(_factorize, [(a.out, a.name, i, a.n_workers) for i in range(a.n_workers)])
    else:
        cn.factorize(worker_i=0, total_workers=1)
    cn.combine()
    cn.k_selection_plot(close_fig=True)                    # <out>/<name>/<name>.k_selection.png (stability vs error)

    for k in a.k:
        cn.consensus(k=k, density_threshold=a.density_threshold, show_clustering=True, close_clustergram_fig=True)
        export(cn, k, a.density_threshold, f"{a.out}/{a.name}/programs_K{k}", a.n_top_genes)
        print(f"K={k}: exported", flush=True)


def _factorize(out, name, i, n):
    from cnmf import cNMF
    cNMF(output_dir=out, name=name).factorize(worker_i=i, total_workers=n)


if __name__ == "__main__":
    main()
