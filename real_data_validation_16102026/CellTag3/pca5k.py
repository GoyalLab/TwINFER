#!/usr/bin/env python3
"""PCA on the top 5,000 highly variable genes for every final dataset + elbow plots.

usage: pca5k.py <dataset> [<dataset> ...]      (see DATASETS below)
Recipe (same as the existing integrated h5ads except for the HVG count and no batch_key):
  raw counts -> normalize_total(1e4) -> log1p -> highly_variable_genes(n_top_genes=5000, flavor='seurat', no batch_key)
  -> scale(max_value=10) on the HVG subset -> PCA(50, arpack, random_state=0)
Outputs: analysis_data/dataset_qc_verification/pca5k/<dataset>_pca5k.npz  (pca coords, variance, variance_ratio, hvg names, cell ids)
         analysis_data/dataset_qc_verification/pca5k/elbow_<dataset>.png and pca5k_summary.json (per-dataset elbow)
The final h5ad / qc_filtered files are NOT modified.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qc_common import FIN, OUT, PETRACE, PETRACE_STAGES, ROOT, REGISTRY, final_counts, read_h5ad_csr, read_h5ad_obs_var

PCA_DIR = f"{OUT}/pca5k"
N_HVG, N_PC = 5000, 50


# ----------------------------------------------------------------------------- count loaders
def load_counts(name):
    """-> (counts csr float32 cells x genes, cell ids, gene names, sample labels, note)"""
    if name in REGISTRY and name != "Watermelon":
        X, cells, genes, obs, _sym = final_counts(name)
        return X.astype(np.float32), cells, genes, obs["sample"].astype(str).to_numpy(), "qc_filtered counts"
    if name.startswith("Watermelon_"):
        st = name.split("_", 1)[1]
        h = f"{FIN}/Watermelon_{st}_data/Watermelon_{st}_integrated.h5ad"
        obs, var = read_h5ad_obs_var(h)
        X = read_h5ad_csr(h, "layers/counts").astype(np.float32)
        return X, list(obs.index), list(var.index), obs["sample"].astype(str).to_numpy(), "layers['counts'] of the stage h5ad"
    if name == "LARRY":
        import scipy.io as sio
        d = f"{FIN}/LARRY_data/filtered"
        X = sio.mmread(f"{d}/larry_qc_counts.mtx").tocsr().astype(np.float32)
        cells = [l.rstrip("\n") for l in open(f"{d}/cells.txt")]
        genes = [l.rstrip("\n") for l in open(f"{d}/genes.txt")]
        obs = pd.read_csv(f"{d}/obs_metadata.csv", index_col=0)
        assert list(obs.index) == cells
        return X, cells, genes, obs["library"].astype(str).to_numpy(), "LARRY_data/filtered raw counts"
    if name == "hPSC_endo_yscher":
        h = f"{FIN}/hPSC_endo_yscher_data/endo_kbnac_spliced_unspliced.h5ad"
        obs, var = read_h5ad_obs_var(h)
        X = read_h5ad_csr(h, "X").astype(np.float32)
        return X, list(obs.index), list(var.index), obs["timepoint"].astype(str).to_numpy(), "X of yscher's endo h5ad (integer-valued)"
    if name == "hPSC_undifferentiated":
        # the copy in finalized_data holds log1pPF in X only; the same cells' raw counts are in the sibling file
        f_final = f"{FIN}/hPSC_undifferentiated_data/hPSC_undifferentiated.h5ad"
        f_cnt = f"{ROOT}/real_data/twinfer2026_run1plus2_counts_mito20_min200genes.h5ad"
        of, vf = read_h5ad_obs_var(f_final)
        oc, vc = read_h5ad_obs_var(f_cnt)
        assert set(of.index) == set(oc.index), "final and counts h5ad have different cells"
        assert list(vf.index) == list(vc.index), "gene lists differ"
        X = read_h5ad_csr(f_cnt, "X")
        pos = pd.Series(np.arange(len(oc)), index=oc.index).loc[of.index].to_numpy()
        X = X[pos].astype(np.float32)
        ok = np.allclose(X.data, np.round(X.data))
        return X, list(of.index), list(vf.index), of["timepoint"].astype(str).to_numpy(), \
            f"counts from {os.path.basename(f_cnt)} reordered to the final obs (integer={ok})"
    if name == "hPSC_Novo_Nordisk":
        base = f"{FIN}/hPSC_Novo_Nordisk_data/qc_filtered"
        mats, cells, lab = [], [], []
        genes = None
        for s in ["hESCs_p2_parent", "hESCs_p3_same", "hESCs_p3_sis", "PS_sis", "DE_same", "DE_sis"]:
            m = sp.load_npz(f"{base}/{s}/counts.npz").tocsr()
            g = [l.rstrip("\n") for l in open(f"{base}/{s}/genes.txt")]
            o = pd.read_csv(f"{base}/{s}/obs_metadata.csv", index_col=0)
            if genes is None:
                genes = g
            assert g == genes and m.shape == (len(o), len(g)), (s, m.shape, len(o), len(g))
            mats.append(m); cells += list(o.index); lab += [s] * len(o)
        return sp.vstack(mats).tocsr().astype(np.float32), cells, genes, np.array(lab), "per-sample counts.npz pooled"
    if name == "Schaffer_naive":   # Schaffer 2026 naive melanoma (GSE279162 Naive1-3 merged = one sample); singletCode singlets, raw GEO counts (staged by real_data_scripts/schaffer_stage1_prepare.py)
        d = f"{ROOT}/analysis_data/stage1_inputs/schaffer"
        X = sp.load_npz(f"{d}/counts.npz").tocsr().astype(np.float32)
        genes = [l.rstrip("\n") for l in open(f"{d}/genes.txt")]
        obs = pd.read_csv(f"{d}/obs_metadata.csv", index_col=0)
        return X, list(obs.index), genes, np.array(["Schaffer_naive"] * len(obs)), "raw GEO counts of the singletCode singlets (Naive1-3 merged)"
    raise KeyError(name)


DATASETS = ["FM01", "FM03", "FM05", "FM06", "FM07", "FM08", "HS054_Sot48h",
            "Watermelon_naive", "Watermelon_lag", "Watermelon_late", "hPSC_20260927",
            "hPSC_endo_yscher", "hPSC_undifferentiated", "hPSC_Novo_Nordisk", "LARRY",
            # not in finalized_data
            "Michaels_et_al", "CellTag3", "SpaceBar", "DTT_mouse", "Schaffer_naive"] + [f"PeTrace_{t}" for t in PETRACE] \
           + [f"PeTrace_{s}" for s in PETRACE_STAGES]      # last 5 = all replicates of a day merged


def elbow(vr):
    """kneedle-style: PC with the largest distance below the chord of the variance-ratio curve."""
    y = np.log(vr)
    x = np.arange(len(y))
    chord = y[0] + (y[-1] - y[0]) * x / (len(y) - 1)
    return int(np.argmax(chord - y)) + 1


def run(name):
    os.makedirs(PCA_DIR, exist_ok=True)
    X, cells, genes, sample, note = load_counts(name)
    n, g = X.shape
    a = sc.AnnData(X)
    a.var_names = pd.Index(genes).astype(str)
    a.var_names_make_unique()
    a.obs_names = pd.Index(cells).astype(str)
    sc.pp.filter_genes(a, min_cells=1)
    d = a.X.data[: min(len(a.X.data), 5_000_000)]
    if np.allclose(d, np.round(d)):
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
    else:  # the available matrix is already normalized (e.g. CellTag3): use as is
        note += "; X is NOT integer counts -> used as shipped (no normalize/log1p)"
    sc.pp.highly_variable_genes(a, n_top_genes=min(N_HVG, a.n_vars - 1), flavor="seurat")
    hv = a.var_names[a.var["highly_variable"]].to_list()
    b = a[:, hv].copy()
    sc.pp.scale(b, max_value=10)
    sc.tl.pca(b, n_comps=N_PC, svd_solver="arpack", random_state=0)
    vr = b.uns["pca"]["variance_ratio"]
    var = b.uns["pca"]["variance"]
    tmp = f"{PCA_DIR}/.{name}_pca5k.tmp.npz"        # atomic: readers never see a half-written file
    np.savez_compressed(tmp, pca=b.obsm["X_pca"].astype(np.float32), variance=var,
                        variance_ratio=vr, hvg=np.array(hv), cells=np.array(cells), sample=np.array(sample))
    os.replace(tmp, f"{PCA_DIR}/{name}_pca5k.npz")
    e = elbow(vr)
    info = dict(dataset=name, n_cells=int(n), n_genes_total=int(g), n_genes_expressed=int(a.n_vars), n_hvg=len(hv),
                source=note, elbow_pc=e, pc1_ratio=float(vr[0]), cum_var_at_elbow=float(vr[:e].sum()),
                cum_var_50=float(vr.sum()), pcs_to_50pct_of_top50=int(np.searchsorted(np.cumsum(vr) / vr.sum(), 0.5) + 1))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(np.arange(1, N_PC + 1), vr * 100, "o-", ms=3)
    ax[0].axvline(e, color="r", ls="--", lw=1, label=f"elbow ~ PC{e}")
    ax[0].set_xlabel("PC"); ax[0].set_ylabel("% variance explained (of scaled 5k-HVG matrix)"); ax[0].legend()
    ax[1].plot(np.arange(1, N_PC + 1), np.cumsum(vr) * 100, "o-", ms=3)
    ax[1].set_xlabel("PC"); ax[1].set_ylabel("cumulative % variance")
    fig.suptitle(f"{name}: PCA on top {len(hv)} HVGs, {n} cells ({note})", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{PCA_DIR}/elbow_{name}.png", dpi=110); plt.close(fig)
    json.dump(info, open(f"{PCA_DIR}/.summary_{name}.tmp", "w"), indent=1)
    os.replace(f"{PCA_DIR}/.summary_{name}.tmp", f"{PCA_DIR}/summary_{name}.json")
    print(f"[{name}] {n} cells, {len(hv)} HVG, elbow PC{e}", flush=True)
    return info, vr


def combine():
    infos, vrs = [], {}
    for n in DATASETS:
        p = f"{PCA_DIR}/summary_{n}.json"
        if os.path.exists(p):
            infos.append(json.load(open(p)))
            vrs[n] = np.load(f"{PCA_DIR}/{n}_pca5k.npz")["variance_ratio"]
    if not infos:
        return
    json.dump(infos, open(f"{PCA_DIR}/.pca5k_summary.tmp", "w"), indent=1)
    os.replace(f"{PCA_DIR}/.pca5k_summary.tmp", f"{PCA_DIR}/pca5k_summary.json")
    pd.DataFrame(infos).to_csv(f"{PCA_DIR}/.pca5k_summary.tmp.csv", index=False)
    os.replace(f"{PCA_DIR}/.pca5k_summary.tmp.csv", f"{PCA_DIR}/pca5k_summary.csv")
    k = len(vrs)
    cols = 5
    rows = int(np.ceil(k / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows), squeeze=False)
    for i, (n, vr) in enumerate(vrs.items()):
        a = ax[i // cols, i % cols]
        e = [x for x in infos if x["dataset"] == n][0]
        a.plot(np.arange(1, N_PC + 1), vr * 100, "o-", ms=2.5)
        a.axvline(e["elbow_pc"], color="r", ls="--", lw=1)
        a.set_title(f"{n} (n={e['n_cells']:,}; elbow PC{e['elbow_pc']})", fontsize=9)
        a.set_xlabel("PC"); a.set_ylabel("% variance")
    for j in range(k, rows * cols):
        ax[j // cols, j % cols].axis("off")
    fig.suptitle("PCA elbow plots, top 5,000 HVGs (log1p CP10k, scaled, 50 PCs)")
    fig.tight_layout(); fig.savefig(f"{PCA_DIR}/.elbow_all.tmp.png", dpi=110); plt.close(fig)
    os.replace(f"{PCA_DIR}/.elbow_all.tmp.png", f"{PCA_DIR}/elbow_all_datasets.png")


if __name__ == "__main__":
    args = sys.argv[1:]
    skip_combine = "--no-combine" in args       # parallel runs: combine once at the end
    args = [a for a in args if a != "--no-combine"]
    if args == ["combine"]:
        combine()
    else:
        for n in args:
            run(n)
        if not skip_combine:
            combine()
