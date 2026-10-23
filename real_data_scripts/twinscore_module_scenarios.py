#!/usr/bin/env python3
"""
twinscore_module_scenarios.py

Apply the TwinScore decomposition scenario table (handoff 2026-10-07, section 3a) to the gene pairs inside
each gene module of a module-network run, one module at a time (no cross-module pairs).

  rho = sqrt(hX hY) rho_s + sqrt((1-hX)(1-hY)) rho_hat_Delta

Per pair (single pre-specified pair view, no BH), all tests two-sided |z| > 1.96 (shared test also at 2.326):
  z_C   shared part rho_s     = sign(S) * C / sd(C)          ("h" in twinscore_paper.py)
  z_rho existence             = rho / sd(rho)
  z_U   unshared part         = U / sd(U)
  shared ns & z_rho ns  -> 1 nothing           shared ns & z_rho sig -> 2 regulation
  shared sig & z_U sig  -> 3 regulation + heterogeneity     shared sig & z_U ns -> 4 heterogeneity
No direction (FM01 is a single time point).

Estimators are those of paper_analysis/fatemap_pipeline/twinscore_paper.py (clone-weighted Spearman, S/C from all
within-clone twin pairs of clones of size 2-MAX_CLONE_SIZE, jackknife SDs over clone groups), merged A+B mode.
This script reuses that file's helpers; it differs only in (a) the gene panel = a module, (b) the jackknife is
accumulated online so memory stays O(G^2), (c) the scenario call instead of Stage I/III BH calls.

Run (from code/TwINFER, after `source env.sh`):
  python real_data_scripts/twinscore_module_scenarios.py --modules-csv <modules.csv> --modules 3 5 6 11 --out-dir <dir>
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import scipy.io as sio

from paper_analysis.fatemap_pipeline import twinscore_supp_helpers as supp
from paper_analysis.fatemap_pipeline.external_yscher import twin_layers_lib as L
from paper_analysis.fatemap_pipeline.twinscore_paper import (
    TWINFER_PROJECT_ROOT, COLLECTRI, SEED, twin_pairs, wsm, get_clone_weights)
from paper_analysis.fatemap_pipeline.twinscore_spec_fm06 import clone_filter, panel_expression

NAMES = {1: "1 nothing", 2: "2 regulation", 3: "3 reg+heterogeneity", 4: "4 heterogeneity"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def scenario(z_C, z_rho, z_U, thr_shared):
    sh = np.abs(z_C) > thr_shared
    rho = np.abs(z_rho) > 1.96
    dl = np.abs(z_U) > 1.96
    code = np.where(~sh & ~rho, 1, np.where(~sh & rho, 2, np.where(sh & dl, 3, 4)))
    return pd.Series(code).map(NAMES).to_numpy()


def load_dataset(dataset):
    label = dataset.lower()
    qc = f"{TWINFER_PROJECT_ROOT}/finalized_data/{dataset}_data/qc_filtered"
    X = sio.mmread(f"{qc}/{label}_qc_counts.mtx").tocsr()
    genes_full = np.array(open(f"{qc}/genes.txt").read().split())
    obs = pd.read_csv(f"{qc}/obs_metadata.csv", index_col=0, dtype={"replicate": str})
    tot = np.asarray(X.sum(axis=1)).ravel().astype(float)
    bc = obs["fatemap_clone_singletcode"].astype(str).to_numpy()
    keep = np.array([len(b) > 0 for b in bc])
    df = pd.DataFrame(dict(cell_id=obs.index.to_numpy(), barcode=bc))[keep].reset_index(drop=True)
    return df, X[keep], tot[keep], genes_full


def run_module(name, genes, df, Xc, tot, genes_full, n_jack, out_dir, edges):
    gidx = {g: i for i, g in enumerate(genes_full)}
    genes = [g for g in genes if g in gidx]
    G = len(genes)
    pidx = np.array([gidx[g] for g in genes])
    rows_ = np.flatnonzero(clone_filter(pd.DataFrame(dict(c=df.barcode)), "c"))
    V = panel_expression(Xc[rows_], tot[rows_], pidx, None, genes_full)
    clone = df.barcode.to_numpy()[rows_]
    cell_id = df.cell_id.to_numpy()[rows_]
    clones_all = np.unique(clone)
    log(f"[{name}] {G} genes, {len(clone):,} cells in {len(clones_all):,} clones (size 2-{supp.MAX_CLONE_SIZE})")

    RHO = wsm(V, get_clone_weights(pd.DataFrame(dict(clone_id=clone))))
    ia, ib, w = twin_pairs(clone, cell_id)
    S, C = L.full_depth_layers(V, ia, ib, w)
    U = S - C
    log(f"[{name}] {len(ia):,} within-clone twin pairs")

    rng = np.random.default_rng(SEED)
    groups = np.array_split(rng.permutation(clones_all), n_jack)
    acc = {k: [np.zeros((G, G)), np.zeros((G, G))] for k in ("rho", "C", "U")}   # sum, sum of squares
    for gi, grp in enumerate(groups):
        km = ~np.isin(clone, grp)
        Vg = V[km]
        r = wsm(Vg, get_clone_weights(pd.DataFrame(dict(clone_id=clone[km]))))
        ia_g, ib_g, w_g = twin_pairs(clone[km], cell_id[km])
        S_g, C_g = L.full_depth_layers(Vg, ia_g, ib_g, w_g)
        for k, m in (("rho", r), ("C", C_g), ("U", S_g - C_g)):
            acc[k][0] += m
            acc[k][1] += m * m
        if (gi + 1) % 10 == 0:
            log(f"[{name}] jackknife {gi + 1}/{n_jack}")

    def jack_sd(k):                       # sqrt((n-1)/n * sum (theta_g - theta_bar)^2)
        s1, s2 = acc[k]
        return np.sqrt(np.maximum((n_jack - 1) / n_jack * (s2 - s1 * s1 / n_jack), 0.0))

    sd_rho, sd_C, sd_U = jack_sd("rho"), jack_sd("C"), jack_sd("U")
    xi, yi = np.triu_indices(G, k=1)          # S, C, U, rho are symmetric: one row per unordered pair
    with np.errstate(divide="ignore", invalid="ignore"):
        z_rho = RHO[xi, yi] / sd_rho[xi, yi]
        z_C = np.sign(S[xi, yi]) * C[xi, yi] / sd_C[xi, yi]
        z_U = U[xi, yi] / sd_U[xi, yi]
    T = pd.DataFrame(dict(gene_1=np.array(genes)[xi], gene_2=np.array(genes)[yi],
                          collectri_edge=[int((a, b) in edges or (b, a) in edges)
                                          for a, b in zip(np.array(genes)[xi], np.array(genes)[yi])],
                          rho=RHO[xi, yi], z_rho=z_rho, S=S[xi, yi], C=C[xi, yi], U=U[xi, yi], z_C=z_C, z_U=z_U))
    T["scenario"] = scenario(z_C, z_rho, z_U, 1.96)
    T["scenario_2326"] = scenario(z_C, z_rho, z_U, 2.326)
    T.to_csv(f"{out_dir}/{name}_pair_scenarios.csv", index=False)

    n = len(T)
    summ = pd.DataFrame(dict(
        n_pairs=T.groupby("scenario").size(),
        frac=T.groupby("scenario").size() / n,
        n_collectri_edges=T.groupby("scenario").collectri_edge.sum(),
        n_2326=T.groupby("scenario_2326").size())).reindex(list(NAMES.values())).fillna(0)
    summ.insert(0, "module", name)
    summ.to_csv(f"{out_dir}/{name}_scenario_summary.csv")
    log(f"[{name}] {n:,} pairs, {int(T.collectri_edge.sum())} CollecTRI edges\n{summ.round(3).to_string()}")
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["FM01", "FM06"], default="FM01")
    ap.add_argument("--modules-csv", required=True, help="modules.csv: gene index, 'module' column")
    ap.add_argument("--modules", type=int, nargs="+", default=None, help="default: every module in the csv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-jack", type=int, default=supp.N_JACK if hasattr(supp, "N_JACK") else 50)
    ap.add_argument("--max-genes", type=int, default=None, help="subsample genes per module (testing)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    mod = pd.read_csv(a.modules_csv, index_col=0).iloc[:, 0]
    ct = pd.read_csv(COLLECTRI, sep="\t")
    edges = set(zip(ct.source_genesymbol, ct.target_genesymbol))
    df, Xc, tot, genes_full = load_dataset(a.dataset)
    todo = a.modules if a.modules else sorted(mod.unique())
    for k in todo:                            # each module separately
        if os.path.exists(f"{a.out_dir}/M{k}_scenario_summary.csv"):
            log(f"[M{k}] already done, skipping")
            continue
        genes = list(mod.index[mod == k])
        if a.max_genes:
            genes = list(np.random.default_rng(0).choice(genes, min(a.max_genes, len(genes)), replace=False))
        run_module(f"M{k}", genes, df, Xc, tot, genes_full, a.n_jack, a.out_dir, edges)


if __name__ == "__main__":
    main()
