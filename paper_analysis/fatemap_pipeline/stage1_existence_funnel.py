#!/usr/bin/env python3
"""Cheap closed-form Stage I (existence) screen over the FULL raw ChIP-Atlas candidate-pair
set (147 TFs x 8,742 targets, 162,251 edges) for hPSC_20260927 -- the funnel step that makes
the full candidate universe tractable, matching the PDF's actual methodology: Stage I
(z_rho = rho/sd(rho), BH-corrected over ALL candidates) is cheap and runs on everything;
only the BH-significant "Called" survivors get the expensive permutation-based Stage 2-4
machinery (infer_with_twinfer), which is the part that's actually O(G^2 x n_shuffles) and
can't scale to thousands of genes.

sd(rho): this repo's existing closed-form clone-aware convention (analytic_zscores.py:
m_eff_step1 = Kish effective sample size of the clone-weighted design, sd = 1/sqrt(m_eff-1)),
not a literal per-pair 50-group jackknife (the PDF's own version) -- the two estimate the same
quantity (clone-structure-corrected sampling variance of a clone-weighted Spearman rho) but
the analytic one is a single global number depending only on the clone-size distribution
(same computation already used unmodified in apply_twinscore_supplement_larry.py etc.), so it
costs nothing extra per pair. rho itself IS the real clone-weighted Spearman correlation
(weight 1/n_c per cell, clone total weight 1) computed exactly, not approximated -- only its
null SD uses the closed-form shortcut instead of a resampling loop.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
import sys

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import rankdata, norm
from statsmodels.stats.multitest import multipletests

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/larry_hematopoiesis_validation")
from twinfer.scoring.analytic_zscores import m_eff_step1

QC_DIR = f'{TWINFER_PROJECT_ROOT}/finalized_data/hPSC_20260927_data/qc_filtered'
EDGES_PATH = f'{TWINFER_PROJECT_ROOT}/analysis_data/hPSC_20260927/data/hpsc_endoderm_chipatlas_candidate_pairs.csv'
OUT_DIR = f'{TWINFER_PROJECT_ROOT}/analysis_data/hPSC_20260927/data'
BH_Q = 0.05
BATCH = 5000


def main():
    print("loading QC-filtered counts + normalizing (log1p CP10k)", flush=True)
    X = sio.mmread(f"{QC_DIR}/hPSC_20260927_qc_counts.mtx").tocsr()
    genes = np.array(open(f"{QC_DIR}/genes.txt").read().split())
    cells = np.array(open(f"{QC_DIR}/cells.txt").read().split())
    obs = pd.read_csv(f"{QC_DIR}/obs_metadata.csv", index_col=0)
    assert list(obs.index) == list(cells)
    clone_ids = obs["fatemap_clone_singletcode"].astype(str).to_numpy()

    total = np.asarray(X.sum(axis=0)).ravel().astype(float)
    total[total == 0] = 1.0

    edges = pd.read_csv(EDGES_PATH)
    needed_genes = sorted(set(edges.TF) | set(edges.target))
    gidx_full = {g: i for i, g in enumerate(genes)}
    present = [g for g in needed_genes if g in gidx_full]
    missing = set(needed_genes) - set(present)
    print(f"{len(edges)} candidate edges, {len(needed_genes)} distinct genes "
          f"({len(missing)} missing from matrix, dropped)", flush=True)
    edges = edges[edges.TF.isin(present) & edges.target.isin(present)].reset_index(drop=True)

    rows = [gidx_full[g] for g in present]
    expr = np.log1p(X[rows, :].toarray().astype(np.float32) / total[None, :].astype(np.float32) * 1e4)
    gidx = {g: i for i, g in enumerate(present)}

    print("clone-weighted rank-standardizing each gene (Z: weighted mean 0, weighted var 1)", flush=True)
    clone_size = pd.Series(clone_ids).map(pd.Series(clone_ids).value_counts()).to_numpy()
    w = (1.0 / clone_size).astype(np.float64)
    wsum = w.sum()

    n_genes_needed = expr.shape[0]
    Z = np.empty((n_genes_needed, expr.shape[1]), dtype=np.float32)
    for i in range(n_genes_needed):
        r = rankdata(expr[i])
        mx = np.sum(w * r) / wsum
        vx = np.sum(w * (r - mx) ** 2) / wsum
        Z[i] = (np.sqrt(w) * (r - mx) / np.sqrt(vx * wsum)).astype(np.float32)
    print(f"Z matrix built: {Z.shape}", flush=True)

    m_eff = m_eff_step1(pd.Series(clone_ids).value_counts().to_numpy())
    sd = 1.0 / np.sqrt(max(m_eff - 1.0, 1e-9))
    print(f"m_eff_step1 = {m_eff:.1f}, closed-form null sd = {sd:.5f}", flush=True)

    tf_idx = edges.TF.map(gidx).to_numpy()
    tgt_idx = edges.target.map(gidx).to_numpy()
    n_pairs = len(edges)
    rho = np.empty(n_pairs, dtype=np.float64)
    for start in range(0, n_pairs, BATCH):
        end = min(start + BATCH, n_pairs)
        rho[start:end] = np.einsum("ij,ij->i", Z[tf_idx[start:end]], Z[tgt_idx[start:end]])
        if start % (BATCH * 10) == 0:
            print(f"  {end}/{n_pairs} pairs scored", flush=True)

    z = rho / sd
    pvals = 2 * norm.sf(np.abs(z))
    reject, qvals, _, _ = multipletests(pvals, alpha=BH_Q, method="fdr_bh")

    edges["rho"] = rho
    edges["z_rho"] = z
    edges["p_val"] = pvals
    edges["q_val"] = qvals
    edges["called"] = reject
    edges.to_csv(f"{OUT_DIR}/hpsc_endoderm_stage1_screen_all.csv", index=False)

    called = edges[edges.called].copy()
    called = called.sort_values("z_rho", key=lambda s: -s.abs())
    called.to_csv(f"{OUT_DIR}/hpsc_endoderm_stage1_called.csv", index=False)

    called_genes = sorted(set(called.TF) | set(called.target))
    json.dump(called_genes, open(f"{OUT_DIR}/hpsc_endoderm_stage1_called_genes.json", "w"))

    print(f"\n=== SUMMARY ===", flush=True)
    print(f"total candidate pairs screened: {n_pairs}", flush=True)
    print(f"BH-called (q<={BH_Q}): {len(called)} ({100*len(called)/n_pairs:.2f}%)", flush=True)
    print(f"distinct TFs in called set: {called.TF.nunique()}", flush=True)
    print(f"distinct targets in called set: {called.target.nunique()}", flush=True)
    print(f"union gene panel size from called pairs: {len(called_genes)}", flush=True)
    print(f"wrote {OUT_DIR}/hpsc_endoderm_stage1_called.csv", flush=True)
    print(f"wrote {OUT_DIR}/hpsc_endoderm_stage1_called_genes.json", flush=True)


if __name__ == "__main__":
    main()
