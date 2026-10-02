#!/usr/bin/env python3
"""Shrink the raw ChIP-Atlas candidate-pair set (147 TFs x 8,742 targets, 162,251 edges --
computationally infeasible for infer_with_twinfer's O(G^2 x B) pairwise-permutation cost)
down to a tractable gene panel: for each of the 147 TFs, rank its ChIP-Atlas-bound candidate
targets by |Spearman correlation| with the TF's own expression across the QC-filtered
endo_T0+endo_T1 cells, and keep the top N_TARGETS_PER_TF (~4) -- same "top-K most-correlated
targets per TF" convention already used in the LARRY gene-sets pipeline (pick_gene_sets.ipynb,
yscher's panel-building approach). All 147 TFs are kept (per the PDF: "TFs are never removed"
-- the shrink only touches the target side).
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import spearmanr

QC_DIR = f'{TWINFER_PROJECT_ROOT}/finalized_data/hPSC_20260927_data/qc_filtered'
EDGES_PATH = f'{TWINFER_PROJECT_ROOT}/analysis_data/hPSC_20260927/data/hpsc_endoderm_chipatlas_candidate_pairs.csv'
OUT_DIR = f'{TWINFER_PROJECT_ROOT}/analysis_data/hPSC_20260927/data'
N_TARGETS_PER_TF = 4


def main():
    print("loading QC-filtered counts + normalizing (log1p CP10k)", flush=True)
    X = sio.mmread(f"{QC_DIR}/hPSC_20260927_qc_counts.mtx").tocsr()
    genes = np.array(open(f"{QC_DIR}/genes.txt").read().split())
    gidx = {g: i for i, g in enumerate(genes)}
    total = np.asarray(X.sum(axis=0)).ravel().astype(float)
    total[total == 0] = 1.0

    edges = pd.read_csv(EDGES_PATH)
    tfs = sorted(edges.TF.unique())
    print(f"{len(tfs)} TFs, {edges.target.nunique()} distinct raw candidate targets, {len(edges)} raw edges", flush=True)

    def normed_row(gene):
        i = gidx.get(gene)
        if i is None:
            return None
        row = np.asarray(X[i, :].todense()).ravel().astype(float)
        return np.log1p(row / total * 1e4)

    kept_edges = []
    missing_tf_expr = 0
    for k, tf in enumerate(tfs):
        tf_vec = normed_row(tf)
        if tf_vec is None or tf_vec.std() == 0:
            missing_tf_expr += 1
            continue
        targets = edges.loc[edges.TF == tf, "target"].tolist()
        targets = [t for t in targets if t != tf]
        scored = []
        for t in targets:
            t_vec = normed_row(t)
            if t_vec is None or t_vec.std() == 0:
                continue
            r, _ = spearmanr(tf_vec, t_vec)
            if np.isfinite(r):
                scored.append((t, r))
        scored.sort(key=lambda x: -abs(x[1]))
        top = scored[:N_TARGETS_PER_TF]
        for t, r in top:
            kept_edges.append((tf, t, r))
        if (k + 1) % 20 == 0:
            print(f"  [{k+1}/{len(tfs)}] {tf}: {len(targets)} candidates -> {len(top)} kept", flush=True)

    print(f"TFs with no/flat expression in QC matrix (skipped): {missing_tf_expr}", flush=True)

    kept_df = pd.DataFrame(kept_edges, columns=["TF", "target", "spearman_r"])
    kept_df.to_csv(f"{OUT_DIR}/hpsc_endoderm_candidate_pairs_top{N_TARGETS_PER_TF}.csv", index=False)

    final_genes = sorted(set(kept_df.TF) | set(kept_df.target))
    json.dump(final_genes, open(f"{OUT_DIR}/hpsc_endoderm_final_gene_panel.json", "w"))

    print(f"\nfinal kept edges: {len(kept_df)}", flush=True)
    print(f"distinct TFs kept: {kept_df.TF.nunique()}", flush=True)
    print(f"distinct targets kept: {kept_df.target.nunique()}", flush=True)
    print(f"final gene panel size (union TF+target): {len(final_genes)}", flush=True)
    print(f"wrote {OUT_DIR}/hpsc_endoderm_candidate_pairs_top{N_TARGETS_PER_TF}.csv", flush=True)
    print(f"wrote {OUT_DIR}/hpsc_endoderm_final_gene_panel.json", flush=True)


if __name__ == "__main__":
    main()
