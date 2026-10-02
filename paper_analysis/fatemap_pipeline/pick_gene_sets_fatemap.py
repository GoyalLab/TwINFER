#!/usr/bin/env python3
"""Adapts larry_hematopoiesis_validation/pick_gene_sets.ipynb's recipe (CollecTRI
edges, |rho| slices, 15-TF/4-target panels -- "THE panel set, one recipe, every
dataset", the same method used to build resources/gene_sets_yscher.json for LARRY)
to FM06/FM08. Two substitutions for a human, single-timepoint, two-replicate dataset:
  - collectri_mouse.tsv -> collectri_human.tsv (FM06/FM08 are human WM989/melanocyte
    cells; fetched from the OmniPath REST API, same schema).
  - LARRY's per-DAY minimum detection floor -> per-REPLICATE (A/B) minimum, since
    FM06/FM08 have one timepoint but two replicates instead of LARRY's three days.
Everything else (MIN_DETECTION_FRAC, N_HVG, N_BAND_EDGES, DETECT_BANDS, VAR_BANDS,
N_HIGH_TFS, N_TARGETS_PER_TF, edges_to_panel/band_edges) is unchanged from the notebook.
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
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
# [2026-10-01 commented out: pointed into the code tree; LARRY resources/ now lives in analysis_data (user)]
# LARRY_RESOURCES = os.path.join(
#     HERE, "..", "larry_hematopoiesis_validation", "resources"
# )
from twinfer.utils.paths import get_data_root as _res_gdr
LARRY_RESOURCES = f"{_res_gdr()}/paper_analysis/larry_hematopoiesis_validation/resources"
COLLECTRI_PATH = os.path.join(LARRY_RESOURCES, "collectri_human.tsv")

MIN_DETECTION_FRAC = 0.05
N_HVG = 2000
SEED = 0
N_BAND_EDGES = 50
DETECT_BANDS = ((0.50, 0.75), (0.75, 0.90), (0.90, 1.00))
VAR_BANDS = ((0.00, 0.33), (0.33, 0.67), (0.67, 1.00))
N_HIGH_TFS = 15
N_TARGETS_PER_TF = 4


def band_edges(band_genes, collectri_pairs):
    b = set(band_genes)
    return [p for p in collectri_pairs if p[0] in b and p[1] in b]


def edges_to_panel(edges, abs_rho, n_tfs=N_HIGH_TFS, n_targets=N_TARGETS_PER_TF,
                    rank_by="count", ascending=False):
    by_tf = {}
    for tf, tgt in edges:
        by_tf.setdefault(tf, []).append(tgt)
    if rank_by == "rho_median":
        ranked = sorted(
            by_tf,
            key=lambda tf: float(np.median([abs_rho[(tf, t)] for t in by_tf[tf]])),
            reverse=not ascending,
        )[:n_tfs]
    else:
        ranked = sorted(
            by_tf,
            key=lambda tf: (len(by_tf[tf]), float(np.mean([abs_rho[(tf, t)] for t in by_tf[tf]]))),
            reverse=True,
        )[:n_tfs]
    genes_out, detail = set(), []
    for tf in ranked:
        genes_out.add(tf)
        tgts = sorted(by_tf[tf], key=lambda t: abs_rho[(tf, t)], reverse=True)[:n_targets]
        genes_out.update(tgts)
        detail.append((tf, [(t, round(abs_rho[(tf, t)], 3)) for t in tgts]))
    return sorted(genes_out), detail


def pick_gene_sets(h5ad_path, out_json, out_detail_json, label):
    A_full = ad.read_h5ad(h5ad_path)
    X = A_full.raw.X  # log1p(CP10k), all genes -- see fatemap_integration_utils.normalize_and_pca
    genes = A_full.raw.var_names
    replicate = A_full.obs["replicate"].to_numpy()
    reps = sorted(set(replicate))
    print(f"[{label}] loaded {X.shape[0]:,} cells x {X.shape[1]:,} genes, replicates={reps}", flush=True)

    # per-replicate detection fraction, worst (minimum) across replicates per gene --
    # LARRY's per-day floor, adapted to FM06/FM08's per-replicate structure.
    per_rep_frac = np.stack([
        np.asarray((X[replicate == r] > 0).sum(axis=0)).ravel() / (replicate == r).sum()
        for r in reps
    ])
    frac_expr = per_rep_frac.min(axis=0)
    gene_mask = frac_expr >= MIN_DETECTION_FRAC
    print(f"[{label}] per-replicate detection floor (min across {len(reps)} reps) "
          f">= {MIN_DETECTION_FRAC:.0%}: {int(gene_mask.sum()):,} / {len(genes):,} genes", flush=True)

    genes = genes[gene_mask]
    X = X[:, gene_mask]
    frac_expr = frac_expr[gene_mask]

    A = ad.AnnData(X=np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X),
                    obs=A_full.obs.copy(), var=pd.DataFrame(index=genes))
    del A_full, X

    collectri_raw = pd.read_csv(COLLECTRI_PATH, sep="\t")
    collectri = collectri_raw[
        collectri_raw["source_genesymbol"] != collectri_raw["target_genesymbol"]
    ]
    print(f"[{label}] CollecTRI human: {collectri['source_genesymbol'].nunique():,} TFs, "
          f"{collectri['target_genesymbol'].nunique():,} distinct targets, "
          f"{len(collectri):,} edges (self-pairs stripped)", flush=True)

    matrix_genes = set(A.var_names)
    collectri_pairs = sorted({
        (r.source_genesymbol, r.target_genesymbol)
        for r in collectri.itertuples(index=False)
        if r.source_genesymbol in matrix_genes and r.target_genesymbol in matrix_genes
    })
    need = sorted({g for p in collectri_pairs for g in p})
    need_idx = {g: k for k, g in enumerate(need)}
    print(f"[{label}] CollecTRI edges with both ends in the detection-floored matrix: "
          f"{len(collectri_pairs):,} over {len(need):,} genes", flush=True)

    _Xs = np.asarray(A[:, need].X)
    _Rk = np.apply_along_axis(rankdata, 0, _Xs).astype(np.float32)
    del _Xs
    _Rk -= _Rk.mean(axis=0)
    _S = np.sqrt((_Rk ** 2).sum(axis=0))
    _S[_S < 1e-12] = 1.0
    _C = np.abs((_Rk.T @ _Rk) / np.outer(_S, _S))
    del _Rk
    abs_rho = {p: float(_C[need_idx[p[0]], need_idx[p[1]]]) for p in collectri_pairs}
    del _C
    print(f"[{label}] |rho| over {len(abs_rho):,} curated edges "
          f"[{min(abs_rho.values()):.3f}, {max(abs_rho.values()):.3f}]", flush=True)

    # 1. Variability -- top-N_HVG genes (batch_key='replicate'), dispersion terciles
    sc.pp.highly_variable_genes(A, n_top_genes=N_HVG, batch_key="replicate")
    hvg_genes = A.var_names[A.var["highly_variable"]]
    hvg_rank = A.var.loc[hvg_genes, "dispersions_norm"].sort_values(ascending=False)
    third = len(hvg_genes) // 3
    variability_pools = {
        "high": hvg_rank.index[:third].tolist(),
        "mid": hvg_rank.index[third:2 * third].tolist(),
        "low": hvg_rank.index[2 * third:].tolist(),
    }
    variability = {}
    variability_detail = {}
    for lvl in ("high", "mid", "low"):
        g, d = edges_to_panel(band_edges(variability_pools[lvl], collectri_pairs), abs_rho)
        variability[lvl], variability_detail[lvl] = g, d
        print(f"[{label}] variability_{lvl:<4}: {len(g)} genes ({len(d)} TFs)", flush=True)

    # 2. Detection -- quantile bands over the CollecTRI-connected genes
    detection_frac = pd.Series(frac_expr, index=A.var_names)
    det_pool = list(need)
    _d = np.array([detection_frac[g] for g in det_pool])
    detection = {}
    detection_detail = {}
    for lvl, (q0, q1) in zip(("low", "mid", "high"), DETECT_BANDS):
        lo, hi = np.quantile(_d, [q0, q1])
        pool = [g for g, val in zip(det_pool, _d) if lo <= val <= hi]
        g, d = edges_to_panel(band_edges(pool, collectri_pairs), abs_rho)
        detection[lvl], detection_detail[lvl] = g, d
        print(f"[{label}] detection_{lvl:<4}: {len(g)} genes ({len(d)} TFs), "
              f"band=[{lo:.1%}, {hi:.1%}]", flush=True)

    # 3. Correlation -- top/mid/bottom N_BAND_EDGES edges by |rho|
    ranked_edges = sorted(collectri_pairs, key=lambda p: abs_rho[p], reverse=True)
    mid, h = len(ranked_edges) // 2, N_BAND_EDGES
    corr_edges = {
        "high": ranked_edges[:h],
        "mid": ranked_edges[mid - h // 2: mid - h // 2 + h],
        "low": ranked_edges[-h:],
    }
    correlation = {}
    correlation_detail = {}
    correlation["high"], correlation_detail["high"] = edges_to_panel(
        corr_edges["high"], abs_rho, rank_by="rho_median")
    correlation["mid"], correlation_detail["mid"] = edges_to_panel(
        corr_edges["mid"], abs_rho, rank_by="rho_median")
    correlation["low"], correlation_detail["low"] = edges_to_panel(
        corr_edges["low"], abs_rho, rank_by="rho_median", ascending=True)
    for lvl in ("high", "mid", "low"):
        print(f"[{label}] correlation_{lvl:<4}: {len(correlation[lvl])} genes "
              f"({len(correlation_detail[lvl])} TFs)", flush=True)

    gene_sets = {
        "variability_high": variability["high"], "variability_mid": variability["mid"],
        "variability_low": variability["low"],
        "detection_high": detection["high"], "detection_mid": detection["mid"],
        "detection_low": detection["low"],
        "correlation_high": correlation["high"], "correlation_mid": correlation["mid"],
        "correlation_low": correlation["low"],
    }
    json.dump(gene_sets, open(out_json, "w"), indent=2)
    print(f"[{label}] wrote {out_json}", flush=True)

    gene_sets_detail = {
        "variability": variability_detail, "detection": detection_detail,
        "correlation": correlation_detail,
    }
    json.dump(gene_sets_detail, open(out_detail_json, "w"), indent=2)
    print(f"[{label}] wrote {out_detail_json}", flush=True)
    return gene_sets


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "FM06"
    h5ad_path = f"{TWINFER_PROJECT_ROOT}/finalized_data/{dataset}_data/{dataset}_integrated.h5ad"
    out_dir = f"{TWINFER_PROJECT_ROOT}/analysis_data/{dataset.lower()}/data"
    os.makedirs(out_dir, exist_ok=True)
    pick_gene_sets(
        h5ad_path,
        os.path.join(out_dir, f"gene_sets_{dataset.lower()}.json"),
        os.path.join(out_dir, f"gene_sets_detail_{dataset.lower()}.json"),
        dataset,
    )
