#!/usr/bin/env python3
"""2026-09-23: SpaceBar version of pick_gene_sets_fatemap.py's correlation_{high,mid,low} recipe
(CollecTRI edges with both ends in the detection-floored matrix -> |Spearman rho| per edge ->
top/mid/bottom N_BAND_EDGES edges -> edges_to_panel: 15 TFs by median |rho|, 4 targets each),
computed natively on SpaceBar's 114-gene targeted panel (Sections 2+4, cells with a called clone,
log1p(CP10k) over the full panel) instead of intersecting FM06's sets with the panel.
Only the correlation family is built: variability/detection bands need genome-wide HVG /
detection quantiles that a 114-gene targeted panel doesn't support.
Differences vs FM06: detection floor is per-SECTION (min over Section2/Section4) instead of per-replicate.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, HERE)
from paper_analysis.fatemap_pipeline.pick_gene_sets_fatemap import (COLLECTRI_PATH, MIN_DETECTION_FRAC, N_BAND_EDGES, edges_to_panel)
from paper_analysis.fatemap_pipeline.run_infer_spacebar import RAW_DIR, SECTIONS
from paper_analysis.fatemap_pipeline.run_infer_spacebar_centroid_ab import _full_gene_panel

OUT_DIR = f'{TWINFER_PROJECT_ROOT}/analysis_data/spacebar/data'


def main():
    genes = _full_gene_panel()
    Xs, sec = [], []
    for i in SECTIONS:
        d = pd.read_csv(f"{RAW_DIR}/Section{i}_cell_by_gene_clustered.csv", usecols=["bc_cluster"] + genes)
        d = d[d["bc_cluster"].notna()]
        Xs.append(d[genes].to_numpy(dtype=float))
        sec += [i] * len(d)
    X = np.concatenate(Xs)
    sec = np.array(sec)
    tot = X.sum(axis=1)
    tot[tot == 0] = 1.0
    X = np.log1p(X / tot[:, None] * 1e4)
    print(f"[SpaceBar] {X.shape[0]:,} cells x {len(genes)} genes", flush=True)

    frac = np.stack([(X[sec == i] > 0).mean(axis=0) for i in SECTIONS]).min(axis=0)
    keep = frac >= MIN_DETECTION_FRAC
    genes = [g for g, k in zip(genes, keep) if k]
    X = X[:, keep]
    print(f"[SpaceBar] detection floor >= {MIN_DETECTION_FRAC:.0%} in every section: {len(genes)} genes", flush=True)

    ct = pd.read_csv(COLLECTRI_PATH, sep="\t")
    ct = ct[ct.source_genesymbol != ct.target_genesymbol]
    gset = set(genes)
    pairs = sorted({(r.source_genesymbol, r.target_genesymbol) for r in ct.itertuples(index=False)
                    if r.source_genesymbol in gset and r.target_genesymbol in gset})
    print(f"[SpaceBar] CollecTRI edges within panel: {len(pairs)}", flush=True)

    R = np.apply_along_axis(rankdata, 0, X)
    C = np.abs(np.corrcoef(R, rowvar=False))
    gi = {g: i for i, g in enumerate(genes)}
    abs_rho = {p: float(C[gi[p[0]], gi[p[1]]]) for p in pairs}
    ranked = sorted(pairs, key=lambda p: abs_rho[p], reverse=True)
    mid, h = len(ranked) // 2, N_BAND_EDGES
    bands = {"high": ranked[:h], "mid": ranked[mid - h // 2: mid - h // 2 + h], "low": ranked[-h:]}
    gene_sets, detail = {}, {}
    for lvl, asc in (("high", False), ("mid", False), ("low", True)):
        g, d = edges_to_panel(bands[lvl], abs_rho, rank_by="rho_median", ascending=asc)
        gene_sets[f"correlation_{lvl}"], detail[lvl] = g, d
        print(f"[SpaceBar] correlation_{lvl}: {len(g)} genes ({len(d)} TFs); "
              f"|rho| range of band edges [{min(abs_rho[p] for p in bands[lvl]):.3f}, "
              f"{max(abs_rho[p] for p in bands[lvl]):.3f}]", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(gene_sets, open(f"{OUT_DIR}/gene_sets_spacebar.json", "w"), indent=2)
    json.dump({"correlation": detail}, open(f"{OUT_DIR}/gene_sets_detail_spacebar.json", "w"), indent=2)
    for k, v in gene_sets.items():
        print(k, v)


if __name__ == "__main__":
    main()
