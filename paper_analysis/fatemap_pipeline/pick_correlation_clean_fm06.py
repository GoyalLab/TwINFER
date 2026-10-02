#!/usr/bin/env python3
"""FM06: rebuild the correlation gene sets with yscher's gene exclusions applied BEFORE the edge selection.

Same recipe as pick_gene_sets_fatemap.py (CollecTRI human edges with both ends detected in >= 5% of the cells of every replicate; |Spearman rho| over
ALL cells of the integrated object; top-50 edges by |rho| -> the 15 TFs with the highest median |rho|, up to 4 targets each), except that the candidate pool
first loses every gene that yscher's spec_v3 leaves out of its universe:
  * cell-cycle / growth flagged   tf_wide/fm06/gene_flags.csv  (|corr| >= 0.2 with the Tirosh S+G2M score; the PC-based growth flag is off for FM06 there)
  * curated cellular machinery    tf_wide/machinery.py  (mitochondrial, ribosome, chaperone, proteasome, mitosis, DNA replication / histones, splicing,
                                  Pol II / general transcription, nuclear transport, core glycolysis, actin, ...)
  * not in her universe           genes absent from gene_flags.csv (detected in < 5% of the clone-filtered cells)
Writes  gene_sets_fm06.json  key  correlation_high_clean  (backup gene_sets_fm06.json.bak_20260925 written first) and
        gene_sets_detail_fm06_clean.json  (TF -> targets with |rho|), and prints old vs new edges / panel.
usage: pick_correlation_clean_fm06.py [--levels high] [--n-tfs 15 --n-targets 4 --suffix '']      (levels: high mid low -> keys correlation_<level>_clean)
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import argparse
import json
import os
import shutil
import sys

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import rankdata

PIPE = os.path.dirname(os.path.abspath(__file__))
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, PIPE)
from paper_analysis.fatemap_pipeline import pick_gene_sets_fatemap as pk

BASE = f'{TWINFER_PROJECT_ROOT}'
# [2026-10-01 commented out: yscher data now copied to clean_data/external_yscher (user yes); see REPOINT_LOG.tsv] Y = "/gpfs/projects/b1255/yscher/Transcriptomic Distance/exports/_probe_tmp"
Y = f"{TWINFER_PROJECT_ROOT}/clean_data/external_yscher/Transcriptomic_Distance/exports/_probe_tmp"
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, f"{Y}/tf_wide")
from paper_analysis.fatemap_pipeline.external_yscher.machinery import is_machinery  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="high")
    ap.add_argument("--n-tfs", type=int, default=pk.N_HIGH_TFS, help="TFs per panel (default 15, the original recipe)")
    ap.add_argument("--n-targets", type=int, default=pk.N_TARGETS_PER_TF, help="targets per TF (default 4)")
    ap.add_argument("--suffix", default="", help="appended to the key: correlation_<level>_clean<suffix>")
    a = ap.parse_args()
    levels = a.levels.split(",")

    FL = pd.read_csv(f"{Y}/tf_wide/fm06/gene_flags.csv")
    in_universe = set(FL.gene)
    cyc = set(FL[(FL.cycle) | (FL.growth)].gene)
    mach = {g for g in in_universe if is_machinery(g)}
    print(f"her FM06 universe {len(in_universe):,} genes; cycle/growth-flagged {len(cyc):,}; machinery {len(mach):,}; excluded (either) {len(cyc | mach):,}")

    A_full = ad.read_h5ad(f"{BASE}/finalized_data/FM06_data/FM06_integrated.h5ad")
    X, genes = A_full.raw.X, A_full.raw.var_names
    replicate = A_full.obs["replicate"].to_numpy()
    reps = sorted(set(replicate))
    frac = np.stack([np.asarray((X[replicate == r] > 0).sum(axis=0)).ravel() / (replicate == r).sum() for r in reps]).min(axis=0)
    keep = frac >= pk.MIN_DETECTION_FRAC
    genes, X = genes[keep], X[:, keep]
    Xd = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    A = ad.AnnData(X=Xd, obs=A_full.obs.copy(), var=pd.DataFrame(index=genes))
    del A_full, X, Xd

    ct = pd.read_csv(pk.COLLECTRI_PATH, sep="\t")
    ct = ct[ct.source_genesymbol != ct.target_genesymbol]
    have = set(A.var_names)
    pairs_all = sorted({(r.source_genesymbol, r.target_genesymbol) for r in ct.itertuples(index=False) if r.source_genesymbol in have and r.target_genesymbol in have})
    ok = lambda g: g in in_universe and g not in cyc and g not in mach
    pairs_clean = [p for p in pairs_all if ok(p[0]) and ok(p[1])]
    print(f"CollecTRI edges with both ends detected: {len(pairs_all):,}; after the exclusions: {len(pairs_clean):,}")

    need = sorted({g for p in pairs_all for g in p})
    idx = {g: k for k, g in enumerate(need)}
    R = np.apply_along_axis(rankdata, 0, np.asarray(A[:, need].X)).astype(np.float32)
    R -= R.mean(axis=0)
    S = np.sqrt((R ** 2).sum(axis=0))
    S[S < 1e-12] = 1.0
    C = np.abs((R.T @ R) / np.outer(S, S))
    abs_rho = {p: float(C[idx[p[0]], idx[p[1]]]) for p in pairs_all}

    gs = json.load(open(f"{BASE}/analysis_data/fm06/data/gene_sets_fm06.json"))
    shutil.copy(f"{BASE}/analysis_data/fm06/data/gene_sets_fm06.json", f"{BASE}/analysis_data/fm06/data/gene_sets_fm06.json.bak_20260925")
    detail = {}
    for lvl in levels:
        ranked = sorted(pairs_clean, key=lambda p: abs_rho[p], reverse=True)
        # edge band scaled with the panel size (original: 50 edges for 15 TFs x 4 targets = 60 slots)
        mid, h = len(ranked) // 2, int(round(pk.N_BAND_EDGES * a.n_tfs * a.n_targets / (pk.N_HIGH_TFS * pk.N_TARGETS_PER_TF)))
        edges = {"high": ranked[:h], "mid": ranked[mid - h // 2: mid - h // 2 + h], "low": ranked[-h:]}[lvl]
        panel, det = pk.edges_to_panel(edges, abs_rho, n_tfs=a.n_tfs, n_targets=a.n_targets, rank_by="rho_median", ascending=(lvl == "low"))
        old = gs[f"correlation_{lvl}"]
        print(f"\ncorrelation_{lvl}: old {len(old)} genes, new {len(panel)} genes; overlap {len(set(old) & set(panel))}")
        print("  new panel:", panel)
        print("  dropped from the old panel:", sorted(set(old) - set(panel)))
        print("  new top edges |rho|:", [(a_, b_, round(abs_rho[(a_, b_)], 3)) for a_, b_ in edges[:10]])
        print(f"  edge band {h}; |rho| of its last edge {round(abs_rho[edges[-1]], 3)}; TFs {len(det)}; edges among the panel genes that are CollecTRI edges: {sum(1 for x in panel for y in panel if (x, y) in abs_rho)}")
        gs[f"correlation_{lvl}_clean{a.suffix}"] = panel
        detail[f"correlation_{lvl}_clean{a.suffix}"] = det
    json.dump(gs, open(f"{BASE}/analysis_data/fm06/data/gene_sets_fm06.json", "w"), indent=2)
    dp = f"{BASE}/analysis_data/fm06/data/gene_sets_detail_fm06_clean.json"
    old_d = json.load(open(dp)) if os.path.exists(dp) else {}
    old_d.update(detail)
    json.dump(old_d, open(dp, "w"), indent=2)
    print("\nwrote gene_sets_fm06.json (keys added:", [f"correlation_{l}_clean{a.suffix}" for l in levels], ") and gene_sets_detail_fm06_clean.json")


if __name__ == "__main__":
    main()
