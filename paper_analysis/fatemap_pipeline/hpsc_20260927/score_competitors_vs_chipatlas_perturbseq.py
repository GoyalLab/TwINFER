#!/usr/bin/env python3
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Score the 4 competitor methods (rho, ppcor, pidc, grnboost2) against OUR OWN ground
truth for hPSC_20260927 -- the ChIP-Atlas candidate pairs (real TF binding evidence,
threshold=100, Pluripotent-stem-cell-class experiments, exact SRX match) restricted to
the final 592-edge candidate set (144 TFs, top-4-correlated targets each, NANOG/POU5F1
expanded to their top-20 Perturb-seq-validated targets) -- NOT CollecTRI (a generic
literature database with no connection to this specific dataset's actual measured
binding/perturbation evidence).

Universe scored: all ordered (TF, target) pairs where TF is one of the 144 real candidate
TFs and target is any other gene in the 489-gene panel (488 x 144 = 70,272 ordered pairs,
matching how run_competitors_fatemap.py's ALL_GENES mode scored every gene as a possible
regulator -- we restrict evaluation to the TF side actually being a real candidate
regulator, not every panel gene).

AUPRC (not AUROC, per repo convention) x baseline rate.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

DATA_DIR = f'{TWINFER_PROJECT_ROOT}/analysis_data/hPSC_20260927/data'
NET_DIR = f"{DATA_DIR}/networks"
METHODS = ["rho", "ppcor", "pidc", "grnboost2"]

final_edges = pd.read_csv(f"{DATA_DIR}/hpsc_endoderm_candidate_pairs_final.csv")
positives = set(zip(final_edges.TF, final_edges.target))
real_tfs = sorted(final_edges.TF.unique())
panel_genes = sorted(set(final_edges.TF) | set(final_edges.target))
print(f"ground truth: {len(positives)} ChIP-Atlas(+Perturb-seq) candidate edges, "
      f"{len(real_tfs)} real TFs, {len(panel_genes)} panel genes", flush=True)

universe = [(tf, g) for tf in real_tfs for g in panel_genes if g != tf]
y = np.array([1 if p in positives else 0 for p in universe])
print(f"scoring universe: {len(universe)} ordered pairs, {y.sum()} positives "
      f"({100*y.mean():.2f}% base rate)\n", flush=True)


def full_report(sc, y):
    sc = np.asarray(sc, float)
    sc = np.nan_to_num(sc, nan=np.nanmin(sc) - 1.0 if np.isfinite(sc).any() else 0.0)
    prec, rec, _ = precision_recall_curve(y, sc)
    auprc = auc(rec, prec)
    rand = float(y.mean())
    return auprc, rand, (auprc / rand if rand > 0 else float("nan"))


results = {}
for method in METHODS:
    df = pd.read_csv(f"{NET_DIR}/{method}_chipatlas_perturbseq_panel_allgenes.csv")
    imp_col = "importance"
    lut = {(r.TF, r.target): getattr(r, imp_col) for r in df.itertuples(index=False)}
    sc = np.array([lut.get(p, np.nan) for p in universe])
    auprc, rand, auprc_x = full_report(sc, y)
    results[method] = dict(auprc=auprc, base_rate=rand, auprc_x=auprc_x)
    print(f"{method:10s}  AUPRC={auprc:.4f}  base_rate={rand:.4f}  AUPRC-x={auprc_x:.3f}x", flush=True)

    ranked = pd.DataFrame({"TF": [p[0] for p in universe], "target": [p[1] for p in universe],
                            "score": sc, "true_candidate": y})
    ranked = ranked.sort_values("score", ascending=False)
    print(f"\n  top 20 {method} edges:")
    print(ranked.head(20).to_string(index=False))
    print()

print("\n=== SUMMARY (AUPRC-x, higher is better, 1.0x = chance) ===")
for m, r in sorted(results.items(), key=lambda kv: -kv[1]["auprc_x"]):
    print(f"{m:10s} {r['auprc_x']:.3f}x")
