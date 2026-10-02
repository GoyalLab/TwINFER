#!/usr/bin/env python3
"""Rescore Watermelon TwinFER results (TwinScore-paper and TwinScore_supp gated-bootstrap) plus the same-universe
competitors against the ChIP-Atlas breast-cell-class ground truth (check_chipatlas_breast_coverage.py's cache)
instead of CollecTRI.

Truth: (TF, target) is positive iff target is in that TF's breast-bound-gene list (score>=250, ChIP-Atlas hg38,
+/-5kb TSS). Universe restricted to pairs whose SOURCE (gene_1) is one of the 82 TFs with >=1 breast-cell-class
ChIP-Atlas experiment (status=="ok") -- for every other TF, breast binding is simply unknown, not negative, so
those pairs are dropped rather than scored as false.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

BASE = f'{TWINFER_PROJECT_ROOT}'
CACHE = f"{BASE}/analysis_data/watermelon_chipatlas_breast/chipatlas_breast_raw.json"
cache = json.load(open(CACHE))
OK_TFS = {tf for tf, e in cache.items() if e.get("status") == "ok"}
BREAST_EDGES = {(tf, g) for tf, e in cache.items() if e.get("status") == "ok" for g in e.get("bound_genes", [])}
print(f"ground truth: {len(OK_TFS)} source TFs with breast ChIP-Atlas data, {len(BREAST_EDGES):,} positive (TF,target) edges")

METHODS = ["rho", "ppcor", "pidc", "genie3", "grnboost2"]
ORDER = ["variability_high", "variability_mid", "variability_low", "detection_high", "detection_mid", "detection_low",
         "correlation_high", "correlation_mid", "correlation_low"]


def met(score, y):
    sc = np.nan_to_num(np.asarray(score, float), nan=np.nanmin(score) - 1 if np.isfinite(score).any() else 0.0)
    if y.sum() < 3 or y.sum() == len(y):
        return None
    pr, rc, _ = precision_recall_curve(y, sc)
    a = auc(rc, pr)
    k = int(y.sum())
    return a, a / y.mean(), int(y[np.argsort(-sc, kind="stable")[:k]].sum()), k


def score_source(dataset, source):
    """source: 'paper' (twinscore_paper pair_terms + _paper-tagged competitor nets) or
    'gated_bootstrap' (twinscore_supp_gated_bootstrap pair_terms + _absplit-tagged competitor nets)."""
    label = dataset.lower()
    DATA = f"{BASE}/analysis_data/{label}/data"
    rows = []
    for gs in ORDER:
        if source == "paper":
            f = f"{DATA}/twinscore_paper/twinscore_paper_{label}_{gs}_pair_terms.csv"
            net_tag = "paper"
        else:
            f = f"{DATA}/twinscore_supp_gated_bootstrap/twinscore_supplement_{label}_{gs}_absplit_gated_bootstrap_pair_terms.csv"
            net_tag = "absplit"
        if not os.path.exists(f):
            continue
        p = pd.read_csv(f)
        mask = p["gene_1"].isin(OK_TFS).to_numpy()
        if mask.sum() < 5:
            continue
        p = p[mask].reset_index(drop=True)
        y = np.array([int((a, b) in BREAST_EDGES) for a, b in zip(p.gene_1, p.gene_2)])
        r = dict(gene_set=gs, pairs=len(p), n_true=int(y.sum()), random=round(y.mean(), 4) if len(y) else np.nan)
        if source == "paper":
            score_called = np.where(p["called"].to_numpy(), np.abs(p["z"].to_numpy()), -np.inf)
            m = met(score_called, y)
            if m:
                r.update(twin_called_auprc=round(m[0], 4), twin_called_x=round(m[1], 3), twin_called_hits=f"{m[2]}/{m[3]}")
            m = met(np.abs(p["z"].to_numpy()), y)
            if m:
                r.update(twin_allz_auprc=round(m[0], 4), twin_allz_x=round(m[1], 3), twin_allz_hits=f"{m[2]}/{m[3]}")
        else:
            m = met(p["TwinScore"].to_numpy(), y)
            if m:
                r.update(twin_x=round(m[1], 3), twin_auprc=round(m[0], 4), twin_hits=f"{m[2]}/{m[3]}")
        key = list(zip(p.gene_1, p.gene_2))
        for meth in METHODS:
            cf = f"{DATA}/networks/{meth}_{gs}_allgenes_{net_tag}.csv"
            if not os.path.exists(cf):
                continue
            c = pd.read_csv(cf)
            imp = {(t, g): v for t, g, v in zip(c.TF, c.target, c.importance)}
            sc = np.array([imp.get(kk, 0.0) for kk in key])
            m = met(sc, y)
            if m:
                r.update({f"{meth}_x": round(m[1], 3), f"{meth}_auprc": round(m[0], 4), f"{meth}_hits": f"{m[2]}/{m[3]}"})
        rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    datasets = sys.argv[1:] or ["Watermelon_naive", "Watermelon_lag", "Watermelon_late"]
    pd.set_option("display.width", 260)
    for ds in datasets:
        for source in ("paper", "gated_bootstrap"):
            df = score_source(ds, source)
            if df.empty:
                print(f"\n{ds} / {source}: no scoreable gene sets"); continue
            xcols = [c for c in df.columns if c.endswith("_x")]
            print(f"\n=== {ds} / TwinScore-{source} vs ChIP-Atlas breast truth ===")
            print(df[["gene_set", "pairs", "n_true", "random"] + xcols].to_string(index=False))
            print("mean:", df[xcols].mean().round(3).to_dict())
            out = f"{BASE}/analysis_data/{ds.lower()}/data/{ds.lower()}_{source}_vs_chipatlas_breast.csv"
            df.to_csv(out, index=False)
            print("wrote", out)
