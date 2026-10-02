#!/usr/bin/env python3
"""Rescore FM01 TwinFER results (TwinScore-paper all-z, and TwinScore_supp gated-bootstrap) plus the same-universe
competitors against the ChIP-Atlas melanoma-cell-class ground truth (check_chipatlas_melanoma_coverage.py's cache,
FM01-only union) instead of CollecTRI. Same convention as score_vs_chipatlas_breast.py: universe restricted to pairs
whose source (gene_1) is one of the 11 TFs with >=1 melanoma-cell-class ChIP-Atlas experiment."""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

BASE = f'{TWINFER_PROJECT_ROOT}'
CACHE = f"{BASE}/analysis_data/melanoma_chipatlas/chipatlas_melanoma_raw.json"
cache = json.load(open(CACHE))
OK_TFS = {tf for tf, e in cache.items() if e.get("status") == "ok"}
MEL_EDGES = {(tf, g) for tf, e in cache.items() if e.get("status") == "ok" for g in e.get("bound_genes", [])}
print(f"ground truth: {len(OK_TFS)} source TFs with melanoma ChIP-Atlas data ({sorted(OK_TFS)}), "
      f"{len(MEL_EDGES):,} positive (TF,target) edges")

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
        n_tf_present = int(p.loc[mask, "gene_1"].nunique())
        if mask.sum() < 5:
            rows.append(dict(gene_set=gs, pairs=int(mask.sum()), n_tf_present=n_tf_present, note="too few pairs (skipped)"))
            continue
        p = p[mask].reset_index(drop=True)
        y = np.array([int((a, b) in MEL_EDGES) for a, b in zip(p.gene_1, p.gene_2)])
        r = dict(gene_set=gs, pairs=len(p), n_tf_present=n_tf_present, n_true=int(y.sum()),
                  random=round(y.mean(), 4) if len(y) else np.nan)
        if y.sum() < 3:
            r["note"] = "too few positives (skipped)"
            rows.append(r)
            continue
        if source == "paper":
            m = met(np.abs(p["z"].to_numpy()), y)
            if m:
                r.update(twin_x=round(m[1], 3), twin_auprc=round(m[0], 4), twin_hits=f"{m[2]}/{m[3]}")
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
    pd.set_option("display.width", 260)
    for source in ("paper", "gated_bootstrap"):
        df = score_source("FM01", source)
        print(f"\n=== FM01 / TwinScore-{source} vs ChIP-Atlas melanoma truth ===")
        print(df.to_string(index=False))
        xcols = [c for c in df.columns if c.endswith("_x")]
        if xcols:
            print("mean (scoreable gene sets only):", df[xcols].mean().round(3).to_dict())
        out = f"{BASE}/analysis_data/fm01/data/fm01_{source}_vs_chipatlas_melanoma.csv"
        df.to_csv(out, index=False)
        print("wrote", out)
