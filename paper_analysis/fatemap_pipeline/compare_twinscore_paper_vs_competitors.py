#!/usr/bin/env python3
"""AUPRC of TwinScore-paper (TwinScore_LARRY_melanoma.pdf method: Stage I-III, jackknife-50, ranked by |z|) vs the
five competitor methods, on the SAME cells/genes (twinfer_input_{label}_{gs}_paper.csv), per gene set, for a dataset.

Two TwinScore-paper scores reported (as the paper itself lists both, Table 2 vs "Pearson rank" comparison):
  called   = |z| for pairs that pass Stage I + Stage III BH (q=0.05); everything else ranked below all of these.
  all_z    = |z| for every candidate pair, ignoring the BH calls (the paper's own "ranked by |z|" list).
Positives = CollecTRI edges (substituted for the paper's ChIP-Atlas ground truth; see twinscore_paper.py docstring).
usage: compare_twinscore_paper_vs_competitors.py <dataset> [gene_set ...]   (default: all 9 gene sets)
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

ds = sys.argv[1]
label = ds.lower()
DATA = f"{TWINFER_PROJECT_ROOT}/analysis_data/{label}/data"
METHODS = ["rho", "ppcor", "pidc", "genie3", "grnboost2"]
ORDER = ["variability_high", "variability_mid", "variability_low", "detection_high", "detection_mid", "detection_low",
         "correlation_high", "correlation_mid", "correlation_low"]
gene_sets = sys.argv[2:] or ORDER


def met(score, y):
    sc = np.nan_to_num(np.asarray(score, float), nan=np.nanmin(score) - 1 if np.isfinite(score).any() else 0.0)
    pr, rc, _ = precision_recall_curve(y, sc)
    a = auc(rc, pr)
    k = int(y.sum())
    return a, a / y.mean(), int(y[np.argsort(-sc, kind="stable")[:k]].sum()), k


rows = []
for gs in gene_sets:
    f = f"{DATA}/twinscore_paper/twinscore_paper_{label}_{gs}_pair_terms.csv"
    if not os.path.exists(f):
        print(f"missing {f}, skip"); continue
    p = pd.read_csv(f)
    y = p["collectri_edge"].to_numpy().astype(int)
    r = dict(gene_set=gs, pairs=len(p), n_true=int(y.sum()), random=round(y.mean(), 4))
    score_called = np.where(p["called"].to_numpy(), np.abs(p["z"].to_numpy()), -np.inf)
    a, x, tp, k = met(score_called, y)
    r.update(paper_called_auprc=round(a, 4), paper_called_x=round(x, 3), paper_called_hits=f"{tp}/{k}")
    a, x, tp, k = met(np.abs(p["z"].to_numpy()), y)
    r.update(paper_allz_auprc=round(a, 4), paper_allz_x=round(x, 3), paper_allz_hits=f"{tp}/{k}")
    key = list(zip(p["gene_1"], p["gene_2"]))
    for m in METHODS:
        cf = f"{DATA}/networks/{m}_{gs}_allgenes_paper.csv"
        if not os.path.exists(cf):
            continue
        c = pd.read_csv(cf)
        imp = {(t, g): v for t, g, v in zip(c["TF"], c["target"], c["importance"])}
        sc = np.array([imp.get(kk, 0.0) for kk in key])
        a, x, tp, k = met(sc, y)
        r.update({f"{m}_auprc": round(a, 4), f"{m}_x": round(x, 3), f"{m}_hits": f"{tp}/{k}"})
    rows.append(r)

df = pd.DataFrame(rows)
df["o"] = df.gene_set.map({g: i for i, g in enumerate(ORDER)})
df = df.sort_values("o").drop(columns="o")
xs = ["paper_called_x", "paper_allz_x"] + [f"{m}_x" for m in METHODS if f"{m}_x" in df]
pd.set_option("display.width", 260)
print(f"=== {ds}: AUPRC-x (AUPRC / random) per gene set, TwinScore-paper's own cell/gene universe")
print(df[["gene_set", "pairs", "n_true", "random"] + xs].to_string(index=False))
print(f"\nmean AUPRC-x over {len(df)} gene sets:", df[xs].mean().round(3).to_dict())
comp_cols = [f"{m}_x" for m in METHODS if f"{m}_x" in df]
best = df[comp_cols].max(axis=1)
print("paper_called beats every competitor in", int((df.paper_called_x > best).sum()), "of", len(df), "gene sets")
print("paper_allz   beats every competitor in", int((df.paper_allz_x > best).sum()), "of", len(df), "gene sets")
out = f"{DATA}/twinscore_paper/{label}_auprc_vs_competitors.csv"
df.to_csv(out, index=False)
print("wrote", out)
