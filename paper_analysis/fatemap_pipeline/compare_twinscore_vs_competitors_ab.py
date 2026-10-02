#!/usr/bin/env python3
"""AUPRC of TwinScore_supplement (gated bootstrap, A/B split) vs the competitor methods on the SAME cells, per gene set.

Universe = every ordered gene pair in the gene set (rows of the pair_terms csv); positives = CollecTRI edges (its
`collectri_edge` column). Competitor networks (`{method}_{gs}_allgenes_absplit.csv`, TF,target,importance) map to
(gene_1, gene_2); a pair missing from a competitor csv gets importance 0 (GENIE3/GRNBoost2 drop zero-importance rows).
Reports AUPRC, AUPRC-x (= AUPRC / fraction of positive pairs) and hits in the top k (k = number of positives).
usage: compare_twinscore_vs_competitors_ab.py <dataset e.g. FM01|Watermelon_naive> [score_subdir]
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

ds = sys.argv[1]
sub = sys.argv[2] if len(sys.argv) > 2 else "twinscore_supp_gated_bootstrap"
label = ds.lower()
DATA = f"{TWINFER_PROJECT_ROOT}/analysis_data/{label}/data"
METHODS = ["rho", "ppcor", "pidc", "genie3", "grnboost2"]


def met(score, y):
    sc = np.nan_to_num(np.asarray(score, float), nan=np.nanmin(score) - 1)
    pr, rc, _ = precision_recall_curve(y, sc)
    a = auc(rc, pr)
    k = int(y.sum())
    return a, a / y.mean(), int(y[np.argsort(-sc, kind="stable")[:k]].sum()), k


rows = []
for f in sorted(glob.glob(f"{DATA}/{sub}/twinscore_supplement_{label}_*_pair_terms.csv")):
    gs = re.search(rf"{label}_(.*?)_absplit", f).group(1)
    p = pd.read_csv(f)
    y = p["collectri_edge"].to_numpy().astype(int)
    r = dict(gene_set=gs, pairs=len(p), n_true=int(y.sum()), random=round(y.mean(), 4))
    a, x, tp, k = met(p["TwinScore"].values, y)
    r.update(TwinScore_auprc=round(a, 4), TwinScore_x=round(x, 3), TwinScore_hits=f"{tp}/{k}")
    key = list(zip(p["gene_1"], p["gene_2"]))
    for m in METHODS:
        cf = f"{DATA}/networks/{m}_{gs}_allgenes_absplit.csv"
        if not os.path.exists(cf):
            continue
        c = pd.read_csv(cf)
        imp = {(t, g): v for t, g, v in zip(c["TF"], c["target"], c["importance"])}
        sc = np.array([imp.get(kk, 0.0) for kk in key])
        a, x, tp, k = met(sc, y)
        r.update({f"{m}_auprc": round(a, 4), f"{m}_x": round(x, 3), f"{m}_hits": f"{tp}/{k}"})
    rows.append(r)

df = pd.DataFrame(rows)
order = ["variability_high", "variability_mid", "variability_low", "detection_high", "detection_mid", "detection_low",
         "correlation_high", "correlation_mid", "correlation_low"]
df["o"] = df.gene_set.map({g: i for i, g in enumerate(order)})
df = df.sort_values("o").drop(columns="o")
xs = ["TwinScore_x"] + [f"{m}_x" for m in METHODS if f"{m}_x" in df]
pd.set_option("display.width", 250)
print("AUPRC-x (AUPRC / random) per gene set")
print(df[["gene_set", "pairs", "n_true", "random"] + xs].to_string(index=False))
print("\nAUPRC")
print(df[["gene_set"] + [c.replace("_x", "_auprc") for c in xs]].to_string(index=False))
print("\nmean AUPRC-x over", len(df), "gene sets:", df[xs].mean().round(3).to_dict())
best = df[xs[1:]].max(axis=1)
print("TwinScore beats the best competitor in", int((df.TwinScore_x > best).sum()), "of", len(df), "gene sets")
df.to_csv(f"{DATA}/{sub}/{label}_auprc_vs_competitors.csv", index=False)
