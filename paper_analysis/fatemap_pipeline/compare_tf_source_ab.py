#!/usr/bin/env python3
"""Same comparison as compare_twinscore_vs_competitors_ab.py, but ALSO with the universe restricted to pairs whose SOURCE
gene (gene_1) is a CollecTRI TF (any gene that appears as source_genesymbol). Non-TF-source pairs can never be positives
(CollecTRI edges are TF -> target), so this removes gene-level effects (e.g. a non-TF source with a large phi lifting all
its outgoing pairs). Scores, positives and competitor tables are unchanged; only the set of pairs ranked changes."""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import os, re, sys, glob
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

DATASETS = ["FM01", "Watermelon_naive", "Watermelon_lag", "Watermelon_late"]
METHODS = ["rho", "ppcor", "pidc", "genie3", "grnboost2"]
ORDER = ["variability_high", "variability_mid", "variability_low", "detection_high", "detection_mid", "detection_low",
         "correlation_high", "correlation_mid", "correlation_low"]
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] CT = pd.read_csv(f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation/resources/collectri_human.tsv', sep="\t")
CT = pd.read_csv(f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/larry_hematopoiesis_validation/resources/collectri_human.tsv', sep="\t")
TFS = set(CT["source_genesymbol"])


def met(score, y):
    sc = np.nan_to_num(np.asarray(score, float), nan=np.nanmin(score) - 1)
    pr, rc, _ = precision_recall_curve(y, sc)
    a = auc(rc, pr)
    k = int(y.sum())
    return a, a / y.mean(), int(y[np.argsort(-sc, kind="stable")[:k]].sum()), k


allrows = []
for ds in DATASETS:
    label = ds.lower()
    D = f"{TWINFER_PROJECT_ROOT}/analysis_data/{label}/data"
    for gs in ORDER:
        p = pd.read_csv(f"{D}/twinscore_supp_gated_bootstrap/twinscore_supplement_{label}_{gs}_absplit_gated_bootstrap_pair_terms.csv")
        y = p["collectri_edge"].to_numpy().astype(int)
        tf = p["gene_1"].isin(TFS).to_numpy()
        key = list(zip(p["gene_1"], p["gene_2"]))
        scores = {"TwinScore": p["TwinScore"].values}
        for m in METHODS:
            c = pd.read_csv(f"{D}/networks/{m}_{gs}_allgenes_absplit.csv")
            imp = {(t, g): v for t, g, v in zip(c["TF"], c["target"], c["importance"])}
            scores[m] = np.array([imp.get(kk, 0.0) for kk in key])
        for scope, mask in [("all_pairs", np.ones(len(p), bool)), ("tf_source", tf)]:
            r = dict(dataset=ds, gene_set=gs, scope=scope, pairs=int(mask.sum()), n_true=int(y[mask].sum()),
                     random=round(y[mask].mean(), 4), n_tf_genes=int(len(set(p.gene_1[tf]))), n_genes=int(p.gene_1.nunique()))
            for name, sc in scores.items():
                a, x, tp, k = met(sc[mask], y[mask])
                r[f"{name}_x"] = round(x, 3); r[f"{name}_auprc"] = round(a, 4); r[f"{name}_hits"] = tp
            allrows.append(r)
df = pd.DataFrame(allrows)
df.to_csv(f'{TWINFER_PROJECT_ROOT}/analysis_data/tf_source_vs_all_pairs_auprc.csv', index=False)
names = ["TwinScore"] + METHODS
xs = [f"{n}_x" for n in names]
pd.set_option("display.width", 250)
print("=== mean AUPRC-x over the 9 gene sets, by dataset and universe")
print(df.groupby(["dataset", "scope"], sort=False)[xs].mean().round(3).to_string())
print("\n=== TwinScore wins (beats every competitor) per dataset / universe")
for (ds, sc), g in df.groupby(["dataset", "scope"], sort=False):
    print(f"{ds:18s} {sc:10s} {int((g.TwinScore_x > g[[f'{m}_x' for m in METHODS]].max(axis=1)).sum())}/9")
print("\n=== correlation_high only")
print(df[df.gene_set == "correlation_high"][["dataset", "scope", "pairs", "n_true", "random"] + xs].to_string(index=False))
print("\n=== TF genes per set (mean) / share of pairs kept")
k = df[df.scope == "tf_source"].groupby("dataset", sort=False)[["n_tf_genes", "n_genes", "pairs"]].mean().round(1)
k["pairs_all"] = df[df.scope == "all_pairs"].groupby("dataset", sort=False)["pairs"].mean().round(1)
print(k.to_string())
