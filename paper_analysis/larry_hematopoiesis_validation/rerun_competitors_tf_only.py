"""Rerun GENIE3/GRNBoost2 with candidate regulators restricted to the panel's actual TF-like genes
(not every gene as a potential source), for a fair comparison against TwinScore's TF-sourced-only
top-K numbers. PIDC/ppcor/rho don't take a regulator list as an algorithmic input (they're
symmetric all-pairs methods), so for those we post-hoc filter their existing ALL_GENES output to
TF-sourced pairs only -- same treatment TwinScore itself got.
"""
import json
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import roc_auc_score, average_precision_score

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, ".")
from paper_analysis.larry_hematopoiesis_validation import apply_twinscore_supplement_larry as m

GS = "correlation_high"
SEED = 0
NCPU = 8

ct = pd.read_csv("resources/collectri_mouse.tsv", sep="\t")
ct = ct[ct.source_genesymbol != ct.target_genesymbol]
CE = set(zip(ct.source_genesymbol, ct.target_genesymbol))
collectri_tfs = set(ct.source_genesymbol.unique())
animaltfdb = set(x.strip() for x in open("resources/mouse_TF_AnimalTFDB3.txt"))
all_tf_like = collectri_tfs | animaltfdb

genes = m.yscher_genes(GS)
TFS = [g for g in genes if g in all_tf_like]
GI = {g: i for i, g in enumerate(genes)}
print(f"{GS}: {len(genes)} genes, {len(TFS)} TF-like candidate regulators: {TFS}", flush=True)

# pooled day2+4 cp10k data, matching run_competitor_methods.py's own convention
import scipy.io as sio
X_full_mm = sio.mmread(f"{m.SOURCE}/larry_qc_counts.mtx").tocsr()
genes_full = np.array(open(f"{m.SOURCE}/genes.txt").read().split())
obs = pd.read_csv(f"{m.SOURCE}/obs_metadata.csv", index_col=0)
day = obs["library"].astype(str).str.extract(r"_d(\d+)_", expand=False).astype(int).to_numpy()
pool_mask = np.isin(day, [2, 4])
cell_total = np.asarray(X_full_mm.sum(axis=1)).ravel().astype(float)
gidx_full = {g: i for i, g in enumerate(genes_full)}
col = [gidx_full[g] for g in genes]
X_full = np.log1p(X_full_mm[pool_mask][:, col].toarray() / cell_total[pool_mask, None] * 1e4)
print(f"pooled day2+4: {X_full.shape[0]:,} cells x {len(genes)} genes", flush=True)

RF_KWARGS = {"n_jobs": 1, "n_estimators": 1000, "max_features": "sqrt"}
SGBM_KWARGS = {"learning_rate": 0.01, "n_estimators": 5000, "max_features": 0.1, "subsample": 0.9}
EARLY_STOP_WINDOW_LENGTH = 25


class EarlyStopMonitor:
    def __init__(self, window_length=EARLY_STOP_WINDOW_LENGTH):
        self.window_length = window_length

    def __call__(self, current_round, regressor, _):
        if current_round >= self.window_length - 1:
            lo = max(0, current_round - self.window_length + 1)
            hi = current_round + 1
            return np.mean(regressor.oob_improvement_[lo:hi]) < 0
        return False


def _fit_one_target(target, regressor_cls, kwargs, with_early_stop):
    tf_names = [t for t in TFS if t != target]
    Xtf = X_full[:, [GI[t] for t in tf_names]]
    y = X_full[:, GI[target]]
    reg = regressor_cls(random_state=SEED, **kwargs)
    if with_early_stop:
        reg.fit(Xtf, y, monitor=EarlyStopMonitor())
        importances = reg.feature_importances_ * len(reg.estimators_)
    else:
        reg.fit(Xtf, y)
        importances = reg.feature_importances_
    return [(tf, target, float(imp)) for tf, imp in zip(tf_names, importances)]


def full_report_topk(pairs, scores, y_lookup):
    y = np.array([y_lookup(p) for p in pairs])
    R = int(y.sum())
    order = np.argsort(-np.asarray(scores), kind="stable")
    topk = order[:R]
    hits = int(y[topk].sum())
    auroc = roc_auc_score(y, scores) if 0 < y.sum() < len(y) else float("nan")
    auprc = average_precision_score(y, scores)
    base = y.mean()
    return dict(n=len(pairs), R=R, hits=hits, prec=hits / R, auroc=auroc, auprc=auprc,
                auprc_x=auprc / base if base > 0 else float("nan"))


t0 = time.time()
genie3_rows = [r for sub in Parallel(n_jobs=NCPU)(
    delayed(_fit_one_target)(g, RandomForestRegressor, RF_KWARGS, False) for g in genes) for r in sub]
print(f"genie3 (TF-only regulators) done in {time.time()-t0:.0f}s", flush=True)

t0 = time.time()
grnboost_rows = [r for sub in Parallel(n_jobs=NCPU)(
    delayed(_fit_one_target)(g, GradientBoostingRegressor, SGBM_KWARGS, True) for g in genes) for r in sub]
print(f"grnboost2 (TF-only regulators) done in {time.time()-t0:.0f}s", flush=True)

pd.DataFrame(genie3_rows, columns=["TF", "target", "importance"]).to_csv(
    f"genie3_{GS}_tfonly.csv", index=False)
pd.DataFrame(grnboost_rows, columns=["TF", "target", "importance"]).to_csv(
    f"grnboost2_{GS}_tfonly.csv", index=False)

# ---------------------------------------------------------------- evaluation
y_lookup = lambda p: 1 if p in CE else 0
U_tf = [(a, b) for a in TFS for b in genes if a != b]  # TF-sourced-only universe, same as TwinScore's

print("\n=== TF-sourced-only universe (n=%d pairs, R=%d true edges) ===" %
      (len(U_tf), sum(y_lookup(p) for p in U_tf)))

for name, rows in [("genie3", genie3_rows), ("grnboost2", grnboost_rows)]:
    mm = {(a, b): v for a, b, v in rows}
    sc = np.array([mm.get(p, 0.0) for p in U_tf])
    rep = full_report_topk(U_tf, sc, y_lookup)
    print(f"{name:10s} (rerun, TF-only regulators): hits={rep['hits']}/{rep['R']} prec={rep['prec']:.3f} "
          f"AUROC={rep['auroc']:.3f} AUPRC={rep['auprc']:.4f} AUPRC_x={rep['auprc_x']:.3f}x")

# post-hoc filter the existing ALL_GENES outputs (genie3/grnboost2 unrestricted, + pidc/ppcor/rho) to
# TF-sourced pairs only: load on the FULL universe (matching how the file was actually scored), then
# subset the resulting array down to the TF-sourced pairs only.
U_full = [(a, b) for a in genes for b in genes if a != b]
idx_full = {p: i for i, p in enumerate(U_full)}
for name in ["genie3", "grnboost2", "pidc", "ppcor", "rho"]:
    try:
        sc_full = m.load_competitor_allgenes(name, GS, U_full, genes)
    except Exception as e:
        print(f"{name}: could not load ALL_GENES version ({e})")
        continue
    if sc_full is None:
        print(f"{name}: no full-universe file found for this exact panel")
        continue
    sc_tf = np.array([sc_full[idx_full[p]] for p in U_tf])
    rep = full_report_topk(U_tf, sc_tf, y_lookup)
    print(f"{name:10s} (existing ALL_GENES run, post-hoc TF-filtered): hits={rep['hits']}/{rep['R']} "
          f"prec={rep['prec']:.3f} AUROC={rep['auroc']:.3f} AUPRC={rep['auprc']:.4f} AUPRC_x={rep['auprc_x']:.3f}x")

# TwinScore itself, same TF-sourced-only universe, for direct comparison
pt = pd.read_csv(f"{GS}_pair_terms.csv").set_index(["gene_1", "gene_2"])
sc_ts = np.array([pt.loc[p, "TwinScore"] for p in U_tf])
rep = full_report_topk(U_tf, sc_ts, y_lookup)
print(f"{'TwinScore':10s} (TF-sourced-only universe): hits={rep['hits']}/{rep['R']} "
      f"prec={rep['prec']:.3f} AUROC={rep['auroc']:.3f} AUPRC={rep['auprc']:.4f} AUPRC_x={rep['auprc_x']:.3f}x")
