#!/usr/bin/env python3
"""Precision / recall of TwinScore-recent, TwinScore(phi) and the competitors at the SAME cut, FM06, one gene set.

Reads   <BASE>/analysis_data/fm06/data/twinscore_spec/twinscore_spec_{merged_unordered,ab}_analytic_fm06_<gs>_pair_terms.csv
Cuts    (a) top-k, k = number of true pairs in the universe: precision = recall = hits / k
        (b) top-n, n = the number of pairs TwinScore-recent calls in that universe (Stage I and |z*| > 2.576; for A/B, with the fan-out dropped):
            precision = hits / n, recall = hits / n_true.  For TwinScore-recent itself this is exactly its called set.
Modes   merged: unordered pairs, true = CollecTRI edge in either direction, source universe = pair with a CollecTRI-source gene; competitors and
                TwinScore(phi) are the larger of the two orientations.
        ab:     ordered pairs, true = edge x -> y, source universe = x is a CollecTRI source.
Writes  twinscore_spec/spec_pr_tables_<gs>.csv and prints the tables.
usage: spec_pr_tables.py [gene_set]
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import os
import sys

import numpy as np
import pandas as pd

PIPE = os.path.dirname(os.path.abspath(__file__))
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, PIPE)
from paper_analysis.fatemap_pipeline.final_fm06_tables import COLLECTRI, COMPETITORS, SCORED, competitor_scores  # noqa: E402

BASE = f'{TWINFER_PROJECT_ROOT}'
D = f"{BASE}/analysis_data/fm06/data/twinscore_spec"
gs = sys.argv[1] if len(sys.argv) > 1 else "correlation_high"


def cut(sc, y, n):
    sc = np.nan_to_num(np.asarray(sc, float), nan=-1e9, neginf=-1e9)
    tp = int(y[np.argsort(-sc, kind="stable")[:n]].sum())
    return tp, tp / n, tp / y.sum()


ct = pd.read_csv(COLLECTRI, sep="\t")
ct = ct[ct.source_genesymbol != ct.target_genesymbol]
sources = set(ct.source_genesymbol)
fin = pd.read_csv(f"{SCORED}/twinscore_supplement_fm06_{gs}_absplit_gated_bootstrap_pair_terms.csv")[["gene_1", "gene_2", "TwinScore"]]
out = []
for mode in ("merged", "ab"):
    if mode == "merged":
        T = pd.read_csv(f"{D}/twinscore_spec_merged_unordered_analytic_fm06_{gs}_pair_terms.csv")
        recent = {"TwinScore-recent [analytic]: R (called)": "analytic: R (called)"}
        mask_src = (T.src_any == 1).to_numpy()
        headline = "TwinScore-recent [analytic]: R (called)"
        old_fin = "final TwinFER" if "final TwinFER" in T.columns else "TwinScore(phi)"
        T = T.rename(columns={old_fin: "TwinScore(phi)", "analytic: R (called)": headline})
        methods = [headline, "TwinScore(phi)", *COMPETITORS]
    else:
        T = pd.read_csv(f"{D}/twinscore_spec_ab_analytic_fm06_{gs}_pair_terms.csv")
        T = T.merge(fin.rename(columns={"TwinScore": "TwinScore(phi)"}), on=["gene_1", "gene_2"], how="left")
        comp = competitor_scores(gs, T)
        for c_ in COMPETITORS:
            T[c_] = comp[c_]
        names = {"analytic: s(R)+coupled*s(gamma), fan-out dropped": "TwinScore-recent [analytic]: s(R)+coupled*s(gamma), fan-out dropped",
                 "analytic: s(R)+coupled*s(gamma)": "TwinScore-recent [analytic]: s(R)+coupled*s(gamma)",
                 "analytic: R only (called)": "TwinScore-recent [analytic]: R only (called)"}
        T = T.rename(columns=names)
        headline = names["analytic: s(R)+coupled*s(gamma), fan-out dropped"]
        mask_src = T.gene_1.isin(sources).to_numpy()
        # methods = [headline, names["analytic: s(R)+coupled*s(gamma)"], names["analytic: R only (called)"], "TwinScore(phi)", *COMPETITORS]   # earlier: also the no-drop and R-only variants
        methods = [headline, "TwinScore(phi)", *COMPETITORS]     # full spec variant only
    y_all = T.collectri_edge.to_numpy()
    for uname, mask in (("all pairs", np.ones(len(T), bool)), ("source-limited", mask_src)):
        y = y_all[mask]
        k = int(y.sum())
        n_called = int((T[headline].to_numpy(float)[mask] > -1e8).sum())
        for m_ in methods:
            sc = T[m_].to_numpy(float)[mask]
            tpk, pk, rk = cut(sc, y, k)
            tpn, pn, rn = cut(sc, y, n_called)
            out.append(dict(mode=mode, universe=uname, n_pairs=int(mask.sum()), n_true=k, random_precision=round(k / mask.sum(), 3), method=m_,
                            hits_top_k=tpk, precision_recall_at_k=round(pk, 3), n_called_cut=n_called, hits_top_n=tpn,
                            precision_at_n=round(pn, 3), recall_at_n=round(rn, 3)))
R = pd.DataFrame(out)
R.to_csv(f"{D}/spec_pr_tables_{gs}.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 75)
for (mode, u), g in R.groupby(["mode", "universe"], sort=False):
    r0 = g.iloc[0]
    print(f"\n== {mode}, {u}: {r0.n_pairs} pairs, {r0.n_true} true (random precision {r0.random_precision}); top-k k={r0.n_true}, top-n n={r0.n_called_cut} (TwinScore-recent's called set) ==")
    print(g[["method", "hits_top_k", "precision_recall_at_k", "hits_top_n", "precision_at_n", "recall_at_n"]].to_string(index=False))
