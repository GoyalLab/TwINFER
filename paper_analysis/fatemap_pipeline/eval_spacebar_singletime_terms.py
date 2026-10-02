#!/usr/bin/env python3
"""Evaluates the single-timepoint TwinScore (S/C/U/h, Stage I-III, apply_singletime_score_
spacebar.py) against CollecTRI, next to the 5 competitors run on the SAME pooled cells
(twinfer_input_spacebar_{gs}_singletime.csv), over the full ordered-pair universe of the gene
set. Same metric/NaN convention as eval_spacebar_centroid_ab_terms.py: AUPRC, AUPRC-x (fold over
random = AUPRC/prevalence), precision@k; competitor pairs never scored get 0.0, NaNs counted and
ranked last (never imputed). z is symmetric per unordered pair (no direction stage), so the same
score is used for both (a,b) and (b,a).
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

D = f'{TWINFER_PROJECT_ROOT}/analysis_data/spacebar/data'
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] COLLECTRI = (f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation/resources/collectri_human.tsv')
COLLECTRI = (f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/larry_hematopoiesis_validation/resources/collectri_human.tsv')
GENE_SETS = ["correlation_high"]
COMP = ["rho", "ppcor", "pidc", "genie3", "grnboost2"]
TERMS = ["z", "S", "C", "U"]
ABS_TERMS = ["S", "C", "U"]  # z is already |z|-ranked in construction but keep signed + abs both


def metrics(score, y):
    score = np.asarray(score, float)
    n_nan = int(np.isnan(score).sum())
    if n_nan:
        score = np.where(np.isnan(score), np.nanmin(score) - 1.0, score)
    ap = average_precision_score(y, score)
    order = np.argsort(-score, kind="stable")
    out = {"AUPRC": ap, "AUPRC_x": ap / y.mean(), "n_nan": n_nan}
    for k in (10, 20, 50):
        out[f"P@{k}"] = y[order[:k]].mean()
    return out


def main():
    ct = pd.read_csv(COLLECTRI, sep="\t")
    ct = ct[ct.source_genesymbol != ct.target_genesymbol]
    CE = set(zip(ct.source_genesymbol, ct.target_genesymbol))
    rows = []
    for gs in GENE_SETS:
        genes = sorted(json.load(open(f"{D}/gene_sets_spacebar.json"))[gs])
        U = pd.DataFrame([(a, b) for a in genes for b in genes if a != b], columns=["gene_1", "gene_2"])
        U["y"] = [1 if (a, b) in CE else 0 for a, b in zip(U.gene_1, U.gene_2)]
        y = U["y"].to_numpy()
        print(f"[{gs}] {len(genes)} genes, {len(U)} ordered pairs, {y.sum()} CollecTRI edges "
              f"(prevalence {y.mean():.3f})", flush=True)

        for m in COMP:
            path = f"{D}/networks/{m}_{gs}_singletime_allgenes.csv"
            if not os.path.exists(path):
                print(f"    ({m} not ready yet)", flush=True)
                continue
            c = pd.read_csv(path)
            sc = dict(zip(zip(c.TF, c.target), c.importance))
            s = [sc.get((a, b), 0.0) for a, b in zip(U.gene_1, U.gene_2)]
            rows.append({"gene_set": gs, "method": m, "kind": "competitor", **metrics(s, y)})

        pt = f"{D}/singletime_score_spacebar_{gs}_pair_terms.csv"
        rp = f"{D}/run_params_spacebar_{gs}_singletime.json"
        if os.path.exists(pt) and os.path.exists(rp):
            dg = json.load(open(rp))
            print(f"    run_params: n_cells={dg['n_cells']:,} n_twin_pairs={dg['n_twin_pairs']:,} "
                  f"n_clones_with_pairs={dg['n_clones_with_pairs']:,} "
                  f"n_genes_inherited={dg['n_genes_inherited']}/{dg['n_genes']}", flush=True)
            P = pd.read_csv(pt)
            assert len(P) == len(U), (len(P), len(U))
            P = U.merge(P, on=["gene_1", "gene_2"], how="left")
            for t in TERMS:
                rows.append({"gene_set": gs, "method": t, "kind": "singletime_term",
                             **metrics(P[t], y)})
            for t in ABS_TERMS:
                rows.append({"gene_set": gs, "method": f"|{t}|", "kind": "singletime_term",
                             **metrics(P[t].abs(), y)})
        else:
            print(f"    (singletime pair_terms for {gs} not ready yet)", flush=True)

    R = pd.DataFrame(rows)
    out = f"{D}/spacebar_singletime_term_performance.csv"
    R.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    for gs in GENE_SETS:
        sub = R[R.gene_set == gs]
        if len(sub):
            print(f"\n=== {gs} ===")
            print(sub.drop(columns="gene_set").round(3).to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
