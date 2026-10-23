#!/usr/bin/env python3
"""
competitors_module_sets.py

Competitor GRN methods (rho, ppcor, PIDC, GENIE3, GRNBoost2) on the gene set of each module of a module-network run,
one module at a time, plus a like-for-like comparison with the TwinScore scenario test.

Same cells and same gene panel as twinscore_module_scenarios.py: clone-filtered cells (clones of size 2-MAX_CLONE_SIZE,
A+B merged), log1p(CP10k) via panel_expression. The competitor implementations are those of
paper_analysis/fatemap_pipeline/run_competitors_fatemap.py (ALL_GENES mode: every module gene is a candidate regulator).

Evaluation (undirected, per module): AUPRC x base rate (1.0 = chance) against CollecTRI edges inside the module.
A pair's score is the larger of its two directed importances. TwinScore pairs are scored by |z_U| (Stage III) and |z_rho|.

PIDC cost grows ~G^3 in pure Python: modules larger than --pidc-max-genes skip PIDC.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

from paper_analysis.fatemap_pipeline import run_competitors_fatemap as rc
from paper_analysis.fatemap_pipeline.twinscore_paper import TWINFER_PROJECT_ROOT, COLLECTRI
from paper_analysis.fatemap_pipeline.twinscore_spec_fm06 import clone_filter, panel_expression
from real_data_scripts.twinscore_module_scenarios import load_dataset

METHODS = ["rho", "ppcor", "pidc", "genie3", "grnboost2"]
log = rc.log


def auprc_x(score, truth):
    score = np.nan_to_num(np.asarray(score, float), nan=-1e9)
    truth = np.asarray(truth, int)
    if truth.sum() < 3:
        return np.nan, int(truth.sum())
    p, r, _ = precision_recall_curve(truth, score)
    return auc(r, p) / truth.mean(), int(truth.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["FM01", "FM06"], default="FM01")
    ap.add_argument("--modules-csv", required=True)
    ap.add_argument("--modules", type=int, nargs="+", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scenario-dir", default=None, help="folder with M{k}_pair_scenarios.csv (TwinScore) to compare")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--n-cores", type=int, default=8)
    ap.add_argument("--pidc-max-genes", type=int, default=300)
    ap.add_argument("--max-genes", type=int, default=None, help="subsample genes per module (testing)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    methods = a.methods.split(",")
    rc.OUT_TAG = ""

    mod = pd.read_csv(a.modules_csv, index_col=0).iloc[:, 0]
    ct = pd.read_csv(COLLECTRI, sep="\t")
    edges = set(zip(ct.source_genesymbol, ct.target_genesymbol))
    df, Xc, tot, genes_full = load_dataset(a.dataset)
    rows_ = np.flatnonzero(clone_filter(pd.DataFrame(dict(c=df.barcode)), "c"))
    gidx = {g: i for i, g in enumerate(genes_full)}
    results = []

    for k in (a.modules if a.modules else sorted(mod.unique())):
        name = f"M{k}"
        summ_path = f"{a.out_dir}/{name}_competitor_auprc.csv"
        if os.path.exists(summ_path):
            log(f"[{name}] already done, skipping")
            continue
        genes = [g for g in mod.index[mod == k] if g in gidx]
        if a.max_genes:
            genes = list(np.random.default_rng(0).choice(genes, min(a.max_genes, len(genes)), replace=False))
        X = panel_expression(Xc[rows_], tot[rows_], np.array([gidx[g] for g in genes]), None, genes_full)
        GI = {g: i for i, g in enumerate(genes)}
        log(f"[{name}] {X.shape[0]:,} cells x {len(genes)} genes")

        for m in methods:
            if os.path.exists(f"{a.out_dir}/{m}_{name}_allgenes.csv"):
                continue
            if m == "pidc" and len(genes) > a.pidc_max_genes:
                log(f"[{name}] PIDC skipped ({len(genes)} genes > {a.pidc_max_genes})")
                continue
            t0 = time.time()
            if m == "rho":
                rc.run_rho(X, genes, GI, genes, a.out_dir, name)
            elif m == "ppcor":
                rc.run_ppcor(X, genes, GI, genes, a.out_dir, name)
            elif m == "pidc":
                rc.run_pidc(X, genes, GI, genes, a.out_dir, name)
            else:
                rc.run_tree_method(X, genes, GI, genes, a.out_dir, name, m, a.n_cores)
            log(f"[{name}] {m} {time.time() - t0:.0f}s")

        # ---- evaluation, undirected pairs ----
        pairs = None
        if a.scenario_dir and os.path.exists(f"{a.scenario_dir}/{name}_pair_scenarios.csv"):
            T = pd.read_csv(f"{a.scenario_dir}/{name}_pair_scenarios.csv")
            pairs = T[["gene_1", "gene_2", "collectri_edge"]].copy()
            pairs["TwinScore |z_U|"] = T.z_U.abs()
            pairs["TwinScore |z_rho|"] = T.z_rho.abs()
        for m in methods:
            p = f"{a.out_dir}/{m}_{name}_allgenes.csv"
            if not os.path.exists(p):
                continue
            L = pd.read_csv(p)
            d = {(r.TF, r.target): r.importance for r in L.itertuples()}
            if pairs is None:
                gl = sorted(genes)
                pairs = pd.DataFrame([(x, y) for i, x in enumerate(gl) for y in gl[i + 1:]], columns=["gene_1", "gene_2"])
                pairs["collectri_edge"] = [int((x, y) in edges or (y, x) in edges) for x, y in zip(pairs.gene_1, pairs.gene_2)]
            pairs[m] = [max(d.get((x, y), 0.0), d.get((y, x), 0.0)) for x, y in zip(pairs.gene_1, pairs.gene_2)]
        if pairs is not None:
            score_cols = [c for c in pairs.columns if c not in ("gene_1", "gene_2", "collectri_edge")]
            out = []
            for c in score_cols:
                v, n_true = auprc_x(pairs[c], pairs.collectri_edge)
                out.append(dict(module=name, method=c, auprc_x_base=v, n_true=n_true, n_pairs=len(pairs)))
            out = pd.DataFrame(out)
            out.to_csv(summ_path, index=False)
            log(f"[{name}] AUPRC x base rate vs CollecTRI (1.0 = chance)\n{out.round(2).to_string(index=False)}")
            results.append(out)

    if results:
        pd.concat(results).to_csv(f"{a.out_dir}/competitor_auprc_all_modules.csv", index=False)


if __name__ == "__main__":
    main()
