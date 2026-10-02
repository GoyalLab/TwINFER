#!/usr/bin/env python3
"""Per-term evaluation of TwinScore_supplement (SpaceBar, centroid A/B) vs CollecTRI, next to
the 5 competitors, over the FULL ordered-pair universe of each gene set (no gates, no NaN
imputation: competitor pairs it never scored get 0.0, same rule as the FM06 comparison).
Metrics: AUPRC, AUPRC-x (fold over random = AUPRC / prevalence), precision@k (k=10,20,50).
Supplement terms are only evaluated if the pair_terms file is newer than this run's z_dagger json
(guards against the stale 12-gene local-test file)."""
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
GENE_SETS = ["correlation_high", "correlation_mid", "correlation_low"]
COMP = ["rho", "ppcor", "pidc", "genie3", "grnboost2"]
TERMS = ["TwinScore", "PAIR", "phi_x", "D", "R", "Wz", "direction_term", "S", "C", "zreg"]


def metrics(score, y):
    score = np.asarray(score, float)
    n_nan = int(np.isnan(score).sum())  # reported, not hidden; NaN pairs are ranked last (finite sentinel, ranking only)
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
        genes = json.load(open(f"{D}/gene_sets_spacebar.json"))[gs]
        U = pd.DataFrame([(a, b) for a in genes for b in genes if a != b], columns=["gene_1", "gene_2"])
        U["y"] = [1 if (a, b) in CE else 0 for a, b in zip(U.gene_1, U.gene_2)]
        y = U["y"].to_numpy()
        print(f"[{gs}] {len(genes)} genes, {len(U)} ordered pairs, {y.sum()} CollecTRI edges "
              f"(prevalence {y.mean():.3f})", flush=True)
        for m in COMP:
            c = pd.read_csv(f"{D}/networks/{m}_{gs}_centroid_ab_allgenes.csv")
            sc = dict(zip(zip(c.TF, c.target), c.importance))
            s = [sc.get((a, b), 0.0) for a, b in zip(U.gene_1, U.gene_2)]
            rows.append({"gene_set": gs, "method": m, "kind": "competitor", **metrics(s, y)})
        # FINAL gated+bootstrap supplement (old ungated-phi files in {D}/ directly are superseded)
        pt = f"{D}/twinscore_supp_gated_bootstrap/twinscore_supplement_spacebar_{gs}_absplit_gated_bootstrap_pair_terms.csv"
        rp = f"{D}/twinscore_supp_gated_bootstrap/run_params_spacebar_{gs}_absplit_gated_bootstrap.json"
        if os.path.exists(pt) and os.path.exists(rp):
            dg = json.load(open(rp))["diagnostics"]
            print(f"    run_params: w={dg['w']:.3f} var_phi_pass={dg['var_phi_pass']:.4f} "
                  f"phi_noise_floor={dg['phi_noise_floor']:.4f} n_gated={dg['n_gated']}/{dg['n_genes']} "
                  f"gate_g={dg['gate_g']:.3f} gate_v={dg['gate_v']:.3f}", flush=True)
            P = pd.read_csv(pt)
            assert len(P) == len(U), (len(P), len(U))
            P = U.merge(P, on=["gene_1", "gene_2"], how="left")
            for t in TERMS:
                rows.append({"gene_set": gs, "method": t, "kind": "supplement_term", **metrics(P[t], y)})
            # |term| variants for signed terms, since CollecTRI direction/sign isn't scored here
            for t in ("R", "Wz", "S", "C", "zreg"):
                rows.append({"gene_set": gs, "method": f"|{t}|", "kind": "supplement_term",
                             **metrics(P[t].abs(), y)})
        else:
            print(f"    (supplement pair_terms for {gs} not ready yet -- competitors only)", flush=True)
    R = pd.DataFrame(rows)
    out = f"{D}/spacebar_centroid_ab_term_performance_gated_bootstrap.csv"
    R.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    for gs in GENE_SETS:
        print(f"\n=== {gs} ===")
        print(R[R.gene_set == gs].drop(columns="gene_set").round(3).to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
