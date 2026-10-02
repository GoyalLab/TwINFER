#!/usr/bin/env python3
"""FM06: score the already-computed TwinFER (TwinScore_supplement, gated + bootstrap, A/B split) and competitor
networks against the melanoma-specific ChIP-Atlas+KnockTF ground truth (build_fm06_melanoma_ground_truth.py)
instead of CollecTRI, and report how the two ground truths compare.

Positives = melanoma ground-truth edges with evidence == 'both' (ChIP binding AND KnockTF DE agree).
Universe (per gene set): ordered pairs (a, b) of panel genes with a in the melanoma-GT TF list AND
(a, b) present in the panel-restricted melanoma GT table (i.e. the pair was testable: both a, b passed the
ChIP-Atlas/KnockTF eligibility, so the pair is either a true positive or a confirmed negative, not merely absent).
This mirrors the CollecTRI-source universe used for the CollecTRI comparison in final_fm06_tables.py.

Reads
  <BASE>/analysis_data/fm06/data/twinscore_supp_gated_bootstrap/twinscore_supplement_fm06_<gs>_absplit_gated_bootstrap_pair_terms.csv
  <BASE>/analysis_data/fm06/data/networks/{pidc,ppcor,rho,genie3,grnboost2}_<gs>_allgenes.csv
  <BASE>/analysis_data/fm06/data/melanoma_tf_network_fm06.tsv            (all evidence tiers, full TF panel)
  <BASE>/analysis_data/paper_analysis/larry_hematopoiesis_validation/resources/collectri_human.tsv
Writes to <BASE>/analysis_data/fatemap_comparison/data/final_fm06/
  summary_melanoma_gt.csv           per gene set, TwinFER vs every competitor, melanoma-GT universe
  melanoma_gt_vs_collectri_edges.csv  per TF: melanoma-GT vs CollecTRI overlap (also written by the builder script)
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

BASE = f'{TWINFER_PROJECT_ROOT}'
DATA = f"{BASE}/analysis_data/fm06/data"
SCORED = f"{DATA}/twinscore_supp_gated_bootstrap"
NETS = f"{DATA}/networks"
MELANOMA_GT = f"{DATA}/melanoma_tf_network_fm06.tsv"
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] COLLECTRI = f"{BASE}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation/resources/collectri_human.tsv"
COLLECTRI = f"{BASE}/analysis_data/paper_analysis/larry_hematopoiesis_validation/resources/collectri_human.tsv"
OUT = f"{BASE}/analysis_data/fatemap_comparison/data/final_fm06"
GENE_SETS = ["correlation_high", "correlation_mid", "correlation_low", "detection_high", "detection_mid",
             "detection_low", "variability_high", "variability_mid", "variability_low"]
COMPETITORS = ["pidc", "ppcor", "rho", "genie3", "grnboost2"]
MELANOMA_TFS = ["MITF", "SOX10", "JUN", "JUND", "FOSL1", "FOSL2", "TEAD1", "TEAD2", "TEAD3", "TEAD4"]


def metrics(score, y):
    score = np.nan_to_num(np.asarray(score, float), nan=-1e9)
    y = np.asarray(y, int)
    prec, rec, _ = precision_recall_curve(y, score)
    a = auc(rec, prec)
    base = float(np.mean(y))
    k = int(np.sum(y))
    tp = int(y[np.argsort(-score, kind="stable")[:k]].sum()) if k > 0 else 0
    return dict(AUPRC=a, random=base, AUPRC_x=(a / base if base > 0 else float("nan")), AUROC=roc_auc_score(y, score),
                k=k, TP_at_k=tp, precision_at_k=(tp / k if k > 0 else float("nan")))


def competitor_scores(gs, p):
    out = {}
    for m in COMPETITORS:
        c = pd.read_csv(f"{NETS}/{m}_{gs}_allgenes.csv")
        d = {(a, b): v for a, b, v in zip(c.TF, c.target, c.importance)}
        out[m] = np.array([d.get((a, b), 0.0) for a, b in zip(p.gene_1, p.gene_2)])
    return out


TIER_EVIDENCE = {"both": {"both"}, "chip": {"chip_only", "both"}, "any": {"chip_only", "kd_only", "both"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=list(TIER_EVIDENCE), default="both",
                     help="'both'=ChIP+KnockTF agree, 'chip'=any ChIP-Atlas binding evidence (ignores KnockTF), "
                          "'any'=ChIP or KnockTF alone")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    gt = pd.read_csv(MELANOMA_GT, sep="\t")
    gt = gt[gt.TF != gt.target]  # drop self-loops (e.g. JUN->JUN, an autoregulation artifact of KnockTF's own-gene signal)
    gt_pos = gt[gt.evidence.isin(TIER_EVIDENCE[args.tier])]
    positive_pairs = set(zip(gt_pos.TF, gt_pos.target))
    # ChIP-Atlas returns a genome-wide binding score for every TF that has any data at all, so once a TF has
    # ChIP-Atlas coverage every (TF, gene) pair is "testable" -- absence from gt just means below-threshold (a
    # true negative), not untested. KnockTF's 'kd_only' tier, in contrast, only covers genes actually measured
    # in a melanoma knockdown experiment, so for the 'any'/'both' tiers keep the assay-tested-pair restriction.
    tfs_with_chip = set(gt[gt.chip_tier != "none"].TF)
    if args.tier == "chip":
        testable_pairs = None  # universe = every (TF, gene) with TF in tfs_with_chip; enforced via MELANOMA_TFS mask below
    else:
        testable_pairs = set(zip(gt.TF, gt.target))
    print(f"Ground-truth tier: '{args.tier}' -> {len(positive_pairs)} positive edges genome-wide "
          f"(self-loops removed)" + (f", {len(testable_pairs)} testable pairs" if testable_pairs is not None else ""))

    rows = []
    top_true_hits = []
    for gs in GENE_SETS:
        p = pd.read_csv(f"{SCORED}/twinscore_supplement_fm06_{gs}_absplit_gated_bootstrap_pair_terms.csv")
        dg = json.load(open(f"{SCORED}/run_params_fm06_{gs}_absplit_gated_bootstrap.json"))["diagnostics"]
        pairs = list(zip(p.gene_1, p.gene_2))
        if testable_pairs is None:
            mask = np.array([(a in MELANOMA_TFS) and (a in tfs_with_chip) for a, b in pairs])
        else:
            mask = np.array([(a in MELANOMA_TFS) and ((a, b) in testable_pairs) for a, b in pairs])
        n_universe = int(mask.sum())
        if n_universe < 5 or mask.sum() == 0:
            print(f"{gs}: only {n_universe} testable pairs in melanoma-GT universe, skipping")
            continue
        y = np.array([1 if (a, b) in positive_pairs else 0 for a, b in pairs])
        if y[mask].sum() == 0:
            print(f"{gs}: {n_universe} testable pairs but 0 positives in melanoma-GT universe, skipping")
            continue
        comp = competitor_scores(gs, p)
        yy = y[mask]
        res = {"TwinFER": metrics(p.TwinScore.to_numpy()[mask], yy)}
        for m in COMPETITORS:
            res[m] = metrics(comp[m][mask], yy)
        best = max(COMPETITORS, key=lambda m: (res[m]["AUPRC_x"] if np.isfinite(res[m]["AUPRC_x"]) else -np.inf))
        r = dict(gene_set=gs, universe_pairs=n_universe, n_true=int(yy.sum()), random=round(float(yy.mean()), 4),
                 n_genes=dg["n_genes"],
                 TwinFER_AUPRC=round(res["TwinFER"]["AUPRC"], 4), TwinFER_AUPRC_x=round(res["TwinFER"]["AUPRC_x"], 3),
                 TwinFER_TP_at_k=res["TwinFER"]["TP_at_k"],
                 best_competitor=best, best_AUPRC=round(res[best]["AUPRC"], 4), best_AUPRC_x=round(res[best]["AUPRC_x"], 3),
                 best_TP_at_k=res[best]["TP_at_k"],
                 TwinFER_minus_best_AUPRC_x=round(res["TwinFER"]["AUPRC_x"] - res[best]["AUPRC_x"], 3))
        for m in COMPETITORS:
            r[f"{m}_AUPRC"] = round(res[m]["AUPRC"], 4)
            r[f"{m}_AUPRC_x"] = round(res[m]["AUPRC_x"], 3)
            r[f"{m}_TP_at_k"] = res[m]["TP_at_k"]
        rows.append(r)

        # every true edge in this universe, with its rank under TwinFER and the best competitor (rank 1 = top score)
        q = p[mask].reset_index(drop=True).copy()
        q["is_true"] = yy.astype(bool)
        q["rank_TwinFER"] = pd.Series(q.TwinScore).rank(ascending=False, method="min").astype(int)
        q["rank_" + best] = pd.Series(comp[best][mask]).rank(ascending=False, method="min").astype(int)
        hits = q[q.is_true][["gene_1", "gene_2", "TwinScore", "rank_TwinFER", "rank_" + best]].sort_values("rank_TwinFER")
        hits.insert(0, "gene_set", gs)
        top_true_hits.append(hits)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(f"{OUT}/summary_melanoma_gt.csv", index=False)
    print(f"\n=== TwinFER vs competitors, melanoma ground truth (tier='{args.tier}') ===")
    print(out_df.to_string(index=False))
    if len(out_df):
        print(f"\nMean AUPRC-x over {len(out_df)} evaluable sets: TwinFER {out_df.TwinFER_AUPRC_x.mean():.3f}, "
              f"best-per-set competitor {out_df.best_AUPRC_x.mean():.3f}")

    if top_true_hits:
        hits_df = pd.concat(top_true_hits, ignore_index=True)
        hits_df.to_csv(f"{OUT}/top_true_hits_melanoma_gt_{args.tier}.csv", index=False)
        print(f"\n=== every true edge in each universe, rank under TwinFER (out of universe_pairs; rank 1 = top score) ===")
        print(hits_df.to_string(index=False))

    # ground-truth vs ground-truth: melanoma GT (both-evidence) vs CollecTRI, restricted to the same 10 TFs
    ct = pd.read_csv(COLLECTRI, sep="\t")
    ct = ct[ct.source_genesymbol != ct.target_genesymbol]
    ct_pairs = set(zip(ct.source_genesymbol, ct.target_genesymbol))
    ct_tf_pairs = {(a, b) for a, b in ct_pairs if a in MELANOMA_TFS}
    print(f"\n=== Ground truth comparison, restricted to the {len(MELANOMA_TFS)} melanoma TFs ===")
    print(f"CollecTRI edges from these TFs (any target, genome-wide): {len(ct_tf_pairs)}")
    print(f"Melanoma GT (tier='{args.tier}') edges from these TFs (genome-wide): {len(positive_pairs)}")
    print(f"Overlap: {len(positive_pairs & ct_tf_pairs)}  "
          f"melanoma-GT-only: {len(positive_pairs - ct_tf_pairs)}  "
          f"CollecTRI-only: {len(ct_tf_pairs - positive_pairs)}")
    print(f"Jaccard: {len(positive_pairs & ct_tf_pairs) / max(1, len(positive_pairs | ct_tf_pairs)):.3f}")


if __name__ == "__main__":
    main()
