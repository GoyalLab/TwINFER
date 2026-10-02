#!/usr/bin/env python3
"""FM06 final tables: TwinFER (gated + bootstrap TwinScore_supplement, A/B split) vs PIDC and the other competitors.

Reads
  <BASE>/analysis_data/fm06/data/twinscore_supp_gated_bootstrap/twinscore_supplement_fm06_<gs>_absplit_gated_bootstrap_{pair,gene}_terms.csv
  <BASE>/analysis_data/fm06/data/twinscore_supp_gated_bootstrap/run_params_fm06_<gs>_absplit_gated_bootstrap.json
  <BASE>/analysis_data/fm06/data/networks/{pidc,ppcor,rho,genie3,grnboost2}_<gs>_allgenes.csv
  <BASE>/analysis_data/paper_analysis/larry_hematopoiesis_validation/resources/collectri_human.tsv
Writes to <BASE>/analysis_data/fatemap_comparison/data/final_fm06/
  summary_allpairs.csv          per gene set, universe = all ordered pairs of the panel
  summary_collectri_sources.csv per gene set, universe = pairs whose SOURCE gene is a CollecTRI source (target = any panel gene)
  overlap_<gs>_<universe>.csv   true edges in the top-k (k = n_true) of TwinFER and/or PIDC, with both ranks and the group
  ablation_correlation_high.csv single-term-alone and leave-one-out ablation of the final score on correlation_high
  effect_of_removing_phi.csv    per gene set and universe: full score vs the score without phi (s(PAIR)+direction) vs PIDC
  no_phi_ranks_correlation_high_<universe>.csv  ranks of every pair (full / no phi / PIDC) for correlation_high

Conventions
  * positives = CollecTRI edges (self-loops removed) among the panel genes, direction source -> target (column collectri_edge).
  * AUPRC = area under the precision-recall curve (sklearn precision_recall_curve + auc); AUPRC-x = AUPRC / random, where
    random = fraction of positives in the universe; TP@k = true edges among the top k = n_true scores (ties broken by row order).
  * competitor pairs missing from a competitor file (GENIE3 / GRNBoost2 drop zero importances) get importance 0.
  * best competitor = the competitor with the highest AUPRC-x in that universe.
  * scores are computed once on the full panel; the CollecTRI-source universe only filters the pairs afterwards
    (no re-standardization); ranks in the overlap files are within the universe.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

BASE = f'{TWINFER_PROJECT_ROOT}'
DATA = f"{BASE}/analysis_data/fm06/data"
SCORED = f"{DATA}/twinscore_supp_gated_bootstrap"
NETS = f"{DATA}/networks"
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] COLLECTRI = f"{BASE}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation/resources/collectri_human.tsv"
COLLECTRI = f"{BASE}/analysis_data/paper_analysis/larry_hematopoiesis_validation/resources/collectri_human.tsv"
OUT = f"{BASE}/analysis_data/fatemap_comparison/data/final_fm06"
GENE_SETS = ["correlation_high", "correlation_mid", "correlation_low", "detection_high", "detection_mid",
             "detection_low", "variability_high", "variability_mid", "variability_low"]
COMPETITORS = ["pidc", "ppcor", "rho", "genie3", "grnboost2"]


def metrics(score, y):
    score = np.nan_to_num(np.asarray(score, float), nan=-1e9)
    prec, rec, _ = precision_recall_curve(y, score)
    a = auc(rec, prec)
    base = float(np.mean(y))
    k = int(np.sum(y))
    tp = int(y[np.argsort(-score, kind="stable")[:k]].sum())
    return dict(AUPRC=a, random=base, AUPRC_x=a / base, AUROC=roc_auc_score(y, score), k=k, TP_at_k=tp,
                precision_at_k=tp / k, precision_x=(tp / k) / base)


def s(v):
    v = np.asarray(v, float)
    f = np.isfinite(v)
    o = np.zeros(len(v))
    if f.sum() > 1:
        o[f] = (v[f] - v[f].mean()) / max(v[f].std(ddof=1), 1e-12)
    o[~f] = (o[f].min() - 1.0) if f.any() else 0.0
    return o


def competitor_scores(gs, p):
    out = {}
    for m in COMPETITORS:
        c = pd.read_csv(f"{NETS}/{m}_{gs}_allgenes.csv")
        d = {(a, b): v for a, b, v in zip(c.TF, c.target, c.importance)}
        out[m] = np.array([d.get((a, b), 0.0) for a, b in zip(p.gene_1, p.gene_2)])
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    ct = pd.read_csv(COLLECTRI, sep="\t")
    ct = ct[ct.source_genesymbol != ct.target_genesymbol]
    sources = set(ct.source_genesymbol)
    rows = {"allpairs": [], "collectri_sources": []}
    for gs in GENE_SETS:
        p = pd.read_csv(f"{SCORED}/twinscore_supplement_fm06_{gs}_absplit_gated_bootstrap_pair_terms.csv")
        dg = json.load(open(f"{SCORED}/run_params_fm06_{gs}_absplit_gated_bootstrap.json"))["diagnostics"]
        p["pair"] = p.gene_1 + "->" + p.gene_2
        comp = competitor_scores(gs, p)
        y = p.collectri_edge.to_numpy()
        universes = {"allpairs": np.ones(len(p), bool), "collectri_sources": p.gene_1.isin(sources).to_numpy()}
        for uname, mask in universes.items():
            yy = y[mask]
            res = {"TwinFER": metrics(p.TwinScore.to_numpy()[mask], yy)}
            for m in COMPETITORS:
                res[m] = metrics(comp[m][mask], yy)
            best = max(COMPETITORS, key=lambda m: res[m]["AUPRC_x"])
            r = dict(gene_set=gs, universe_pairs=int(mask.sum()), n_true=int(yy.sum()), random=round(float(yy.mean()), 4),
                     n_genes=dg["n_genes"], genes_passing_gate=dg["n_genes"] - dg["n_gated"], w=round(dg["w"], 4),
                     TwinFER_AUPRC=round(res["TwinFER"]["AUPRC"], 4), TwinFER_AUPRC_x=round(res["TwinFER"]["AUPRC_x"], 3),
                     TwinFER_AUROC=round(res["TwinFER"]["AUROC"], 3), TwinFER_TP_at_k=res["TwinFER"]["TP_at_k"],
                     TwinFER_precision_at_k=round(res["TwinFER"]["precision_at_k"], 3),
                     best_competitor=best, best_AUPRC=round(res[best]["AUPRC"], 4), best_AUPRC_x=round(res[best]["AUPRC_x"], 3),
                     best_TP_at_k=res[best]["TP_at_k"],
                     TwinFER_minus_best_AUPRC_x=round(res["TwinFER"]["AUPRC_x"] - res[best]["AUPRC_x"], 3))
            for m in COMPETITORS:
                r[f"{m}_AUPRC"] = round(res[m]["AUPRC"], 4)
                r[f"{m}_AUPRC_x"] = round(res[m]["AUPRC_x"], 3)
                r[f"{m}_TP_at_k"] = res[m]["TP_at_k"]
            rows[uname].append(r)
            # overlap of true edges in the top-k, TwinFER vs PIDC
            q = p[mask].copy()
            q["PIDC"] = comp["pidc"][mask]
            q["rank_TwinFER"] = q.TwinScore.rank(ascending=False, method="min").astype(int)
            q["rank_PIDC"] = q.PIDC.rank(ascending=False, method="min").astype(int)
            k = int(q.collectri_edge.sum())
            inT, inP, T = q.rank_TwinFER <= k, q.rank_PIDC <= k, q.collectri_edge == 1
            grp = np.where(inT & inP, "common", np.where(inT, "TwinFER_only", "PIDC_only"))
            o = q[T & (inT | inP)][["pair", "rank_TwinFER", "rank_PIDC"]].copy()
            o["group"] = grp[(T & (inT | inP)).to_numpy()]
            o["k"] = k
            o.sort_values(["group", "rank_TwinFER"]).to_csv(f"{OUT}/overlap_{gs}_{uname}.csv", index=False)
    for uname, r in rows.items():
        pd.DataFrame(r).to_csv(f"{OUT}/summary_{uname}.csv", index=False)

    # ablation, correlation_high (components of the final score: w*s(phi) + s(PAIR) + direction)
    gs = "correlation_high"
    p = pd.read_csv(f"{SCORED}/twinscore_supplement_fm06_{gs}_absplit_gated_bootstrap_pair_terms.csv")
    dg = json.load(open(f"{SCORED}/run_params_fm06_{gs}_absplit_gated_bootstrap.json"))["diagnostics"]
    w, gg, gv = dg["w"], p.gate_g.iloc[0], p.gate_v.iloc[0]
    sD, sR, sW, sph, dr = s(p.D), s(p.R), s(p.Wz), s(p.phi_x), p.direction_term.to_numpy()

    def full(phi=1, D=1, R=1, W=1, d=1):
        return w * phi * sph + s(D * sD + R * gg * sR + W * gg * gv * sW) + d * dr

    assert np.abs(full() - p.TwinScore).max() < 1e-8, "score reconstruction failed"
    y = p.collectri_edge.to_numpy()
    src = p.gene_1.isin(sources).to_numpy()
    items = [("phi alone", sph), ("D (PIDC) alone", sD), ("R alone", sR), ("Wz alone", sW), ("direction alone", dr),
             ("PAIR = D+R+Wz", s(sD + gg * sR + gg * gv * sW)),
             ("FULL", full()), ("FULL - phi", full(phi=0)), ("FULL - D", full(D=0)), ("FULL - R", full(R=0)),
             ("FULL - Wz", full(W=0)), ("FULL - direction", full(d=0))]
    ab = []
    for name, sc in items:
        for uname, mask in (("allpairs", np.ones(len(p), bool)), ("collectri_sources", src)):
            m_ = metrics(np.asarray(sc)[mask], y[mask])
            ab.append(dict(term=name, universe=uname, AUPRC=round(m_["AUPRC"], 4), AUPRC_x=round(m_["AUPRC_x"], 3),
                           TP_at_k=m_["TP_at_k"], k=m_["k"]))
    pd.DataFrame(ab).to_csv(f"{OUT}/ablation_correlation_high.csv", index=False)
    # effect of removing phi: score without the phi term = s(PAIR) + direction_term (i.e. w set to 0), same pairs and universes
    from scipy.stats import spearmanr
    eff = []
    for gs in GENE_SETS:
        p = pd.read_csv(f"{SCORED}/twinscore_supplement_fm06_{gs}_absplit_gated_bootstrap_pair_terms.csv")
        dg = json.load(open(f"{SCORED}/run_params_fm06_{gs}_absplit_gated_bootstrap.json"))["diagnostics"]
        p["pair"] = p.gene_1 + "->" + p.gene_2
        p["NoPhi"] = s(p.PAIR) + p.direction_term.to_numpy()
        p["PIDC"] = competitor_scores(gs, p)["pidc"]
        y = p.collectri_edge.to_numpy()
        for uname, mask in (("allpairs", np.ones(len(p), bool)), ("collectri_sources", p.gene_1.isin(sources).to_numpy())):
            mf, mn, mp = (metrics(p[c].to_numpy()[mask], y[mask]) for c in ("TwinScore", "NoPhi", "PIDC"))
            q = p[mask].copy()
            kk = int(q.collectri_edge.sum())
            top = lambda c: set(q.nlargest(kk, c).pair)
            eff.append(dict(gene_set=gs, universe=uname, w=round(dg["w"], 4), n_true=kk,
                            full_AUPRC=round(mf["AUPRC"], 4), full_AUPRC_x=round(mf["AUPRC_x"], 3), full_TP_at_k=mf["TP_at_k"],
                            no_phi_AUPRC=round(mn["AUPRC"], 4), no_phi_AUPRC_x=round(mn["AUPRC_x"], 3), no_phi_TP_at_k=mn["TP_at_k"],
                            pidc_AUPRC=round(mp["AUPRC"], 4), pidc_AUPRC_x=round(mp["AUPRC_x"], 3), pidc_TP_at_k=mp["TP_at_k"],
                            delta_AUPRC_x_full_minus_no_phi=round(mf["AUPRC_x"] - mn["AUPRC_x"], 3),
                            spearman_full_vs_no_phi=round(spearmanr(q.TwinScore, q.NoPhi)[0], 3),
                            spearman_no_phi_vs_pidc=round(spearmanr(q.NoPhi, q.PIDC)[0], 3),
                            topk_overlap_full_vs_no_phi=len(top("TwinScore") & top("NoPhi")),
                            topk_overlap_no_phi_vs_pidc=len(top("NoPhi") & top("PIDC"))))
            if gs == "correlation_high":
                q["rank_full"] = q.TwinScore.rank(ascending=False, method="min").astype(int)
                q["rank_no_phi"] = q.NoPhi.rank(ascending=False, method="min").astype(int)
                q["rank_PIDC"] = q.PIDC.rank(ascending=False, method="min").astype(int)
                q[["pair", "collectri_edge", "rank_full", "rank_no_phi", "rank_PIDC"]].to_csv(
                    f"{OUT}/no_phi_ranks_correlation_high_{uname}.csv", index=False)
    pd.DataFrame(eff).to_csv(f"{OUT}/effect_of_removing_phi.csv", index=False)
    print(f"wrote tables to {OUT}")


if __name__ == "__main__":
    main()
