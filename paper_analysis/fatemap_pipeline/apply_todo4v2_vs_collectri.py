#!/usr/bin/env python3
"""Ports larry_hematopoiesis_validation/apply_todo4v2_allpairs_with_competitors.py's
TODO4v2 formula and CollecTRI-based validation to FM06/FM08.

TODO4v2 score (same structure as the LARRY reference, gate: max(z_abs_t1,z_abs_t2) > 2.576
AND z_reg_gated > 2.326):
    z_abs_rho_t1 + Cc + divp + new_gamma + s(z_flux) + hinge_stable + hinge_het + s(|z_dagger|)

One deliberate substitution: the reference script's z_dagger is
`z_signed(rho_cross_xy, DEFAULT_SD["cross"])`, an analytic formula calibrated to LARRY's own
null standard deviation. That constant is LARRY-specific (depends on LARRY's clone-size/
effective-sample-size structure) and would be silently wrong for FM06/FM08's very different
clone sizes, so z_dagger here is instead the empirical null z-score infer_with_twinfer already
computed per dataset (run_infer_fatemap.py saves it to z_dagger_{label}_{geneset}.json).
Everything else in the formula is unchanged.

Validation: same as the reference -- human CollecTRI (source_genesymbol, target_genesymbol)
pairs are the truth set; AUPRC vs. random-baseline fold enrichment (auprc_x) and top-k
precision (k = number of true edges in the candidate universe), same metric as
apply_todo4v2_allpairs_with_competitors.py's full_report().
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

COLLECTRI_PATH = (
    # [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation/resources/collectri_human.tsv'
    f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/larry_hematopoiesis_validation/resources/collectri_human.tsv'
)
GENE_SETS = [
    "variability_high", "variability_mid", "variability_low",
    "detection_high", "detection_mid", "detection_low",
    "correlation_high", "correlation_mid", "correlation_low",
]
DATASETS = ["FM06", "FM08"]

Z_HET_THR_NEW = -2.326
Z_TWO_SIDED = 2.576  # norm.ppf(1 - 0.01/2)
Z_ONE_SIDED = 2.326  # norm.ppf(1 - 0.01)


def s(v):
    v = np.asarray(v, float)
    f = np.isfinite(v)
    o = np.zeros(len(v))
    if f.sum() > 1:
        o[f] = (v[f] - v[f].mean()) / max(v[f].std(ddof=1), 1e-12)
    o[~f] = (o[f].min() - 1.0) if f.any() else 0.0
    return o


def full_report(sc, y):
    sc = np.asarray(sc, float)
    sc = np.nan_to_num(sc, nan=np.nanmin(sc) - 1.0 if np.isfinite(sc).any() else 0.0)
    prec, rec, _ = precision_recall_curve(y, sc)
    auprc = auc(rec, prec)
    rand = float(y.mean())
    k = int(y.sum())
    order = np.argsort(-sc, kind="stable")
    top_k = order[:k]
    tp = int((y[top_k] == 1).sum())
    # top-k precision, k = n_true: by construction predicted-positives == n_true here, so
    # precision == recall == F1 at this operating point (not three independent numbers).
    topk_prec = tp / k if k else float("nan")
    f1_at_topk = topk_prec
    # best F1 achievable anywhere along the full precision-recall curve (independent of the
    # top-k=n_true operating point above) -- the usual threshold-free F1 summary.
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_curve = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
    max_f1 = float(np.nanmax(f1_curve)) if len(f1_curve) else float("nan")
    return dict(auprc=auprc, auprc_random=rand, auprc_x=auprc / rand if rand > 0 else float("nan"),
                topk_precision=topk_prec, f1_at_topk=f1_at_topk, max_f1=max_f1, tp=tp, k=k)


def lookup(d, a, b, default=np.nan):
    if f"{a}__{b}" in d:
        return d[f"{a}__{b}"]
    if f"{b}__{a}" in d:
        return d[f"{b}__{a}"]
    return default


def todo4v2_score(dd, z_reg_map, z_dagger_map):
    z_abs_t1 = dd.z_abs_rho_t1.to_numpy()
    z_abs_t2 = dd.z_abs_rho_t2.to_numpy()
    z_abs_change = dd.z_abs_rho_change.to_numpy()
    z_het = dd.z_het.to_numpy()
    z_div = dd.z_div.to_numpy()
    z_gamma = dd.z_gamma.to_numpy()
    z_stable = -dd.z_rho_change.to_numpy()

    z_reg = np.array([lookup(z_reg_map, a, b) for a, b in zip(dd.gene_1, dd.gene_2)])
    # z_dagger: empirical null z-score of rho_cross(x->y) for THIS directed pair (see module
    # docstring -- replaces the reference script's LARRY-calibrated analytic version).
    z_dagger = np.array([lookup(z_dagger_map, a, b) for a, b in zip(dd.gene_1, dd.gene_2)])

    # z_flux: per-gene net outward-vs-inward directional signal, built from |z_dagger| (magnitude
    # of cross-time asymmetry, not its sign -- same "abs-REG" construction the LARRY reference
    # settled on for TODO4v2's flux term).
    tmp = dd.assign(_zd=np.abs(z_dagger))
    regd = tmp.groupby("gene_1")["_zd"].mean() - tmp.groupby("gene_2")["_zd"].mean()
    REG = regd.to_dict()
    z_flux = np.array([REG.get(a, 0.0) - REG.get(b, 0.0) for a, b in zip(dd.gene_1, dd.gene_2)])

    gate = (np.maximum(z_abs_t1, z_abs_t2) > Z_TWO_SIDED) & (np.nan_to_num(z_reg, nan=-np.inf) > Z_ONE_SIDED)

    Cc = -s(z_abs_change)
    divp = -np.abs(z_div) * (z_het < Z_HET_THR_NEW).astype(float)
    s_zg = s(z_gamma)
    new_gamma = (np.abs(s_zg) >= 1.0).astype(float) * s_zg
    hinge_stable = -np.where(z_stable > Z_ONE_SIDED, z_stable, 0.0)
    hinge_het = np.where(z_het < -Z_ONE_SIDED, z_het, 0.0)
    s_zdagger = s(np.abs(z_dagger))

    score = z_abs_t1 + Cc + divp + new_gamma + s(z_flux) + hinge_stable + hinge_het + s_zdagger
    score = np.where(gate, score, -np.inf)
    return score, int(gate.sum())


def main():
    ct = pd.read_csv(COLLECTRI_PATH, sep="\t")
    ct = ct[ct.source_genesymbol != ct.target_genesymbol]
    CE = set(zip(ct.source_genesymbol, ct.target_genesymbol))
    print(f"human CollecTRI: {len(CE):,} directed edges", flush=True)

    rows = []
    for dataset in DATASETS:
        label = dataset.lower()
        data_dir = f"{TWINFER_PROJECT_ROOT}/analysis_data/{label}/data"
        for gs in GENE_SETS:
            re_path = os.path.join(data_dir, f"ranked_edges_{label}_{gs}.csv")
            zreg_path = os.path.join(data_dir, f"z_reg_gated_{label}_{gs}.json")
            zdag_path = os.path.join(data_dir, f"z_dagger_{label}_{gs}.json")
            if not (os.path.exists(re_path) and os.path.exists(zreg_path) and os.path.exists(zdag_path)):
                print(f"[{dataset}/{gs}] SKIP -- outputs not found yet", flush=True)
                continue

            dd = pd.read_csv(re_path)
            dd = dd[dd.gene_1 != dd.gene_2].reset_index(drop=True)
            if dd.empty:
                print(f"[{dataset}/{gs}] SKIP -- no directed edge candidates", flush=True)
                continue
            z_reg_map = json.load(open(zreg_path))
            z_dagger_map = json.load(open(zdag_path))

            U = list(zip(dd.gene_1, dd.gene_2))
            y = np.array([1 if p in CE else 0 for p in U])
            if y.sum() < 3:
                print(f"[{dataset}/{gs}] SKIP -- only {int(y.sum())} true CollecTRI edges "
                      f"in {len(U)} candidates", flush=True)
                continue

            score, n_pass_gate = todo4v2_score(dd, z_reg_map, z_dagger_map)
            m = full_report(score, y)

            # twinScore (the current, already-computed default formula) for reference.
            base_m = full_report(dd["twinScore"].to_numpy(), y) if "twinScore" in dd.columns else None

            rows.append(dict(
                dataset=dataset, gene_set=gs, n_pairs=len(U), n_true=int(y.sum()),
                n_pass_gate=n_pass_gate,
                todo4v2_auprc=m["auprc"], todo4v2_auprc_random=m["auprc_random"],
                todo4v2_auprc_x=m["auprc_x"], todo4v2_topk_precision=m["topk_precision"],
                todo4v2_f1_at_topk=m["f1_at_topk"], todo4v2_max_f1=m["max_f1"],
                todo4v2_topk_hits=f"{m['tp']}/{m['k']}",
                twinscore_auprc=base_m["auprc"] if base_m else np.nan,
                twinscore_auprc_random=base_m["auprc_random"] if base_m else np.nan,
                twinscore_auprc_x=base_m["auprc_x"] if base_m else np.nan,
                twinscore_topk_precision=base_m["topk_precision"] if base_m else np.nan,
                twinscore_f1_at_topk=base_m["f1_at_topk"] if base_m else np.nan,
                twinscore_max_f1=base_m["max_f1"] if base_m else np.nan,
                twinscore_topk_hits=f"{base_m['tp']}/{base_m['k']}" if base_m else "",
            ))
            print(f"[{dataset}/{gs}] n_true={int(y.sum())}/{len(U)}  n_pass_gate={n_pass_gate}  "
                  f"TODO4v2 auprc_x={m['auprc_x']:.3f}x top-k_prec={m['topk_precision']:.3f} "
                  f"max_F1={m['max_f1']:.3f}  |  twinScore auprc_x={base_m['auprc_x']:.3f}x "
                  f"top-k_prec={base_m['topk_precision']:.3f} max_F1={base_m['max_f1']:.3f}"
                  if base_m else "n/a", flush=True)

    df = pd.DataFrame(rows)
    out_csv = f'{TWINFER_PROJECT_ROOT}/analysis_data/fatemap_todo4v2_vs_collectri.csv'
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}", flush=True)
    print("\n" + df.to_string(index=False), flush=True)

    for dataset in DATASETS:
        sub = df[df.dataset == dataset]
        if sub.empty:
            continue
        print(f"\n[{dataset}] mean TODO4v2 auprc_x = {sub.todo4v2_auprc_x.mean():.3f}x  "
              f"mean twinScore auprc_x = {sub.twinscore_auprc_x.mean():.3f}x  "
              f"({len(sub)}/{len(GENE_SETS)} gene sets with >=3 true edges)")


if __name__ == "__main__":
    main()
