"""Which TwinScore terms actually track CollecTRI edges, and what composite works best per gene
set. Uses the analytic twin_score_inputs (resources/analytic_infer/{gs}/) -- no permutations, so
an exhaustive-ish search is cheap.

For each gene set:
  universe   = (gene-set TFs) x (panel genes), self excluded
  y_true     = CollecTRI directed edge?
  candidate terms (each panel-standardized with s(), like calculate_twin_score):
     abs_rho_t1, abs_rho_t2, abs_rho_change, rho_change,
     z_div, z_het, z_d_het, abs_gamma, z_gamma, abs_z_gamma,
     min_abs_rho_cross (INH), rho_cross_xy
  1. term-alone discriminative power: AUPRC / random-AUPRC and EPR for +term and -term.
  2. forward selection of a signed sum of s(term)s maximising mean AUPRC-ratio (5-fold CV over
     the pairs, to avoid overfitting the ~50-200 positives).
Writes term_discriminative_power.csv, best_formula_per_geneset.csv.
"""
import zlib  # [2026-10-01 added]
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

HERE = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation'
from twinfer.utils.paths import get_data_root as _res_gdr  # [2026-10-01 LARRY resources/ moved out of the code tree to analysis_data/paper_analysis/larry_hematopoiesis_validation/resources (user)]
RES_HERE = f"{_res_gdr()}/paper_analysis/larry_hematopoiesis_validation"
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] R = f"{HERE}/resources"
R = f"{RES_HERE}/resources"
OUT = f"{R}/benchmark"
GENE_SETS = ["variability_high", "variability_mid", "variability_low",
             "detection_high", "detection_mid", "detection_low",
             "correlation_high", "correlation_mid", "correlation_low"]
N_RANDOM = 300
SEED = 0

ct = pd.read_csv(f"{R}/collectri_mouse.tsv", sep="\t")
ct = ct[ct.source_genesymbol != ct.target_genesymbol]
CE = set(zip(ct.source_genesymbol, ct.target_genesymbol))
gene_sets = json.load(open(f"{R}/gene_sets.json"))
detail = json.load(open(f"{R}/gene_sets_detail.json"))


def s(v):
    v = np.asarray(v, float); f = np.isfinite(v)
    o = np.zeros(len(v))
    if f.sum() > 1:
        o[f] = (v[f] - v[f].mean()) / max(v[f].std(ddof=1), 1e-12)
    o[~f] = (o[f].min() - 1.0) if f.any() else 0.0
    return o


def terms_for(gs):
    d = pd.read_csv(f"{R}/analytic_infer/{gs}/twin_score_inputs.csv")
    crit, lvl = gs.rsplit("_", 1)
    panel = gene_sets[gs]
    tfs = set(tf for tf, _ in detail[crit][lvl] if tf in panel)
    d = d[d.gene_1.isin(tfs) & (d.gene_1 != d.gene_2)].reset_index(drop=True)
    U = list(zip(d.gene_1, d.gene_2))
    y = np.array([1 if p in CE else 0 for p in U])
    T = pd.DataFrame({
        "abs_rho_t1": d.rho_t1.abs(), "abs_rho_t2": d.rho_t2.abs(),
        "abs_rho_change": d.rho_change.abs(), "rho_change": d.rho_change,
        "z_div": d.z_div, "z_het": d.z_het, "z_d_het": d.z_d_het,
        "abs_gamma": d.gamma.abs(), "z_gamma": d.z_gamma, "abs_z_gamma": d.z_gamma.abs(),
        "min_abs_rho_cross": np.minimum(d.rho_cross_xy.abs(), d.rho_cross_yx.abs()),
        "rho_cross_xy": d.rho_cross_xy.abs(),
    })
    return T.apply(s), y, U


def auprc_ratio(y, score, seed):
    au = average_precision_score(y, score)
    rng = np.random.default_rng(seed)
    au_r = np.mean([average_precision_score(y, rng.permutation(score)) for _ in range(N_RANDOM)])
    return au / au_r if au_r > 0 else np.nan, au


def epr(y, score):
    k = int(y.sum()); n = len(y)
    idx = np.argsort(-score, kind="stable")[:k]
    return (y[idx].sum() / k) / (k / n)


# ---- 1. term-alone discriminative power --------------------------------------------------
disc = []
term_tables = {}
for gs in GENE_SETS:
    T, y, U = terms_for(gs)
    term_tables[gs] = (T, y)
    if y.sum() < 3:
        continue
    for col in T.columns:
        for sgn in (1, -1):
            # [2026-10-01 commented out: hash() of strings is randomised per process (PYTHONHASHSEED) so this seed was not reproducible; zlib.crc32 of the repr is stable] r, au = auprc_ratio(y, sgn * T[col].to_numpy(), SEED + hash((gs, col, sgn)) % 2**31)
            r, au = auprc_ratio(y, sgn * T[col].to_numpy(), SEED + zlib.crc32(repr((gs, col, sgn)).encode()) % 2**31)
            e = epr(y, sgn * T[col].to_numpy())
            mt = T.loc[y == 1, col].mean(); mf = T.loc[y == 0, col].mean()
            disc.append(dict(gene_set=gs, term=col, sign=sgn, auprc=au, auprc_ratio=r, epr=e,
                             mean_true=mt, mean_false=mf))
disc = pd.DataFrame(disc)
disc.to_csv(f"{OUT}/term_discriminative_power.csv", index=False)

# best single-sign orientation per term, averaged over gene sets
best_sign = (disc.loc[disc.groupby(["gene_set", "term"]).auprc_ratio.idxmax()]
             [["gene_set", "term", "sign", "auprc_ratio", "epr"]])
term_summary = (best_sign.groupby("term")
                .agg(auprc_ratio_mean=("auprc_ratio", "mean"),
                     epr_mean=("epr", "mean"),
                     sign_mode=("sign", lambda x: int(x.mode().iloc[0])),
                     n_gs_ratio_gt_1=("auprc_ratio", lambda x: int((x > 1).sum())))
                .sort_values("auprc_ratio_mean", ascending=False))
print("=== per-term discriminative power (best sign per gene set, then averaged) ===")
print(term_summary.round(3).to_string())

# ---- 2. forward selection of a signed composite, per gene set ---------------------------
def cv_ratio(T, y, cols_signs):
    if not cols_signs:
        return -np.inf
    score = np.zeros(len(y))
    for c, sg in cols_signs:
        score = score + sg * T[c].to_numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rs = []
    for _, te in skf.split(score, y):
        yt, st = y[te], score[te]
        if yt.sum() < 1 or yt.sum() == len(yt):
            continue
        au = average_precision_score(yt, st)
        rng = np.random.default_rng(SEED)
        au_r = np.mean([average_precision_score(yt, rng.permutation(st)) for _ in range(80)])
        rs.append(au / au_r if au_r > 0 else np.nan)
    return float(np.nanmean(rs)) if rs else -np.inf


rows_best = []
for gs in GENE_SETS:
    T, y = term_tables[gs]
    if y.sum() < 5:
        rows_best.append(dict(gene_set=gs, formula="(too few curated edges)", cv_auprc_ratio=np.nan))
        continue
    chosen, cur = [], -np.inf
    pool = [(c, sg) for c in T.columns for sg in (1, -1)]
    while True:
        cand = [(cs, cv_ratio(T, y, chosen + [cs])) for cs in pool
                if cs[0] not in {c for c, _ in chosen}]
        cs_best, v_best = max(cand, key=lambda kv: kv[1])
        if v_best <= cur + 0.01:
            break
        chosen.append(cs_best); cur = v_best
    # full-data ratio + EPR for the chosen composite
    score = sum(sg * T[c].to_numpy() for c, sg in chosen)
    r_full, _ = auprc_ratio(y, score, SEED + 12345)
    rows_best.append(dict(
        gene_set=gs,
        formula=" ".join(f"{'+' if sg > 0 else '-'} s({c})" for c, sg in chosen).strip(),
        cv_auprc_ratio=cur, full_auprc_ratio=r_full, full_epr=epr(y, score),
        n_terms=len(chosen)))
best = pd.DataFrame(rows_best)
best.to_csv(f"{OUT}/best_formula_per_geneset.csv", index=False)
print("\n=== best signed composite per gene set (5-fold CV over pairs) ===")
print(best.round(2).to_string(index=False))

# how often each term appears in a best formula, and with which sign
import re
picks = []
for _, r in best.iterrows():
    if not isinstance(r.formula, str) or "too few" in r.formula:
        continue
    for sg, name in re.findall(r"([+-])\s*s\(([a-z_0-9]+)\)", r.formula):
        picks.append((name, +1 if sg == "+" else -1))
pk = pd.DataFrame(picks, columns=["term", "sign"])
print("\n=== term selection frequency across the 9 best formulas ===")
print(pk.groupby("term").agg(times_picked=("sign", "size"),
                             sign=("sign", lambda x: int(x.mode().iloc[0]))).sort_values(
    "times_picked", ascending=False).to_string())
