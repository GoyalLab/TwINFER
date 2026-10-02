"""LARRY equivalent of summary_table_todo4v2.py (same target format: variant, dataset_id,
method, f1_topk, precision_topk, auprc -- natural/GMM dropped per instruction), TODO4v2(nogate)
plus LARRY's 5 available competitor methods (rho/ppcor/pidc/genie3/grnboost2 -- no PEARSON/
SCODE/SCSGL/PPCOR-as-BEELINE here, these are yscher's own precomputed importance files, not a
BEELINE rerun), across all 9 gene sets x 2 twin_defs.

signed_directed IS included: CollecTRI does carry sign (consensus_stimulation/
consensus_inhibition), reused via apply_current_score_signed.py's existing load_signed_true_edges
(stim-only -> +1, inhib-only -> -1, ambiguous/neither dropped from the signed ground-truth
universe). Predicted sign: the 5 LARRY competitor files (rho/ppcor/pidc/genie3/
grnboost2_*_allgenes.csv) are ALL magnitude-only here (verified: every one's importance column is
non-negative, unlike a live BEELINE PEARSON/PPCOR rerun), so there is no per-method native sign to
use -- a single shared sign proxy is needed for all 5. First attempt used TwINFER's own rho_t1
(matching apply_current_score_signed.py's existing convention) with TODO4v2 using its own
sign(rho_cross_xy); TODO4v2's signed retention (auprc_signed/auprc_unsigned) came out at 0.46 vs
the 5 competitors' near-identical 0.67-0.69 -- suspicious, since 5 different methods clustering
that tightly suggested the shared rho_t1 sign proxy (not each method's own ranking) was driving
the retention number. Per user instruction, switched the competitor sign proxy to genuine Pearson
correlation (raw log1p(cp10k) expression, computed fresh here from
resources/twinfer_input_yscher_cp10k{,_annotfilter}/<gene_set>.csv -- NOT TwINFER's rho_t1, which
is a weighted-twin-pair Spearman statistic, not a plain Pearson correlation), and ALSO added
TODO4v2-pearson-sign as a second TODO4v2 row to see whether TODO4v2 improves under the same sign
source the competitors now use.

"dataset_id" here = gene_set (twin_def is a separate column, since unlike the sim benchmarks
LARRY has two twin-pairing definitions per gene_set, not multiple simulation replicates).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from itertools import combinations, permutations
from sklearn.metrics import auc, precision_recall_curve

from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: project root for the clean_data/ repointing, see REPOINT_LOG.tsv]
HERE = os.path.dirname(os.path.abspath(__file__))
from twinfer.utils.paths import get_data_root as _res_gdr  # [2026-10-01 LARRY resources/ moved out of the code tree to analysis_data/paper_analysis/larry_hematopoiesis_validation/resources (user)]
RES_HERE = f"{_res_gdr()}/paper_analysis/larry_hematopoiesis_validation"
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, HERE)
from twinfer.scoring.analytic_zscores import DEFAULT_SD, z_signed
from paper_analysis.larry_hematopoiesis_validation import apply_todo4v2_allpairs_with_competitors as A

R = A.R
GENE_SETS = A.GENE_SETS
METHODS = A.METHODS
TWIN_DEFS = A.TWIN_DEFS


def precision_recall_f1(selected, true_set):
    if not selected:
        return 0.0, 0.0, 0.0
    tp = len(selected & true_set)
    precision = tp / len(selected)
    recall = tp / len(true_set) if true_set else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def top_k_tie_aware(score_map, universe, true_set, k):
    """score_map: {pair: score}, universe: list of all scoreable pairs (any type)."""
    if k == 0 or not universe:
        return set()
    scores = np.array([score_map.get(p, 0.0) for p in universe])
    order = np.argsort(-scores, kind="stable")
    maxk = min(len(universe), k)
    boundary = scores[order[maxk - 1]]
    nonzero = scores[scores > 0]
    non_zero_min = nonzero.min() if len(nonzero) else 0.0
    best_val = max(non_zero_min, boundary)
    selected = {universe[i] for i in range(len(universe)) if scores[i] >= best_val}
    return selected


def auprc_from_scores(score_map, universe, true_set):
    scores = np.array([score_map.get(p, 0.0) for p in universe])
    labels = np.array([1 if p in true_set else 0 for p in universe])
    if labels.sum() == 0:
        return float("nan")
    precision, recall, _ = precision_recall_curve(labels, scores)
    return float(auc(recall, precision))


def todo4v2_score_nogate(dd, U, z_reg_map):
    """Same terms as A.todo4v2_score, but WITHOUT the existence/regulation gate -- established
    this session (todo4v2_sim_scoring.py's three-variant comparison) as decisively the best
    variant on every simulated benchmark (1.4-1.7x better than the gated 'old'/'new' variants),
    because the gate discards most pairs on small networks. apply_todo4v2_allpairs_with_
    competitors.py only implements the gated version, so this reimplements the same math minus
    the final `np.where(gate, score, -np.inf)` line."""
    z_flux = A._reg_and_flux(dd)
    z_abs_t1 = dd.z_abs_rho_t1.to_numpy()
    z_abs_change = dd.z_abs_rho_change.to_numpy()
    z_het = dd.z_het.to_numpy(); z_div = dd.z_div.to_numpy()
    z_gamma = dd.z_gamma.to_numpy()
    z_stable = -dd.z_rho_change.to_numpy()
    z_dagger = z_signed(dd.rho_cross_xy.to_numpy(), DEFAULT_SD["cross"])

    Cc = -A.s(z_abs_change)
    divp = -np.abs(z_div) * (z_het < A.Z_HET_THR_NEW).astype(float)
    s_zg = A.s(z_gamma)
    new_gamma = (np.abs(s_zg) >= 1.0).astype(float) * s_zg
    hinge_stable = -np.where(z_stable > A.Z_ONE_SIDED, z_stable, 0.0)
    hinge_het = np.where(z_het < -A.Z_ONE_SIDED, z_het, 0.0)
    s_zdagger = A.s(np.abs(z_dagger))

    return z_abs_t1 + Cc + divp + new_gamma + A.s(z_flux) + hinge_stable + hinge_het + s_zdagger


_PEARSON_CACHE = {}


def pearson_sign_map(twin_def, gs, genes):
    key = (twin_def, gs)
    if key in _PEARSON_CACHE:
        return _PEARSON_CACHE[key]
    # [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] expr_dir = f"{HERE}/resources/twinfer_input_yscher_cp10k" + ("_annotfilter" if twin_def == "annotfilter" else "")
    expr_dir = f"{RES_HERE}/resources/twinfer_input_yscher_cp10k" + ("_annotfilter" if twin_def == "annotfilter" else "")
    path = f"{expr_dir}/{gs}.csv"
    cols = [f"{g}_mRNA" for g in genes]
    df = pd.read_csv(path, usecols=cols)
    corr = df.corr(method="pearson")
    sign_map = {}
    for a in genes:
        for b in genes:
            if a == b:
                continue
            v = corr.loc[f"{a}_mRNA", f"{b}_mRNA"]
            sign_map[(a, b)] = np.sign(v) if np.isfinite(v) else 1.0
    _PEARSON_CACHE[key] = sign_map
    return sign_map


def auprc_signed_from_scores(mag_map, sign_map, universe, true_signed):
    signed_positions = [(a, b, s) for (a, b) in universe for s in (1, -1)]
    labels = np.array([1 if p in true_signed else 0 for p in signed_positions])
    if labels.sum() == 0:
        return float("nan"), None
    scores = np.array([mag_map.get((a, b), 0.0) if sign_map.get((a, b)) == s else 0.0
                        for a, b, s in signed_positions])
    precision, recall, _ = precision_recall_curve(labels, scores)
    return float(auc(recall, precision)), (signed_positions, scores, labels)


def top_k_signed(signed_positions, scores, labels, k):
    if k == 0:
        return 0.0, 0.0
    order = np.argsort(-scores, kind="stable")
    maxk = min(len(scores), k)
    boundary = scores[order[maxk - 1]]
    nonzero = scores[scores > 0]
    non_zero_min = nonzero.min() if len(nonzero) else 0.0
    best_val = max(non_zero_min, boundary)
    selected = scores >= best_val
    tp = int((labels[selected] == 1).sum())
    n_sel = int(selected.sum())
    precision = tp / n_sel if n_sel else 0.0
    recall = tp / labels.sum() if labels.sum() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, f1


def score_one_panel(twin_def, gs, CE, true_signed):
    cfg = TWIN_DEFS[twin_def]
    dpath = f"{R}/{cfg['analytic_dir']}/{gs}/twin_score_inputs.csv"
    zpath = f"{R}/infer_results/{gs}_t2_t4_allpairs{cfg['infer_suffix']}/z_scores_by_step.json"
    if not os.path.exists(dpath) or not os.path.exists(zpath):
        return []
    dd = pd.read_csv(dpath)
    dd = dd[dd.gene_1 != dd.gene_2].reset_index(drop=True)
    z_reg_map = A.load_zreg_map(gs, cfg["infer_suffix"])
    U = list(zip(dd.gene_1, dd.gene_2))
    genes = sorted(set(dd.gene_1) | set(dd.gene_2))
    U_full = [(a, b) for a in genes for b in genes if a != b]
    true_directed = CE & set(U_full)
    true_undirected = {frozenset(p) for p in true_directed}
    possible_und = {frozenset(p) for p in combinations(genes, 2)}
    U_full_und = list(possible_und)

    rows = []

    # --- TODO4v2(nogate) ---
    sc = todo4v2_score_nogate(dd, U, z_reg_map)
    finite = np.isfinite(sc)
    floor = (np.min(sc[finite]) - 1.0) if finite.any() else 0.0
    sc = np.where(finite, sc, floor)  # NaN terms (e.g. z_div undefined for a pair) -> treat as no-evidence, not propagate
    sc = sc - np.min(sc)  # shift to non-negative so |EdgeWeight| ranks the same as descending sc
    score_map_shifted = dict(zip(U, sc))
    auprc = auprc_from_scores(score_map_shifted, U_full, true_directed)
    sel = top_k_tie_aware(score_map_shifted, U_full, true_directed, len(true_directed))
    p_topk, _, f1_topk = precision_recall_f1(sel, true_directed)

    und_score = {}
    for (a, b), v in score_map_shifted.items():
        key = frozenset((a, b))
        und_score[key] = max(und_score.get(key, -np.inf), v)
    auprc_u = auprc_from_scores(und_score, U_full_und, true_undirected)
    sel_u = top_k_tie_aware(und_score, U_full_und, true_undirected, len(true_undirected))
    p_topk_u, _, f1_topk_u = precision_recall_f1(sel_u, true_undirected)

    rows.append(dict(twin_def=twin_def, dataset_id=gs, method="TODO4v2", variant="directed_unsigned",
                      auprc=auprc, precision_topk=p_topk, f1_topk=f1_topk))
    rows.append(dict(twin_def=twin_def, dataset_id=gs, method="TODO4v2", variant="undirected",
                      auprc=auprc_u, precision_topk=p_topk_u, f1_topk=f1_topk_u))

    n_true_signed_pairs = sum(1 for p in U_full for s in (1, -1) if (p[0], p[1], s) in true_signed)
    pearson_sign = pearson_sign_map(twin_def, gs, genes)

    todo_sign_map = dict(zip(zip(dd.gene_1, dd.gene_2), np.sign(dd.rho_cross_xy.to_numpy())))
    auprc_s, packed = auprc_signed_from_scores(score_map_shifted, todo_sign_map, U_full, true_signed)
    p_topk_s, f1_topk_s = top_k_signed(*packed, k=n_true_signed_pairs) if packed is not None else (float("nan"), float("nan"))
    rows.append(dict(twin_def=twin_def, dataset_id=gs, method="TODO4v2", variant="signed_directed",
                      auprc=auprc_s, precision_topk=p_topk_s, f1_topk=f1_topk_s))

    todo_pearson_sign_map = {p: pearson_sign.get(p, 1.0) for p in U}
    auprc_sp, packed_p = auprc_signed_from_scores(score_map_shifted, todo_pearson_sign_map, U_full, true_signed)
    p_topk_sp, f1_topk_sp = top_k_signed(*packed_p, k=n_true_signed_pairs) if packed_p is not None else (float("nan"), float("nan"))
    rows.append(dict(twin_def=twin_def, dataset_id=gs, method="TODO4V2_PEARSONSIGN", variant="signed_directed",
                      auprc=auprc_sp, precision_topk=p_topk_sp, f1_topk=f1_topk_sp))

    # --- competitors ---
    for m in METHODS:
        try:
            sc_comp = A.load_competitor_allgenes(m, gs, U)
        except FileNotFoundError:
            continue
        score_map_c = dict(zip(U, np.nan_to_num(sc_comp, nan=0.0)))
        auprc_c = auprc_from_scores(score_map_c, U_full, true_directed)
        sel_c = top_k_tie_aware(score_map_c, U_full, true_directed, len(true_directed))
        p_topk_c, _, f1_topk_c = precision_recall_f1(sel_c, true_directed)

        und_c = {}
        for (a, b), v in score_map_c.items():
            key = frozenset((a, b))
            und_c[key] = max(und_c.get(key, -np.inf), v)
        auprc_cu = auprc_from_scores(und_c, U_full_und, true_undirected)
        sel_cu = top_k_tie_aware(und_c, U_full_und, true_undirected, len(true_undirected))
        p_topk_cu, _, f1_topk_cu = precision_recall_f1(sel_cu, true_undirected)

        rows.append(dict(twin_def=twin_def, dataset_id=gs, method=m.upper(), variant="directed_unsigned",
                          auprc=auprc_c, precision_topk=p_topk_c, f1_topk=f1_topk_c))
        rows.append(dict(twin_def=twin_def, dataset_id=gs, method=m.upper(), variant="undirected",
                          auprc=auprc_cu, precision_topk=p_topk_cu, f1_topk=f1_topk_cu))

        comp_sign_map = {p: pearson_sign.get(p, 1.0) for p in U}
        auprc_cs, packed_c = auprc_signed_from_scores(score_map_c, comp_sign_map, U_full, true_signed)
        p_topk_cs, f1_topk_cs = top_k_signed(*packed_c, k=n_true_signed_pairs) if packed_c is not None else (float("nan"), float("nan"))
        rows.append(dict(twin_def=twin_def, dataset_id=gs, method=m.upper(), variant="signed_directed",
                          auprc=auprc_cs, precision_topk=p_topk_cs, f1_topk=f1_topk_cs))

    return rows


def load_signed_true_edges(ct):
    """Verbatim convention from apply_current_score_signed.py: unambiguous stim-only/inhib-only
    CollecTRI edges only; edges marked both or neither are dropped from the signed universe."""
    stim_only = ct.consensus_stimulation & ~ct.consensus_inhibition
    inhib_only = ct.consensus_inhibition & ~ct.consensus_stimulation
    signed = set()
    for row, sgn in [(ct[stim_only], 1), (ct[inhib_only], -1)]:
        for a, b in zip(row.source_genesymbol, row.target_genesymbol):
            signed.add((a, b, sgn))
    return signed


def main():
    ct = pd.read_csv(f"{R}/collectri_mouse.tsv", sep="\t")
    ct = ct[ct.source_genesymbol != ct.target_genesymbol]
    CE = set(zip(ct.source_genesymbol, ct.target_genesymbol))
    true_signed = load_signed_true_edges(ct)
    print(f"signed true edges (unambiguous stim-only/inhib-only): {len(true_signed)}")

    all_rows = []
    for twin_def in TWIN_DEFS:
        for gs in GENE_SETS:
            all_rows.extend(score_one_panel(twin_def, gs, CE, true_signed))

    df = pd.DataFrame(all_rows)

    blocks = []
    for twin_def in df.twin_def.unique():
        for variant in df.variant.unique():
            sub = df[(df.twin_def == twin_def) & (df.variant == variant)]
            for dataset_id in list(sub.dataset_id.unique()) + ["ALL"]:
                s = sub if dataset_id == "ALL" else sub[sub.dataset_id == dataset_id]
                for method in sorted(s.method.unique()):
                    m = s[s.method == method]
                    blocks.append(dict(twin_def=twin_def, variant=variant, dataset_id=dataset_id,
                                        method=method, f1_topk=m.f1_topk.mean(),
                                        precision_topk=m.precision_topk.mean(), auprc=m.auprc.mean()))
    table = pd.DataFrame(blocks).round(4)
    # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] out_csv = f"{HERE}/summary_metrics_table_larry.csv"
    out_csv = f"{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/larry_hematopoiesis_validation/summary_metrics_table_larry.csv"
    table.to_csv(out_csv, index=False)
    print(f"LARRY: {len(table)} rows -> {out_csv}")
    print(table[(table.variant == "directed_unsigned") & (table.dataset_id == "ALL")].to_string(index=False))


if __name__ == "__main__":
    main()
