#!/usr/bin/env python3
"""HS054_Sot48h: TwinScore exactly as specified in TwinScore_LARRY_melanoma.pdf (Sept 2026),
the "melanoma" (one-sample) case -- Stages I-III, jackknife-over-50-clone-group SDs, BH at
q=0.05, ranked by |z| (Stage III). No direction stage (one sample: "regulation not
established", matching the PDF's stated limitation for melanoma FM06 itself).

Deliberate substitutions (no ChIP-Atlas TF network exists for this pancreatic cell type):
  - Candidate pairs: all ordered pairs among the gene-set panel (not ChIP-Atlas-bound pairs).
  - Truth/evaluation: CollecTRI edges (not ChIP-Atlas), same as every other HS054 comparison
    in this pipeline -- this is an evaluation substitution only, not part of the scoring
    formula itself.
  - Weighted Pearson of normal-score-transformed ranks (the PDF's spec) is implemented as
    weighted Spearman (_weighted_spearman_matrix / wsm), matching every other correlation in
    this repo including yscher's own twin_layers_lib.full_depth_layers ("every correlation is
    gzu's clone-weighted Spearman") -- the two are documented in this codebase as equivalent.

Formula (all clone-weighted; twin pairs = ALL C(n,2) within-clone cell pairs, clone weight 1
split evenly over its pairs; jackknife over 50 random clone groups, sd(theta) =
sqrt(49/50 * sum_g(theta_(-g) - theta_bar)^2), referred to t_49):
  Stage I (existence):   z_rho = rho / sd(rho) over all cells; BH over candidate pairs (q=0.05).
  Twin correlations:     S (same-cell), C (twin/cross), U = S - C (unshared), from
                         full_depth_layers over ALL within-clone twin pairs.
  Stage II (inherited):  h = sign(S) * C / sd(C); inherited state if h > 2.405.
  Stage III:             z = U / sd(U); BH over Stage-I-called pairs (q=0.05); ranked by |z|.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import t as t_dist
from sklearn.metrics import auc, precision_recall_curve

PIPE = os.path.dirname(os.path.abspath(__file__))
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, PIPE)
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/package")
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/yscher/Transcriptomic Distance/exports/_probe_tmp")
from paper_analysis.fatemap_pipeline import twinscore_supp_helpers as supp
from paper_analysis.fatemap_pipeline.external_yscher import twin_layers_lib as L
from twinfer.inference.correlation_functions import (  # noqa: E402
    assign_twin_id, split_twins, get_unit_weights, get_clone_weights,
    _weighted_spearman_matrix as wsm,
)
from paper_analysis.fatemap_pipeline.twinscore_spec_fm06 import clone_filter, panel_expression  # noqa: E402

log = supp.log
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] COLLECTRI = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation/resources/collectri_human.tsv'
COLLECTRI = f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/larry_hematopoiesis_validation/resources/collectri_human.tsv'
QC_DIR = f'{TWINFER_PROJECT_ROOT}/finalized_data/HS054_Sot48h_data/qc_filtered'
QC_MTX = f"{QC_DIR}/hs054_sot48h_qc_counts.mtx"
GENE_SETS_JSON = f'{TWINFER_PROJECT_ROOT}/analysis_data/hs054/data/gene_sets_hs054.json'
OUT_DIR = f'{TWINFER_PROJECT_ROOT}/analysis_data/hs054/data/twinscore_paper'
N_JACK = 50
H_THR = 2.405
SEED = 0


def load_raw_hs054(gs):
    all_sets = json.load(open(GENE_SETS_JSON))
    if gs == "union":
        genes = sorted(set().union(*all_sets.values()))
    else:
        genes = sorted(all_sets[gs])
    X = sio.mmread(QC_MTX).tocsr()
    genes_full = np.array(open(f"{QC_DIR}/genes.txt").read().split())
    obs = pd.read_csv(f"{QC_DIR}/obs_metadata.csv", index_col=0)
    gidx = {g: i for i, g in enumerate(genes_full)}
    tot = np.asarray(X.sum(axis=1)).ravel().astype(float)
    bc = obs["fatemap_clone_singletcode"].astype(str).to_numpy()
    keep = np.array([len(b) > 0 for b in bc])
    df = pd.DataFrame(dict(cell_id=obs.index.to_numpy(), barcode=bc))[keep].reset_index(drop=True)
    return df, X[keep], tot[keep], genes, np.array([gidx[g] for g in genes]), genes_full


def twin_pairs(clone, cell_id):
    fr = pd.DataFrame(dict(cell_id=cell_id, clone_id=clone))
    tw = assign_twin_id(fr)
    a0, b0 = split_twins(tw)
    pos = pd.Series(np.arange(len(fr)), index=cell_id)
    ia = pos[a0.cell_id.to_numpy()].to_numpy()
    ib = pos[b0.cell_id.to_numpy()].to_numpy()
    w = get_unit_weights(a0, unit="clone")
    return ia, ib, w


def benjamini_hochberg(pvals, q=0.05):
    """Returns a boolean array: which p-values are BH-significant at level q."""
    p = np.asarray(pvals, float)
    n = len(p)
    out = np.zeros(n, dtype=bool)
    idx = np.flatnonzero(np.isfinite(p))
    m = len(idx)
    if m == 0:
        return out
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    thresh = (np.arange(1, m + 1) / m) * q
    below = ranked <= thresh
    if below.any():
        kmax = int(np.flatnonzero(below).max())
        out[order[: kmax + 1]] = True
    return out


def two_sided_p(z, df):
    return 2.0 * t_dist.sf(np.abs(z), df=df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gene_set")
    ap.add_argument("--n-jack", type=int, default=N_JACK)
    ap.add_argument("--out-dir", default=OUT_DIR)
    a = ap.parse_args()
    gs = a.gene_set
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    df, Xc, tot, genes, pidx, genes_full = load_raw_hs054(gs)
    G = len(genes)
    m = clone_filter(pd.DataFrame(dict(c=df.barcode)), "c")
    rows_ = np.flatnonzero(m)
    V = panel_expression(Xc[rows_], tot[rows_], pidx, None, genes_full)
    clone = df.barcode.to_numpy()[rows_]
    cell_id = df.cell_id.to_numpy()[rows_]
    n_cells = len(clone)
    clones_all = np.unique(clone)
    log(f"[{gs}] {n_cells:,} cells, {len(clones_all):,} clones (size 2-{supp.MAX_CLONE_SIZE})")

    cw_full = get_clone_weights(pd.DataFrame(dict(clone_id=clone)))
    RHO = wsm(V, cw_full)
    ia, ib, w = twin_pairs(clone, cell_id)
    S, C = L.full_depth_layers(V, ia, ib, w)
    U = S - C
    n_pairs_twin = len(ia)
    log(f"[{gs}] {n_pairs_twin:,} within-clone twin pairs (all C(n,2))")

    # ---- jackknife over N_JACK random clone groups ----
    shuffled = rng.permutation(clones_all)
    groups = np.array_split(shuffled, a.n_jack)
    RHO_j = np.empty((a.n_jack, G, G))
    U_j = np.empty((a.n_jack, G, G))
    C_j = np.empty((a.n_jack, G, G))
    for gi, grp in enumerate(groups):
        keep_mask = ~np.isin(clone, grp)
        cw_g = get_clone_weights(pd.DataFrame(dict(clone_id=clone[keep_mask])))
        RHO_j[gi] = wsm(V[keep_mask], cw_g)
        ia_g, ib_g, w_g = twin_pairs(clone[keep_mask], cell_id[keep_mask])
        Vg = V[keep_mask]
        S_g, C_g = L.full_depth_layers(Vg, ia_g, ib_g, w_g)
        U_j[gi] = S_g - C_g
        C_j[gi] = C_g
    log(f"[{gs}] jackknife done ({a.n_jack} clone groups)")

    def jack_sd(vals_j):
        theta_bar = vals_j.mean(axis=0)
        return np.sqrt((a.n_jack - 1) / a.n_jack * np.sum((vals_j - theta_bar) ** 2, axis=0))

    sd_rho = jack_sd(RHO_j)
    sd_C = jack_sd(C_j)
    sd_U = jack_sd(U_j)

    ct = pd.read_csv(COLLECTRI, sep="\t")
    ct = ct[ct.source_genesymbol != ct.target_genesymbol]
    edges = set(zip(ct.source_genesymbol, ct.target_genesymbol))

    gi_ = {g: i for i, g in enumerate(genes)}
    pairs = [(x, y) for x in genes for y in genes if x != y]
    xi = np.array([gi_[x] for x, _ in pairs])
    yi = np.array([gi_[y] for _, y in pairs])

    z_rho = RHO[xi, yi] / sd_rho[xi, yi]
    p_rho = two_sided_p(z_rho, df=a.n_jack - 1)
    stage1 = benjamini_hochberg(p_rho, q=0.05)

    h = np.sign(S[xi, yi]) * C[xi, yi] / sd_C[xi, yi]
    inherited = h > H_THR

    z_U = U[xi, yi] / sd_U[xi, yi]
    p_U = np.full(len(pairs), np.nan)
    p_U[stage1] = two_sided_p(z_U[stage1], df=a.n_jack - 1)
    called = np.zeros(len(pairs), dtype=bool)
    called[stage1] = benjamini_hochberg(p_U[stage1], q=0.05)

    T = pd.DataFrame(dict(gene_1=[x for x, _ in pairs], gene_2=[y for _, y in pairs]))
    T["collectri_edge"] = [int(p in edges) for p in pairs]
    T["rho"] = RHO[xi, yi]
    T["z_rho"] = z_rho
    T["stage1_called"] = stage1
    T["S"] = S[xi, yi]
    T["C"] = C[xi, yi]
    T["U"] = U[xi, yi]
    T["h"] = h
    T["inherited_state"] = inherited
    T["z"] = z_U
    T["called"] = called

    n_stage1 = int(stage1.sum())
    n_inherited = int((inherited & stage1).sum())
    n_called = int(called.sum())
    log(f"[{gs}] Stage I called: {n_stage1}/{len(pairs)}; of those, inherited-state: {n_inherited}; "
        f"Stage III called (BH over Stage I): {n_called}")

    y_true = T["collectri_edge"].to_numpy()
    score_called = np.where(T["called"], np.abs(T["z"]), -np.inf)
    score_ranked_all = np.abs(T["z"].to_numpy())  # unfiltered |z| ranking, for the "ranked by |z|" list

    def full_report(sc, y):
        sc = np.nan_to_num(np.asarray(sc, float), nan=np.nanmin(sc) - 1.0 if np.isfinite(sc).any() else 0.0)
        prec, rec, _ = precision_recall_curve(y, sc)
        auprc = auc(rec, prec)
        rand = float(y.mean())
        k = int(y.sum())
        tp = int((y[np.argsort(-sc, kind="stable")[:k]] == 1).sum())
        return dict(auprc_x=auprc / rand if rand > 0 else np.nan, tp=tp, k=k)

    if y_true.sum() >= 3:
        m_called = full_report(score_called, y_true)
        m_all = full_report(score_ranked_all, y_true)
        log(f"[{gs}] vs CollecTRI (n_true={int(y_true.sum())}/{len(y_true)}): "
            f"called-only auprc_x={m_called['auprc_x']:.3f}x hits={m_called['tp']}/{m_called['k']}  |  "
            f"all-ranked-by-|z| auprc_x={m_all['auprc_x']:.3f}x hits={m_all['tp']}/{m_all['k']}")

    out_csv = f"{a.out_dir}/twinscore_paper_hs054_{gs}_pair_terms.csv"
    T.sort_values("z", key=abs, ascending=False).to_csv(out_csv, index=False)
    log(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
