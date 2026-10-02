#!/usr/bin/env python3
"""FM06: the TwinScore pair procedure (spec of 2026-09-25, as implemented in yscher's fanout_delta/spec_v3.py), scored against CollecTRI.

Two cases (--mode)
  merged  A and B pooled into ONE sample (clone id = barcode, pooled over dishes, so twin pairs may cross dishes). Score = R.
  ab      A = t1, B = t2. Steps 1-4 at the LARGER replicate (more cells in clones of 2-50); cross-sample terms from A cells paired with their B
          clone-mates (rho_dagger(x->y) = corr(x in A, y in B)); score = s(R) + 1[coupled] s(gamma); fan-out (bidirectional triangle) dropped.
Steps (all Spearman, clone weighted)
  I    |z_rho| = |rho| / sd_rho > 2.576                                 rho over all cells of the sample
  II   z_het = (rho_Delta - centre) / sd_het, h = -z_het sign(centre), heterogeneous if h > 2.326; centre = mean of 20 random re-pairings
  III  lambda = min(1, max(0,h)/2.33), z_reg = (S - lambda C) / sqrt(sd_S^2 + lambda^2 sd_C^2); g = max(0, 1 - 1/var_pairs(C/sd_C));
       z* = g z_reg + (1-g) z_rho; called if |z*| > 2.576
  IV   R = sqrt(max(0,u_x)^2 + max(0,u_y)^2), u = (|z*| - row mean)/max(row sd, sqrt(1-2/pi)) over the full row of x / of y (all panel genes)
  S = same-cell correlation over the twin cells; C = sister cross correlation (both symmetrised), from full_depth_layers of yscher's twin_layers_lib.
Nulls (--null variants are all computed in one run and reported side by side)
  analytic   sd_rho = 1/sqrt(N_eff-1); sd_het = sd_C = 1/sqrt(m_eff-1), m_eff = K^2/sum(1/C(n_c,2)); sd_S = 1/sqrt(N_tw-1)   (yscher's spec_v3)
  clone      the same with the number of clones K in place of every effective size (clone = the independent unit)
  bootstrap  per-pair sd from n_null draws: rho, S : partner taken from a random cell of another clone; C : second sister replaced by a cell of
             another clone; rho_Delta : random re-pairing of the twin differences (centre still the mean of the first 20 re-pairings)
Reports AUPRC-x, AP-x and hits at k=n_true against CollecTRI (merged: unordered pairs, true = edge in either direction, source universe = pair with a CollecTRI-source gene; ab: ordered pairs, x a source), next to the final
TwinFER, the earlier merged-sample score and the competitors on the same pairs.

usage: twinscore_spec_fm06.py <gene_set> --mode merged|ab [--n-null 100] [--out-dir DIR]
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.metrics import average_precision_score

PIPE = os.path.dirname(os.path.abspath(__file__))
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, PIPE)
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/package")
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/larry_hematopoiesis_validation")
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/yscher/Transcriptomic Distance/exports/_probe_tmp")
from twinfer.scoring import analytic_zscores as AZ
from paper_analysis.fatemap_pipeline.external_yscher import twin_layers_lib as L
from paper_analysis.fatemap_pipeline import twinscore_supp_helpers as supp
from twinfer.inference.correlation_functions import (_build_cross_time_twins, _prepare_delta_rank_permutation_cache,  # noqa: E402
                                                      assign_twin_id, get_clone_weights, get_unit_weights, split_twins)
from twinfer.inference.correlation_functions import _weighted_spearman_matrix as wsm  # noqa: E402
from paper_analysis.fatemap_pipeline.final_fm06_tables import COLLECTRI, COMPETITORS, NETS, SCORED, competitor_scores, metrics, s  # noqa: E402

log = supp.log
BASE = f'{TWINFER_PROJECT_ROOT}'
Z, ZH, FLOOR = 2.576, 2.326, np.sqrt(1 - 2 / np.pi)


def kish(v):
    v = np.asarray(v, float)
    return v.sum() ** 2 / (v ** 2).sum()


# [2026-10-01 commented out: yscher data now copied to clean_data/external_yscher (user yes); see REPOINT_LOG.tsv] YROOT = "/gpfs/projects/b1255/yscher/Transcriptomic Distance/exports/_probe_tmp"
YROOT = f"{TWINFER_PROJECT_ROOT}/clean_data/external_yscher/Transcriptomic_Distance/exports/_probe_tmp"


def load_raw(gs):
    """barcoded cells, the full count matrix (csr, cells x all genes), per-cell totals, the panel genes and their column positions."""
    qc = f"{BASE}/finalized_data/FM06_data/qc_filtered"
    genes = sorted(json.load(open(f"{BASE}/analysis_data/fm06/data/gene_sets_fm06.json"))[gs])
    X = sio.mmread(f"{qc}/fm06_qc_counts.mtx").tocsr()
    genes_full = np.array(open(f"{qc}/genes.txt").read().split())
    obs = pd.read_csv(f"{qc}/obs_metadata.csv", index_col=0)
    gidx = {g: i for i, g in enumerate(genes_full)}
    tot = np.asarray(X.sum(axis=1)).ravel().astype(float)
    bc = obs["fatemap_clone_singletcode"].astype(str).to_numpy()
    keep = np.array([len(b) > 0 for b in bc])
    df = pd.DataFrame(dict(cell_id=obs.index.to_numpy(), barcode=bc, dish=obs["replicate"].astype(str).to_numpy()))[keep].reset_index(drop=True)
    return df, X[keep], tot[keep], genes, np.array([gidx[g] for g in genes]), genes_full


def covariate_universe(Xc, genes_full):
    """yscher's spec_v3 gene universe: detected in >= 5% of the cells, not cell-cycle / growth-flagged / curated machinery."""
    # [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
    # sys.path.insert(0, f"{YROOT}/tf_wide")
    from paper_analysis.fatemap_pipeline.external_yscher.machinery import is_machinery
    FL = pd.read_csv(f"{YROOT}/tf_wide/fm06/gene_flags.csv")
    bad = set(FL[(FL.cycle) | (FL.growth)].gene) | {g for g in FL.gene if is_machinery(g)}
    det = np.asarray((Xc > 0).mean(0)).ravel()
    return np.array([i for i, g in enumerate(genes_full) if det[i] >= 0.05 and g not in bad])


def panel_expression(Xc, tot, pidx, uni=None, genes_full=None):
    """log1p(count / cell total * 1e4) of the panel genes; with `uni` given, log total counts and the S, G2M and immediate-early scores (computed on
    the universe genes, as spec_v3 does) are regressed out of every gene (per-cell covariates, so the twin structure is untouched)."""
    Vp = np.log1p(Xc[:, pidx].toarray().astype(float) / np.where(tot == 0, 1, tot)[:, None] * 1e4)
    if uni is None:
        return Vp
    # [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
    # sys.path.insert(0, f"{YROOT}/hive_fm06/work/fm06-builder-contrarian/scripts")
    from paper_analysis.fatemap_pipeline.external_yscher.gene_lists import G2M_GENES, S_GENES
    VB = np.log1p(Xc[:, uni].toarray().astype(float) / np.where(tot == 0, 1, tot)[:, None] * 1e4)
    idx = {genes_full[i].upper(): k for k, i in enumerate(uni)}
    up = lambda gl: {g.upper() for g in gl}
    IEG = ["FOS", "FOSB", "JUN", "JUNB", "JUND", "EGR1", "EGR2", "EGR3", "IER2", "IER3", "ATF3", "DUSP1", "ZFP36", "NR4A1"]
    covs = [np.log1p(np.expm1(VB).sum(1))]
    for gl in (S_GENES, G2M_GENES, IEG):
        cols = [idx[n] for n in up(gl) if n in idx]
        if cols:
            sd_ = VB[:, cols].std(0)
            covs.append(((VB[:, cols] - VB[:, cols].mean(0)) / np.where(sd_ > 0, sd_, 1.0)).mean(1))
    Zc = np.column_stack([np.ones(len(VB))] + covs)
    del VB
    beta = np.linalg.lstsq(Zc, Vp, rcond=None)[0]
    log(f"    covariates regressed out ({len(covs)}: log total counts, S, G2M, immediate-early scores; universe {len(uni)} genes); mean R^2 over panel genes "
        f"{np.mean(1 - ((Vp - Zc @ beta) ** 2).sum(0) / ((Vp - Vp.mean(0)) ** 2).sum(0)):.3f}")
    return Vp - Zc @ beta + Vp.mean(0)


def clone_filter(df, key):
    n = df.groupby(key).transform("size").to_numpy()
    return (n >= 2) & (n <= supp.MAX_CLONE_SIZE)


class Sample:
    """one sample: cells, clone weights, twin pairs, S, C, rho, rho_Delta and its centre."""

    def __init__(self, X, clone, cell_id, rng, n_centre=20):
        self.X, self.rng = X, rng
        fr = pd.DataFrame(dict(cell_id=cell_id, clone_id=clone))
        self.clone_code = pd.factorize(clone)[0]
        self.n = len(fr)
        self.cw = get_clone_weights(fr)
        tw = assign_twin_id(fr)
        a0, b0 = split_twins(tw)
        pos = pd.Series(np.arange(self.n), index=cell_id)
        self.ia, self.ib = pos[a0.cell_id.to_numpy()].to_numpy(), pos[b0.cell_id.to_numpy()].to_numpy()
        self.w = get_unit_weights(a0, unit="clone")
        self.mw = 0.5 * np.concatenate([self.w, self.w])
        sizes = fr.groupby("clone_id").size().to_numpy()
        self.K = len(sizes)
        self.N_eff = kish(self.cw)
        self.m_twin = AZ.m_eff_twin(sizes[sizes >= 2])
        self.N_tw = kish(self.mw)
        self.G = X.shape[1]
        self.RHO = wsm(X, self.cw)
        self.S, self.C = L.full_depth_layers(X, self.ia, self.ib, self.w)
        self.RD = wsm(X[self.ia] - X[self.ib], self.w)
        np.fill_diagonal(self.RD, 0.0)
        acc = np.zeros((self.G, self.G))
        for _ in range(n_centre):
            acc += wsm(X[self.ia] - X[self.ib[rng.permutation(len(self.ib))]], self.w)
        self.CEN = acc / n_centre
        log(f"    sample: {self.n:,} cells, {self.K:,} clones, {len(self.ia):,} twin pairs; N_eff {self.N_eff:.0f}, m_twin {self.m_twin:.0f}, N_tw {self.N_tw:.0f}")

    def other_clone(self, cells):
        out = self.rng.integers(self.n, size=len(cells))
        bad = self.clone_code[out] == self.clone_code[cells]
        while bad.any():
            out[bad] = self.rng.integers(self.n, size=int(bad.sum()))
            bad = self.clone_code[out] == self.clone_code[cells]
        return out

    def bootstrap_sds(self, n_null):
        X, G = self.X, self.G
        r, c, s_, d = [], [], [], []
        allc = np.arange(self.n)
        tw_cells = np.concatenate([self.ia, self.ib])
        for _ in range(n_null):
            j = self.other_clone(allc)
            r.append(wsm(np.hstack([X, X[j]]), self.cw)[:G, G:])
            j2 = self.other_clone(self.ia)
            P = np.vstack([X[self.ia], X[j2]])
            Q = np.vstack([X[j2], X[self.ia]])
            c.append(wsm(np.hstack([P, Q]), self.mw)[:G, G:])
            j3 = self.other_clone(tw_cells)
            s_.append(wsm(np.hstack([X[tw_cells], X[j3]]), self.mw)[:G, G:])
            d.append(wsm(X[self.ia] - X[self.ib[self.rng.permutation(len(self.ib))]], self.w))
        sd = lambda a: np.maximum(np.std(a, axis=0, ddof=1), 1e-6)
        sym = lambda m: (m + m.T) / 2
        return dict(sd_rho=sym(sd(np.array(r))), sd_C=sym(sd(np.array(c))), sd_S=sym(sd(np.array(s_))), sd_het=sym(sd(np.array(d))))

    def clone_groups(self):
        """clones grouped by size: list of (n_clones, size) arrays of cell indices."""
        order = np.argsort(self.clone_code, kind="stable")
        cc = self.clone_code[order]
        starts = np.r_[0, np.flatnonzero(np.diff(cc)) + 1]
        sizes = np.diff(np.r_[starts, len(cc)])
        return [order[starts[sizes == sz][:, None] + np.arange(sz)[None, :]] for sz in np.unique(sizes)]

    def within_shuffle(self, grp):
        """every gene independently shuffled across the cells of each clone: clone means (the heterogeneity) kept, within-cell coupling removed."""
        Y = np.empty_like(self.X)
        for g in range(self.G):
            for Cm in grp:
                idx = np.argsort(self.rng.random(Cm.shape), axis=1)
                Y[Cm, g] = self.X[Cm[np.arange(len(Cm))[:, None], idx], g]
        return Y

    def zreg_null(self, lam, n_null):
        """corrected z_reg null: centre and sd of S - lam*C (lam fixed at its observed value) under the within-clone shuffle = 'heterogeneity, no regulation'."""
        grp = self.clone_groups()
        W = []
        t0 = time.time()
        for _ in range(n_null):
            S_, C_ = L.full_depth_layers(self.within_shuffle(grp), self.ia, self.ib, self.w)
            W.append(S_ - lam * C_)
        W = np.array(W)
        sym = lambda m: (m + m.T) / 2
        log(f"    within-clone shuffle null for z_reg: {n_null} draws in {time.time() - t0:.0f}s")
        return dict(centre=sym(W.mean(0)), sd=sym(np.maximum(W.std(0, ddof=1), 1e-6)))

    def null_sds(self, kind, boot=None):
        G = self.G
        full = lambda v: np.full((G, G), float(v))
        if kind in ("analytic", "zreg_corrected"):       # zreg_corrected: the same analytic sds for rho, h, C, S; z_reg gets the within-clone null (steps(reg=...))
            return dict(sd_rho=full(1 / np.sqrt(self.N_eff - 1)), sd_het=full(1 / np.sqrt(self.m_twin - 1)), sd_C=full(1 / np.sqrt(self.m_twin - 1)),
                        sd_S=full(1 / np.sqrt(self.N_tw - 1)))
        if kind == "clone":
            v = full(1 / np.sqrt(self.K - 1))
            return dict(sd_rho=v, sd_het=v, sd_C=v, sd_S=v)
        return boot


def steps(smp, sd, reg=None):
    """steps I-IV on one sample with the given null sds; returns matrices over all ordered pairs of the panel."""
    G = smp.G
    off = ~np.eye(G, dtype=bool)
    z_rho = smp.RHO / sd["sd_rho"]
    z_het = (smp.RD - smp.CEN) / sd["sd_het"]
    h = -z_het * np.sign(smp.CEN)
    lam = np.minimum(1.0, np.maximum(0.0, h) / 2.33)
    Wm = smp.S - lam * smp.C
    zreg = Wm / np.sqrt(sd["sd_S"] ** 2 + (lam * sd["sd_C"]) ** 2)
    if reg is not None:                                   # corrected: (S - lam C - null mean) / null sd, both from the within-clone shuffle
        zreg = (Wm - reg["centre"]) / reg["sd"]
    v = float(np.var((smp.C / sd["sd_C"])[off]))
    g = max(0.0, 1.0 - 1.0 / v) if v > 0 else 0.0
    zstar = g * zreg + (1 - g) * z_rho
    az = np.abs(zstar)
    np.fill_diagonal(az, np.nan)
    mu_r, sd_r = np.nanmean(az, 1), np.maximum(np.nanstd(az, 1), FLOOR)
    UX = (az - mu_r[:, None]) / sd_r[:, None]
    UY = (az - mu_r[None, :]) / sd_r[None, :]      # the panel has one gene list, so the column statistics are the row statistics of y
    R = np.sqrt(np.maximum(UX, 0) ** 2 + np.maximum(UY, 0) ** 2)
    np.fill_diagonal(R, np.nan)
    return dict(z_rho=z_rho, h=h, lam=lam, W=Wm, zreg=zreg, g=g, zstar=zstar, R=R, stage1=np.abs(z_rho) > Z, hetero=h > ZH, called=np.abs(zstar) > Z)


def cross_terms(dfA, XA, dfB, XB, genes):
    """rho_dagger(x->y) = corr(x in A, y in B) over cross-sample twins (A cell x B clone-mate), s. yscher spec_v3."""
    cols = [f"{g}_mRNA" for g in genes]
    def fr(d, V):
        return assign_twin_id(pd.concat([d[["cell_id", "clone_id"]].reset_index(drop=True), pd.DataFrame(V, columns=cols)], axis=1))
    f1, f2 = fr(dfA, XA), fr(dfB, XB)
    at1, at2 = _build_cross_time_twins(f1, f2)
    Xa, Yb = at1[cols].to_numpy(float), at2[cols].to_numpy(float)
    wx = get_unit_weights(at1, unit="clone")
    ca, cb = _prepare_delta_rank_permutation_cache(Xa, wx), _prepare_delta_rank_permutation_cache(Yb, wx)
    num = ca["cx"].T @ (wx[:, None] * cb["cx"])
    den = np.outer(ca["sx"], cb["sx"])
    with np.errstate(divide="ignore", invalid="ignore"):
        rdag = np.where(den > 0, num / den, np.nan)
    n1, n2 = f1.groupby("clone_id").size(), f2.groupby("clone_id").size()
    bi = n1.index.intersection(n2.index)
    m_cross = len(bi) ** 2 / np.sum(1.0 / (n1.loc[bi] * n2.loc[bi]).to_numpy(float))
    log(f"    cross-sample: {len(bi):,} clones in both, m_cross {m_cross:.0f}, sd_dag {1 / np.sqrt(m_cross - 1):.4f}")
    return rdag, 1 / np.sqrt(m_cross - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gene_set")
    ap.add_argument("--mode", choices=["merged", "ab"], required=True)
    ap.add_argument("--n-null", type=int, default=100)
    ap.add_argument("--nulls", default="analytic,clone,bootstrap", help="comma list of null variants to score")
    ap.add_argument("--no-covariates", action="store_true", help="skip the covariate regression (default: regress out log total counts, S, G2M, immediate-early)")
    ap.add_argument("--out-dir", default=f"{BASE}/analysis_data/fm06/data/twinscore_spec")
    a = ap.parse_args()
    gs = a.gene_set
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(supp.SEED)
    df, Xc, tot, genes, pidx, genes_full = load_raw(gs)
    G = len(genes)
    ct = pd.read_csv(COLLECTRI, sep="\t")
    ct = ct[ct.source_genesymbol != ct.target_genesymbol]
    edges, sources = set(zip(ct.source_genesymbol, ct.target_genesymbol)), set(ct.source_genesymbol)
    tf_idx = np.array([g in sources for g in genes])

    if a.mode == "merged":
        # clone = (df.dish + ":" + df.barcode).to_numpy()   # earlier: clones within a dish, twin pairs within a dish (8,517 cells)
        clone = df.barcode.to_numpy()                          # clone = barcode pooled over dishes A and B: twin pairs may cross dishes (yscher's setup)
        m = clone_filter(pd.DataFrame(dict(c=clone)), "c")
        rows_ = np.flatnonzero(m)
        uni = None if a.no_covariates else covariate_universe(Xc[rows_], genes_full)
        V = panel_expression(Xc[rows_], tot[rows_], pidx, uni, genes_full)
        smp = Sample(V, clone[m], df.cell_id.to_numpy()[m], rng)
        rdag = None
    else:
        parts, mm_ = {}, {}
        for d_ in ("A", "B"):
            sel = np.flatnonzero((df.dish == d_).to_numpy())
            parts[d_] = df.iloc[sel].reset_index(drop=True)
            mm_[d_] = sel
        both = set(parts["A"].barcode) & set(parts["B"].barcode)
        nc = {d_: int(clone_filter(parts[d_], "barcode").sum()) for d_ in parts}
        big = max(nc, key=nc.get)
        log(f"    cells in clones of 2-{supp.MAX_CLONE_SIZE}: {nc}; steps I-IV at replicate {big}")
        bigrows = mm_[big][clone_filter(parts[big], "barcode")]
        uni = None if a.no_covariates else covariate_universe(Xc[bigrows], genes_full)
        Vd = {d_: panel_expression(Xc[mm_[d_]], tot[mm_[d_]], pidx, uni, genes_full) for d_ in parts}   # each replicate regressed on its own covariates
        dB, XB0 = parts[big], Vd[big]
        m = clone_filter(dB, "barcode")
        smp = Sample(XB0[m], dB.barcode.to_numpy()[m], dB.cell_id.to_numpy()[m], rng)
        dA, XA0 = parts["A"], Vd["A"]
        dB2, XB2 = parts["B"], Vd["B"]
        kA, kB = dA.barcode.isin(both).to_numpy(), dB2.barcode.isin(both).to_numpy()
        rdag, sd_dag = cross_terms(dA[kA].rename(columns={"barcode": "clone_id"}), XA0[kA], dB2[kB].rename(columns={"barcode": "clone_id"}), XB2[kB], genes)
        rdag = np.nan_to_num(rdag)
        zdag = rdag / sd_dag

    kinds = a.nulls.split(",")
    boot = None
    if "bootstrap" in kinds:
        t0 = time.time()
        boot = smp.bootstrap_sds(a.n_null)
        log(f"    bootstrap sds from {a.n_null} draws in {time.time() - t0:.0f}s; median bootstrap/analytic: " + ", ".join(
            f"{k} {np.median((boot[k] / smp.null_sds('analytic')[k])[~np.eye(G, dtype=bool)]):.2f}" for k in boot))

    fin_path = f"{SCORED}/twinscore_supplement_fm06_{gs}_absplit_gated_bootstrap_pair_terms.csv"
    fin = pd.read_csv(fin_path) if os.path.exists(fin_path) else None      # TwinScore(phi): only for gene sets it has been scored on
    prev = f"{BASE}/analysis_data/fm06/data/twinscore_v2_merged/twinscore_v2_merged_fm06_{gs}_pair_terms.csv"
    variants_all = {}
    rows = [(x, y) for x in range(G) for y in range(G) if x != y]
    T = pd.DataFrame(dict(gene_1=[genes[x] for x, _ in rows], gene_2=[genes[y] for _, y in rows]))
    T["collectri_edge"] = [int((a_, b_) in edges) for a_, b_ in zip(T.gene_1, T.gene_2)]
    xi, yi = np.array([x for x, _ in rows]), np.array([y for _, y in rows])
    for kind in kinds:
        sd = smp.null_sds(kind, boot)
        reg = None
        if kind == "zreg_corrected":
            reg = smp.zreg_null(steps(smp, smp.null_sds("analytic"))["lam"], a.n_null)
        r = steps(smp, sd, reg)
        R_, called, het, stage1 = r["R"], r["called"], r["hetero"], r["stage1"]
        ok = stage1 & called
        if a.mode == "ab":
            zxy, zyx = zdag, zdag.T
            coup = np.maximum(np.abs(zxy), np.abs(zyx)) > Z
            gam = np.abs(rdag) - np.abs(rdag.T)
            sel = ok.copy()
            sR = np.zeros((G, G)); sG = np.zeros((G, G))
            sR[sel] = s(R_[sel]); sG[sel] = s(gam[sel])
            score = np.where(sel, sR + coup * sG, -np.inf)
            # fan-out: called heterogeneous pair with a TF called with both genes; bidirectional => dropped
            calledm = sel
            bidir = (np.abs(zxy) > Z) & (np.abs(zyx) > Z)
            drop = np.zeros((G, G), bool)
            for x in range(G):
                for y in range(G):
                    if x != y and calledm[x, y] and het[x, y] and bidir[x, y]:
                        if any(calledm[x, w_] and calledm[y, w_] for w_ in np.flatnonzero(tf_idx) if w_ not in (x, y)):
                            drop[x, y] = True
            score_drop = np.where(drop, -np.inf, score)
            T[f"TwinScore-recent [{kind}]: s(R)+coupled*s(gamma), fan-out dropped"] = score_drop[xi, yi]
            T[f"TwinScore-recent [{kind}]: s(R)+coupled*s(gamma)"] = score[xi, yi]
            T[f"TwinScore-recent [{kind}]: R only (called)"] = np.where(ok, R_, -np.inf)[xi, yi]
        else:
            T[f"TwinScore-recent [{kind}]: R (called)"] = np.where(ok, R_, -np.inf)[xi, yi]
        T[f"TwinScore-recent [{kind}]: R ungated"] = R_[xi, yi]
        for k_ in ("z_rho", "h", "lam", "zreg", "zstar"):
            T[f"{kind}_{k_}"] = r[k_][xi, yi]
        log(f"    [{kind}] g {r['g']:.3f}; stage I {int(stage1[~np.eye(G, dtype=bool)].sum())}, heterogeneous {int(het[~np.eye(G, dtype=bool)].sum())}, called {int(called[~np.eye(G, dtype=bool)].sum())}, both {int(ok[~np.eye(G, dtype=bool)].sum())} of {G * (G - 1)}")
    T.to_csv(f"{a.out_dir}/twinscore_spec_{a.mode}_{'_'.join(kinds)}_fm06_{gs}_pair_terms.csv", index=False)

    if fin is not None:
        T = T.merge(fin[["gene_1", "gene_2", "TwinScore"]].rename(columns={"TwinScore": "TwinScore(phi)"}), on=["gene_1", "gene_2"], how="left")
    else:
        log("    no TwinScore(phi) file for this gene set: not compared")
    if os.path.exists(prev):
        P = pd.read_csv(prev)[["gene_1", "gene_2", "TwinScoreV2", "s_zreg"]].rename(columns={"TwinScoreV2": "earlier merged score (as written)", "s_zreg": "earlier merged s(z_reg)"})
        T = T.merge(P, on=["gene_1", "gene_2"], how="left")
    have_comp = [m_ for m_ in COMPETITORS if os.path.exists(f"{NETS}/{m_}_{gs}_allgenes.csv")]
    comp = {}
    if have_comp:
        from paper_analysis.fatemap_pipeline import final_fm06_tables as _ft
        _sv, _ft.COMPETITORS = _ft.COMPETITORS, have_comp
        comp = competitor_scores(gs, T)
        _ft.COMPETITORS = _sv
    else:
        log("    no competitor networks for this gene set: not compared")
    for c_ in have_comp:
        T[c_] = comp[c_]
    if a.mode == "merged":
        # one sample -> each pair listed once (R is symmetric in x, y). Truth: a CollecTRI edge in EITHER direction; the earlier / competitor scores are
        # collapsed to the larger of the two orientations; the source universe = pairs with at least one CollecTRI-source gene.
        gi_ = {g: i for i, g in enumerate(genes)}
        T["src_any"] = (T.gene_1.isin(sources) | T.gene_2.isin(sources)).astype(int)
        ia_, ib_ = T.gene_1.map(gi_).to_numpy(), T.gene_2.map(gi_).to_numpy()
        T["_lo"], T["_hi"] = np.minimum(ia_, ib_), np.maximum(ia_, ib_)
        T = T.drop(columns=["gene_1", "gene_2"]).groupby(["_lo", "_hi"], as_index=False).max()
        T["gene_1"], T["gene_2"] = [genes[i] for i in T._lo], [genes[i] for i in T._hi]
        T = T.drop(columns=["_lo", "_hi"])
        T.to_csv(f"{a.out_dir}/twinscore_spec_merged_unordered_{'_'.join(kinds)}_fm06_{gs}_pair_terms.csv", index=False)
        log(f"    unordered pairs: {len(T)}; CollecTRI edges (either direction): {int(T.collectri_edge.sum())}")
    score_cols = [c for c in T.columns if (": " in c and not c.startswith(tuple(f"{k}_" for k in ("analytic", "clone", "bootstrap")))) or c in
                  ("TwinScore(phi)", "earlier merged score (as written)", "earlier merged s(z_reg)", *COMPETITORS) and c in T.columns]
    # report only the full spec variant of TwinScore-recent (merged: R (called); ab: s(R)+coupled*s(gamma) with the fan-out dropped)
    score_cols = [c for c in score_cols if not (c.endswith("R ungated") or c.endswith("R only (called)") or c.endswith("s(gamma)"))]
    y = T.collectri_edge.to_numpy()
    out = []
    for uname, mask in (("allpairs", np.ones(len(T), bool)), ("collectri_sources", (T.src_any == 1).to_numpy() if a.mode == "merged" else T.gene_1.isin(sources).to_numpy())):
        for c_ in score_cols:
            sc = np.nan_to_num(T[c_].to_numpy(float), nan=-1e9, neginf=-1e9)[mask]
            mm = metrics(sc, y[mask])
            base = float(np.mean(y[mask]))
            called_ = sc > -1e8                      # spec scores are -inf for pairs that are not called; ungated scores / competitors rank every pair
            npred = int(called_.sum()) if not called_.all() else np.nan
            tp_called = int(y[mask][called_].sum()) if not called_.all() else np.nan
            out.append(dict(gene_set=gs, mode=a.mode, universe=uname, method=c_, AUPRC=round(mm["AUPRC"], 4), AUPRC_x=round(mm["AUPRC_x"], 3),
                            AP_x=round(average_precision_score(y[mask], sc) / base, 3), hits=mm["TP_at_k"], k=mm["k"],
                            precision_at_k=round(mm["precision_at_k"], 3), recall_at_k=round(mm["precision_at_k"], 3),
                            n_called=npred, precision_called=round(tp_called / npred, 3) if npred == npred and npred > 0 else np.nan,
                            recall_called=round(tp_called / mm["k"], 3) if npred == npred else np.nan, random=round(base, 4), AUROC=round(mm["AUROC"], 3)))
    R = pd.DataFrame(out)
    R.to_csv(f"{a.out_dir}/twinscore_spec_{a.mode}_{'_'.join(kinds)}_fm06_{gs}_metrics.csv", index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 70)
    print(R.to_string(index=False))


if __name__ == "__main__":
    main()
