#!/usr/bin/env python3
"""Which null sd is right for the spec statistics? FM06, merged sample (A+B, one sample), one gene set.

Candidate null sds, all per unordered pair of panel genes
  analytic   1/sqrt(N_eff-1) (rho), 1/sqrt(m_eff-1) (h, C), 1/sqrt(N_tw-1) (S), quadrature sqrt(sd_S^2 + lam^2 sd_C^2) for z_reg      (yscher spec_v3)
  clone      the same with the number of clones K as the size
  cellboot   my first bootstrap: partner value from a random cell of ANOTHER clone (destroys the clone structure of the partner gene)
  block      clone-BLOCK permutation, per gene independently, clones of equal size exchanged as blocks: keeps every gene's clone structure and
             within-clone spread, makes genes independent.  Null for rho (existence).
  within     within-clone shuffle of every gene independently, clone means kept: keeps clone-level covariation (heterogeneity), removes within-cell
             coupling.  Null for z_reg = (S - lam C)/sd: sd of S - lam_obs C over draws, and its mean.
  repair     random re-pairing of the twin differences: null for h (z_het).  (sd over draws)

Checks on synthetic data with a KNOWN truth (built from the real data, several seeds)
  block-permuted data   no x-y association at all      -> z_rho must have sd 1 and |z|>2.576 in ~1% of pairs; h ~ N(0,1) too
  within-clone-shuffled no regulation, heterogeneity kept -> z_reg (lam fixed at the real-data value) must have sd 1, |z|>2.576 in ~1%
A null sd is right if it gives sd(z) = 1 there.  Real-data ratios (each null sd / analytic sd) are printed as well.

usage: null_calibration_fm06.py <gene_set> [--n-null 100] [--n-syn 3]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

PIPE = os.path.dirname(os.path.abspath(__file__))
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, PIPE)
from paper_analysis.fatemap_pipeline import twinscore_spec_fm06 as sp

wsm, L, log = sp.wsm, sp.L, sp.log
Z = sp.Z


def groups(smp):
    """clones grouped by size: list of (n_clones, size) arrays of cell indices, in the sample's own cell order."""
    order = np.argsort(smp.clone_code, kind="stable")
    cc = smp.clone_code[order]
    starts = np.r_[0, np.flatnonzero(np.diff(cc)) + 1]
    sizes = np.diff(np.r_[starts, len(cc)])
    out = []
    for sz in np.unique(sizes):
        st = starts[sizes == sz]
        out.append(order[st[:, None] + np.arange(sz)[None, :]])
    return out


def block_perm(X, grp, rng):
    Y = np.empty_like(X)
    for g in range(X.shape[1]):
        for Cm in grp:
            Y[Cm, g] = X[Cm[rng.permutation(len(Cm))], g]
    return Y


def within_shuffle(X, grp, rng):
    Y = np.empty_like(X)
    for g in range(X.shape[1]):
        for Cm in grp:
            idx = np.argsort(rng.random(Cm.shape), axis=1)
            Y[Cm, g] = X[Cm[np.arange(len(Cm))[:, None], idx], g]
    return Y


def sym(m):
    return (m + m.T) / 2


def real_nulls(smp, lam, grp, n_null):
    """block-permutation sd of rho, within-clone mean/sd of S - lam C, re-pairing sd of rho_Delta, cell-bootstrap sds, on the real sample."""
    X, G = smp.X, smp.G
    rb, wr = [], []
    for _ in range(n_null):
        rb.append(wsm(block_perm(X, grp, smp.rng), smp.cw))
        S_, C_ = L.full_depth_layers(within_shuffle(X, grp, smp.rng), smp.ia, smp.ib, smp.w)
        wr.append(S_ - lam * C_)
    sd = lambda a: np.maximum(np.std(a, axis=0, ddof=1), 1e-6)
    boot = smp.bootstrap_sds(n_null)
    return dict(sd_rho_block=sym(sd(np.array(rb))), wr_mean=np.mean(wr, axis=0), sd_reg_within=sd(np.array(wr)), boot=boot)


def summarise(name, z, up):
    z = z[up]
    z = z[np.isfinite(z)]
    return dict(check=name, n=len(z), mean=round(float(z.mean()), 3), sd=round(float(z.std(ddof=1)), 3), rate_2p576=round(float((np.abs(z) > Z).mean()), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gene_set")
    ap.add_argument("--n-null", type=int, default=100)
    ap.add_argument("--n-syn", type=int, default=3)
    ap.add_argument("--no-covariates", action="store_true")
    ap.add_argument("--out-dir", default=f"{sp.BASE}/analysis_data/fm06/data/twinscore_spec")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(sp.supp.SEED)
    df, Xc, tot, genes, pidx, genes_full = sp.load_raw(a.gene_set)
    G = len(genes)
    up = np.triu(np.ones((G, G), bool), 1)
    # clone = (df.dish + ":" + df.barcode).to_numpy()   # earlier: within-dish clones
    clone = df.barcode.to_numpy()                          # barcode pooled over dishes A and B (twin pairs may cross dishes)
    m = sp.clone_filter(pd.DataFrame(dict(c=clone)), "c")
    rows_ = np.flatnonzero(m)
    uni = None if a.no_covariates else sp.covariate_universe(Xc[rows_], genes_full)
    V = sp.panel_expression(Xc[rows_], tot[rows_], pidx, uni, genes_full)
    smp = sp.Sample(V, clone[m], df.cell_id.to_numpy()[m], rng)
    grp = groups(smp)
    ana = smp.null_sds("analytic")
    r_ana = sp.steps(smp, ana)
    lam = r_ana["lam"]

    t0 = time.time()
    rn = real_nulls(smp, lam, grp, a.n_null)
    log(f"    real-data null draws ({a.n_null}) in {time.time() - t0:.0f}s")
    boot = rn["boot"]
    # re-pairing sd of rho_Delta is boot['sd_het']; its centre is smp.CEN (20 re-pairings)
    ratios = []
    med = lambda x: round(float(np.median(x[up])), 3)
    for nm, num, den in (("rho: cellboot/analytic", boot["sd_rho"], ana["sd_rho"]), ("rho: block/analytic", rn["sd_rho_block"], ana["sd_rho"]),
                         ("rho: block/cellboot", rn["sd_rho_block"], boot["sd_rho"]),
                         ("h (rho_Delta): repair/analytic", boot["sd_het"], ana["sd_het"]),
                         ("C: cellboot/analytic", boot["sd_C"], ana["sd_C"]), ("S: cellboot/analytic", boot["sd_S"], ana["sd_S"])):
        ratios.append(dict(stat=nm, median_ratio=med(num / den), p10=round(float(np.percentile((num / den)[up], 10)), 3), p90=round(float(np.percentile((num / den)[up], 90)), 3)))
    sd_reg_ana = np.sqrt(ana["sd_S"] ** 2 + (lam * ana["sd_C"]) ** 2)
    sd_reg_indep = np.sqrt(boot["sd_S"] ** 2 + (lam * boot["sd_C"]) ** 2)
    for nm, num, den in (("z_reg: within-clone shuffle / analytic quadrature", rn["sd_reg_within"], sd_reg_ana), ("z_reg: within-clone / independent-bootstrap quadrature", rn["sd_reg_within"], sd_reg_indep)):
        ratios.append(dict(stat=nm, median_ratio=med(num / den), p10=round(float(np.percentile((num / den)[up], 10)), 3), p90=round(float(np.percentile((num / den)[up], 90)), 3)))
    R1 = pd.DataFrame(ratios)
    print("\nREAL DATA: null sd ratios (per unordered pair; median and 10-90% range)")
    print(R1.to_string(index=False))
    print(f"within-clone null mean of S - lam*C: median {np.median(rn['wr_mean'][up]):.4f}, median |mean|/sd {np.median((np.abs(rn['wr_mean']) / rn['sd_reg_within'])[up]):.3f}")

    rows = []
    for k in range(a.n_syn):
        for kind, fn in (("block-permuted (no x-y association)", block_perm), ("within-clone shuffled (heterogeneity kept, no regulation)", within_shuffle)):
            Xs = fn(smp.X, grp, np.random.default_rng(1000 * k + (1 if fn is block_perm else 2)))
            ss = sp.Sample(Xs, clone[m], df.cell_id.to_numpy()[m], np.random.default_rng(k), n_centre=20)
            if fn is block_perm:
                for nm, sdd in (("analytic", ana["sd_rho"]), ("clone K", sp.np.full((G, G), 1 / np.sqrt(smp.K - 1))), ("cell-bootstrap", boot["sd_rho"]),
                                ("clone-block permutation", rn["sd_rho_block"])):
                    rows.append(dict(seed=k, data=kind, stat="z_rho", null=nm, **summarise("", ss.RHO / sdd, up)))
                for nm, sdd in (("analytic", ana["sd_het"]), ("clone K", sp.np.full((G, G), 1 / np.sqrt(smp.K - 1))), ("re-pairing", boot["sd_het"])):
                    zh = (ss.RD - ss.CEN) / sdd
                    rows.append(dict(seed=k, data=kind, stat="z_het", null=nm, **summarise("", zh, up)))
            else:
                W = ss.S - lam * ss.C
                for nm, sdd, mu in (("analytic quadrature", sd_reg_ana, 0.0), ("independent-bootstrap quadrature", sd_reg_indep, 0.0),
                                    ("within-clone shuffle sd (centred)", rn["sd_reg_within"], rn["wr_mean"]), ("within-clone shuffle sd (uncentred)", rn["sd_reg_within"], 0.0)):
                    rows.append(dict(seed=k, data=kind, stat="z_reg (lam fixed at real)", null=nm, **summarise("", (W - mu) / sdd, up)))
    Rs = pd.DataFrame(rows).drop(columns="check")
    agg = Rs.groupby(["data", "stat", "null"], sort=False).agg(n=("n", "first"), mean=("mean", "mean"), sd=("sd", "mean"), false_call_rate=("rate_2p576", "mean")).round(3).reset_index()
    print("\nSYNTHETIC DATA WITH KNOWN TRUTH: z should have mean 0, sd 1, false-call rate |z|>2.576 = 0.010 (h one-tailed 2.326: 0.010)")
    pd.set_option("display.width", 230)
    pd.set_option("display.max_colwidth", 70)
    print(agg.to_string(index=False))
    R1.to_csv(f"{a.out_dir}/null_calibration_{a.gene_set}_real_ratios.csv", index=False)
    agg.to_csv(f"{a.out_dir}/null_calibration_{a.gene_set}_synthetic.csv", index=False)


if __name__ == "__main__":
    main()
