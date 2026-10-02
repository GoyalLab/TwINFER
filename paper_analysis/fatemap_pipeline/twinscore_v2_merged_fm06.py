#!/usr/bin/env python3
"""Test of a new TwinScore on FM06 with replicates A and B merged into one sample.

  TwinScore(x->y) = I( z(|rho|) >= 2.326 ) * I( lambda*C_xy / rho_xy < 1/2 ) * I( z_reg >= 2.326 ) * s( z_reg )

Definitions as implemented (choices where the written definition leaves room are marked [choice]):
  sample   one giant sample: all QC cells with a clone barcode from dish A and dish B; time_step = 0 for every cell.
           clone id = "<dish>:<barcode>", so sister pairs are formed WITHIN a dish and pooled over A and B; clones of 2-50 cells per dish [choice: not
           restricted to clones spanning both dishes]. Expression = log1p(count / cell total * 1e4) of the panel genes (same as the scorer).
  rho      clone-weighted Spearman (weights: every clone total weight 1) of x and y across cells; the package's _weighted_spearman_matrix.
  C_xy     sister-1 x vs sister-2 y over all within-clone cell pairs of both dishes, clone weighted, symmetrised over the two sisters
           (identical to sister_matrix_from_frame; checked with --check).
  lambda   min(1, |z_het| / 2.326), z_het = the package's heterogeneity z of the twin difference correlation vs the random-pair null
           (compute_zhet_zrho, n_shuffles permutations).
  z(|rho|) and z(.) nulls: within-sample clone-scramble nulls [choice]: for every draw each cell's y-value is taken from a random cell of a DIFFERENT
           clone of the same dish (rho null), and the second member of every sister pair is replaced by a random cell of a different clone of the same dish
           (C null), keeping the observed clone weights. z = (observed - null mean) / null sd per pair, over n_null draws.
  z_reg    z( |rho| - lambda*sign(rho)*C ) = z( sign(rho) * M_xy ), M = rho - lambda*C (sign clip removed), after conditioning on any third gene w coupled to
           both x and y [choice: w coupled = z(|rho_xw|) >= 2.326 and z(|rho_yw|) >= 2.326]: the statistic is the smallest of sign(rho_xy) * M_xy and
           sign(rho_xy) * partial_xy|w over the coupled w, with the first-order partial of the matrix M,
           partial_xy|w = (M_xy - M_xw M_yw) / sqrt((1 - M_xw^2)(1 - M_yw^2)). The same conditioning (same coupled sets) is applied to every null draw.
  s(.)     z-score of z_reg over all ordered pairs of the panel.
The score is symmetric in (x, y) (no direction term), so both orientations of a pair get the same value.
Reports AUPRC / AUPRC-x / hits at k against CollecTRI for: the score as written (product of the three indicators and s(z_reg)), the continuous s(z_reg),
the unconditioned z_reg, and the indicator product without s(); next to the final TwinFER and the competitors on the same pairs.

usage: twinscore_v2_merged_fm06.py <gene_set> [--n-null 200] [--n-shuffles 500] [--n-cores 8] [--check] [--out-dir DIR]
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

PIPE = os.path.dirname(os.path.abspath(__file__))
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, PIPE)
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/package")
from paper_analysis.fatemap_pipeline import twinscore_supp_helpers as supp
from twinfer.inference.correlation_functions import _weighted_spearman_matrix as wsm  # noqa: E402
from twinfer.inference.correlation_functions import assign_twin_id, get_unit_weights, split_twins  # noqa: E402
from paper_analysis.fatemap_pipeline.final_fm06_tables import COLLECTRI, COMPETITORS, SCORED, competitor_scores, metrics, s  # noqa: E402

log = supp.log
BASE = f'{TWINFER_PROJECT_ROOT}'
Z_CRIT = 2.326


def load_merged(gs):
    qc = f"{BASE}/finalized_data/FM06_data/qc_filtered"
    genes = sorted(json.load(open(f"{BASE}/analysis_data/fm06/data/gene_sets_fm06.json"))[gs])
    X = sio.mmread(f"{qc}/fm06_qc_counts.mtx").tocsr()
    genes_full = np.array(open(f"{qc}/genes.txt").read().split())
    obs = pd.read_csv(f"{qc}/obs_metadata.csv", index_col=0)
    gidx = {g: i for i, g in enumerate(genes_full)}
    tot = np.asarray(X.sum(axis=1)).ravel().astype(float)
    mat = np.log1p(X[:, [gidx[g] for g in genes]].toarray().astype(float) / np.where(tot == 0, 1, tot)[:, None] * 1e4)
    df = pd.DataFrame(mat, columns=[f"{g}_mRNA" for g in genes])
    df.insert(0, "time_step", 0)
    df.insert(0, "cell_id", obs.index.to_numpy())
    barcode = obs["fatemap_clone_singletcode"].astype(str).to_numpy()
    dish = obs["replicate"].astype(str).to_numpy()
    df.insert(0, "clone_id", np.array([f"{d}:{b}" for d, b in zip(dish, barcode)], dtype=object))
    df["dish"] = dish
    df = df[np.array([len(b) > 0 for b in barcode])].reset_index(drop=True)
    sizes = df.groupby("clone_id").size()
    keep = sizes[(sizes >= 2) & (sizes <= supp.MAX_CLONE_SIZE)].index
    df = df[df.clone_id.isin(keep)].reset_index(drop=True)
    log(f"[FM06/{gs}] merged sample: {len(df):,} cells ({(df.dish == 'A').sum():,} A, {(df.dish == 'B').sum():,} B), {df.clone_id.nunique():,} within-dish clones")
    return df, genes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gene_set")
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--n-shuffles", type=int, default=500)
    ap.add_argument("--n-cores", type=int, default=8)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out-dir", default=f"{BASE}/analysis_data/fm06/data/twinscore_v2_merged")
    a = ap.parse_args()
    gs = a.gene_set
    os.makedirs(a.out_dir, exist_ok=True)
    raw, genes = load_merged(gs)
    G, n = len(genes), len(raw)
    X = raw[[f"{g}_mRNA" for g in genes]].to_numpy(float)
    rng = np.random.default_rng(supp.SEED)

    # ---- observed rho and C
    w = get_unit_weights(raw, unit="clone")
    rho = wsm(X, w)
    tw = assign_twin_id(raw)
    t0, t1 = split_twins(tw)
    pos = pd.Series(np.arange(n), index=raw.cell_id.to_numpy())
    i0, i1 = pos[t0.cell_id.to_numpy()].to_numpy(), pos[t1.cell_id.to_numpy()].to_numpy()
    wp = get_unit_weights(t0, unit="clone")
    w2 = np.concatenate([wp, wp]) / 2.0
    log(f"    {len(i0):,} within-dish sister pairs")

    def c_matrix(j1):
        P = np.vstack([X[i0], X[j1]])
        Q = np.vstack([X[j1], X[i0]])
        return wsm(np.hstack([P, Q]), w2)[:G, G:]

    C = c_matrix(i1)
    if a.check:
        S_ref = supp.same_cell_matrix(raw, genes)
        C_ref, _ = supp.sister_matrix_from_frame(raw, genes)
        off = ~np.eye(G, dtype=bool)
        log(f"    check vs package: max|rho - same_cell_matrix| = {np.abs(rho - S_ref.to_numpy())[off].max():.2e}; max|C - sister_matrix| = {np.abs(C - C_ref.to_numpy())[off].max():.2e}")

    # ---- lambda from the package's heterogeneity z
    log(f"    z_het ({a.n_shuffles} shuffles)")
    zhet, _, _ = supp.compute_zhet_zrho(raw, genes, n_shuffles=a.n_shuffles, n_cores=a.n_cores)
    lam = np.minimum(1.0, np.abs(zhet.to_numpy()) / Z_CRIT)
    lam = np.nan_to_num(lam, nan=0.0)

    # ---- clone-scramble nulls
    clone_code = pd.factorize(raw.clone_id.to_numpy())[0]
    dish_code = pd.factorize(raw.dish.to_numpy())[0]
    dish_members = [np.flatnonzero(dish_code == d) for d in range(dish_code.max() + 1)]

    def other_clone_cell(cells):
        """for every entry of `cells` a random cell of the same dish belonging to a different clone."""
        out = np.empty(len(cells), dtype=int)
        for d, mem in enumerate(dish_members):
            sel = np.flatnonzero(dish_code[cells] == d)
            if not len(sel):
                continue
            j = mem[rng.integers(len(mem), size=len(sel))]
            bad = clone_code[j] == clone_code[cells[sel]]
            while bad.any():
                j[bad] = mem[rng.integers(len(mem), size=int(bad.sum()))]
                bad = clone_code[j] == clone_code[cells[sel]]
            out[sel] = j
        return out

    t_start = time.time()
    rho_null = np.empty((a.n_null, G, G))
    C_null = np.empty((a.n_null, G, G))
    all_cells = np.arange(n)
    for b in range(a.n_null):
        j = other_clone_cell(all_cells)
        r = wsm(np.hstack([X, X[j]]), w)[:G, G:]
        rho_null[b] = (r + r.T) / 2.0
        C_null[b] = c_matrix(other_clone_cell(i0))
    log(f"    {a.n_null} null draws in {time.time() - t_start:.0f}s")

    off = ~np.eye(G, dtype=bool)
    abs_null = np.abs(rho_null)
    z_abs = (np.abs(rho) - abs_null.mean(0)) / np.maximum(abs_null.std(0, ddof=1), 1e-12)
    np.fill_diagonal(z_abs, 0.0)
    strong = z_abs >= Z_CRIT
    coupled = strong[:, None, :] & strong[None, :, :]          # (x, y, w)
    for k in range(G):
        coupled[k, :, k] = False
        coupled[:, k, k] = False
        coupled[k, k, :] = False

    def cond_q(rho_, C_, sgn):
        M = rho_ - lam * C_
        np.fill_diagonal(M, 0.0)
        q0 = sgn * M
        Mxw, Myw, Mxy = M[:, None, :], M[None, :, :], M[:, :, None]
        den = np.sqrt(np.clip((1.0 - Mxw ** 2) * (1.0 - Myw ** 2), 1e-12, None))
        part = (Mxy - Mxw * Myw) / den
        val = np.where(coupled, sgn[:, :, None] * part, np.inf).min(axis=2)
        return np.minimum(q0, val), q0

    sgn_obs = np.sign(rho)
    q_obs, q0_obs = cond_q(rho, C, sgn_obs)
    qn = np.empty((a.n_null, G, G))
    q0n = np.empty((a.n_null, G, G))
    for b in range(a.n_null):
        qn[b], q0n[b] = cond_q(rho_null[b], C_null[b], np.sign(rho_null[b]))
    z_reg = (q_obs - qn.mean(0)) / np.maximum(qn.std(0, ddof=1), 1e-12)
    z_reg0 = (q0_obs - q0n.mean(0)) / np.maximum(q0n.std(0, ddof=1), 1e-12)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(rho) > 1e-12, lam * C / rho, np.inf)
    I1 = z_abs >= Z_CRIT
    I2 = ratio < 0.5
    I3 = z_reg >= Z_CRIT
    sz = np.zeros((G, G))
    sz[off] = s(z_reg[off])
    score = I1 * I2 * I3 * sz

    ct = pd.read_csv(COLLECTRI, sep="\t")
    edges = set(zip(ct.source_genesymbol, ct.target_genesymbol))
    rows = []
    for x in range(G):
        for y in range(G):
            if x == y:
                continue
            rows.append(dict(gene_1=genes[x], gene_2=genes[y], rho=rho[x, y], C=C[x, y], lam=lam[x, y], z_abs=z_abs[x, y], z_reg=z_reg[x, y],
                             z_reg_uncond=z_reg0[x, y], I1=int(I1[x, y]), I2=int(I2[x, y]), I3=int(I3[x, y]), s_zreg=sz[x, y],
                             TwinScoreV2=score[x, y], collectri_edge=int((genes[x], genes[y]) in edges)))
    T = pd.DataFrame(rows)
    T.to_csv(f"{a.out_dir}/twinscore_v2_merged_fm06_{gs}_pair_terms.csv", index=False)
    log(f"    pairs {len(T)}; I1 {int(I1[off].sum())}, I2 {int(I2[off].sum())}, I3 {int(I3[off].sum())}, all three {int((I1 & I2 & I3)[off].sum())} of {int(off.sum())}")

    # ---- comparison against the final TwinFER and the competitors on the same pairs
    fin = pd.read_csv(f"{SCORED}/twinscore_supplement_fm06_{gs}_absplit_gated_bootstrap_pair_terms.csv")
    T = T.merge(fin[["gene_1", "gene_2", "TwinScore"]], on=["gene_1", "gene_2"], how="left")
    comp = competitor_scores(gs, T)
    src = set(ct[ct.source_genesymbol != ct.target_genesymbol].source_genesymbol)
    y = T.collectri_edge.to_numpy()
    variants = {"V2 as written": T.TwinScoreV2.to_numpy(), "V2 continuous s(z_reg)": T.s_zreg.to_numpy(), "V2 unconditioned z_reg": T.z_reg_uncond.to_numpy(),
                "V2 indicators only (I1*I2*I3)": (T.I1 * T.I2 * T.I3).to_numpy(float), "final TwinFER": T.TwinScore.to_numpy()}
    variants.update({m: comp[m] for m in COMPETITORS})
    out = []
    for uname, mask in (("allpairs", np.ones(len(T), bool)), ("collectri_sources", T.gene_1.isin(src).to_numpy())):
        for name, sc in variants.items():
            m = metrics(sc[mask], y[mask])
            out.append(dict(gene_set=gs, universe=uname, method=name, AUPRC=round(m["AUPRC"], 4), AUPRC_x=round(m["AUPRC_x"], 3),
                            hits=m["TP_at_k"], k=m["k"], AUROC=round(m["AUROC"], 3)))
    R = pd.DataFrame(out)
    R.to_csv(f"{a.out_dir}/twinscore_v2_merged_fm06_{gs}_metrics.csv", index=False)
    pd.set_option("display.width", 200)
    print(R.to_string(index=False))


if __name__ == "__main__":
    main()
