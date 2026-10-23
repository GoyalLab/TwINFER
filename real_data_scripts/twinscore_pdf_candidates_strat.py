#!/usr/bin/env python3
"""
twinscore_pdf_candidates_strat.py  (COPY of twinscore_pdf_candidates.py with an optional STRATIFIED-RANK adjustment; the original is unchanged)

Adjustment: with --strata-csv/--strata-col every gene is ranked WITHIN strata of the confounders (cell-cycle phase, library complexity, replicate, mito fraction),
the within-stratum weighted mid-rank being rescaled to the full-sample rank scale (x W_total / W_stratum). All later steps are unchanged.

twinscore_pdf_candidates.py

TwinScore exactly as written in TwinScore_procedure_and_top100.pdf (30 Sept 2026), one sample, on a candidate-pair universe
(TF x ChIP-Atlas-bound target, from yscher's chip_candidates.py). Used to check whether the PDF's FM06 (melanoma) listed pairs
can be recovered.

  values v = 1e4 x / sum_g x ; weighted mid-ranks (cell weight 1/n_c, clone weight 1); clone-weighted Spearman
  Stage I   z_rho = rho / sigma_rho,  sigma_rho^2 = (1/N1^2) sum_c [1/n_c + (1-1/n_c) h_x h_y];  BH over ALL candidate pairs, q<=.05
  twin terms (all within-clone cell pairs, weight 1/P_c): S, C, U=S-C, h_x = C(x,x), rho_Delta
  sigma_C^2  = (1/N_t^2) sum_{n_c>=2} [1+S^2+(n_c-2)(h_x+h_y)+(n_c^2-3n_c+3) h_x h_y]/(2 P_c)
  sigma_D^2  = (1/N_t^2) sum_{n_c>=2} 1/(n_c-1)
  Stage II  h = sign(S) C/sigma_C ; inherited if h > 2.326
  Stage III z_U = rho_Delta/sigma_D ; BH over the Stage I pairs ; called if q_U <= .05
  gene filter (non-TF targets only; after the BH corrections, before ranking): GO/Tirosh/histone exclusion list
  listed pairs = Stage I pairs after the gene filter, ranked by |z_U| (rank by |U| and by |rho| alongside)
Not implemented: shared/unshared driver tests (sections 9-10) and direction (section 11: needs two samples).
"""
import argparse, json, os, re, time
import numpy as np, pandas as pd, scipy.io as sio
from scipy.stats import norm

TF_FILE = "real_data/humanTFs/TF_names_v_1.01.txt"
EXCL = "clean_data/paper_analysis/fatemap_pipeline/hpsc_20260927/gene_exclusion_list.json"
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def wrank(v, w):
    u, inv = np.unique(v, return_inverse=True)
    sw = np.bincount(inv, weights=w)
    return (np.cumsum(sw) - sw + 0.5 * sw)[inv]


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((n * p[o] / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="FM06")
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-clone", type=int, default=None, help="optional cap on clone size for twin pairs (PDF text gives none)")
    ap.add_argument("--chunk", type=int, default=400)
    ap.add_argument("--excl", default=EXCL, help="gene exclusion list json with key _union_all")
    ap.add_argument("--tag", default="", help="suffix for output files")
    ap.add_argument("--mtx-name", default=None, help="qc counts file name (default <dataset lower>_qc_counts.mtx)")
    ap.add_argument("--transpose", action="store_true", help="mtx is genes x cells")
    ap.add_argument("--qc-dir", default=None, help="folder with the counts mtx, genes.txt and obs_metadata.csv (default finalized_data/<dataset>_data/qc_filtered)")
    ap.add_argument("--sample-col", default=None); ap.add_argument("--sample-value", default=None, help="score only cells with obs[sample_col]==value (the PDF's sample B)")
    ap.add_argument("--strata-csv", default=None, help="CSV (index = cell ids of obs_metadata.csv) holding the stratum label column")
    ap.add_argument("--strata-col", default=None)
    ap.add_argument("--min-det", type=float, default=0.05, help="genes must be detected in this fraction of the scored cells")
    ap.add_argument("--tf-min-det", type=float, default=None, help="detection floor for the regulator (TF) side only; default = --min-det. Lower it (e.g. 0.01) for lowly detected TFs and flag the pairs as low power")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    root = os.environ.get("TWINFER_PROJECT_ROOT", "/gpfs/projects/b1255/hzhang/TwINFER_KA")
    os.chdir(root)

    # ---- cells / clones --------------------------------------------------------------------------------------------
    label = a.dataset.lower(); qc = a.qc_dir or f"finalized_data/{a.dataset}_data/qc_filtered"
    X = sio.mmread(f"{qc}/{a.mtx_name or label + '_qc_counts.mtx'}")
    X = (X.T if a.transpose else X).tocsr().astype(np.float32)
    genes_full = np.array(open(f"{qc}/genes.txt").read().split())
    obs = pd.read_csv(f"{qc}/obs_metadata.csv", index_col=0, dtype={"replicate": str})
    bc = obs["fatemap_clone_singletcode"].astype(str).to_numpy()
    ok = ((pd.Series(bc).str.len() > 0) & (pd.Series(bc) != "nan")).to_numpy()
    if a.sample_col:
        ok &= (obs[a.sample_col].astype(str) == str(a.sample_value)).to_numpy()
    keep = np.flatnonzero(ok)
    strata = None
    if a.strata_csv:
        strata = pd.read_csv(a.strata_csv, index_col=0).reindex(obs.index)[a.strata_col].astype(str).to_numpy()[keep]
    X, bc = X[keep], bc[keep]
    tot = np.asarray(X.sum(1)).ravel(); tot[tot == 0] = 1
    nc_clone = pd.Series(bc).value_counts()
    n_c = pd.Series(bc).map(nc_clone).to_numpy()
    w = 1.0 / n_c                                                         # cell weight 1/n_c
    if strata is not None:
        groups = [np.flatnonzero(strata == u) for u in pd.unique(strata)]; Wg = [w[g].sum() for g in groups]; Wtot = w.sum()
        log(f"stratified ranks: {len(groups)} strata of {a.strata_col}; cells per stratum min {min(len(g) for g in groups)}, median {int(np.median([len(g) for g in groups]))}")
    N1 = len(nc_clone)
    tw_clones = nc_clone[nc_clone >= 2] if a.max_clone is None else nc_clone[(nc_clone >= 2) & (nc_clone <= a.max_clone)]
    Nt = len(tw_clones); nct = tw_clones.to_numpy(float); Pc = nct * (nct - 1) / 2
    log(f"{len(bc):,} cells, N1={N1:,} clones, N_t={Nt:,} clones with n_c>=2, max clone {int(nc_clone.max())}, within-clone pairs {int(Pc.sum()):,}")
    # twin pairs
    order = np.argsort(bc, kind="stable"); bs = bc[order]
    ia, ib, wp = [], [], []
    starts = np.flatnonzero(np.r_[True, bs[1:] != bs[:-1]]); ends = np.r_[starts[1:], len(bs)]
    for s, e in zip(starts, ends):
        n = e - s
        if n < 2 or (a.max_clone and n > a.max_clone):
            continue
        ii, jj = np.triu_indices(n, 1)
        ia.append(order[s + ii]); ib.append(order[s + jj]); wp.append(np.full(len(ii), 1.0 / (n * (n - 1) / 2)))
    ia, ib, wp = np.concatenate(ia), np.concatenate(ib), np.concatenate(wp)
    # variance constants
    A_rho, B_rho = float((1.0 / nc_clone.to_numpy()).sum()), float((1 - 1.0 / nc_clone.to_numpy()).sum())
    a0, a1, a2 = (1 / (2 * Pc)).sum(), ((nct - 2) / (2 * Pc)).sum(), ((nct ** 2 - 3 * nct + 3) / (2 * Pc)).sum()
    sig_D = np.sqrt((1.0 / (nct - 1)).sum()) / Nt

    # ---- candidates ------------------------------------------------------------------------------------------------
    cand = pd.read_csv(a.candidates, sep="\t", header=None, names=["TF", "gene"])
    gi = {g: i for i, g in enumerate(genes_full)}
    det = np.asarray((X > 0).mean(0)).ravel(); detected = set(genes_full[det >= a.min_det])        # genes detected in >=5% of the scored cells
    detected_tf = detected if a.tf_min_det is None else set(genes_full[det >= a.tf_min_det])      # regulator-side floor
    # --- replaced: cand = cand[cand.TF.isin(gi) & cand.gene.isin(gi) & cand.TF.isin(detected) & cand.gene.isin(detected)].reset_index(drop=True)
    cand = cand[cand.TF.isin(gi) & cand.gene.isin(gi) & cand.TF.isin(detected_tf) & cand.gene.isin(detected)].reset_index(drop=True)
    tfs = sorted(cand.TF.unique()); tgt = sorted(set(cand.gene) | set(tfs))
    log(f"{len(cand):,} candidate pairs, {len(tfs)} TFs, {len(tgt):,} genes")
    tcol = {g: i for i, g in enumerate(tfs)}; gcol = {g: i for i, g in enumerate(tgt)}

    def ranks_block(genes):                                               # weighted mid-ranks of the CP10k values, per gene
        cols = X[:, [gi[g] for g in genes]].toarray()
        V = cols * (1e4 / tot)[:, None]
        if strata is None:
            return np.stack([wrank(V[:, j], w) for j in range(V.shape[1])], 1)
        out = np.empty(V.shape)
        for j in range(V.shape[1]):
            r = np.empty(V.shape[0])
            for g, wg in zip(groups, Wg):
                r[g] = wrank(V[g, j], w[g]) * (Wtot / wg)
            out[:, j] = r
        return out

    def prep(R):                                                          # centre over twin cells, M_g, h_g
        mu = (wp[:, None] * 0.5 * (R[ia] + R[ib])).sum(0) / Nt
        ra, rb = R[ia] - mu, R[ib] - mu
        M = (wp[:, None] * 0.5 * (ra ** 2 + rb ** 2)).sum(0)
        return ra, rb, M

    RT = ranks_block(tfs); raT, rbT, MT = prep(RT)
    hT = np.maximum(0.0, (wp[:, None] * raT * rbT).sum(0) / MT)
    RTc = RT - (w[:, None] * RT).sum(0) / w.sum(); vT = (w[:, None] * RTc ** 2).sum(0)
    dT = raT - rbT
    Z = {k: np.zeros((len(tfs), len(tgt)), np.float32) for k in ("rho", "S", "C", "rhoD")}
    hG = np.zeros(len(tgt))
    for s in range(0, len(tgt), a.chunk):
        gs = tgt[s:s + a.chunk]
        R = ranks_block(gs); ra, rb, M = prep(R)
        hG[s:s + len(gs)] = np.maximum(0.0, (wp[:, None] * ra * rb).sum(0) / M)
        Rc = R - (w[:, None] * R).sum(0) / w.sum(); vG = (w[:, None] * Rc ** 2).sum(0)
        Z["rho"][:, s:s + len(gs)] = ((RTc * w[:, None]).T @ Rc) / np.sqrt(np.outer(vT, vG))
        nrm = np.sqrt(np.outer(MT, M))
        Z["S"][:, s:s + len(gs)] = 0.5 * ((raT * wp[:, None]).T @ ra + (rbT * wp[:, None]).T @ rb) / nrm
        Z["C"][:, s:s + len(gs)] = 0.5 * ((raT * wp[:, None]).T @ rb + (rbT * wp[:, None]).T @ ra) / nrm
        dG = ra - rb
        Z["rhoD"][:, s:s + len(gs)] = ((dT * wp[:, None]).T @ dG) / np.sqrt(np.outer((wp[:, None] * dT ** 2).sum(0), (wp[:, None] * dG ** 2).sum(0)))
        log(f"  genes {min(s + a.chunk, len(tgt))}/{len(tgt)}")

    # ---- statistics for the candidate pairs ----------------------------------------------------------------------
    xi = cand.TF.map(tcol).to_numpy(); yi = cand.gene.map(gcol).to_numpy()
    hx, hy = hT[xi], hG[yi]
    T = cand.copy()
    T["rho"] = Z["rho"][xi, yi]; T["S"] = Z["S"][xi, yi]; T["C"] = Z["C"][xi, yi]; T["U"] = T.S - T.C; T["rho_delta"] = Z["rhoD"][xi, yi]
    T["h_x"], T["h_y"] = hx, hy
    sig_rho = np.sqrt((A_rho + B_rho * hx * hy)) / N1
    sig_C = np.sqrt((1 + T.S ** 2) * a0 + (hx + hy) * a1 + hx * hy * a2) / Nt
    T["z_rho"] = T.rho / sig_rho; T["h"] = np.sign(T.S) * T.C / sig_C; T["z_U"] = T.rho_delta / sig_D
    T["p_rho"] = 2 * norm.sf(np.abs(T.z_rho)); T["q_rho"] = bh(T.p_rho.fillna(1)); T["stage1"] = T.q_rho <= 0.05
    S1 = T.stage1.to_numpy()
    T["q_U"] = np.nan; T.loc[S1, "q_U"] = bh(2 * norm.sf(np.abs(T.z_U[S1])))
    T["called"] = T.q_U <= 0.05; T["inherited"] = T.h > 2.326
    tfcat = {l.strip() for l in open(TF_FILE) if l.strip()}
    excl = set(json.load(open(a.excl))["_union_all"])
    tech = lambda g: g in {"MALAT1", "NEAT1", "XIST", "FIRRE"} or g.startswith("MT-")
    T["filtered_out"] = ~T.gene.isin(tfcat) & (T.gene.isin(excl) | T.gene.map(tech))
    T.to_csv(f"{a.out}/pdf_procedure_all_candidate_pairs{a.tag}.csv", index=False)
    L = T[S1 & ~T.filtered_out].copy()
    L["rank_absU"] = L.U.abs().rank(ascending=False).astype(int); L["rank_absrho"] = L.rho.abs().rank(ascending=False).astype(int)
    L = L.reindex(L.z_U.abs().sort_values(ascending=False).index); L.insert(0, "rank_absz", range(1, len(L) + 1))
    L.to_csv(f"{a.out}/pdf_procedure_listed_pairs{a.tag}.csv", index=False)
    summ = dict(candidate_pairs=len(T), stage1_pairs=int(S1.sum()), removed_by_gene_filter=int((S1 & T.filtered_out).sum()), listed=len(L),
                called=int(L.called.sum()), inherited=int(L.inherited.sum()), N1=int(N1), N_t=int(Nt), pairs_within_clone=int(Pc.sum()))
    json.dump(summ, open(f"{a.out}/summary{a.tag}.json", "w"), indent=1); log(json.dumps(summ))


if __name__ == "__main__":
    main()
