#!/usr/bin/env python3
"""Exact implementation of the TwinScore procedure per TwinScore_procedure_and_top100.pdf
(2026-09-30), applied to hPSC_20260927's 592-edge ChIP-Atlas+Perturb-seq candidate panel
(489 genes). B = endo_T1 (scored sample), A = endo_T0 (earlier sample, direction only) --
matches the PDF's own "endo: T15 scored, T0 for direction" structure exactly (our dataset
only has two timepoints, endo_T0/endo_T1, which map onto their A/B directly).

Implements sections 1-8 and 11 verbatim (clone-weighted ranks w_i=1/n_c(i), self-twin
heritability h_x = C(x,x), closed-form sigma_rho/sigma_C/sigma_Delta, Stage I/II/III,
direction stage with a real D>=20-draw pooled permutation null). Sections 9-10 (shared/
unshared driver, jackknife over G=50 clone groups) are NOT implemented in this first pass
-- they only annotate fan-out pairs for Table 1/2/3 display, not the core TwinScore
ranking; flagged as a follow-up if needed.

Values v_gi = 1e4*x_gi/sum_g(x_gi) (CP10k, not log -- correlations are rank-based so this
is numerically identical to log1p(CP10k) for all rank statistics here).
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-10-01 added: replaces hardcoded project-root paths]
import json

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import norm

# QC_DIR = "/gpfs/projects/b1255/hzhang/TwINFER_KA/finalized_data/hPSC_20260927_data/qc_filtered"   # [2026-10-01 replaced by TWINFER_PROJECT_ROOT]
QC_DIR = f'{TWINFER_PROJECT_ROOT}/finalized_data/hPSC_20260927_data/qc_filtered'
# DATA_DIR = "/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/hPSC_20260927/data"   # [2026-10-01 replaced by TWINFER_PROJECT_ROOT]
DATA_DIR = f'{TWINFER_PROJECT_ROOT}/analysis_data/hPSC_20260927/data'
SEED = 101010
N_DRAWS = 20
H_THRESH = 2.326
Z_DAGGER_THRESH = 2.576
rng = np.random.default_rng(SEED)


def weighted_rank(v, w):
    """R_i = sum_{j:v_j<v_i} w_j + 0.5*sum_{j:v_j=v_i} w_j, per gene row. v,w: (n_cells,)."""
    order = np.argsort(v, kind="mergesort")
    v_sorted = v[order]
    w_sorted = w[order]
    cum = np.cumsum(w_sorted)
    # for each distinct value block, rank = (cum before block) + 0.5*(sum within block)
    n = len(v)
    R_sorted = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j < n and v_sorted[j] == v_sorted[i]:
            j += 1
        before = cum[i - 1] if i > 0 else 0.0
        within = w_sorted[i:j].sum()
        R_sorted[i:j] = before + 0.5 * within
        i = j
    R = np.empty(n)
    R[order] = R_sorted
    return R


def weighted_corr(u, v, w):
    """Weighted Pearson correlation. u,v,w: (n,) arrays (or (n,k) for u/v with shared w)."""
    wsum = w.sum()
    ubar = (w * u).sum(axis=0) / wsum
    vbar = (w * v).sum(axis=0) / wsum
    du, dv = u - ubar, v - vbar
    num = (w * (du * dv.T).T if du.ndim > 1 else w * du * dv).sum(axis=0) if False else None
    return num


def bh_q(p):
    """BH q-values; NaN p-values excluded from the family, returned as NaN."""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    finite = np.isfinite(p)
    n = finite.sum()
    if n == 0:
        return q
    idx = np.where(finite)[0]
    order = idx[np.argsort(p[idx])]
    ranked_p = p[order]
    q_sorted = np.minimum.accumulate((ranked_p * n / np.arange(1, n + 1))[::-1])[::-1]
    q[order] = np.clip(q_sorted, 0, 1)
    return q


def wcorr_vec(U, V, w):
    """U,V: (n_cells, k) matrices (k genes), w: (n_cells,). Returns (k,) weighted pearson corr
    of column i of U against column i of V (paired, elementwise per column), weights w shared."""
    wsum = w.sum()
    ubar = (w[:, None] * U).sum(axis=0) / wsum
    vbar = (w[:, None] * V).sum(axis=0) / wsum
    du, dv = U - ubar, V - vbar
    num = (w[:, None] * du * dv).sum(axis=0)
    den = np.sqrt((w[:, None] * du**2).sum(axis=0) * (w[:, None] * dv**2).sum(axis=0))
    return num / den


def main():
    genes_panel = json.load(open(f"{DATA_DIR}/hpsc_endoderm_final_gene_panel.json"))
    pairs_df = pd.read_csv(f"{DATA_DIR}/hpsc_endoderm_candidate_pairs_final.csv")
    print(f"candidate (listed) pairs: {len(pairs_df)}, panel genes: {len(genes_panel)}", flush=True)

    X = sio.mmread(f"{QC_DIR}/hPSC_20260927_qc_counts.mtx").tocsr()
    all_genes = np.array(open(f"{QC_DIR}/genes.txt").read().split())
    cells = np.array(open(f"{QC_DIR}/cells.txt").read().split())
    obs = pd.read_csv(f"{QC_DIR}/obs_metadata.csv", index_col=0)
    assert np.array_equal(cells, obs.index.to_numpy())

    gidx_all = {g: i for i, g in enumerate(all_genes)}
    panel_rows = [gidx_all[g] for g in genes_panel]
    cp10k_total = np.asarray(X.sum(axis=0)).ravel().astype(float)
    cp10k_total[cp10k_total == 0] = 1.0
    V_panel = (X[panel_rows, :].toarray().astype(float) / cp10k_total[None, :]) * 1e4  # genes x all_cells

    is_A = (obs["time_step"].to_numpy() == 0)  # endo_T0
    is_B = (obs["time_step"].to_numpy() == 1)  # endo_T1
    clone = obs["fatemap_clone_singletcode"].astype(str).to_numpy()

    V_A, V_B = V_panel[:, is_A], V_panel[:, is_B]
    clone_A, clone_B = clone[is_A], clone[is_B]
    print(f"A (endo_T0): {V_A.shape[1]} cells, B (endo_T1): {V_B.shape[1]} cells", flush=True)

    # ---- clone sizes ----
    sizeB = pd.Series(clone_B).value_counts()
    sizeA = pd.Series(clone_A).value_counts()
    n_c_B = sizeB.reindex(clone_B).to_numpy().astype(float)  # per-B-cell clone size
    n_c_A = sizeA.reindex(clone_A).to_numpy().astype(float)  # per-A-cell clone size
    N1 = len(sizeB)
    clones_with_twins_B = sizeB[sizeB >= 2].index
    Nt = len(clones_with_twins_B)
    clones_both = sorted(set(sizeA.index) & set(sizeB.index))
    Nc = len(clones_both)
    print(f"N1 (B clones)={N1}, Nt (B clones nc>=2)={Nt}, Nc (clones in both A&B)={Nc}", flush=True)

    # ---- Stage I weighted ranks in B (weight 1/n_c per B cell) ----
    w_B = 1.0 / n_c_B
    n_genes = V_B.shape[0]
    R_B = np.empty_like(V_B)
    for g in range(n_genes):
        R_B[g] = weighted_rank(V_B[g], w_B)

    # ---- A-local weighted ranks (weight 1/n_c^A per A cell) ----
    w_A = 1.0 / n_c_A
    R_A = np.empty_like(V_A)
    for g in range(n_genes):
        R_A[g] = weighted_rank(V_A[g], w_A)

    # ---- within-sample B twin pairs (clones with nc>=2) ----
    clone_to_bidx = {}
    for idx, c in enumerate(clone_B):
        clone_to_bidx.setdefault(c, []).append(idx)
    pair_a_idx, pair_b_idx, pair_w, pair_clone_size = [], [], [], []
    for c in clones_with_twins_B:
        idxs = clone_to_bidx[c]
        nc = len(idxs)
        Pc = nc * (nc - 1) / 2
        for i in range(nc):
            for j in range(i + 1, nc):
                pair_a_idx.append(idxs[i]); pair_b_idx.append(idxs[j])
                pair_w.append(1.0 / Pc); pair_clone_size.append(nc)
    pair_a_idx = np.array(pair_a_idx); pair_b_idx = np.array(pair_b_idx)
    w_p = np.array(pair_w); nc_p = np.array(pair_clone_size, dtype=float)
    P = len(pair_a_idx)
    print(f"within-B twin pairs: {P}", flush=True)

    # ---- centre Stage I ranks at twin-pair cells, per gene ----
    Ra_all = R_B[:, pair_a_idx]  # genes x P
    Rb_all = R_B[:, pair_b_idx]
    mu_g = (w_p * (Ra_all + Rb_all) / 2.0).sum(axis=1) / Nt  # genes,
    ra_all = Ra_all - mu_g[:, None]
    rb_all = Rb_all - mu_g[:, None]
    M_g = (w_p * (ra_all**2 + rb_all**2) / 2.0).sum(axis=1)  # genes,

    # ---- self-twin heritability h_x = C(x,x), floored at 0 (NaN/0 for M_g=0 genes --
    # zero expression variance among within-B twin cells, can't estimate heritability) ----
    gene_testable = M_g > 0
    M_g_safe = np.where(gene_testable, M_g, 1.0)
    h_gene = (w_p * ra_all * rb_all).sum(axis=1) / M_g_safe
    h_gene = np.where(gene_testable, h_gene, np.nan)
    h_gene_floored = np.clip(h_gene, 0, None)  # NaN propagates through np.clip
    gidx_panel = {g: i for i, g in enumerate(genes_panel)}

    # ---- restrict candidate pairs to those with both genes in panel ----
    pairs_df = pairs_df[pairs_df["TF"].isin(gidx_panel) & pairs_df["target"].isin(gidx_panel)].reset_index(drop=True)
    ix = pairs_df["TF"].map(gidx_panel).to_numpy()
    iy = pairs_df["target"].map(gidx_panel).to_numpy()
    N = len(pairs_df)
    print(f"candidate pairs with both genes in panel (N tested): {N}", flush=True)

    rax, rbx = ra_all[ix], rb_all[ix]
    ray, rby = ra_all[iy], rb_all[iy]
    Mx, My = M_g[ix], M_g[iy]
    hx, hy = h_gene_floored[ix], h_gene_floored[iy]

    # genes with zero expression variance across the (small) within-B twin-pair set have
    # M_g=0 -- undefined S/C/U for any pair involving them (not enough twin-cell signal to
    # test). Mark untestable, exclude from BH family below (noted in output as all-NaN rows).
    testable = (Mx > 0) & (My > 0)
    n_untestable = (~testable).sum()
    if n_untestable:
        print(f"WARNING: {n_untestable} candidate pairs untestable (M_g=0 for a gene with "
              f"zero expression among the {P} within-B twin-pair cells) -- excluded from "
              f"ranking/BH, left as NaN in output", flush=True)
    Mx_safe = np.where(testable, Mx, 1.0)
    My_safe = np.where(testable, My, 1.0)

    S = (w_p * (rax * ray + rbx * rby) / 2.0).sum(axis=1) / np.sqrt(Mx_safe * My_safe)
    C = (w_p * (rax * rby + rbx * ray) / 2.0).sum(axis=1) / np.sqrt(Mx_safe * My_safe)
    S = np.where(testable, S, np.nan)
    C = np.where(testable, C, np.nan)
    U = S - C
    dx, dy = rax - rbx, ray - rby
    denom = np.sqrt((w_p * dx**2).sum(axis=1) * (w_p * dy**2).sum(axis=1))
    denom_safe = np.where(denom > 0, denom, 1.0)
    rho_delta = (w_p * dx * dy).sum(axis=1) / denom_safe
    rho_delta = np.where(testable & (denom > 0), rho_delta, np.nan)

    # ---- closed-form null SDs (sum over distinct clone sizes nc>=2, grouped) ----
    uniq_nc, nc_count = np.unique(nc_p, return_counts=True)
    Pc_of_nc = uniq_nc * (uniq_nc - 1) / 2
    # map each twin-pair's clone back to its clone id to count distinct CLONES per nc (not pairs)
    pair_clone_id = np.array([c for c in clones_with_twins_B for _ in range(len(clone_to_bidx[c]) * (len(clone_to_bidx[c]) - 1) // 2)])
    clones_per_nc = {}
    for c in clones_with_twins_B:
        nc = len(clone_to_bidx[c])
        clones_per_nc[nc] = clones_per_nc.get(nc, 0) + 1

    def sigma_C2(S_pair, hx_pair, hy_pair):
        total = 0.0
        for nc, n_clones in clones_per_nc.items():
            Pc = nc * (nc - 1) / 2
            term = (1 + S_pair**2 + (nc - 2) * (hx_pair + hy_pair) + (nc**2 - 3 * nc + 3) * hx_pair * hy_pair) / (2 * Pc)
            total += n_clones * term
        return total / Nt**2

    sigma_Delta2 = sum(n_clones / (nc - 1) for nc, n_clones in clones_per_nc.items()) / Nt**2
    sigma_Delta = np.sqrt(sigma_Delta2)
    m_t = Nt**2 / sum(n_clones / (nc * (nc - 1) / 2) for nc, n_clones in clones_per_nc.items())

    sigma_C2_vec = np.array([sigma_C2(S[i], hx[i], hy[i]) for i in range(N)])
    sigma_C_vec = np.sqrt(sigma_C2_vec)

    h_pair = np.sign(S) * C / sigma_C_vec
    inherited = h_pair > H_THRESH

    z_U = rho_delta / sigma_Delta
    p_U = 2 * norm.sf(np.abs(z_U))
    q_U = bh_q(p_U)
    called = q_U <= 0.05

    # ---- Stage I rho (full-sample, all B cells, not just twins) ----
    Rx_full, Ry_full = R_B[ix], R_B[iy]
    rho_full = wcorr_vec(Rx_full.T, Ry_full.T, w_B)
    # sigma_rho^2 = (1/N1^2) sum_c [1/nc + (1-1/nc)*hx*hy]
    clone_sizes_B = sizeB.to_numpy().astype(float)
    base = (1.0 / clone_sizes_B).sum()
    extra_coeff = (1.0 - 1.0 / clone_sizes_B).sum()
    sigma_rho2 = (base + extra_coeff * (hx[:, None] * hy[:, None]).ravel()) / N1**2  # broadcast per pair
    # (hx,hy are per-pair scalars; base/extra_coeff are pair-independent sums over clones)
    sigma_rho2 = (base + extra_coeff * hx * hy) / N1**2
    z_rho = rho_full / np.sqrt(sigma_rho2)
    p_rho = 2 * norm.sf(np.abs(z_rho))
    q_rho = bh_q(p_rho)

    # ---- direction stage: cross-sample twin pairs (A,B same clone) ----
    bidx_by_clone = clone_to_bidx
    aidx_by_clone = {}
    for idx, c in enumerate(clone_A):
        aidx_by_clone.setdefault(c, []).append(idx)
    q_a_idx, q_b_idx, q_w = [], [], []
    for c in clones_both:
        a_list = aidx_by_clone[c]; b_list = bidx_by_clone.get(c, [])
        if not b_list:
            continue
        ncA, ncB = len(a_list), len(b_list)
        wq = 1.0 / (ncA * ncB)
        for a in a_list:
            for b in b_list:
                q_a_idx.append(a); q_b_idx.append(b); q_w.append(wq)
    q_a_idx = np.array(q_a_idx); q_b_idx = np.array(q_b_idx); w_q = np.array(q_w)
    Qn = len(q_a_idx)
    print(f"cross-sample twin pairs (direction): {Qn}", flush=True)

    m_c = Nc**2 / sum(1.0 / (len(aidx_by_clone[c]) * len(bidx_by_clone[c])) for c in clones_both)

    RA_q = R_A[:, q_a_idx]  # genes x Qn
    RB_q = R_B[:, q_b_idx]

    def rho_dagger(gi, gj):
        """rho^dagger(gi -> gj): corr_wq(R^A_gi,a, R^B_gj,b)."""
        return wcorr_vec(RA_q[gi][:, None], RB_q[gj][:, None], w_q)[0]

    rho_xy = np.array([rho_dagger(ix[i], iy[i]) for i in range(N)])
    rho_yx = np.array([rho_dagger(iy[i], ix[i]) for i in range(N)])
    delta = rho_xy - rho_yx
    z_dag_xy = rho_xy * np.sqrt(m_c - 1)
    z_dag_yx = rho_yx * np.sqrt(m_c - 1)
    coupled = np.maximum(np.abs(z_dag_xy), np.abs(z_dag_yx)) > Z_DAGGER_THRESH

    # ---- permutation null for delta: D draws, clone labels permuted so A cells pair with
    # B cells of a DIFFERENT clone; pooled over all N pairs ----
    clones_both_arr = np.array(clones_both)
    delta_null_pool = []
    for d in range(N_DRAWS):
        perm = rng.permutation(len(clones_both_arr))
        # avoid any fixed point (A clone mapped to itself)
        fixed = perm == np.arange(len(clones_both_arr))
        while fixed.any():
            swap_with = rng.permutation(len(clones_both_arr))
            idxs = np.where(fixed)[0]
            for k in idxs:
                perm[k] = swap_with[k]
            fixed = perm == np.arange(len(clones_both_arr))
        clone_map = dict(zip(clones_both_arr, clones_both_arr[perm]))

        qa2, qb2, w2 = [], [], []
        for c in clones_both:
            a_list = aidx_by_clone[c]; b_list = bidx_by_clone[clone_map[c]]
            ncA, ncB = len(a_list), len(b_list)
            wq = 1.0 / (ncA * ncB)
            for a in a_list:
                for b in b_list:
                    qa2.append(a); qb2.append(b); w2.append(wq)
        qa2 = np.array(qa2); qb2 = np.array(qb2); w2 = np.array(w2)
        RA2 = R_A[:, qa2]; RB2 = R_B[:, qb2]
        rho_xy_n = wcorr_vec(RA2[ix].T, RB2[iy].T, w2)
        rho_yx_n = wcorr_vec(RA2[iy].T, RB2[ix].T, w2)
        delta_null_pool.append(rho_xy_n - rho_yx_n)
        print(f"  direction null draw {d+1}/{N_DRAWS} done", flush=True)

    delta_null_pool = np.concatenate(delta_null_pool)  # size D*N
    abs_delta = np.abs(delta)
    abs_pool = np.abs(delta_null_pool)
    # p_delta_i = (1 + #{|delta_null| >= |delta_i|}) / (D*N+1), pooled over ALL draws+pairs
    sorted_pool = np.sort(abs_pool)
    finite_delta = np.isfinite(abs_delta)
    counts_ge = np.full(N, np.nan)
    counts_ge[finite_delta] = len(sorted_pool) - np.searchsorted(sorted_pool, abs_delta[finite_delta], side="left")
    p_delta = np.where(finite_delta, (1 + counts_ge) / (N_DRAWS * N + 1), np.nan)
    q_delta = bh_q(p_delta)
    directed = q_delta <= 0.05  # NaN-safe: comparison with NaN is False
    regulator_is_x = np.sign(delta) == np.sign(U)  # NaN-safe: False when either is NaN/untestable
    direction_label = np.where(directed & regulator_is_x, pairs_df["TF"] + "->" + pairs_df["target"],
                                np.where(directed & ~regulator_is_x, pairs_df["target"] + "->" + pairs_df["TF"], ""))
    reg_type = np.where(directed, np.where(U > 0, "activation", "repression"), "")
    symmetric = coupled & ~directed & finite_delta

    untestable = ~testable
    scenario = np.where(
        untestable, "untestable (zero expression among within-B twin cells)",
        np.where(called & directed, "regulation",
                 np.where(symmetric & inherited, "fan-out (shared driver untested)",
                          np.where(symmetric & ~inherited, "unshared fan-out (driver untested)",
                                   "regulation not established"))))

    out = pairs_df.copy()
    out["rho"] = rho_full; out["z_rho"] = z_rho; out["q_rho"] = q_rho
    out["S"] = S; out["C"] = C; out["U"] = U
    out["hx"] = hx; out["hy"] = hy
    out["rho_delta"] = rho_delta; out["z_U"] = z_U; out["q_U"] = q_U; out["called"] = called
    out["h_pair"] = h_pair; out["inherited_state"] = inherited
    out["rho_dag_xy"] = rho_xy; out["rho_dag_yx"] = rho_yx; out["delta"] = delta
    out["z_dag_xy"] = z_dag_xy; out["z_dag_yx"] = z_dag_yx; out["coupled"] = coupled
    out["p_delta"] = p_delta; out["q_delta"] = q_delta; out["directed"] = directed
    out["direction"] = direction_label
    out["type"] = reg_type
    out["scenario"] = scenario
    out["rank_by_zU"] = pd.Series(np.abs(z_U)).rank(ascending=False, method="min")
    out["rank_by_rho"] = pd.Series(np.abs(rho_full)).rank(ascending=False, method="min")
    out = out.sort_values("rank_by_zU", na_position="last")
    out.to_csv(f"{DATA_DIR}/hpsc_endoderm_twinscore_exact.csv", index=False)

    print(f"\nN tested = {N}", flush=True)
    print(f"Stage I (rho) called at q<=0.05: {(q_rho<=0.05).sum()}", flush=True)
    print(f"Stage III called (q_U<=0.05): {called.sum()}", flush=True)
    print(f"Inherited state (h>{H_THRESH}): {inherited.sum()}", flush=True)
    print(f"Coupled (max|z_dag|>{Z_DAGGER_THRESH}): {coupled.sum()}", flush=True)
    print(f"Directed (q_delta<=0.05): {directed.sum()}", flush=True)
    print(f"Scenario counts:\n{pd.Series(scenario).value_counts()}", flush=True)
    print(f"\ntop 20 by |z_U|:", flush=True)
    print(out[["TF", "target", "z_U", "q_U", "called", "inherited_state", "directed", "direction", "type"]]
          .head(20).to_string(index=False), flush=True)
    print(f"\nwrote {DATA_DIR}/hpsc_endoderm_twinscore_exact.csv", flush=True)


if __name__ == "__main__":
    main()
