#!/usr/bin/env python3
"""2026-09-22 user request: rerun TwinScore_supplement (the formula WITH phi/persistence,
distinct from the default package twinScore) under both the pooled (t1=t2) and real A/B-split
(replicate A=t1, B=t2) constructions, with no gates and no NaN-placeholder-filling -- matching
the "no gates/nan, recalculate if they do not exist" standard already applied to the default
twinScore investigation in this session.

Two gates removed vs apply_twinscore_supplement_fatemap.py:
  1. persistence()'s heritability significance threshold (h1g>thr1 and h2g>thr2) -- previously
     genes failing it got the PANEL MEDIAN as a placeholder phi value, not a real number. Removed:
     phi is now computed for every gene where h1g>0 and h2g>0 (the only mathematically necessary
     condition for sqrt(h1g*h2g) to be real and positive), and left NaN only when truly undefined.
  2. z_dagger_map completeness: the original script reads z_dagger_{label}_{gs}.json, which was
     built from a GATED run_infer_fatemap.py run (only pairs that passed Stage 3's one-sided
     z_reg_gated test -- see conversation -- reach Step 4 and get a z_dagger entry). This reads
     the new z_dagger_{label}_{gs}_[absplit_]allpairs.json instead (from run_infer_fatemap_allpairs.py
     with bypass_stage3_gate=True), which has full (or near-full) coverage.

Imports everything else (S/zhet, sister/cross correlation, D/PIDC, Wz, clr_calibrate, signal_share,
s) unchanged from the original module.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apply_twinscore_supplement_fatemap as supp

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/home/gzu5140/TwINFER_KA/code/TwINFER/package")
from twinfer.inference.correlation_functions import calculate_pairwise_gene_gene_correlation_matrix

log = supp.log
s = supp.s
signal_share = supp.signal_share
clr_calibrate = supp.clr_calibrate
null_sd = supp.null_sd  # 2026-09-23: analytic 1/sqrt(meff-1) formula. Checked directly against a real
# clone-label-permutation null on FM06/correlation_high and FM08/correlation_mid: underestimates the
# true noise floor by 1.5x-2.9x (worse for the smaller-sample dataset). Kept only as a fallback for
# genes/pairs where the bootstrap below can't be computed (e.g. all-NaN draws) -- see conversation.
# Every live use of null_sd() in this file has been replaced by bootstrap_null_sd() below.
m_eff_step1 = supp.m_eff_step1
compute_zhet_zrho = supp.compute_zhet_zrho
same_cell_matrix = supp.same_cell_matrix
sister_matrix_from_frame = supp.sister_matrix_from_frame
cross_matrix = supp.cross_matrix
bootstrap_phi_noise_floor = supp.bootstrap_phi_noise_floor
compute_D = supp.compute_D
compute_Wz = supp.compute_Wz
R0 = supp.R0
Z_DIRECTION_GATE = supp.Z_DIRECTION_GATE
MAX_CLONE_SIZE = supp.MAX_CLONE_SIZE
SEED = supp.SEED


PHI_NEGFLOOR_DATASET_NOTES = {
    "FM01": "* FM01 A/B split rests on few clones: 94 clones / 1,555 cells (A 716, B 839) span both replicates, with no\n"
            "  per-side clone-size cap (largest side 138 cells); FM06 has 2,101 clones. Expect noisy phi.\n",
}

PHI_NEGFLOOR_README = """# TwinScore_supplement, phi "neg-only floor" variant

Everything in this folder was produced with `apply_twinscore_supplement_fatemap_allpairs.py --phi-neg-floor-only`
(and `--absplit` where the file name says `absplit`). Nothing here is mixed with the default (v3) outputs, which
live one level up in `../` (no `_phinegfloor` tag, no subfolder).

## What is computed
TwinScore_supplement = w_rel * s(phi_x) + s(PAIR) + direction_term, for every ordered gene pair (a, b) in a gene set
(no gates), with PAIR = s(D) + gate_g * s(R) + gate_g * v_gate * s(Wz)  (D = PIDC, R = CLR-calibrated
S - lambda*C, Wz = split-half partial correlation). `s()` is a z-score across pairs; NaN gets (min - 1).
phi_x = phi[a] is a GENE-level persistence term (the source gene's within-clone cross-cell correlation
relative to its heritability), repeated for every pair starting at gene a. In an A/B split, A = t1 and B = t2.

phi_g = rho_dagger_gg / sqrt(h1_used * h2_used), clipped to +/-10, where h1/h2 = heritability of gene g at t1/t2:
  * h <= 0  -> replaced by that gene's own bootstrap null SD (clone-label permutation), THAT SIDE ONLY
  * h > 0   -> used as is: NEVER floored, however small
  * every gene therefore gets a real phi (NaN only if h or rho_dagger_gg is itself non-finite)

## Why this variant
The default (v3) sets phi = NaN whenever either h <= 0 and floors every positive h at its noise floor. On the
FM06/FM08 A/B runs that left phi NaN for 23% / 42% of genes (h estimated <= 0 from noise), and `s()` scores a NaN
below every finite value, so all pairs of those genes were pushed to the bottom of the ranking. This variant keeps
those genes in play using a noise-scale denominator instead of dropping them, and does not inflate positive-but-small h.

## Caveats
* Discontinuity: h = -0.05 uses the null SD (~0.03) as its denominator, while h = +0.001 uses 0.001, so the
  barely-positive gene gets the larger |phi|. The +/-10 clip is the only bound on that. The run log reports how many
  positive h values fell below their own noise floor and were left unfloored.
{DATASET_NOTES}* Clone label = concatenation of every barcode a cell carries (as in FM06/FM08), so cells sharing one barcode but
  differing in another are split into different "clones" (in FM01, 11% of cells carry 2-5 barcodes).
* Results are not directly comparable with FM06/FM08 files made with v3 unless those are rerun with the same flag.
* w_rel (reliability weight) is computed identically to v3 from the spread of phi.

Per-gene-set settings and diagnostics: `run_params_*.json`.
"""


def bootstrap_null_sd_diag(raw_t, genes, seed, batch=30, min_batches=3, max_perm=210, tol=0.05):
    """Empirical null SD of the heritability diagonal h_g, via clone-label permutation (breaks
    real clonal structure -> true null of 'no heritable signal'). Replaces null_sd(meff_twin) for
    the phi noise floor -- verified null_sd underestimates this by 1.5x (FM06) to 2.9x (FM08).
    Runs in batches of `batch` permutations, stops once the per-gene SD estimate stops moving by
    more than `tol` (relative) between batches, capped at max_perm. Returns {gene: empirical_sd}."""
    rng = np.random.default_rng(seed)
    draws = {g: [] for g in genes}
    prev = None
    n_done = 0
    while n_done < max_perm:
        for _ in range(batch):
            shuffled = raw_t.copy()
            shuffled["clone_id"] = rng.permutation(shuffled["clone_id"].to_numpy())
            try:
                C_perm, _ = sister_matrix_from_frame(shuffled, genes)
            except ValueError:
                continue
            for g in genes:
                v = C_perm.loc[g, g]
                if np.isfinite(v):
                    draws[g].append(v)
        n_done += batch
        cur = {g: (np.std(draws[g], ddof=1) if len(draws[g]) > 5 else np.nan) for g in genes}
        if n_done >= min_batches * batch and prev is not None:
            finite_pairs = [(cur[g], prev[g]) for g in genes if np.isfinite(cur[g]) and np.isfinite(prev[g])]
            if finite_pairs:
                rel = np.median([abs(c - p) / p for c, p in finite_pairs if p > 0])
                if rel < tol:
                    prev = cur
                    break
        prev = cur
    return prev, n_done


def bootstrap_null_sd_offdiag(raw_t, genes, seed, kind, extra=None, batch=8, min_batches=2, max_perm=48, tol=0.03):
    """Empirical null SD for an off-diagonal matrix statistic, via clone-label permutation. Each
    permutation already yields O(n_genes^2) pair values, so far fewer permutation draws are needed
    than the diagonal case above for the same precision. kind='C' -> sister off-diagonal (replaces
    null_sd(meff_twin_t2) for sd_twin_t2/gate_g). kind='reg' -> S_perm - lam*C_perm numerator
    (replaces null_sd(meff_step1) for sd_reg/R); `extra` must be the real (non-permuted) lambda
    shrinkage matrix in that case -- lambda itself isn't being null-tested, just reused as a fixed
    weight. Returns (scalar empirical SD, n_perm_used)."""
    rng = np.random.default_rng(seed)
    off_mask = ~np.eye(len(genes), dtype=bool)
    vals = []
    n_done = 0
    prev_sd = None
    while n_done < max_perm:
        for _ in range(batch):
            shuffled = raw_t.copy()
            shuffled["clone_id"] = rng.permutation(shuffled["clone_id"].to_numpy())
            try:
                C_perm, _ = sister_matrix_from_frame(shuffled, genes)
                if kind == "C":
                    v = C_perm.to_numpy()[off_mask]
                else:
                    S_perm = same_cell_matrix(shuffled, genes)
                    num = S_perm.to_numpy() - extra.to_numpy() * C_perm.to_numpy()
                    v = num[off_mask]
            except ValueError:
                continue
            vals.extend(v[np.isfinite(v)].tolist())
        n_done += batch
        cur_sd = np.std(vals, ddof=1) if len(vals) > 5 else np.nan
        if n_done >= min_batches * batch and prev_sd is not None and np.isfinite(cur_sd) and prev_sd > 0:
            if abs(cur_sd - prev_sd) / prev_sd < tol:
                prev_sd = cur_sd
                break
        prev_sd = cur_sd
    return prev_sd, n_done


def persistence_no_gate(h1, h2, rho_dagger, genes, noise_floor_h1_g, noise_floor_h2_g, t1_raw, t2_raw,
                        neg_floor_only=False):
    """Same as supp.persistence but WITHOUT the h1g>thr1/h2g>thr2 significance gate --
    real phi wherever mathematically defined (h1g>0 and h2g>0), no panel-median placeholder.

    2026-09-22 fix, v2: phi's denominator sqrt(h1g*h2g) is unstable when h1g/h2g are small
    relative to THEIR OWN estimation noise, not relative to the panel median (v1 used
    0.05*panel_median, which never triggered). Floor h1g/h2g individually at their own noise
    floor before combining them.

    2026-09-23 fix, v3: v2 used the analytic null_sd(meff_twin) as that per-gene noise floor.
    Checked directly against a real clone-label-permutation null (see conversation): null_sd
    underestimates the true noise floor by 1.5x (FM06) to 2.9x (FM08) -- worse for the
    smaller-sample dataset. `noise_floor_h1_g`/`noise_floor_h2_g` are now the genuine per-gene
    bootstrap noise floors (bootstrap_null_sd_diag, clone-label permutation), passed in from
    run_gene_set. Falls back to the old analytic null_sd for any gene the bootstrap couldn't
    resolve (e.g. too few finite permutation draws)."""
    PHI_CLIP = 10.0
    phi = {}
    n_floored = 0
    if neg_floor_only:
        # 2026-09-23 "neg-only floor" variant (user spec): only a NON-POSITIVE h is replaced, by its own
        # bootstrap null SD, and only that side; a positive h is never floored, however tiny; the other
        # side is used as is. phi = rho_dagger_gg / sqrt(h1_used * h2_used), clipped to +/-PHI_CLIP.
        # Every gene gets a real phi (NaN only if h or rho_dagger_gg itself is non-finite or the null SD is <= 0).
        n_rep1 = n_rep2 = n_pos_below_nf = n_nan = 0
        for g in genes:
            h1g, h2g, rdg = h1[g], h2[g], rho_dagger[g]
            nf1, nf2 = noise_floor_h1_g[g], noise_floor_h2_g[g]
            if not (np.isfinite(h1g) and np.isfinite(h2g) and np.isfinite(rdg)):
                phi[g] = np.nan; n_nan += 1
                continue
            h1_used, h2_used = h1g, h2g
            if h1g <= 0:
                h1_used = nf1; n_rep1 += 1
            elif h1g < nf1:
                n_pos_below_nf += 1
            if h2g <= 0:
                h2_used = nf2; n_rep2 += 1
            elif h2g < nf2:
                n_pos_below_nf += 1
            denom = np.sqrt(h1_used * h2_used) if (h1_used > 0 and h2_used > 0) else np.nan
            phi_raw_g = rdg / denom if np.isfinite(denom) and denom > 0 else np.nan
            phi[g] = float(np.clip(phi_raw_g, -PHI_CLIP, PHI_CLIP)) if np.isfinite(phi_raw_g) else np.nan
            n_nan += int(not np.isfinite(phi[g]))
        vals = np.array([v for v in phi.values() if np.isfinite(v)])
        log(f"    phi (neg-only floor): h1<=0 replaced by null SD for {n_rep1}/{len(genes)} genes, "
            f"h2<=0 for {n_rep2}/{len(genes)}; {n_pos_below_nf} positive h side(s) below own noise floor left unfloored; "
            f"{n_nan} NaN; |phi|>={PHI_CLIP} (clipped): {int((np.abs(vals) >= PHI_CLIP).sum())}/{len(vals)}")
        return _finish_phi(phi, genes, noise_floor_h1_g, noise_floor_h2_g, t1_raw, t2_raw)
    for g in genes:
        h1g, h2g, rdg = h1[g], h2[g], rho_dagger[g]
        ok = np.isfinite(h1g) and np.isfinite(h2g) and h1g > 0 and h2g > 0
        if not ok:
            phi[g] = np.nan
            continue
        nf1, nf2 = noise_floor_h1_g[g], noise_floor_h2_g[g]
        h1_safe = max(h1g, nf1)
        h2_safe = max(h2g, nf2)
        n_floored += int(h1g < nf1 or h2g < nf2)
        denom = np.sqrt(h1_safe * h2_safe)
        phi_raw_g = rdg / denom if denom > 0 else np.nan
        phi[g] = float(np.clip(phi_raw_g, -PHI_CLIP, PHI_CLIP)) if np.isfinite(phi_raw_g) else np.nan
    n_nan = sum(1 for v in phi.values() if not np.isfinite(v))
    if n_nan:
        log(f"    phi: {n_nan}/{len(genes)} genes genuinely undefined (h1<=0 or h2<=0), left as NaN")
    if n_floored:
        log(f"    phi: {n_floored}/{len(genes)} genes had h below its own bootstrap noise floor, "
            f"floored+clipped to +/-{PHI_CLIP}")
    return _finish_phi(phi, genes, noise_floor_h1_g, noise_floor_h2_g, t1_raw, t2_raw)


def _finish_phi(phi, genes, noise_floor_h1_g, noise_floor_h2_g, t1_raw, t2_raw):
    """Shared tail of persistence_no_gate (both modes): reliability weight w_rel from the spread of phi."""
    phi_vals = np.array([v for v in phi.values() if np.isfinite(v)])
    var_phi = phi_vals.var(ddof=1) if len(phi_vals) > 1 else np.nan
    thr1 = {g: 2.0 * noise_floor_h1_g[g] for g in genes}
    thr2 = {g: 2.0 * noise_floor_h2_g[g] for g in genes}
    # bootstrap_phi_noise_floor takes a single scalar thr1/thr2 (median gate) -- kept as-is, only
    # feeds the separate w_rel reliability weight, not phi itself.
    noise_var = bootstrap_phi_noise_floor(t1_raw, t2_raw, genes,
                                           float(np.median(list(thr1.values()))),
                                           float(np.median(list(thr2.values()))))
    floor = float(np.median(list(noise_var.values()))) if noise_var else 0.0
    w = max(0.0, 1.0 - floor / var_phi) if (np.isfinite(var_phi) and var_phi > 0) else 0.0
    return phi, w


def load_raw_pooled(dataset, gene_set_name):
    return supp.load_raw(dataset, gene_set_name)


def load_raw_ab(dataset, gene_set_name):
    """Real A/B split: replicate A=t1(0), B=t2(1); only clones present in both, same
    MAX_CLONE_SIZE-per-side cap as run_infer_fatemap_ab_split.py."""
    label = dataset.lower()
    qc_dir = f"{TWINFER_PROJECT_ROOT}/finalized_data/{dataset}_data/qc_filtered"
    gene_sets = json.load(open(f"{TWINFER_PROJECT_ROOT}/analysis_data/{label}/data/gene_sets_{label}.json"))
    genes = sorted(gene_sets[gene_set_name])

    import scipy.io as sio
    X = sio.mmread(f"{qc_dir}/{label}_qc_counts.mtx").tocsr()
    genes_full = np.array(open(f"{qc_dir}/genes.txt").read().split())
    obs = pd.read_csv(f"{qc_dir}/obs_metadata.csv", index_col=0)
    assert X.shape[0] == len(obs)
    gidx = {g: i for i, g in enumerate(genes_full)}
    col = [gidx[g] for g in genes]
    cell_total = np.asarray(X.sum(axis=1)).ravel().astype(float)
    mat = np.log1p(X[:, col].toarray().astype(float) / np.where(cell_total == 0, 1, cell_total)[:, None] * 1e4)

    df = pd.DataFrame(mat, columns=[f"{g}_mRNA" for g in genes])
    df.insert(0, "time_step", (obs["replicate"].to_numpy() == "B").astype(int))
    df.insert(0, "cell_id", obs.index.to_numpy())
    clone = obs["fatemap_clone_singletcode"].astype(str)
    df.insert(0, "clone_id", clone.to_numpy())
    df = df[clone.str.len().to_numpy() > 0].reset_index(drop=True)

    span = df.groupby("clone_id")["time_step"].nunique()
    spanning = span[span == 2].index
    n_before = len(df)
    df = df[df["clone_id"].isin(spanning)].reset_index(drop=True)

    from paper_analysis.fatemap_pipeline.run_infer_fatemap_ab_split import ab_max_clone_size
    cap = ab_max_clone_size(dataset)  # None = no cap (FM01), else MAX_CLONE_SIZE
    if cap is not None:
        per_side = df.groupby(["clone_id", "time_step"]).size()
        oversized = per_side[per_side > cap].index.get_level_values("clone_id").unique()
        df = df[~df["clone_id"].isin(oversized)].reset_index(drop=True)
    log(f"[{dataset}/{gene_set_name}] A/B-spanning-clone filter: {n_before:,} -> {len(df):,} cells "
        f"({df['clone_id'].nunique():,} clones, t1=A: {(df.time_step==0).sum():,}, "
        f"t2=B: {(df.time_step==1).sum():,})")
    return df, genes, qc_dir, genes_full, gidx


def run_gene_set(dataset, gs, ce_edges, absplit, n_shuffles=2000, n_cores=8, compute_wz=True, phi_neg_floor_only=False):
    label = dataset.lower()
    if absplit:
        full, genes, qc_dir, genes_full, gidx = load_raw_ab(dataset, gs)
        t1_raw = full[full.time_step == 0].reset_index(drop=True)
        t2_raw = full[full.time_step == 1].reset_index(drop=True)
        raw = full  # used for zhet/S (same-cell, pooled-timepoint stat -- no t1/t2 distinction there)
    else:
        raw, genes, qc_dir, genes_full, gidx = load_raw_pooled(dataset, gs)
        t1_raw = t2_raw = raw

    log(f"[{dataset}/{gs}] real S, z_het (fresh permutation, this dataset)")
    zhet_matrix, S, twin_delta = compute_zhet_zrho(raw, genes, n_shuffles=n_shuffles, n_cores=n_cores)
    S_t1 = S_t2 = S

    log(f"[{dataset}/{gs}] C, heritability (fresh weighted-Spearman)")
    C_t1, meff_twin_t1 = sister_matrix_from_frame(t1_raw, genes)
    if absplit:
        C_t2, meff_twin_t2 = sister_matrix_from_frame(t2_raw, genes)
    else:
        C_t2, meff_twin_t2 = C_t1, meff_twin_t1
    h_t1 = {g: C_t1.loc[g, g] for g in genes}
    h_t2 = {g: C_t2.loc[g, g] for g in genes}

    log(f"[{dataset}/{gs}] cross-cell (within-clone sibling) correlation -- persistence numerator")
    rho_dagger, meff_cross_ = cross_matrix(t1_raw, t2_raw, genes)
    rho_dagger_diag = {g: rho_dagger.loc[g, g] for g in genes}

    log(f"[{dataset}/{gs}] bootstrap noise floors (clone-label permutation null, replaces null_sd everywhere)")
    boot_h1, n_boot_h1 = bootstrap_null_sd_diag(t1_raw, genes, seed=SEED)
    boot_h2, n_boot_h2 = bootstrap_null_sd_diag(t2_raw, genes, seed=SEED + 1)
    fallback_h1, fallback_h2 = null_sd(meff_twin_t1), null_sd(meff_twin_t2)
    noise_floor_h1_g = {g: (boot_h1[g] if np.isfinite(boot_h1[g]) else fallback_h1) for g in genes}
    noise_floor_h2_g = {g: (boot_h2[g] if np.isfinite(boot_h2[g]) else fallback_h2) for g in genes}
    n_fallback = sum(1 for g in genes if not np.isfinite(boot_h1[g]) or not np.isfinite(boot_h2[g]))
    log(f"    phi h-floor: {n_boot_h1}/{n_boot_h2} bootstrap perms (t1/t2), "
        f"{n_fallback}/{len(genes)} genes fell back to analytic null_sd")

    log(f"[{dataset}/{gs}] persistence phi_g (NO GATE), reliability w")
    phi, w_rel = persistence_no_gate(h_t1, h_t2, rho_dagger_diag, genes, noise_floor_h1_g, noise_floor_h2_g, t1_raw, t2_raw,
                                neg_floor_only=phi_neg_floor_only)

    log(f"[{dataset}/{gs}] zreg, R, twin-signal gate g")
    n_genes = len(genes)
    meff_step1 = m_eff_step1(raw.groupby("clone_id").size().to_numpy())  # kept for diagnostics only
    lam = np.minimum(1.0, zhet_matrix.abs() / 2.33)
    sd_reg, n_boot_reg = bootstrap_null_sd_offdiag(t2_raw, genes, seed=SEED + 2, kind="reg", extra=lam)
    sd_twin_t2, n_boot_twin = bootstrap_null_sd_offdiag(t2_raw, genes, seed=SEED + 3, kind="C")
    if not np.isfinite(sd_reg) or sd_reg <= 0:
        sd_reg = null_sd(m_eff_step1(raw.groupby("clone_id").size().to_numpy()))
    if not np.isfinite(sd_twin_t2) or sd_twin_t2 <= 0:
        sd_twin_t2 = null_sd(meff_twin_t2)
    log(f"    sd_reg (bootstrap, {n_boot_reg} perms) = {sd_reg:.4g}, "
        f"sd_twin_t2 (bootstrap, {n_boot_twin} perms) = {sd_twin_t2:.4g}")
    zreg = (S_t2 - lam * C_t2) / sd_reg
    R_fn = clr_calibrate(zreg, genes)
    zC = C_t2 / sd_twin_t2
    gate_g = signal_share(zC.to_numpy()[~np.eye(n_genes, dtype=bool)])

    if compute_wz:
        log(f"[{dataset}/{gs}] D (real PIDC) and Wz (split-half partial correlation)")
    else:
        log(f"[{dataset}/{gs}] D (real PIDC)")
    D_mat = compute_D(t2_raw, genes)
    if compute_wz:
        Wz_mat, v_gate, a_star = compute_Wz(raw, genes, qc_dir, label, genes_full, gidx)
    else:
        Wz_mat = pd.DataFrame(0.0, index=genes, columns=genes)
        v_gate, a_star = 0.0, np.nan

    log(f"[{dataset}/{gs}] direction: z^dagger (from allpairs/gate-bypassed json), gamma, kappa_gamma, q")
    suffix = "_absplit_allpairs" if absplit else "_allpairs"
    z_dagger_path = (f"{TWINFER_PROJECT_ROOT}/analysis_data/{label}/data/"
                      f"z_dagger_{label}_{gs}{suffix}.json")
    z_dagger_map = json.load(open(z_dagger_path)) if os.path.exists(z_dagger_path) else {}
    log(f"    z_dagger coverage: {len(z_dagger_map)} entries from {z_dagger_path}")
    U = [(a, b) for a in genes for b in genes if a != b]

    def zdag(a, b):
        return z_dagger_map.get(f"{a}__{b}", np.nan)

    z_dag_xy = np.array([zdag(a, b) for a, b in U])
    z_dag_yx = np.array([zdag(b, a) for a, b in U])
    gamma_raw = (np.abs(z_dag_xy) - np.abs(z_dag_yx)) / np.sqrt(2 * (1 - 2 / np.pi))
    kappa_gamma = signal_share(gamma_raw)
    q = 0.5 + (R0 - 0.5) * (2 * norm.cdf(np.abs(gamma_raw)) - 1) * np.sign(gamma_raw)
    q = np.clip(np.nan_to_num(q, nan=0.5), 1e-6, 1 - 1e-6)
    gate_dir = (np.nan_to_num(np.maximum(np.abs(z_dag_xy), np.abs(z_dag_yx)), nan=0.0)
                > Z_DIRECTION_GATE).astype(float)
    direction_term = kappa_gamma * gate_dir * np.log(q)

    log(f"[{dataset}/{gs}] assembling PAIR and TwinScore")
    D_vec = np.array([D_mat.loc[a, b] for a, b in U])
    R_vec = np.array([R_fn(a, b) for a, b in U])
    Wz_vec = np.array([Wz_mat.loc[a, b] for a, b in U])
    PAIR = s(D_vec) + gate_g * s(R_vec) + gate_g * v_gate * s(Wz_vec)
    phi_x = np.array([phi[a] for a, b in U])
    score = w_rel * s(phi_x) + s(PAIR) + direction_term

    pair_df = pd.DataFrame({
        "gene_1": [a for a, b in U], "gene_2": [b for a, b in U],
        "S": [S_t2.loc[a, b] for a, b in U], "C": [C_t2.loc[a, b] for a, b in U],
        "zhet": [zhet_matrix.loc[a, b] for a, b in U], "lambda": [lam.loc[a, b] for a, b in U],
        "zreg": [zreg.loc[a, b] for a, b in U], "R": R_vec, "gate_g": gate_g,
        "D": D_vec, "Wz": Wz_vec, "gate_v": v_gate,
        "rho_dagger_xy": [rho_dagger.loc[a, b] for a, b in U],
        "rho_dagger_yx": [rho_dagger.loc[b, a] for a, b in U],
        "z_dagger_xy": z_dag_xy, "z_dagger_yx": z_dag_yx, "gamma": gamma_raw, "kappa_gamma": kappa_gamma,
        "q": q, "direction_term": direction_term, "PAIR": PAIR, "TwinScore": score,
        "phi_x": phi_x,
        "collectri_edge": [1 if p in ce_edges else 0 for p in U],
    })
    gene_df = pd.DataFrame({
        "gene": genes, "h_t1": [h_t1[g] for g in genes], "h_t2": [h_t2[g] for g in genes],
        "rho_dagger_gg": [rho_dagger_diag[g] for g in genes], "phi": [phi[g] for g in genes],
    })
    diagnostics = dict(dataset=dataset, gs=gs, absplit=absplit, n_genes=n_genes, n_pairs=len(U),
                        meff_twin_t1=meff_twin_t1, meff_twin_t2=meff_twin_t2, meff_cross=meff_cross_,
                        meff_step1=meff_step1, w=w_rel, gate_g=gate_g, gate_v=v_gate,
                        kappa_gamma=kappa_gamma, wz_shrinkage_a=a_star,
                        n_phi_nan=int(sum(1 for v in phi.values() if not np.isfinite(v))),
                        z_dagger_coverage=len(z_dagger_map))
    return pair_df, gene_df, diagnostics


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["FM01", "FM06", "FM08"])
    ap.add_argument("--gene-set", default="correlation_high")
    ap.add_argument("--n-shuffles", type=int, default=500)
    ap.add_argument("--n-cores", type=int, default=8)
    ap.add_argument("--absplit", action="store_true")
    ap.add_argument("--no-wz", action="store_true")
    ap.add_argument("--phi-neg-floor-only", action="store_true",
                    help="phi: only a non-positive h is replaced (by its bootstrap null SD, that side only); positive h never floored; clip +/-10; outputs go to <data>/twinscore_supp_phinegfloor/ with a README and run_params json")
    args = ap.parse_args()

    ct = pd.read_csv(supp.COLLECTRI_PATH, sep="\t")
    ct = ct[ct.source_genesymbol != ct.target_genesymbol]
    CE = set(zip(ct.source_genesymbol, ct.target_genesymbol))

    label = args.dataset.lower()
    out_dir = f"{TWINFER_PROJECT_ROOT}/analysis_data/{label}/data"
    if args.phi_neg_floor_only:
        out_dir = os.path.join(out_dir, "twinscore_supp_phinegfloor")  # own folder, never mixed with v3 outputs
    os.makedirs(out_dir, exist_ok=True)
    if args.phi_neg_floor_only:
        with open(os.path.join(out_dir, "README.md"), "w") as fh:
            fh.write(PHI_NEGFLOOR_README.replace("{DATASET_NOTES}", PHI_NEGFLOOR_DATASET_NOTES.get(args.dataset, "")))
    suffix = "_absplit_allpairs" if args.absplit else "_allpairs"

    pair_df, gene_df, diag = run_gene_set(
        args.dataset, args.gene_set, CE, args.absplit, n_shuffles=args.n_shuffles,
        n_cores=args.n_cores, compute_wz=not args.no_wz, phi_neg_floor_only=args.phi_neg_floor_only,
    )
    out_suffix = suffix + ("_phinegfloor" if args.phi_neg_floor_only else "")
    pair_out = os.path.join(out_dir, f"twinscore_supplement_{label}_{args.gene_set}{out_suffix}_pair_terms.csv")
    gene_out = os.path.join(out_dir, f"twinscore_supplement_{label}_{args.gene_set}{out_suffix}_gene_terms.csv")
    pair_df.to_csv(pair_out, index=False)
    gene_df.to_csv(gene_out, index=False)
    log(f"wrote {pair_out}")
    log(f"wrote {gene_out}")
    log(f"diagnostics: {diag}")
    import datetime
    params = dict(dataset=args.dataset, gene_set=args.gene_set, absplit=args.absplit, n_shuffles=args.n_shuffles,
                  compute_wz=not args.no_wz, phi_mode="neg_floor_only" if args.phi_neg_floor_only else "v3",
                  phi_clip=10.0, script=os.path.basename(__file__),
                  z_dagger_input=f"{TWINFER_PROJECT_ROOT}/analysis_data/{label}/data/z_dagger_{label}_{args.gene_set}{suffix}.json",
                  written=datetime.datetime.now().isoformat(timespec="seconds"), diagnostics=diag)
    with open(os.path.join(out_dir, f"run_params_{label}_{args.gene_set}{out_suffix}.json"), "w") as fh:
        json.dump(params, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
