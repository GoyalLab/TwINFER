#!/usr/bin/env python3
"""TwinScore_supplement, FINAL gated + bootstrapped version, PARALLEL-BOOTSTRAP variant (2026-09-23).

Same method and same numbers as apply_twinscore_supplement_fatemap_gated_bootstrap.py, but the four bootstrap floors run on
--n-cores workers (permutations pre-drawn in the parent in the original order, so the results do not depend on the number
of cores). Use this one for large panels; default outputs go to twinscore_supp_gated_bootstrap_parallel/.

Supersedes every other twin_supp runner (see the SUPERSEDED_* files next to this one).

What this version does
  * Every noise scale that used to be the analytic null_sd(meff) = 1/sqrt(meff-1) is a clone-label
    PERMUTATION null instead (checked directly: the analytic formula understates the true noise by
    1.5x on FM06 and 2.9x on FM08):
      - per-gene heritability floors nf1_g / nf2_g  (bootstrap_null_sd_diag, replicate A / B)
      - sd_reg   = null SD of S - lambda*C           (R / zreg)
      - sd_twin_t2 = null SD of the sister off-diagonal C  (gate_g)
      - the gate threshold used inside bootstrap_phi_noise_floor (w_rel) = median of 2*nf
    The analytic null_sd is only a fallback if a bootstrap value is NaN / <= 0.
  * phi is GATED with those bootstrap floors: a gene keeps phi_g = rho_dagger_gg / sqrt(h1_g * h2_g)
    (not clipped: the gate already removes small-h genes, so no extreme ratio reaches the score) only if h1_g > 2*nf1_g AND h2_g > 2*nf2_g. Any other gene (h at or below twice
    its own noise, positive or negative) gets the median phi of the passing genes. No NaN phi.
  * w_rel = 1 - floor/var(phi) with var(phi) and the bootstrap noise floor taken over the genes that PASSED the gate only
    (median-placeholder genes excluded). PAIR (D, R, Wz), direction_term and the score are unchanged:
      TwinScore = w_rel * s(phi_x) + s(PAIR) + direction_term ; phi_x = phi[source gene].
Outputs go to <analysis_data>/<dataset>/data/twinscore_supp_gated_bootstrap/ (own folder + README +
run_params json), never mixed with other variants.
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
from paper_analysis.fatemap_pipeline import twinscore_supp_helpers as supp

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


GATED_README = '# TwinScore_supplement, FINAL: bootstrapped noise floors + h-gate on phi\n\nProduced by `apply_twinscore_supplement_fatemap_gated_bootstrap.py` (add `--absplit` for the A/B split).\nThis is the only twin_supp variant to use; all others are renamed `SUPERSEDED_*` and must not be used.\n\n## Formula\nTwinScore = w_rel * s(phi_x) + s(PAIR) + direction_term for every ordered pair (a, b), no pair gates.\nPAIR = s(D) + gate_g*s(R) + gate_g*v_gate*s(Wz) (D = PIDC, R = CLR-calibrated (S - lambda*C)/sd_reg,\nWz = split-half partial correlation). s() = z-score across pairs (NaN -> min-1). phi_x = phi[a] is a\nGENE-level term repeated for every pair starting at a. A/B split: A = t1, B = t2.\n\n## Bootstrap (replaces every analytic null_sd)\nClone labels are permuted (breaks all clonal structure) and the statistic is recomputed; the SD across\npermutations is the noise scale. Batches with a relative-change stopping rule, seeds SEED, SEED+1..3.\n  * nf1_g, nf2_g : per-gene SD of the heritability h_g in replicate A / B (`bootstrap_null_sd_diag`)\n  * sd_reg       : SD of S - lambda*C over off-diagonal pairs (lambda held at its real value)\n  * sd_twin_t2   : SD of the sister off-diagonal C (gate_g)\n  * w_rel         : 1 - floor/var(phi), BOTH taken over the genes that passed the gate only (placeholders excluded);\n                 gate threshold in the phi bootstrap = median of 2*nf\nThe analytic 1/sqrt(meff-1) is used only if a bootstrap value is NaN or <= 0 (count logged).\n\n## phi gate\nGene passes iff h_A > 2*nf1 and h_B > 2*nf2. Passing genes: phi = rho_dagger_gg/sqrt(h_A*h_B),\nnot clipped (the gate already removes small-h genes, so no extreme ratio reaches the score). Failing genes (either h at or below 2x its own noise, positive or negative): phi =\nmedian phi of the passing genes. `gene_terms` has phi, phi_raw (ungated, NaN if h<=0), gated, nf1, nf2.\n\n## Caveats\n* phi is per source gene, so it promotes or demotes every outgoing pair of a gene together.\n* Only pairs between the genes of a gene set are scored; positives = CollecTRI edges among them.\n'


# ---------------------------------------------------------------------------------------------------------
# Parallel bootstraps (2026-09-23). Every bootstrap draw is independent, so the draws are made in the parent in
# the SAME order as the old serial code (identical random stream) and only the per-draw statistic is computed in
# worker processes. Results are therefore identical for any n_cores (n_cores=1 runs serially, no pool).
# The previous serial versions are preserved in SUPERSEDED_apply_twinscore_supplement_fatemap_allpairs.py.
import concurrent.futures as _cf
import multiprocessing as _mp

_CTX = {}  # set in the parent before a pool is created; forked workers inherit it (no per-task data copies)


def _pmap(fn, items, n_cores):
    items = list(items)
    if n_cores is None or n_cores <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with _cf.ProcessPoolExecutor(max_workers=min(int(n_cores), len(items)), mp_context=_mp.get_context("fork")) as ex:
        return list(ex.map(fn, items, chunksize=1))


def _diag_worker(perm):
    shuffled = _CTX["raw"].copy()
    shuffled["clone_id"] = perm
    try:
        C_perm, _ = sister_matrix_from_frame(shuffled, _CTX["genes"])
    except ValueError:
        return None
    return np.array([C_perm.loc[g, g] for g in _CTX["genes"]], float)


def bootstrap_null_sd_diag(raw_t, genes, seed, batch=30, min_batches=3, max_perm=210, tol=0.05, n_cores=1):
    """Empirical null SD of the heritability diagonal h_g, via clone-label permutation (breaks
    real clonal structure -> true null of 'no heritable signal'). Replaces null_sd(meff_twin) for
    the phi noise floor -- verified null_sd underestimates this by 1.5x (FM06) to 2.9x (FM08).
    Runs in batches of `batch` permutations, stops once the per-gene SD estimate stops moving by
    more than `tol` (relative) between batches, capped at max_perm. Permutations within a batch are evaluated on
    `n_cores` workers. Returns ({gene: empirical_sd}, n_perm)."""
    rng = np.random.default_rng(seed)
    draws = {g: [] for g in genes}
    prev = None
    n_done = 0
    _CTX.update(raw=raw_t, genes=list(genes))
    while n_done < max_perm:
        perms = [rng.permutation(raw_t["clone_id"].to_numpy()) for _ in range(batch)]
        for res in _pmap(_diag_worker, perms, n_cores):
            if res is None:
                continue
            for g, v in zip(genes, res):
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


def _offdiag_worker(perm):
    shuffled = _CTX["raw"].copy()
    shuffled["clone_id"] = perm
    off_mask = _CTX["off_mask"]
    try:
        C_perm, _ = sister_matrix_from_frame(shuffled, _CTX["genes"])
        if _CTX["kind"] == "C":
            v = C_perm.to_numpy()[off_mask]
        else:
            S_perm = same_cell_matrix(shuffled, _CTX["genes"])
            v = (S_perm.to_numpy() - _CTX["extra"].to_numpy() * C_perm.to_numpy())[off_mask]
    except ValueError:
        return None
    return v


def bootstrap_null_sd_offdiag(raw_t, genes, seed, kind, extra=None, batch=8, min_batches=2, max_perm=48, tol=0.03,
                              n_cores=1):
    """Empirical null SD for an off-diagonal matrix statistic, via clone-label permutation. Each
    permutation already yields O(n_genes^2) pair values, so far fewer permutation draws are needed
    than the diagonal case above for the same precision. kind='C' -> sister off-diagonal (replaces
    null_sd(meff_twin_t2) for sd_twin_t2/gate_g). kind='reg' -> S_perm - lam*C_perm numerator
    (replaces null_sd(meff_step1) for sd_reg/R); `extra` must be the real (non-permuted) lambda
    shrinkage matrix in that case -- lambda itself isn't being null-tested, just reused as a fixed
    weight. Permutations within a batch run on `n_cores` workers. Returns (scalar empirical SD, n_perm_used)."""
    rng = np.random.default_rng(seed)
    off_mask = ~np.eye(len(genes), dtype=bool)
    vals = []
    n_done = 0
    prev_sd = None
    _CTX.update(raw=raw_t, genes=list(genes), off_mask=off_mask, kind=kind, extra=extra)
    while n_done < max_perm:
        perms = [rng.permutation(raw_t["clone_id"].to_numpy()) for _ in range(batch)]
        for v in _pmap(_offdiag_worker, perms, n_cores):
            if v is not None:
                vals.extend(v[np.isfinite(v)].tolist())
        n_done += batch
        cur_sd = np.std(vals, ddof=1) if len(vals) > 5 else np.nan
        if n_done >= min_batches * batch and prev_sd is not None and np.isfinite(cur_sd) and prev_sd > 0:
            if abs(cur_sd - prev_sd) / prev_sd < tol:
                prev_sd = cur_sd
                break
        prev_sd = cur_sd
    return prev_sd, n_done


def _phi_boot_worker(drawn):
    t1_raw, t2_raw, genes = _CTX["t1_raw"], _CTX["t2_raw"], _CTX["genes"]
    ci1, ci2, thr1, thr2 = _CTX["ci1"], _CTX["ci2"], _CTX["thr1"], _CTX["thr2"]
    t1_idx, t1_cid, t2_idx, t2_cid = [], [], [], []
    for i, c in enumerate(drawn):
        r1 = ci1.get(c)
        if r1 is not None and len(r1):
            t1_idx.append(r1); t1_cid.append(np.full(len(r1), i))
        r2 = ci2.get(c)
        if r2 is not None and len(r2):
            t2_idx.append(r2); t2_cid.append(np.full(len(r2), i))
    if not t1_idx or not t2_idx:
        return None
    t1_b = t1_raw.iloc[np.concatenate(t1_idx)].reset_index(drop=True).copy()
    t1_b["clone_id"] = np.concatenate(t1_cid)
    t2_b = t2_raw.iloc[np.concatenate(t2_idx)].reset_index(drop=True).copy()
    t2_b["clone_id"] = np.concatenate(t2_cid)
    try:
        C1b, _ = sister_matrix_from_frame(t1_b, genes)
        C2b, _ = sister_matrix_from_frame(t2_b, genes)
        xcb, _ = cross_matrix(t1_b, t2_b, genes)
    except ValueError:
        return None
    out = {}
    for g in genes:
        h1g, h2g, rdg = C1b.loc[g, g], C2b.loc[g, g], xcb.loc[g, g]
        ok = np.isfinite(h1g) and np.isfinite(h2g) and h1g > thr1 and h2g > thr2
        denom = np.sqrt(h1g * h2g) if (ok and h1g > 0 and h2g > 0) else np.nan
        phib = rdg / denom if (np.isfinite(denom) and denom > 0) else np.nan
        if np.isfinite(phib):
            out[g] = phib
    return out


def bootstrap_phi_noise_floor_par(t1_raw, t2_raw, genes, thr1, thr2, n_boot=20, seed=SEED, n_cores=1):
    """Same as supp.bootstrap_phi_noise_floor (clone-level cluster bootstrap of phi; per-gene variance across
    replicates, >= 4 finite draws) with the replicates evaluated on n_cores workers; identical random stream."""
    ci1 = t1_raw.groupby("clone_id").indices
    ci2 = t2_raw.groupby("clone_id").indices
    universe = sorted(set(t1_raw.clone_id) | set(t2_raw.clone_id))
    rng = np.random.default_rng(seed)
    draws = [rng.choice(universe, size=len(universe), replace=True) for _ in range(n_boot)]
    _CTX.update(t1_raw=t1_raw, t2_raw=t2_raw, genes=list(genes), ci1=ci1, ci2=ci2, thr1=thr1, thr2=thr2)
    phi_draws = {g: [] for g in genes}
    for res in _pmap(_phi_boot_worker, draws, n_cores):
        if res is None:
            continue
        for g, v in res.items():
            phi_draws[g].append(v)
    noise_var = {}
    for g in genes:
        vals = np.array(phi_draws[g])
        if len(vals) >= 4:
            noise_var[g] = float(np.var(vals, ddof=1))
    return noise_var


def persistence_gated_bootstrap(h1, h2, rho_dagger, genes, noise_floor_h1_g, noise_floor_h2_g, t1_raw, t2_raw,
                                gate_mult=2.0, n_cores=1):
    """Gated phi with bootstrap noise floors. Gene passes iff h1 > gate_mult*nf1 and h2 > gate_mult*nf2 (both
    finite); passing genes keep phi = rho_dagger_gg/sqrt(h1*h2), not clipped; failing genes get the
    median phi of the passing genes (panel-median placeholder, as in the original gated persistence(),
    but with bootstrap instead of analytic thresholds). Returns phi, w_rel and an info dict."""
    # PHI_CLIP = 10.0  # removed 2026-09-23: the h-gate already excludes small-h genes; no clipped value in any FM06 set
    raw, passed = {}, {}
    for g in genes:
        h1g, h2g, rdg = h1[g], h2[g], rho_dagger[g]
        nf1, nf2 = noise_floor_h1_g[g], noise_floor_h2_g[g]
        ok = (np.isfinite(h1g) and np.isfinite(h2g) and np.isfinite(rdg) and np.isfinite(nf1) and np.isfinite(nf2)
              and nf1 > 0 and nf2 > 0 and h1g > gate_mult * nf1 and h2g > gate_mult * nf2)
        passed[g] = bool(ok)
        # raw[g] = float(np.clip(rdg / np.sqrt(h1g * h2g), -PHI_CLIP, PHI_CLIP)) if ok else np.nan  # old clipped version
        raw[g] = float(rdg / np.sqrt(h1g * h2g)) if ok else np.nan
    vals = np.array([v for v in raw.values() if np.isfinite(v)])
    if len(vals):
        placeholder = float(np.median(vals))
    else:
        placeholder = 0.0
        log("    WARNING: no gene passed the h-gate; phi placeholder set to 0.0")
    phi = {g: (raw[g] if passed[g] else placeholder) for g in genes}
    gated = [g for g in genes if not passed[g]]
    log(f"    phi gate (h > {gate_mult:g}x bootstrap floor, both replicates): {len(gated)}/{len(genes)} genes gated "
        f"-> median placeholder phi = {placeholder:.3f}; gated: {', '.join(gated) if gated else '-'}")
    phi_out, w, w_info = _finish_phi(phi, genes, noise_floor_h1_g, noise_floor_h2_g, t1_raw, t2_raw,
                                     use_genes=[g for g in genes if passed[g]], n_cores=n_cores)
    return phi_out, w, dict(gated=gated, placeholder=placeholder, gate_mult=gate_mult, n_pass=len(vals), **w_info)


def _finish_phi(phi, genes, noise_floor_h1_g, noise_floor_h2_g, t1_raw, t2_raw, use_genes=None, n_cores=1):
    """Reliability weight w_rel = max(0, 1 - floor/var(phi)). 2026-09-23: var(phi) AND the noise floor are taken over
    `use_genes` only (the genes that PASSED the h-gate), so median-placeholder genes neither shrink the variance nor
    enter the floor (matches yscher's nanvar(phi) before the median is filled in). Returns phi, w, info."""
    ug = list(genes) if use_genes is None else list(use_genes)
    phi_vals = np.array([phi[g] for g in ug if np.isfinite(phi[g])])
    var_phi = phi_vals.var(ddof=1) if len(phi_vals) > 1 else np.nan
    thr1 = {g: 2.0 * noise_floor_h1_g[g] for g in genes}
    thr2 = {g: 2.0 * noise_floor_h2_g[g] for g in genes}
    # bootstrap_phi_noise_floor takes a single scalar thr1/thr2 (median gate) -- kept as-is, only
    # feeds the separate w_rel reliability weight, not phi itself.
    noise_var = bootstrap_phi_noise_floor_par(t1_raw, t2_raw, genes,
                                               float(np.median(list(thr1.values()))),
                                               float(np.median(list(thr2.values()))), n_cores=n_cores)
    # old (all genes, placeholders included): floor = float(np.median(list(noise_var.values()))) if noise_var else 0.0
    nv = [noise_var[g] for g in ug if g in noise_var and np.isfinite(noise_var[g])] if noise_var else []
    floor = float(np.median(nv)) if nv else 0.0
    w = max(0.0, 1.0 - floor / var_phi) if (np.isfinite(var_phi) and var_phi > 0) else 0.0
    return phi, w, dict(var_phi=float(var_phi) if np.isfinite(var_phi) else np.nan, noise_floor=floor, n_w_genes=int(len(phi_vals)))


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

    from paper_analysis.fatemap_pipeline.run_infer_fatemap_ab_split import AB_SUBSAMPLE_PER_SIDE, ab_max_clone_size, subsample_per_side
    if AB_SUBSAMPLE_PER_SIDE.get(dataset):  # Watermelon stages: same per-side subsampling as the inference loader
        df = subsample_per_side(df, AB_SUBSAMPLE_PER_SIDE[dataset])
    cap = ab_max_clone_size(dataset)  # None = no cap (FM01), else MAX_CLONE_SIZE
    if cap is not None:
        per_side = df.groupby(["clone_id", "time_step"]).size()
        oversized = per_side[per_side > cap].index.get_level_values("clone_id").unique()
        df = df[~df["clone_id"].isin(oversized)].reset_index(drop=True)
    log(f"[{dataset}/{gene_set_name}] A/B-spanning-clone filter: {n_before:,} -> {len(df):,} cells "
        f"({df['clone_id'].nunique():,} clones, t1=A: {(df.time_step==0).sum():,}, "
        f"t2=B: {(df.time_step==1).sum():,})")
    return df, genes, qc_dir, genes_full, gidx


def run_gene_set(dataset, gs, ce_edges, absplit, n_shuffles=2000, n_cores=8, compute_wz=True):
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
    boot_h1, n_boot_h1 = bootstrap_null_sd_diag(t1_raw, genes, seed=SEED, n_cores=n_cores)
    boot_h2, n_boot_h2 = bootstrap_null_sd_diag(t2_raw, genes, seed=SEED + 1, n_cores=n_cores)
    fallback_h1, fallback_h2 = null_sd(meff_twin_t1), null_sd(meff_twin_t2)
    noise_floor_h1_g = {g: (boot_h1[g] if np.isfinite(boot_h1[g]) else fallback_h1) for g in genes}
    noise_floor_h2_g = {g: (boot_h2[g] if np.isfinite(boot_h2[g]) else fallback_h2) for g in genes}
    n_fallback = sum(1 for g in genes if not np.isfinite(boot_h1[g]) or not np.isfinite(boot_h2[g]))
    log(f"    phi h-floor: {n_boot_h1}/{n_boot_h2} bootstrap perms (t1/t2), "
        f"{n_fallback}/{len(genes)} genes fell back to analytic null_sd")

    log(f"[{dataset}/{gs}] persistence phi_g (h-GATE with bootstrap floors), reliability w")
    phi, w_rel, phi_info = persistence_gated_bootstrap(h_t1, h_t2, rho_dagger_diag, genes, noise_floor_h1_g,
                                                       noise_floor_h2_g, t1_raw, t2_raw, n_cores=n_cores)

    log(f"[{dataset}/{gs}] zreg, R, twin-signal gate g")
    n_genes = len(genes)
    meff_step1 = m_eff_step1(raw.groupby("clone_id").size().to_numpy())  # kept for diagnostics only
    lam = np.minimum(1.0, zhet_matrix.abs() / 2.33)
    sd_reg, n_boot_reg = bootstrap_null_sd_offdiag(t2_raw, genes, seed=SEED + 2, kind="reg", extra=lam, n_cores=n_cores)
    sd_twin_t2, n_boot_twin = bootstrap_null_sd_offdiag(t2_raw, genes, seed=SEED + 3, kind="C", n_cores=n_cores)
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
        "phi_raw": [(rho_dagger_diag[g] / np.sqrt(h_t1[g] * h_t2[g])) if (h_t1[g] > 0 and h_t2[g] > 0) else np.nan for g in genes],
        "gated": [g in set(phi_info["gated"]) for g in genes],
        "nf_h1": [noise_floor_h1_g[g] for g in genes], "nf_h2": [noise_floor_h2_g[g] for g in genes],
    })
    diagnostics = dict(dataset=dataset, gs=gs, absplit=absplit, n_genes=n_genes, n_pairs=len(U),
                        meff_twin_t1=meff_twin_t1, meff_twin_t2=meff_twin_t2, meff_cross=meff_cross_,
                        meff_step1=meff_step1, w=w_rel, gate_g=gate_g, gate_v=v_gate,
                        kappa_gamma=kappa_gamma, wz_shrinkage_a=a_star,
                        n_phi_nan=int(sum(1 for v in phi.values() if not np.isfinite(v))),
                        n_gated=len(phi_info["gated"]), gated_genes=phi_info["gated"],
                        phi_placeholder=phi_info["placeholder"], gate_mult=phi_info["gate_mult"],
                        var_phi_pass=phi_info["var_phi"], phi_noise_floor=phi_info["noise_floor"],
                        boot_perms=dict(h1=n_boot_h1, h2=n_boot_h2, sd_reg=n_boot_reg, sd_twin_t2=n_boot_twin),
                        sd_reg=float(sd_reg), sd_twin_t2=float(sd_twin_t2),
                        z_dagger_coverage=len(z_dagger_map))
    return pair_df, gene_df, diagnostics


def main():
    import argparse
    import datetime
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["FM01", "FM06", "FM08", "Watermelon_naive", "Watermelon_lag", "Watermelon_late"])
    ap.add_argument("--gene-set", default="correlation_high")
    ap.add_argument("--n-shuffles", type=int, default=500)
    ap.add_argument("--n-cores", type=int, default=8)
    ap.add_argument("--absplit", action="store_true")
    ap.add_argument("--no-wz", action="store_true")
    ap.add_argument("--out-dir", default=None, help="override output folder (default: <data>/twinscore_supp_gated_bootstrap_parallel/)")
    args = ap.parse_args()

    ct = pd.read_csv(supp.COLLECTRI_PATH, sep="\t")
    ct = ct[ct.source_genesymbol != ct.target_genesymbol]
    CE = set(zip(ct.source_genesymbol, ct.target_genesymbol))

    label = args.dataset.lower()
    out_dir = args.out_dir or f"{TWINFER_PROJECT_ROOT}/analysis_data/{label}/data/twinscore_supp_gated_bootstrap_parallel"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "README.md"), "w") as fh:
        fh.write(GATED_README)
    z_suffix = "_absplit_allpairs" if args.absplit else "_allpairs"  # z_dagger json from the bypass-gate inference run
    suffix = "_absplit_gated_bootstrap" if args.absplit else "_gated_bootstrap"

    pair_df, gene_df, diag = run_gene_set(
        args.dataset, args.gene_set, CE, args.absplit, n_shuffles=args.n_shuffles,
        n_cores=args.n_cores, compute_wz=not args.no_wz,
    )
    pair_out = os.path.join(out_dir, f"twinscore_supplement_{label}_{args.gene_set}{suffix}_pair_terms.csv")
    gene_out = os.path.join(out_dir, f"twinscore_supplement_{label}_{args.gene_set}{suffix}_gene_terms.csv")
    pair_df.to_csv(pair_out, index=False)
    gene_df.to_csv(gene_out, index=False)
    log(f"wrote {pair_out}")
    log(f"wrote {gene_out}")
    log(f"diagnostics: {diag}")
    params = dict(dataset=args.dataset, gene_set=args.gene_set, absplit=args.absplit, n_shuffles=args.n_shuffles,
                  compute_wz=not args.no_wz, phi_mode="gated_bootstrap", gate_mult=2.0, phi_clip=None,
                  script=os.path.basename(__file__),
                  z_dagger_input=f"{TWINFER_PROJECT_ROOT}/analysis_data/{label}/data/z_dagger_{label}_{args.gene_set}{z_suffix}.json",
                  written=datetime.datetime.now().isoformat(timespec="seconds"), diagnostics=diag)
    with open(os.path.join(out_dir, f"run_params_{label}_{args.gene_set}{suffix}.json"), "w") as fh:
        json.dump(params, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
