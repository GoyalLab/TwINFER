#!/usr/bin/env python3
# SUPERSEDED 2026-09-23 -- DO NOT USE. Use apply_twinscore_supplement_fatemap_gated_bootstrap.py (serial) or ..._gated_bootstrap_parallel.py.
# Kept only so older numbers can be reproduced.
"""SpaceBar analog of apply_twinscore_supplement_fatemap_allpairs.py, on the centroid-split A/B
construction (run_infer_spacebar_centroid_ab.py) instead of FM06/FM08's real replicate A/B.

All the S/zhet, C/heritability, cross-correlation (persistence numerator), D (PIDC), gamma/
direction_term machinery in apply_twinscore_supplement_fatemap.py operates on a generic
clone_id/time_step/{gene}_mRNA dataframe -- reused verbatim here. persistence_no_gate (the
2026-09-22 phi fix, floored per-gene at 1 null-SD of its own heritability estimate rather than
gated at a panel-median placeholder) is likewise dataset-agnostic and imported from the FM06
allpairs module rather than duplicated.

Wz (split-half partial correlation): 2026-09-23 correction -- SpaceBar's raw per-section CSVs
(Section{i}_cell_by_gene_clustered.csv) ARE the QC'd raw-count data (same role FM06/FM08's
qc_filtered .mtx plays), not something SpaceBar lacks. run_infer_spacebar_centroid_ab.py's
build_ab_split_with_full_panel already extracts the full 114-gene raw-count matrix aligned to
the exact same retained-cell set/order as the target-gene twinfer input, so Wz is computed here
via an in-memory variant of compute_Wz (_one_split_half_W itself is fully dataset-agnostic --
only compute_Wz's file-reading wrapper was FM06/FM08-specific) instead of via qc_dir/mtx files.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_analysis.fatemap_pipeline import SUPERSEDED_apply_twinscore_supplement_fatemap as supp
from paper_analysis.fatemap_pipeline.SUPERSEDED_apply_twinscore_supplement_fatemap_allpairs import (
    persistence_no_gate, bootstrap_null_sd_diag, bootstrap_null_sd_offdiag,
)
from paper_analysis.fatemap_pipeline.run_infer_spacebar_centroid_ab import build_ab_split_with_full_panel, MAX_SIDE
from paper_analysis.fatemap_pipeline.run_infer_spacebar_centroid_ab import GENE_SETS

log = supp.log
s = supp.s
signal_share = supp.signal_share
clr_calibrate = supp.clr_calibrate
null_sd = supp.null_sd
m_eff_step1 = supp.m_eff_step1
compute_zhet_zrho = supp.compute_zhet_zrho
sister_matrix_from_frame = supp.sister_matrix_from_frame
cross_matrix = supp.cross_matrix
compute_D = supp.compute_D
_one_split_half_W = supp._one_split_half_W
R0 = supp.R0
Z_DIRECTION_GATE = supp.Z_DIRECTION_GATE
SEED = 101010


def load_raw_ab_spacebar(gene_set_name):
    genes = GENE_SETS[gene_set_name]
    df, X_full, genes_full = build_ab_split_with_full_panel(gene_set_name)
    return df, genes, X_full, genes_full


def compute_Wz_spacebar(raw, genes, X_full, genes_full, seed=SEED, n_boot=20):
    """In-memory analog of apply_twinscore_supplement_fatemap.compute_Wz: X_full is already the
    raw-count matrix aligned 1:1 to `raw`'s rows/order (built together in
    build_ab_split_with_full_panel), so no qc_dir/mtx file read or cell_id re-matching is needed."""
    import time
    t0 = time.time()
    clone_ids = raw["clone_id"].to_numpy()
    n = len(genes)
    W_draws, meff_twin_list = [], []
    for b in range(n_boot):
        Wb, meff_twin_b = _one_split_half_W(X_full, clone_ids, genes, genes_full, seed=seed + b)
        W_draws.append(Wb)
        meff_twin_list.append(meff_twin_b)
    W_draws = np.stack(W_draws, axis=0)
    meff_twin_ = float(np.mean(meff_twin_list))
    log(f"    Wz: {n_boot} bootstrap split-half draws done ({time.time()-t0:.0f}s), {X_full.shape[0]:,} cells")

    W_mean = W_draws.mean(axis=0)
    W_var_boot = W_draws.var(axis=0, ddof=1)
    iu = np.triu_indices(n, k=1)
    denom = np.sum(W_mean[iu] ** 2)
    a_star = float(np.clip(W_var_boot[iu].sum() / denom, 0.0, 1.0)) if denom > 0 else 1.0

    def shrink_invert(Wm):
        Wshrunk = (1 - a_star) * Wm + a_star * np.eye(n)
        P = -np.linalg.pinv(Wshrunk + 1e-8 * np.eye(n))
        d = np.sqrt(np.clip(np.outer(np.diag(P), np.diag(P)), 1e-12, None))
        PC = P / d
        np.fill_diagonal(PC, 0.0)
        return PC

    PC_mean = shrink_invert(W_mean)
    PC_draws = np.stack([shrink_invert(W_draws[b]) for b in range(n_boot)], axis=0)
    se = PC_draws.std(axis=0, ddof=1)
    iu_full = ~np.eye(n, dtype=bool)
    se_floor = max(1e-9, 0.05 * float(np.median(np.abs(PC_mean[iu_full]))))
    se_safe = np.where(se > se_floor, se, se_floor)
    Wz_raw = PC_mean / se_safe
    Wz_clipped = np.clip(Wz_raw, -20.0, 20.0)
    n_clipped = int(np.sum((Wz_raw != Wz_clipped) & iu_full))
    if n_clipped:
        log(f"    Wz: clipped {n_clipped} outlier z-value(s) to +/-20 (se_floor={se_floor:.4g})")
    Wz = pd.DataFrame(Wz_clipped, index=genes, columns=genes)
    v = signal_share(Wz.to_numpy()[~np.eye(n, dtype=bool)])
    log(f"    Wz: a*={a_star:.3f}  v={v:.3f}  (meff_twin={meff_twin_:.0f})")
    return Wz, v, a_star


def run_gene_set(gs, ce_edges, n_shuffles=2000, n_cores=8, compute_wz=True):
    full, genes, X_full, genes_full = load_raw_ab_spacebar(gs)
    t1_raw = full[full.time_step == 0].reset_index(drop=True)  # central (A)
    t2_raw = full[full.time_step == 1].reset_index(drop=True)  # peripheral (B)
    raw = full  # same-cell stats (S, zhet) are timepoint-agnostic

    log(f"[SpaceBar/{gs}] real S, z_het (fresh permutation)")
    zhet_matrix, S, twin_delta = compute_zhet_zrho(raw, genes, n_shuffles=n_shuffles, n_cores=n_cores)
    S_t2 = S

    log(f"[SpaceBar/{gs}] C, heritability (fresh weighted-Spearman, each side)")
    C_t1, meff_twin_t1 = sister_matrix_from_frame(t1_raw, genes)
    C_t2, meff_twin_t2 = sister_matrix_from_frame(t2_raw, genes)
    h_t1 = {g: C_t1.loc[g, g] for g in genes}
    h_t2 = {g: C_t2.loc[g, g] for g in genes}

    log(f"[SpaceBar/{gs}] cross-cell (within-clone, central-vs-peripheral) correlation -- persistence numerator")
    rho_dagger, meff_cross_ = cross_matrix(t1_raw, t2_raw, genes)
    rho_dagger_diag = {g: rho_dagger.loc[g, g] for g in genes}

    log(f"[SpaceBar/{gs}] bootstrap noise floors (clone-label permutation null, replaces null_sd everywhere)")
    boot_h1, n_boot_h1 = bootstrap_null_sd_diag(t1_raw, genes, seed=SEED)
    boot_h2, n_boot_h2 = bootstrap_null_sd_diag(t2_raw, genes, seed=SEED + 1)
    fallback_h1, fallback_h2 = null_sd(meff_twin_t1), null_sd(meff_twin_t2)
    noise_floor_h1_g = {g: (boot_h1[g] if np.isfinite(boot_h1[g]) else fallback_h1) for g in genes}
    noise_floor_h2_g = {g: (boot_h2[g] if np.isfinite(boot_h2[g]) else fallback_h2) for g in genes}
    n_fallback = sum(1 for g in genes if not np.isfinite(boot_h1[g]) or not np.isfinite(boot_h2[g]))
    log(f"    phi h-floor: {n_boot_h1}/{n_boot_h2} bootstrap perms (t1/t2), "
        f"{n_fallback}/{len(genes)} genes fell back to analytic null_sd")

    log(f"[SpaceBar/{gs}] persistence phi_g (NO GATE), reliability w")
    phi, w_rel = persistence_no_gate(h_t1, h_t2, rho_dagger_diag, genes, noise_floor_h1_g, noise_floor_h2_g, t1_raw, t2_raw)

    log(f"[SpaceBar/{gs}] zreg, R, twin-signal gate g")
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
        log(f"[SpaceBar/{gs}] D (real PIDC) and Wz (split-half partial correlation, in-memory)")
    else:
        log(f"[SpaceBar/{gs}] D (real PIDC)")
    D_mat = compute_D(t2_raw, genes)
    if compute_wz:
        Wz_mat, v_gate, a_star = compute_Wz_spacebar(raw, genes, X_full, genes_full)
    else:
        Wz_mat = pd.DataFrame(0.0, index=genes, columns=genes)
        v_gate, a_star = 0.0, np.nan

    log(f"[SpaceBar/{gs}] direction: z^dagger (from centroid_ab allpairs json), gamma, kappa_gamma, q")
    z_dagger_path = (f"{TWINFER_PROJECT_ROOT}/analysis_data/spacebar/data/"
                      f"z_dagger_spacebar_{gs}_centroid_ab_allpairs.json")
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

    log(f"[SpaceBar/{gs}] assembling PAIR and TwinScore")
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
    diagnostics = dict(dataset="SPACEBAR", gs=gs, n_genes=n_genes, n_pairs=len(U),
                        meff_twin_t1=meff_twin_t1, meff_twin_t2=meff_twin_t2, meff_cross=meff_cross_,
                        meff_step1=meff_step1, w=w_rel, gate_g=gate_g, gate_v=v_gate,
                        kappa_gamma=kappa_gamma, wz_shrinkage_a=a_star,
                        n_phi_nan=int(sum(1 for v in phi.values() if not np.isfinite(v))),
                        z_dagger_coverage=len(z_dagger_map))
    return pair_df, gene_df, diagnostics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene-set", choices=list(GENE_SETS), default="correlation_high")
    ap.add_argument("--n-shuffles", type=int, default=500)
    ap.add_argument("--n-cores", type=int, default=8)
    ap.add_argument("--no-wz", action="store_true")
    args = ap.parse_args()

    ct = pd.read_csv(supp.COLLECTRI_PATH, sep="\t")
    ct = ct[ct.source_genesymbol != ct.target_genesymbol]
    CE = set(zip(ct.source_genesymbol, ct.target_genesymbol))

    out_dir = f'{TWINFER_PROJECT_ROOT}/analysis_data/spacebar/data'
    os.makedirs(out_dir, exist_ok=True)

    pair_df, gene_df, diag = run_gene_set(args.gene_set, CE, n_shuffles=args.n_shuffles,
                                           n_cores=args.n_cores, compute_wz=not args.no_wz)
    pair_out = os.path.join(out_dir, f"twinscore_supplement_spacebar_{args.gene_set}_centroid_ab_allpairs_pair_terms.csv")
    gene_out = os.path.join(out_dir, f"twinscore_supplement_spacebar_{args.gene_set}_centroid_ab_allpairs_gene_terms.csv")
    pair_df.to_csv(pair_out, index=False)
    gene_df.to_csv(gene_out, index=False)
    log(f"wrote {pair_out}")
    log(f"wrote {gene_out}")
    log(f"diagnostics: {diag}")


if __name__ == "__main__":
    main()
