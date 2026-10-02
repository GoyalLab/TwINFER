#!/usr/bin/env python
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_a2a3302f_2026-09-28_why_twinfer_fails; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Run the REAL TwinScore_supplement gated-bootstrap pipeline (not a hand-rolled proxy) on mCAD's
canonical Gillespie simulation, at two (t1,t2) conventions: (1,10) [current project standard] vs
(10,30) [proposed shift]. Uses the production helper functions (twinscore_supp_helpers +
apply_twinscore_supplement_fatemap_gated_bootstrap's bootstrap_null_sd_diag/offdiag and
persistence_gated_bootstrap) exactly as apply_twinscore_supplement_sim_gated_bootstrap.py does,
just with the file pointed at the raw Gillespie real_data file directly (already has native
gene_N_mRNA columns, so no renaming needed) and T1T2 overridden.

Direction_term (z_dagger) is SKIPPED -- it needs a full ALL_PAIRS TwINFER inference run, a separate,
much heavier job. This only reports the phi/persistence-gate machinery and the R/D/PAIR
existence-relevant metrics, which is what t1 vs t2 actually affects.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline")
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/package")
from paper_analysis.fatemap_pipeline import twinscore_supp_helpers as supp
from paper_analysis.fatemap_pipeline import apply_twinscore_supplement_fatemap_gated_bootstrap as gb

R = f'{TWINFER_PROJECT_ROOT}'
FILE = f"{R}/simulation_data/real_data/df_rows_0_0_0_0_0_12072026_005933_ncells_6000_mCAD_1_0_c733c5af.csv"
MATRIX = f"{R}/simulation_data/twinfer_format/mCAD/interaction_matrix.txt"
genes = [f"gene_{i+1}" for i in range(5)]
M = np.loadtxt(MATRIX, delimiter=",")

df_full = pd.read_csv(FILE)
print("available time_steps:", sorted(df_full.time_step.unique()))


def existence_auprc_x(vals, iu):
    und = ((M != 0) | (M.T != 0))[iu].astype(int)
    return average_precision_score(und, np.nan_to_num(vals)) / und.mean()


def run_at(t1, t2, seed_offset=0):
    print(f"\n=== t1={t1}  t2={t2} ===")
    avail = sorted(df_full.time_step.unique())
    t1r = min(avail, key=lambda x: abs(x - t1))
    t2r = min(avail, key=lambda x: abs(x - t2))
    t1_raw = df_full[df_full.time_step == t1r].reset_index(drop=True)
    t2_raw = df_full[df_full.time_step == t2r].reset_index(drop=True)
    print(f"  using actual saved steps t1={t1r} t2={t2r}  ({len(t1_raw)} cells @t1, {len(t2_raw)} cells @t2)")

    S = supp.same_cell_matrix(t2_raw, genes)
    C_t1, meff_t1 = supp.sister_matrix_from_frame(t1_raw, genes)
    C_t2, meff_t2 = supp.sister_matrix_from_frame(t2_raw, genes)
    h1 = {g: C_t1.loc[g, g] for g in genes}
    h2 = {g: C_t2.loc[g, g] for g in genes}
    rho_dagger, meff_cross = supp.cross_matrix(t1_raw, t2_raw, genes)
    rdd = {g: rho_dagger.loc[g, g] for g in genes}

    b1, n1 = gb.bootstrap_null_sd_diag(t1_raw, genes, seed=supp.SEED + seed_offset)
    b2, n2 = gb.bootstrap_null_sd_diag(t2_raw, genes, seed=supp.SEED + 1 + seed_offset)
    nf1 = {g: (b1[g] if np.isfinite(b1[g]) else supp.null_sd(meff_t1)) for g in genes}
    nf2 = {g: (b2[g] if np.isfinite(b2[g]) else supp.null_sd(meff_t2)) for g in genes}
    phi, w_rel, phi_info = gb.persistence_gated_bootstrap(h1, h2, rdd, genes, nf1, nf2, t1_raw, t2_raw)

    print(f"  h1 (heritability @t1): {[round(h1[g],3) for g in genes]}")
    print(f"  h2 (heritability @t2): {[round(h2[g],3) for g in genes]}")
    print(f"  rho_dagger (cross-time persistence): {[round(rdd[g],3) for g in genes]}")
    print(f"  nf1 (noise floor @t1): {[round(nf1[g],4) for g in genes]}")
    print(f"  nf2 (noise floor @t2): {[round(nf2[g],4) for g in genes]}")
    print(f"  phi (persistence score, gated): {[round(phi[g],3) for g in genes]}")
    print(f"  w_rel (reliability weight): {w_rel:.3f}")
    print(f"  gate info: {phi_info}")

    # existence-relevant part: R via zreg (CLR-calibrated), D via PIDC
    zhet_matrix, S_check, _ = supp.compute_zhet_zrho(t2_raw, genes, n_shuffles=300, n_cores=8)
    lam = np.minimum(1.0, zhet_matrix.abs() / 2.33)
    sd_reg, _ = gb.bootstrap_null_sd_offdiag(t2_raw, genes, seed=supp.SEED + 2 + seed_offset, kind="reg", extra=lam)
    zreg = (S - lam * C_t2) / sd_reg
    R_fn = supp.clr_calibrate(zreg, genes)
    D_mat = supp.compute_D(t2_raw, genes)

    n = len(genes)
    iu = np.triu_indices(n, 1)
    U = [(a, b) for a in genes for b in genes if a != b]
    D_vec = np.array([D_mat.loc[a, b] for a, b in U])
    R_vec = np.array([R_fn(a, b) for a, b in U])
    phi_x = np.array([phi[a] for a, b in U])
    sd_tw, _ = gb.bootstrap_null_sd_offdiag(t2_raw, genes, seed=supp.SEED + 3 + seed_offset, kind="C")
    gate_g = supp.signal_share((C_t2 / (sd_tw if np.isfinite(sd_tw) and sd_tw > 0 else supp.null_sd(meff_t2))).to_numpy()[~np.eye(n, dtype=bool)])
    PAIR = supp.s(D_vec) + gate_g * supp.s(R_vec)
    score_no_direction = w_rel * supp.s(phi_x) + supp.s(PAIR)

    gi = {g: i for i, g in enumerate(genes)}
    # AUPRC over all directed pairs U (existence truth is undirected: either direction counts)
    und_dir = np.array([1 if (M[gi[a],gi[b]]!=0 or M[gi[b],gi[a]]!=0) else 0 for a,b in U])
    met2 = {}
    for name, v in [("PAIR", PAIR), ("R", R_vec), ("D(PIDC)", D_vec), ("score(no direction)", score_no_direction)]:
        vv = np.nan_to_num(np.asarray(v, float), nan=np.nanmin(v) - 1 if np.isfinite(v).any() else 0)
        met2[name] = average_precision_score(und_dir, vv) / und_dir.mean()
    print(f"  AUPRC-x (directed pairs, undirected truth): {met2}")
    return dict(h1=h1, h2=h2, rho_dagger=rdd, phi=phi, w_rel=w_rel, phi_info=phi_info, auprc=met2)


r_1_10 = run_at(1, 10, seed_offset=0)
r_10_30 = run_at(10, 30, seed_offset=100)

print("\n=== SUMMARY: (t1=1,t2=10) vs (t1=10,t2=30) ===")
print(f"w_rel:  {r_1_10['w_rel']:.3f}  ->  {r_10_30['w_rel']:.3f}")
for k in r_1_10["auprc"]:
    print(f"AUPRC-x {k:22s}: {r_1_10['auprc'][k]:.2f}  ->  {r_10_30['auprc'][k]:.2f}")
