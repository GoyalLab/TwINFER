#!/usr/bin/env python3
"""PDF-exact final results for hPSC_20260927, Stage I survivors only (4,219 pairs):
S (same-cell correlation), C (twin/sister correlation), U = S-C (unshared correlation),
Stage II h = sign(S)*C/sd(C) (inherited state if h>2.405), Stage III z = U/sd(U)
(BH-corrected over Stage-I-called pairs, ranked by |z| -- this IS the PDF's headline
"Called pairs ranked by |z|" list, Table 1/2), plus the direction stage (z_dagger/gamma,
already computed in run_stage234_hpsc_endoderm.py).

C computed via a pairs-restricted loop (not the O(G^2) full-matrix helper in
twinscore_supp_helpers.py, which would re-hit the same scaling wall Stage II/IV avoided
by restricting to declared pairs) -- same fix pattern as run_stage234_hpsc_endoderm.py.

sd(C), sd(U): closed-form clone-effective-sample-size SD (this repo's own
analytic_zscores.py convention: null_sd(m_eff) = 1/sqrt(m_eff-1)), consistent with how
Stage I's own null is characterized and with the repo's established practice of using
this closed-form SD wherever a per-dataset calibrated constant hasn't been measured.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/package")
from twinfer.inference.correlation_functions import (
    assign_twin_id, split_twins, get_unit_weights, weighted_spearman,
    calculate_pairwise_gene_gene_correlation_matrix,
)

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/larry_hematopoiesis_validation")
from twinfer.scoring.analytic_zscores import m_eff_twin

LABEL = "hpsc_20260927"
GENE_SET = "chipatlas_perturbseq_panel"
DATA_DIR = f'{TWINFER_PROJECT_ROOT}/analysis_data/hPSC_20260927/data'
BH_Q = 0.05
Z_INHERITED = 2.405


def log(m):
    print(m, flush=True)


def null_sd(meff):
    return 1.0 / np.sqrt(max(meff - 1.0, 1e-9))


def main():
    df_all = pd.read_csv(f"{DATA_DIR}/twinfer_input_{LABEL}_{GENE_SET}.csv")
    all_genes = [c[:-5] for c in df_all.columns if c.endswith("_mRNA")]
    called = pd.read_csv(f"{DATA_DIR}/hpsc_endoderm_stage1_called.csv")
    pairs = list(zip(called.gene_1, called.gene_2))
    genes = sorted(set(called.gene_1) | set(called.gene_2))
    log(f"[hPSC_20260927] {len(pairs)} Stage-I-called pairs, {len(genes)} genes")

    t1_raw = df_all[df_all.time_step == 0].reset_index(drop=True)  # endo_T0 (more cells)

    # ---- S: same-cell correlation (fast, BLAS-vectorized, full matrix over `genes` is fine) ----
    t0 = time.time()
    S = calculate_pairwise_gene_gene_correlation_matrix(t1_raw, genes, use_clone=True)
    log(f"[hPSC_20260927] S matrix: {time.time()-t0:.1f}s")

    # ---- C: twin/sister correlation, restricted to called pairs (+ each gene's own diagonal
    # C(g,g) = heritability, needed for Stage II's h) -- NOT the O(G^2) full double loop. ----
    t0 = time.time()
    tw = assign_twin_id(t1_raw)
    twin_0, twin_1 = split_twins(tw)
    w = get_unit_weights(twin_0, unit="clone")
    w2 = np.concatenate([w, w]) / 2.0
    vals0 = {g: twin_0[f"{g}_mRNA"].to_numpy() for g in genes}
    vals1 = {g: twin_1[f"{g}_mRNA"].to_numpy() for g in genes}

    def C_of(gx, gy):
        px = np.concatenate([vals0[gx], vals1[gx]])
        py = np.concatenate([vals1[gy], vals0[gy]])
        return weighted_spearman(px, py, w2)

    C_diag = {g: C_of(g, g) for g in genes}  # heritability h_g
    C_pair = {}
    for a, b in pairs:
        C_pair[(a, b)] = C_of(a, b)
    log(f"[hPSC_20260927] C (heritability + {len(pairs)} pair correlations): {time.time()-t0:.1f}s")

    clone_sizes = t1_raw.groupby("clone_id").size().to_numpy()
    meff_twin = m_eff_twin(clone_sizes)
    sd_C = null_sd(meff_twin)
    sd_U = np.sqrt(2) * sd_C  # U = S - C; S's sampling noise is much smaller (large N, not twin-limited) so
    # the dominant variance is C's -- Var(U) ~= Var(S) + Var(C); using 2*Var(C) as a conservative
    # (slightly inflated) proxy since S's own null_sd at N_cells>>N_twin-pairs is negligible by comparison.

    rows = []
    for a, b in pairs:
        s_ab = float(S.loc[a, b])
        c_ab = C_pair[(a, b)]
        u_ab = s_ab - c_ab
        h_a, h_b = C_diag[a], C_diag[b]
        h_pair = np.sign(s_ab) * c_ab / sd_C if sd_C > 0 else np.nan
        inherited = abs(h_pair) > Z_INHERITED if np.isfinite(h_pair) else False
        z_u = u_ab / sd_U if sd_U > 0 else np.nan
        rows.append(dict(gene_1=a, gene_2=b, S=s_ab, C=c_ab, U=u_ab,
                          h_a=h_a, h_b=h_b, h_pair=h_pair, inherited_state=inherited, z=z_u))

    out = pd.DataFrame(rows)
    out["p_val"] = 2 * norm.sf(np.abs(out["z"].to_numpy()))
    finite = out["p_val"].notna()
    reject = np.zeros(len(out), dtype=bool)
    qvals = np.full(len(out), np.nan)
    reject[finite.to_numpy()], qvals[finite.to_numpy()], _, _ = multipletests(
        out.loc[finite, "p_val"], alpha=BH_Q, method="fdr_bh")
    out["q_val"] = qvals
    out["called_stage3"] = reject

    # ---- merge in the direction stage already computed ----
    direction = pd.read_csv(f"{DATA_DIR}/ranked_edges_{LABEL}_{GENE_SET}_stage234.csv")
    out = out.merge(direction[["gene_1", "gene_2", "direction", "z_gamma_ab", "z_gamma_ba"]],
                     on=["gene_1", "gene_2"], how="left")

    final_edges = pd.read_csv(f"{DATA_DIR}/hpsc_endoderm_candidate_pairs_final.csv")
    positives = set(zip(final_edges.TF, final_edges.target)) | set(zip(final_edges.target, final_edges.TF))
    out["true_candidate"] = [(a, b) in positives for a, b in zip(out.gene_1, out.gene_2)]

    out = out.sort_values("z", key=lambda s: -s.abs())
    out_path = f"{DATA_DIR}/hpsc_endoderm_PDF_FINAL_stage3.csv"
    out.to_csv(out_path, index=False)

    log(f"\n=== FINAL (PDF Stage III: z=U/sd(U), BH q<={BH_Q}) ===")
    log(f"total called (Stage I) pairs scored: {len(out)}")
    log(f"Stage III BH-called (q<={BH_Q}): {int(out.called_stage3.sum())} "
        f"({100*out.called_stage3.mean():.2f}%)")
    log(f"inherited-state pairs (|h|>{Z_INHERITED}): {int(out.inherited_state.sum())}")
    log(f"true ChIP-Atlas(+Perturb-seq) candidates among Stage-III-called: "
        f"{int(out[out.called_stage3].true_candidate.sum())}/{int(out.called_stage3.sum())}")

    log(f"\ntop 20 by |z| (all Stage-I-called pairs, ranked exactly as PDF Table 1/2):")
    log(out[["gene_1", "gene_2", "S", "C", "U", "z", "q_val", "called_stage3",
             "inherited_state", "direction", "true_candidate"]].head(20).to_string(index=False))

    log(f"\ntop 20 Stage-III-called pairs that are TRUE ChIP-Atlas candidates:")
    tc = out[out.called_stage3 & out.true_candidate]
    log(tc[["gene_1", "gene_2", "S", "C", "U", "z", "direction"]].head(20).to_string(index=False))

    log(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
