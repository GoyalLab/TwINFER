#!/usr/bin/env python
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_807d072d_2026-09-20_real_network_sim; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""
Two label-free (no ground truth) alternatives to todo4v2(nogate)'s unweighted
equal-variance term combination, applied to the 5 terms that get summed via
s(.): z_abs_t1 (existence), z_abs_rho_change (-> Cc), z_gamma (-> new_gamma),
z_flux, |z_dagger|. divp/hinge_stable/hinge_het are left exactly as nogate --
they already have their own gated sparsity.

1. excess-variance-over-null: each of these is already a z-score against an
   analytic null (variance ~1 under "no real signal"). A term whose empirical
   variance in THIS panel exceeds 1 is picking up real structure; one that
   sits at ~1 is indistinguishable from its own null here. Weight by
   max(var(term) - 1, floor), normalized so mean weight = 1 (same total
   "budget" as the unweighted nogate, just redistributed).

2. unsupervised ensemble agreement (Parisi et al. 2014 PNAS spectral
   meta-learner idea): under conditional independence given the (unknown)
   true label, the covariance matrix of M voters has structure
   Cov ~ D (per-term noise) + lambda*lambda^T (rank-1, lambda_i ~ term i's
   discriminative power). Zero the diagonal (it's contaminated by each
   term's own noise variance) and take the leading eigenvector of the
   off-diagonal-only matrix -> that's lambda up to scale/sign. Sign-fixed
   against z_abs_t1 (existence has an unambiguous "more positive = more
   likely a true edge" direction) and normalized to mean(weight)=1 the same
   way as (1), for a fair comparison.

No y (true edges) is touched by either weighting scheme -- y is used only
afterward, to EVALUATE which scheme ranks pairs better.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob
import json
import sys

import numpy as np
import pandas as pd

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/work_in_progress/benchmark")
from benchmarks.network_benchmarks.score.todo4v2_sim_scoring import s, full_report, load_true_edges, _reg_and_flux, Z_HET_THR_NEW, REAL_NETWORK_MATRIX
from twinfer.scoring.analytic_zscores import DEFAULT_SD, z_signed

ROOT = f'{TWINFER_PROJECT_ROOT}'
JSON_GLOB = f"{ROOT}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs/*_all_results.json"

TERM_NAMES = ["z_abs_t1", "z_abs_change", "z_gamma", "z_flux", "z_dagger_abs"]


def _base_terms(dd):
    """The 5 raw z-score terms + the 3 untouched gated terms' ingredients."""
    z_abs_t1 = dd.z_abs_rho_t1.to_numpy()
    z_abs_change = dd.z_abs_rho_change.to_numpy()
    z_gamma = dd.z_gamma.to_numpy()
    z_flux = _reg_and_flux(dd, abs_valued=False)
    z_dagger = z_signed(dd.rho_cross_xy.to_numpy(), DEFAULT_SD["cross"])
    z_dagger_abs = np.abs(z_dagger)
    z_het = dd.z_het.to_numpy()
    z_div = dd.z_div.to_numpy()
    z_stable = -dd.z_rho_change.to_numpy()
    terms = dict(z_abs_t1=z_abs_t1, z_abs_change=z_abs_change, z_gamma=z_gamma,
                 z_flux=z_flux, z_dagger_abs=z_dagger_abs)
    fixed = dict(z_het=z_het, z_div=z_div, z_stable=z_stable)
    return terms, fixed


def _fixed_terms_score(fixed):
    z_het, z_div, z_stable = fixed["z_het"], fixed["z_div"], fixed["z_stable"]
    divp = -np.abs(z_div) * (z_het < Z_HET_THR_NEW).astype(float)
    hinge_stable = -np.where(z_stable > 2.326, z_stable, 0.0)
    hinge_het = np.where(np.abs(z_het) > 2.576, np.abs(z_het), 0.0)
    return divp + hinge_stable + hinge_het


def score_baseline(dd):
    terms, fixed = _base_terms(dd)
    z_abs_t1 = terms["z_abs_t1"]
    Cc = -s(terms["z_abs_change"])
    s_zg = s(terms["z_gamma"])
    new_gamma = (np.abs(s_zg) >= 1.0).astype(float) * s_zg
    return z_abs_t1 + Cc + new_gamma + s(terms["z_flux"]) + s(terms["z_dagger_abs"]) + _fixed_terms_score(fixed)


def score_excess_variance(dd, floor=0.05):
    terms, fixed = _base_terms(dd)
    standardized = {k: s(v) for k, v in terms.items()}
    raw_var = {k: np.nanvar(v) for k, v in terms.items()}  # raw (pre-s()) variance vs null-expected 1
    excess = {k: max(raw_var[k] - 1.0, floor) for k in TERM_NAMES}
    mean_excess = np.mean(list(excess.values()))
    weight = {k: excess[k] / mean_excess for k in TERM_NAMES}  # mean weight = 1, same "budget" as baseline

    score = (weight["z_abs_t1"] * standardized["z_abs_t1"]
             + weight["z_abs_change"] * -standardized["z_abs_change"]
             + weight["z_gamma"] * ((np.abs(standardized["z_gamma"]) >= 1.0).astype(float) * standardized["z_gamma"])
             + weight["z_flux"] * standardized["z_flux"]
             + weight["z_dagger_abs"] * standardized["z_dagger_abs"]
             + _fixed_terms_score(fixed))
    return score, weight


def score_spectral(dd, floor=0.05):
    terms, fixed = _base_terms(dd)
    standardized = {k: s(v) for k, v in terms.items()}
    X = np.column_stack([standardized[k] for k in TERM_NAMES])  # (n_pairs, 5)
    n = X.shape[0]

    if n < 10:  # too few pairs for a stable covariance estimate -- fall back to unweighted
        weight = {k: 1.0 for k in TERM_NAMES}
    else:
        C = np.cov(X, rowvar=False)
        np.fill_diagonal(C, 0.0)  # diagonal is contaminated by each term's own noise variance
        evals, evecs = np.linalg.eigh(C)
        lam = evecs[:, np.argmax(np.abs(evals))]  # leading eigenvector of the off-diagonal-only matrix
        exist_idx = TERM_NAMES.index("z_abs_t1")
        if lam[exist_idx] < 0:  # sign-fix against the one term with an unambiguous semantic direction
            lam = -lam
        lam = np.clip(lam, floor, None)  # a term voting the "wrong" way still gets a small positive floor
        mean_lam = np.mean(lam)
        weight = {k: lam[i] / mean_lam for i, k in enumerate(TERM_NAMES)}

    score = (weight["z_abs_t1"] * standardized["z_abs_t1"]
             + weight["z_abs_change"] * -standardized["z_abs_change"]
             + weight["z_gamma"] * ((np.abs(standardized["z_gamma"]) >= 1.0).astype(float) * standardized["z_gamma"])
             + weight["z_flux"] * standardized["z_flux"]
             + weight["z_dagger_abs"] * standardized["z_dagger_abs"]
             + _fixed_terms_score(fixed))
    return score, weight


def main():
    rows = []
    for p in sorted(glob.glob(JSON_GLOB)):
        d = json.load(open(p))
        sim_type = d.get("sim_type", "")
        tsi = d["twin_score_inputs"]
        dd = pd.DataFrame(tsi["data"], columns=tsi["columns"])
        dd = dd[dd.gene_1 != dd.gene_2].reset_index(drop=True)
        dd = dd[dd.rho_cross_xy.notna()].reset_index(drop=True)
        if len(dd) == 0:
            continue
        matrix_path = f"{ROOT}/input_data/real_world_networks/{REAL_NETWORK_MATRIX.get(sim_type, '')}"
        true_edges = load_true_edges(matrix_path, d["gene_names"])
        U = list(zip(dd.gene_1, dd.gene_2))
        y = np.array([1 if pr in true_edges else 0 for pr in U])
        if y.sum() < 1:
            continue

        m_base = full_report(score_baseline(dd), y)
        score_ev, w_ev = score_excess_variance(dd)
        m_ev = full_report(score_ev, y)
        score_sp, w_sp = score_spectral(dd)
        m_sp = full_report(score_sp, y)

        rows.append(dict(sim_type=sim_type, n_pairs=len(U),
                          baseline=m_base["auprc_x"], excess_var=m_ev["auprc_x"], spectral=m_sp["auprc_x"]))

    df = pd.DataFrame(rows)
    print("=== per sim_type ===")
    for st, g in df.groupby("sim_type"):
        print(f"{st:<20} n={len(g):3d}  baseline={g.baseline.mean():.3f}x  "
              f"excess_var={g.excess_var.mean():.3f}x  spectral={g.spectral.mean():.3f}x")

    print(f"\n=== TOTAL (n={len(df)}) ===")
    print(f"mean baseline    = {df.baseline.mean():.3f}x")
    print(f"mean excess_var  = {df.excess_var.mean():.3f}x  (wins {100*(df.excess_var>df.baseline).mean():.0f}%)")
    print(f"mean spectral    = {df.spectral.mean():.3f}x  (wins {100*(df.spectral>df.baseline).mean():.0f}%)")
    # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] df.to_csv(f"{ROOT}/code/TwINFER/work_in_progress/benchmark/todo4v2_weighted_variants_real_networks.csv", index=False)
    df.to_csv(f"{TWINFER_PROJECT_ROOT}/clean_data/benchmarks/network_benchmarks/score_results/todo4v2_weighted_variants_real_networks.csv", index=False)


if __name__ == "__main__":
    main()
