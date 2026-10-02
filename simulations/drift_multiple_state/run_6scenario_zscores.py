"""
Run infer_with_twinfer FROM SCRATCH on 6 scenarios x 20 replicates and save all
Step 1/2/3/4 metrics and the five z-scores for the gene_1-gene_2 pair:

    step1_z     gene-gene rho vs cell-scramble null
    step2_z_het twin rho_Delta(t1) vs random-pair rho_Delta  (heterogeneity)
    z_div       fixed-Delta divergence shuffle (Yuval)         (heterogeneity)
    step3_z_d   d = rho_Delta(t2) - rho_Delta(t1) vs its null
    step4_z     signed cross-correlation rho_cross(g1->g2) vs scramble

Scenarios
    no_regulation        yscher figure_2_simulations_1000 / A_B           (single file)
    A_to_B               yscher figure_2_simulations_1000 / A_to_B        (single file)
    multistate_A_to_B    yscher .../A_to_B_high_k_on + A_to_B_low_k_on    (merged)
    multistate_A_B       yscher .../A_B_high_k_on   + A_B_low_k_on        (merged)
    drift_A_to_B         hzhang drift_simulation / *_A_to_B_2_states       (single file)
    drift_A_B            hzhang drift_simulation / *_A_B_no_reg_2_states   (single file)

The gene_1-gene_2 pair is forced through every step (alpha_gene_gene_corr ~ 1,
z_score_threshold_two_states = 0) so the z-scores exist for every scenario;
the thresholds only affect routing, not the z-score values. Step 4 is also
computed directly (unconditionally) so it is defined even when the pipeline
would not reach it.

One JSON per task -> --output-dir ; aggregated by aggregate_6scenario.py.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import argparse
import glob
import json
import os
import re
import time
import traceback

import numpy as np
import pandas as pd
import numba
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from twinfer.inference.infer import infer_with_twinfer  # noqa: F401  (kept: canonical cross-check / legacy path)
from twinfer.inference.correlation_functions import (
    read_input_matrix,
    split_and_merge_simulations,
    assign_twin_id,
    calculate_pairwise_gene_gene_correlation_matrix,
    check_gene_gene_correlation_threshold,
    calculate_twin_random_correlations,
    differentiate_single_state_reg_and_multiple_states,
    identify_reg_if_multiple_states,
    get_cross_correlations,
    identify_actual_directed_edges,
    _build_cross_time_twins,
    calculate_gated_regulation_statistic,
)
from twinfer.utils.paths import get_repo_root

# T1, T2 = 1, 20
T1 = int(os.environ.get("DRIFT_T1", "1"))
T2 = int(os.environ.get("DRIFT_T2", "20"))
PAIR = ("gene_1", "gene_2")
N_REPS = 20
N_SHUFFLES = 5000          # null draws for every step (was: pipeline auto-count / 10k)

REPO = get_repo_root()
YSCHER = "/projects/b1255/yscher/Transcriptomic Distance/simulation_data/figure_2_simulations_1000"
DRIFT = f'{TWINFER_PROJECT_ROOT}/simulation_data/drift_simulation'          # archived (drift_old_calc_wrong)
DRIFT_LOWMID = f'{TWINFER_PROJECT_ROOT}/simulation_data/drift_lowmid'        # rebuilt low->mid drift (K_frozen / K_ramp)

BASE_CONFIG = {
    "n_cells": 6000,
    "simulation_time_before_division": 1000,
    "twin_simulation_time_after_division": 48,
    "twin_measurement_resolution": 1,
    "path_to_connectivity_matrix": f"{REPO}/simulation_example_input_data/connectivity_matrix_A_to_B.txt",
    "param_csv": f"{REPO}/simulation_example_input_data/median_parameter.csv",
    "rows_to_use": [[0, 1]],
}


class NpEnc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def _rep(fn):
    m = re.search(r"_rep_(\d+)", fn) or re.search(r"df_rows_0_0_(\d+)_", fn)
    return int(m.group(1)) if m else None


def _first_n_by_rep(files, n):
    by = {}
    for f in sorted(files):
        r = _rep(os.path.basename(f))
        if r is not None:
            by.setdefault(r, f)
    return [(r, by[r]) for r in sorted(by)[:n]]


def collect_tasks():
    tasks = []  # (scenario, rep_id, path or [paths])

    for scen, sub, pat in [
        ("no_regulation", "A_B", "df_rows_0_1_*_ncells_6000_A_B_rep_*.csv"),
        ("A_to_B", "A_to_B", "df_rows_0_1_*_ncells_6000_A_to_B_rep_*.csv"),
        # rebuilt low->mid drift (k_on burn-in = K-calibration param; absolute
        # k_on: low=0.12, mid=0.66); single file per rep, state col = low/mid.
        ("kramp_A_B", "__lowmid_kramp__",
         "df_rows_0_0_*_ncells_6000_A_B_no_reg_lowmid_K_ramp_*.csv"),
        ("kramp_A_to_B", "__lowmid_kramp__",
         "df_rows_0_0_*_ncells_6000_A_to_B_lowmid_K_ramp_*.csv"),
        ("kfrozen_A_B", "__lowmid_kfrozen__",
         "df_rows_0_0_*_ncells_6000_A_B_no_reg_lowmid_K_frozen_*.csv"),
        ("kfrozen_A_to_B", "__lowmid_kfrozen__",
         "df_rows_0_0_*_ncells_6000_A_to_B_lowmid_K_frozen_*.csv"),
        # ---- older scenarios, kept for reference (sim data archived to drift_old_calc_wrong) ----
        # ("drift_A_to_B", None, "df_rows_0_0_*_ncells_6000_A_to_B_2_states_*.csv"),
        # ("drift_A_B", None, "df_rows_0_0_*_ncells_6000_A_B_no_reg_2_states_*.csv"),
        # # scenario 7 / 8: static mix of drift "up" sub-population + baseline A_B
        # ("mix_upDrift_A_B", "__mix__",
        #  "df_rows_0_0_*_ncells_6000_mix_upDrift_A_B_2_states_*.csv"),
        # ("mix_upDrift_A_to_B", "__mix__",
        #  "df_rows_0_0_*_ncells_6000_mix_upDrift_A_to_B_2_states_*.csv"),
        # # scenario 9-12: alternative drift regimes (see regenerate_drift_variant.py)
        # #   fast5h  = same endpoints, k_on ramp completes in 5 h (not 15 h)
        # #   recover = k_on RISES 0.12x -> 1.0x ("up") vs stays 0.12x ("down")
        # ("drift_A_B_fast5h", "__var__",
        #  "df_rows_0_0_*_ncells_6000_A_B_no_reg_2_states_fast5h_*.csv"),
        # ("drift_A_to_B_fast5h", "__var__",
        #  "df_rows_0_0_*_ncells_6000_A_to_B_2_states_fast5h_*.csv"),
        # ("drift_A_B_recover", "__var__",
        #  "df_rows_0_0_*_ncells_6000_A_B_no_reg_2_states_recover_*.csv"),
        # ("drift_A_to_B_recover", "__var__",
        #  "df_rows_0_0_*_ncells_6000_A_to_B_2_states_recover_*.csv"),
    ]:
        if sub is None:
            root = DRIFT
        elif sub == "__mix__":
            root = f"{DRIFT}/scenario78_mix"
        elif sub == "__var__":
            root = f"{DRIFT}_variants"
        elif sub == "__lowmid_kramp__":
            root = f"{DRIFT_LOWMID}/K_ramp"
        elif sub == "__lowmid_kfrozen__":
            root = f"{DRIFT_LOWMID}/K_frozen"
        else:
            root = f"{YSCHER}/{sub}"
        for r, f in _first_n_by_rep(glob.glob(os.path.join(root, pat)), N_REPS):
            tasks.append((scen, r, f))

    for scen, hi, lo in [
        ("multistate_A_to_B", "A_to_B_high_k_on", "A_to_B_low_k_on"),
        ("multistate_A_B", "A_B_high_k_on", "A_B_low_k_on"),
    ]:
        hf = {_rep(os.path.basename(x)): x
              for x in glob.glob(f"{YSCHER}/{hi}/df_*.csv") if _rep(os.path.basename(x))}
        lf = {_rep(os.path.basename(x)): x
              for x in glob.glob(f"{YSCHER}/{lo}/df_*.csv") if _rep(os.path.basename(x))}
        common = sorted(set(hf) & set(lf))[:N_REPS]
        for r in common:
            tasks.append((scen, r, [hf[r], lf[r]]))

    # 50/50 merge of the LOW-k_on sub-population with its matching non-drift
    # baseline -- not part of the current requested scenario set; kept for reference.
    # for scen, lo_dir, base_dir in [
    #     ("multistate_lowbase_A_B",    "A_B_low_k_on",    "A_B"),
    #     ("multistate_lowbase_A_to_B", "A_to_B_low_k_on", "A_to_B"),
    # ]:
    #     lof = {_rep(os.path.basename(x)): x
    #            for x in glob.glob(f"{YSCHER}/{lo_dir}/df_*.csv")
    #            if _rep(os.path.basename(x)) is not None}
    #     baf = {_rep(os.path.basename(x)): x
    #            for x in glob.glob(f"{YSCHER}/{base_dir}/df_rows_0_1_*_ncells_6000_{base_dir}_rep_*.csv")
    #            if _rep(os.path.basename(x)) is not None}
    #     common = sorted(set(lof) & set(baf))[:N_REPS]
    #     for r in common:
    #         tasks.append((scen, r, [lof[r], baf[r]]))

    return tasks


def _load(path):
    return pd.read_csv(path) if isinstance(path, str) else split_and_merge_simulations(path)


def _flatten_matrix(mat, prefix):
    out = {}
    if mat is None:
        return out
    m = np.asarray(mat)
    genes = list(getattr(mat, "index", [])) or [f"g{i+1}" for i in range(m.shape[0])]
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            out[f"{prefix}{genes[i]}_{genes[j]}"] = float(m[i, j])
    return out


def _pv(d, pair=PAIR):
    if d is None:
        return None
    for k in (pair, (pair[1], pair[0])):
        if k in d:
            return d[k]
    return None


GENES = ["gene_1", "gene_2"]
SEED = 101010


# ======================================================================
# LEGACY PATH (kept, commented out per instruction -- do not delete).
#
# Single forced infer_with_twinfer() call. Problem: with gene_1-gene_2 forced
# through routing, the pipeline does not populate the t2 gene-gene matrix or a
# t2 Step-2 z-score, and Step 4 is skipped whenever Step 3 calls "no reg".
# Replaced by process_replicate() below, which reproduces infer.py's Steps 1-4
# on infer.py's own clone partition/seeds and evaluates Step 1 + Step 2
# (heterogeneity + divergence) at BOTH t1 and t2.
# ----------------------------------------------------------------------
# def step4_direct(sim, n_cores=1):
#     reps = sorted(sim["replicate"].unique())
#     left = sim[(sim["time_step"] == T1) & (sim["replicate"] == reps[0])].reset_index(drop=True)
#     right = sim[(sim["time_step"] == T2) & (sim["replicate"] == reps[1])].reset_index(drop=True)
#     at1, at2 = _build_cross_time_twins(left, right)
#     dpairs = [("gene_1", "gene_2"), ("gene_2", "gene_1"), ("gene_1", "gene_1"), ("gene_2", "gene_2")]
#     dmat = get_cross_correlations(at1, at2, gene_pairs=dpairs, unit="clone")
#     _, zc, _ = identify_actual_directed_edges(
#         at1, at2, dmat, gene_pairs=dpairs,
#         z_score_threshold=2.5, use_scramble=True,
#         n_shuffles=N_SHUFFLES, n_cores_to_use=n_cores, verbose=False,
#         return_z_scores=True, return_rho_cross_null=True,
#         prepare_rho_cross_null=True, unit="clone",
#     )
#     return {
#         "step4_z_1to2": zc.get(("gene_1", "gene_2")),
#         "step4_z_2to1": zc.get(("gene_2", "gene_1")),
#         "step4_rho_cross_1to2": float(dmat.loc["gene_1", "gene_2"]),
#         "step4_rho_cross_2to1": float(dmat.loc["gene_2", "gene_1"]),
#     }
#
# def process_replicate_via_infer(path, scenario, rep_id, n_cores=1):
#     numba.set_num_threads(1)
#     with threadpool_limits(limits=1):
#         sim = _load(path)
#         res = infer_with_twinfer(
#             None, data=sim, is_simulation_data=True, merge_to_multiple_states=False,
#             base_config=BASE_CONFIG, t1=T1, t2=T2,
#             match_sim_details=False, check_for_steady_state=False,
#             seed=101010, plot=False, verbose=False, ranked_list=False,
#             return_diagnostics=True, n_cores=n_cores,
#             n_shuffles_step1=N_SHUFFLES, n_shuffles_step2=N_SHUFFLES,
#             n_shuffles_stage3=N_SHUFFLES, n_shuffles_direction=N_SHUFFLES,
#             n_shuffles_fanout=N_SHUFFLES,
#             alpha_gene_gene_corr=0.999, z_score_threshold_two_states=0.0,
#             use_scramble_cross_correlation=True,
#         )
#         diag = res["diagnostics"]
#         rec = {"scenario": scenario, "rep_id": rep_id}
#         rec["step1_z"]     = _pv(diag["step1"]["z_scores"])
#         rec["step2_z_het"] = _pv(diag["step2"].get("z_het") or diag["step2"].get("z_scores"))
#         rec["z_div"]       = _pv(diag["step2"].get("z_div") or diag["step2"].get("z_score_div"))
#         s3 = _pv(res.get("stage3", {})) or {}
#         rec["step3_z_d"] = s3.get("z_d")
#         rec.update(step4_direct(sim, n_cores=n_cores))
#         rec["step4_z"] = rec["step4_z_1to2"]
#     return rec
# ======================================================================


def _partition(sim, seed=SEED):
    """Reproduce infer.py's simulation clone split (infer.py lines ~388-432).

    seed 101010 -> permute clones -> first 1/4 = t1 twins, second 1/4 = t2 twins,
    second 1/2 = cross-time (one rep-0 cell at t1 paired to one rep-1 cell at t2).
    rho measurements = within-time twins + the cross-time timepoint cells.
    """
    rng = np.random.default_rng(seed)
    clone_ids = sim["clone_id"].drop_duplicates().to_numpy()
    sh = rng.permutation(clone_ids)
    n = len(sh) // 4
    t1c, t2c, ac = sh[:n], sh[n:2 * n], sh[2 * n:]
    reps = sorted(sim["replicate"].unique())

    t1_twins_raw = sim[sim["clone_id"].isin(t1c) & (sim["time_step"] == T1)].copy()
    t2_twins_raw = sim[sim["clone_id"].isin(t2c) & (sim["time_step"] == T2)].copy()
    ac_left = sim[sim["clone_id"].isin(ac) & (sim["time_step"] == T1)
                  & (sim["replicate"] == reps[0])].copy()
    ac_right = sim[sim["clone_id"].isin(ac) & (sim["time_step"] == T2)
                   & (sim["replicate"] == reps[1])].copy()

    return {
        "t1_twins": assign_twin_id(t1_twins_raw).reset_index(drop=True),
        "t2_twins": assign_twin_id(t2_twins_raw).reset_index(drop=True),
        "rho_t1": pd.concat([t1_twins_raw, ac_left], ignore_index=True),
        "rho_t2": pd.concat([t2_twins_raw, ac_right], ignore_index=True),
        "ac_left": ac_left.reset_index(drop=True),
        "ac_right": ac_right.reset_index(drop=True),
    }


def _step1(rho_meas, base_seed):
    mat = calculate_pairwise_gene_gene_correlation_matrix(rho_meas, GENES, use_clone=True)
    no_reg, pot, null_stats, z = check_gene_gene_correlation_threshold(
        rho_meas, mat, GENES, use_scramble=True, z_score_threshold=2.576,
        verbose=False, use_clone=True, n_shuffles=N_SHUFFLES, base_seed=base_seed,
    )
    ns = null_stats.get(PAIR)
    return {"z": _pv(z), "rho": float(mat.loc[PAIR]),
            "null_mean": float(ns[0]) if ns else None,
            "null_std": float(ns[1]) if ns else None}, mat


def _step2(rho_meas, twins, rand_seed, div_seed, n_cores):
    twin_mat, rand_mat = calculate_twin_random_correlations(
        rho_meas, twins, GENES, random_state=rand_seed, unit="clone")
    _, _, null_stats, z_het, div = differentiate_single_state_reg_and_multiple_states(
        rho_meas, [PAIR], twin_mat, rand_mat, GENES,
        z_score_threshold=1e9, verbose=False, unit="clone",
        n_shuffles=N_SHUFFLES, n_cores_to_use=n_cores,
        divergence_random_state=div_seed, return_divergence_details=True,
    )
    ns = null_stats.get(PAIR)
    dv = _pv(div) or {}
    return {
        "z_het": _pv(z_het),
        "z_div": dv.get("z_div"),
        "rho_delta": float(twin_mat.loc[PAIR]),
        "rho_delta_random": float(rand_mat.loc[PAIR]),
        "het_null_mean": float(ns[0]) if ns else None,
        "het_null_std": float(ns[1]) if ns else None,
        "div_null_mean": dv.get("null_mean"),
        "div_null_std": dv.get("null_std"),
    }, twin_mat, rand_mat


def _step4(ac_left, ac_right, n_cores):
    at1, at2 = _build_cross_time_twins(ac_left, ac_right)
    dpairs = [("gene_1", "gene_2"), ("gene_2", "gene_1"),
              ("gene_1", "gene_1"), ("gene_2", "gene_2")]
    dmat = get_cross_correlations(at1, at2, gene_pairs=dpairs, unit="clone")
    _, zc, xnull = identify_actual_directed_edges(
        at1, at2, dmat, gene_pairs=dpairs, z_score_threshold=2.5,
        use_scramble=True, n_shuffles=N_SHUFFLES, n_cores_to_use=n_cores,
        verbose=False, return_z_scores=True, return_rho_cross_null=True,
        prepare_rho_cross_null=True, unit="clone",
    )

    # z_gamma: asymmetry between the two cross-correlation directions,
    #   gamma = |rho_cross(g1->g2)| - |rho_cross(g2->g1)|   (infer.py Step 4).
    # Computed here unconditionally from the permutation null (the pipeline only
    # forms it when BOTH directions are individually significant).
    z_gamma_1to2 = None
    try:
        d12 = xnull.get(("gene_1", "gene_2")) or {}
        d21 = xnull.get(("gene_2", "gene_1")) or {}
        n12 = np.abs(np.asarray(d12["null_values"], dtype=float))
        n21 = np.abs(np.asarray(d21["null_values"], dtype=float))
        m = min(len(n12), len(n21))
        gnull = (n12[:m] - n21[:m])
        gnull = gnull[np.isfinite(gnull)]
        gamma_obs = abs(float(dmat.loc["gene_1", "gene_2"])) - abs(float(dmat.loc["gene_2", "gene_1"]))
        sd = float(np.std(gnull, ddof=1))
        if sd > 0 and gnull.size > 1:
            z_gamma_1to2 = (gamma_obs - float(np.mean(gnull))) / sd
    except Exception:  # noqa: BLE001
        pass

    return {
        "step4_z_1to2": zc.get(("gene_1", "gene_2")),
        "step4_z_2to1": zc.get(("gene_2", "gene_1")),
        "step4_rho_cross_1to2": float(dmat.loc["gene_1", "gene_2"]),
        "step4_rho_cross_2to1": float(dmat.loc["gene_2", "gene_1"]),
        # gamma is a signed asymmetry; the reverse direction is its negation.
        "z_gamma_1to2": z_gamma_1to2,
        "z_gamma_2to1": (None if z_gamma_1to2 is None else -z_gamma_1to2),
    }


def process_replicate(path, scenario, rep_id, n_cores=1):
    """Faithful reproduction of infer.py Steps 1-4 for gene_1-gene_2, with Step 1
    and Step 2 (heterogeneity + divergence) evaluated at BOTH t1 and t2."""
    numba.set_num_threads(1)
    with threadpool_limits(limits=1):
        sim = _load(path)
        P = _partition(sim)

        rec = {
            "scenario": scenario, "rep_id": rep_id,
            "t1": T1, "t2": T2,
            "path": os.path.basename(path) if isinstance(path, str)
            else [os.path.basename(p) for p in path],
        }

        # ---- Step 1: gene-gene rho vs cell-scramble null (t1, t2) ----
        s1a, gene_t1 = _step1(P["rho_t1"], SEED + 1)
        s1b, gene_t2 = _step1(P["rho_t2"], SEED + 2)
        rec["step1_z_t1"], rec["step1_z_t2"] = s1a["z"], s1b["z"]
        rec["rho_t1"], rec["rho_t2"] = s1a["rho"], s1b["rho"]
        rec["step1_null_mean_t1"], rec["step1_null_std_t1"] = s1a["null_mean"], s1a["null_std"]
        rec["step1_null_mean_t2"], rec["step1_null_std_t2"] = s1b["null_mean"], s1b["null_std"]

        # ---- Step 2: twin rho_Delta vs random (het) + divergence (t1, t2) ----
        s2a, twin_t1, rand_t1 = _step2(P["rho_t1"], P["t1_twins"], SEED + 271829, SEED + 271832, n_cores)
        s2b, twin_t2, rand_t2 = _step2(P["rho_t2"], P["t2_twins"], SEED + 271830, SEED + 271833, n_cores)
        rec["step2_z_het_t1"], rec["step2_z_het_t2"] = s2a["z_het"], s2b["z_het"]
        rec["z_div_t1"], rec["z_div_t2"] = s2a["z_div"], s2b["z_div"]
        rec["rho_delta_t1"], rec["rho_delta_t2"] = s2a["rho_delta"], s2b["rho_delta"]
        rec["rho_delta_random_t1"], rec["rho_delta_random_t2"] = s2a["rho_delta_random"], s2b["rho_delta_random"]
        for k in ("het_null_mean", "het_null_std", "div_null_mean", "div_null_std"):
            rec[f"{k}_t1"], rec[f"{k}_t2"] = s2a[k], s2b[k]

        # ---- Step 2b: heterogeneity-gated regulation statistic (t1, t2) ----
        #   rho_reg(lambda) = gene-gene rho (2n) - lambda * cross-sibling rho (2n),
        #   lambda = min(1, |z_het|/2.33); one null serves every lambda.
        g_t1 = calculate_gated_regulation_statistic(
            P["t1_twins"], PAIR[0], PAIR[1], s2a["z_het"],
            n_shuffles=N_SHUFFLES, random_state=SEED + 271840)
        g_t2 = calculate_gated_regulation_statistic(
            P["t2_twins"], PAIR[0], PAIR[1], s2b["z_het"],
            n_shuffles=N_SHUFFLES, random_state=SEED + 271841)
        rec["z_reg_gated_t1"], rec["z_reg_gated_t2"] = g_t1["z_reg_gated"], g_t2["z_reg_gated"]
        rec["lambda_gated_t1"], rec["lambda_gated_t2"] = g_t1["lambda"], g_t2["lambda"]
        rec["rho_reg_gated_t1"], rec["rho_reg_gated_t2"] = g_t1["rho_reg_gated"], g_t2["rho_reg_gated"]
        rec["rho_gene_gene_2n_t1"], rec["rho_gene_gene_2n_t2"] = g_t1["rho_gene_gene_2n"], g_t2["rho_gene_gene_2n"]
        rec["rho_cross_2n_t1"], rec["rho_cross_2n_t2"] = g_t1["rho_cross_2n"], g_t2["rho_cross_2n"]

        # ---- Step 3: d = rho_Delta(t2) - rho_Delta(t1) vs null (both timepoints) ----
        _, _, stage3 = identify_reg_if_multiple_states(
            twin_t1, twin_t2, rand_t1, rand_t2, [PAIR], GENES,
            P["t1_twins"], P["t2_twins"], t1_raw=P["rho_t1"], t2_raw=P["rho_t2"],
            alpha=0.01, n_shuffles=N_SHUFFLES, shuffle_seed=SEED + 271831,
            unit="clone", n_cores_to_use=n_cores,
        )
        s3 = _pv(stage3) or {}
        rec["step3_z_d"] = s3.get("z_d")
        rec["step3_z_d_div"] = s3.get("z_d_div")
        rec["step3_d"] = s3.get("d")
        rec["step3_null_mean"] = s3.get("null_mean")
        rec["step3_null_std"] = s3.get("null_std")
        rec["step3_z_critical"] = s3.get("z_critical")

        # ---- Step 4: signed cross-correlation z, both directions (both timepoints) ----
        rec.update(_step4(P["ac_left"], P["ac_right"], n_cores))
        rec["step4_z"] = rec["step4_z_1to2"]        # canonical: g1 -> g2

        # ---- flattened correlation matrices ----
        for name, mat in [("gene_t1", gene_t1), ("gene_t2", gene_t2),
                          ("twin_delta_t1", twin_t1), ("twin_delta_t2", twin_t2),
                          ("random_delta_t1", rand_t1), ("random_delta_t2", rand_t2)]:
            rec.update(_flatten_matrix(mat, f"{name}_"))

        # ---- threshold-derived classification (pipeline defaults) ----
        s1 = rec["step1_z_t1"]
        het = rec["step2_z_het_t1"]
        zd, zdc = rec["step3_z_d"], rec["step3_z_critical"]
        if s1 is None or abs(s1) <= 2.576:
            rec["derived_classification"] = "no_regulation"
        elif het is not None and abs(het) > 5.0:
            rec["derived_classification"] = ("multiple_states_and_reg"
                                             if (zd is not None and zdc is not None and zd > zdc)
                                             else "multiple_states_no_reg")
        else:
            rec["derived_classification"] = "single_state_regulation"

    return rec


def run_one(path, scenario, rep_id, out_dir, n_cores):
    out = os.path.join(out_dir, f"{scenario}_rep_{rep_id}.json")
    if os.path.exists(out):
        return out, "skipped"
    try:
        rec = process_replicate(path, scenario, rep_id, n_cores=n_cores)
        os.makedirs(out_dir, exist_ok=True)
        with open(out, "w") as f:
            json.dump(rec, f, cls=NpEnc, indent=2)
        def _f(x):
            return "None" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.2f}"
        return out, (f"OK  s1(t1/t2)={_f(rec['step1_z_t1'])}/{_f(rec['step1_z_t2'])}  "
                     f"s2het(t1/t2)={_f(rec['step2_z_het_t1'])}/{_f(rec['step2_z_het_t2'])}  "
                     f"zdiv(t1/t2)={_f(rec['z_div_t1'])}/{_f(rec['z_div_t2'])}  "
                     f"zreg_gated(t1/t2)={_f(rec['z_reg_gated_t1'])}/{_f(rec['z_reg_gated_t2'])}  "
                     f"s3={_f(rec['step3_z_d'])}  s4(1>2/2>1)={_f(rec['step4_z_1to2'])}/{_f(rec['step4_z_2to1'])}")
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return out, f"ERROR {type(e).__name__}: {e}"


def _aggregate(out_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(out_dir, "*_rep_*.json"))):
        with open(f) as fh:
            rows.append(json.load(fh))
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values(["scenario", "rep_id"])
    csv = os.path.join(out_dir, "six_scenario_zscores.csv")
    df.to_csv(csv, index=False)
    print(f"\naggregated {len(df)} rows -> {csv}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n-cores", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=1, help="parallel replicates (joblib)")
    ap.add_argument("--only", type=str, default=None, help="scenario:rep,scenario:rep")
    ap.add_argument("--list-tasks", action="store_true")
    args = ap.parse_args()

    tasks = collect_tasks()
    if args.list_tasks:
        for s, r, _ in tasks:
            print(f"{s}:{r}")
        raise SystemExit(0)
    if args.only:
        want = set(args.only.split(","))
        tasks = [t for t in tasks if f"{t[0]}:{t[1]}" in want]

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{len(tasks)} tasks -> {args.output_dir} ({args.jobs} parallel)  "
          f"t1={T1}h t2={T2}h", flush=True)

    def _do(idx, scen, rep, path):
        t0 = time.time()
        out, status = run_one(path, scen, rep, args.output_dir, args.n_cores)
        print(f"[{idx+1}/{len(tasks)}] {scen} rep {rep}: {status}  ({time.time()-t0:.1f}s)", flush=True)

    if args.jobs > 1:
        Parallel(n_jobs=args.jobs, backend="loky")(
            delayed(_do)(i, s, r, p) for i, (s, r, p) in enumerate(tasks))
    else:
        for i, (s, r, p) in enumerate(tasks):
            _do(i, s, r, p)

    _aggregate(args.output_dir)
