"""
For the 100 h "low/mid" drift (`recover` variant: one twin sub-population holds
k_on at 0.12x, the other ramps 0.12x -> 1.0x baseline over 15 h), track every
TwINFER z-score and its underlying correlation for the gene_1-gene_2 pair as a
function of twin time -- one point per hour, mean of the 3 replicates.

Per hour t (Step 3 / Step 4 are referenced to a fixed t1 = REF_T1):
  rho_gene            Step 1 cell-level rho(g1,g2)                     -> z_gene
  rho_delta           Step 2 twin  hat{rho}_Delta(g1,g2)              -> z_het
  rho_delta_random    Step 2 random-pair null rho_Delta
                      Step 2 divergence shuffle                        -> z_reg (= z_div)
  d = rho_D(t)-rho_D(t1)   Step 3                                      -> z_d
  rho_cross(g1->g2 , t1->t)  Step 4                                    -> z_1to2 / z_2to1 / z_gamma

Input : simulation_data/drift_simulation_variants/lowmid_100h/
Output: analysis_data/drift_inference/lowmid_100h/
        lowmid_100h_zscores_vs_time{,_perrep}.csv  +  two PNGs
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from simulations.drift_multiple_state import run_6scenario_zscores as R
R.N_SHUFFLES = 400
from simulations.drift_multiple_state.run_6scenario_zscores import (_step1, _step2, _step4, SEED, GENES, PAIR, _pv)
from twinfer.inference.correlation_functions import (
    assign_twin_id, calculate_twin_random_correlations, identify_reg_if_multiple_states)

SIM = f'{TWINFER_PROJECT_ROOT}/simulation_data/drift_simulation_variants/lowmid_100h'
OUT = f'{TWINFER_PROJECT_ROOT}/analysis_data/drift_inference/lowmid_100h'
NETS = [("A, B (unregulated)", "A_B_no_reg_2_states_recover"),
        ("A->B (regulated)",   "A_to_B_2_states_recover")]
TMAX = 100
REF_T1 = 1


def _rep(f):
    return int(re.search(r"df_rows_0_0_(\d+)_", os.path.basename(f)).group(1))


def one_file(net_label, path):
    df = pd.read_csv(path)
    rep = _rep(path)
    reps = sorted(df["replicate"].unique())

    raw1 = df[df["time_step"] == REF_T1].reset_index(drop=True)
    tw1 = assign_twin_id(raw1).reset_index(drop=True)
    twin1_mat, rand1_mat = calculate_twin_random_correlations(
        raw1, tw1, GENES, random_state=SEED + 271829, unit="clone")
    ac_left = df[(df["time_step"] == REF_T1) & (df["replicate"] == reps[0])].reset_index(drop=True)

    rows = []
    for t in [x for x in range(1, TMAX + 1) if (df["time_step"] == x).any()]:
        raw = df[df["time_step"] == t].reset_index(drop=True)
        tw = assign_twin_id(raw).reset_index(drop=True)
        s1, _ = _step1(raw, SEED + 1)
        s2, twin_mat, rand_mat = _step2(raw, tw, SEED + 271829, SEED + 271832, n_cores=4)

        _, _, stage3 = identify_reg_if_multiple_states(
            twin1_mat, twin_mat, rand1_mat, rand_mat, [PAIR], GENES,
            tw1, tw, t1_raw=raw1, t2_raw=raw,
            alpha=0.01, n_shuffles=R.N_SHUFFLES, shuffle_seed=SEED + 271831,
            unit="clone", n_cores_to_use=4)
        s3 = _pv(stage3) or {}

        ac_right = df[(df["time_step"] == t) & (df["replicate"] == reps[1])].reset_index(drop=True)
        try:
            s4 = _step4(ac_left, ac_right, n_cores=4)
        except Exception:  # noqa: BLE001
            s4 = {}

        rows.append({
            "network": net_label, "rep": rep, "time": t,
            "rho_gene": s1["rho"], "z_gene": s1["z"],
            "rho_delta": s2["rho_delta"], "rho_delta_random": s2["rho_delta_random"],
            "z_het": s2["z_het"], "z_reg": s2["z_div"],
            "step3_d": s3.get("d"), "z_d": s3.get("z_d"),
            "rho_cross_1to2": s4.get("step4_rho_cross_1to2"),
            "rho_cross_2to1": s4.get("step4_rho_cross_2to1"),
            "z_1to2": s4.get("step4_z_1to2"), "z_2to1": s4.get("step4_z_2to1"),
            "z_gamma": s4.get("z_gamma_1to2"),
        })
    print(f"  [{net_label} rep {rep}] {len(rows)} timepoints", flush=True)
    return rows


CORR_SPECS = [("rho_gene", "gene-gene rho", "#000000"),
              ("rho_delta", "twin rho_Delta", "#0072B2"),
              ("rho_delta_random", "random-pair rho_Delta", "#D55E00"),
              ("step3_d", "d = rho_D(t) - rho_D(t1)", "#009E73"),
              ("rho_cross_1to2", "cross rho g1->g2", "#CC79A7"),
              ("rho_cross_2to1", "cross rho g2->g1", "#56B4E9")]
Z_SPECS = [("z_gene", "Step 1  z (gene-gene)"),
           ("z_het", "Step 2  z_het"),
           ("z_reg", "Step 2  z_regulation (divergence)"),
           ("z_d", "Step 3  z_d"),
           ("z_1to2", "Step 4  z  g1->g2"),
           ("z_2to1", "Step 4  z  g2->g1"),
           ("z_gamma", "Step 4  z_gamma")]


def series_fig(agg, specs, ylabel, title, stem, thr=None):
    n = len(specs)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.3 * ncol, 3.4 * nrow), squeeze=False, sharex=True)
    for ax, spec in zip(axes.flat, specs):
        col = spec[0]
        for net_label, _ in NETS:
            a = agg[agg.network == net_label].sort_values("time")
            m, s = a[f"{col}_mean"], a[f"{col}_std"]
            cc = "#0072B2" if "regulated" in net_label and "un" not in net_label else "#D55E00"
            ax.plot(a.time, m, "-o", ms=3, color=cc, label=net_label)
            ax.fill_between(a.time, m - s, m + s, color=cc, alpha=0.13)
        ax.axhline(0, color="0.7", lw=0.8)
        if thr:
            for y in (thr, -thr):
                ax.axhline(y, ls="--", lw=1, color="0.5")
        ax.set_title(spec[1], fontsize=10)
        ax.grid(alpha=0.25)
        ax.set_ylabel(ylabel)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("twin time after division (h)")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(OUT, stem)
    fig.savefig(p, dpi=170, bbox_inches="tight")
    print("saved", p)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    tasks = [(nl, f) for nl, tag in NETS
             for f in sorted(glob.glob(os.path.join(SIM, f"df_rows_0_0_*_{tag}_*.csv")), key=_rep)]
    print(f"{len(tasks)} sim files -> {OUT}  (Step3/4 referenced to t1={REF_T1})", flush=True)

    results = Parallel(n_jobs=min(6, len(tasks)), backend="loky")(
        delayed(one_file)(nl, f) for nl, f in tasks)
    df = pd.DataFrame([r for sub in results for r in sub])
    df.to_csv(os.path.join(OUT, "lowmid_100h_zscores_vs_time_perrep.csv"), index=False)

    val_cols = [c for c in df.columns if c not in ("network", "rep", "time")]
    agg = df.groupby(["network", "time"])[val_cols].agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(os.path.join(OUT, "lowmid_100h_zscores_vs_time.csv"), index=False)

    series_fig(agg, CORR_SPECS, r"$\rho$",
               f"Low/mid drift (recover), 100 h  -  correlations vs time  -  mean of 3 reps  "
               f"(Step3/4 ref t1={REF_T1} h)",
               "lowmid_100h_correlations_vs_time.png")
    series_fig(agg, Z_SPECS, "z",
               f"Low/mid drift (recover), 100 h  -  z-scores vs time  -  mean of 3 reps  "
               f"(Step3/4 ref t1={REF_T1} h)",
               "lowmid_100h_zscores_vs_time.png", thr=2.33)
