"""
Low/mid drift (`recover` variant), 100 h: Step-1 and Step-2 z-scores + their
correlations for gene_1-gene_2, one point per hour, mean of the 3 replicates.
Both networks (A_B unregulated, A_to_B regulated).  (Step 3/4 omitted.)

  Step 1 : gene-gene rho(g1,g2)              -> z_gene
  Step 2 : twin  hat{rho}_Delta(g1,g2)       -> z_het
           random-pair rho_Delta (null)
           divergence shuffle                -> z_reg  (= z_div)

Output: analysis_data/drift_inference/lowmid_100h/
        lowmid_100h_step12_vs_time{,_perrep}.csv  +  two PNGs
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
R.N_SHUFFLES = 5000
from simulations.drift_multiple_state.run_6scenario_zscores import _step1, _step2, SEED
from twinfer.inference.correlation_functions import assign_twin_id

import argparse as _argparse
_ap = _argparse.ArgumentParser()
_ap.add_argument("--sim-dir", default=f'{TWINFER_PROJECT_ROOT}/simulation_data/drift_simulation_variants/lowmid_100h')
_ap.add_argument("--out-dir", default=f'{TWINFER_PROJECT_ROOT}/analysis_data/drift_inference/lowmid_100h')
_ap.add_argument("--tmax", type=int, default=100)
_ap.add_argument("--every", type=int, default=5, help="timepoint spacing in hours")
_ap.add_argument("--jobs", type=int, default=6)
_a = _ap.parse_args()

SIM = _a.sim_dir
OUT = _a.out_dir
NETS = [("A, B (unregulated)", "A_B_no_reg_2_states_recover", "#D55E00"),
        ("A->B (regulated)",   "A_to_B_2_states_recover",   "#0072B2")]
TIMES = list(range(_a.every, _a.tmax + 1, _a.every))


def _rep(f):
    return int(re.search(r"df_rows_0_0_(\d+)_", os.path.basename(f)).group(1))


def one_file(net_label, path):
    df = pd.read_csv(path)
    rep = _rep(path)
    rows = []
    for t in [x for x in TIMES if (df["time_step"] == x).any()]:
        raw = df[df["time_step"] == t].reset_index(drop=True)
        tw = assign_twin_id(raw).reset_index(drop=True)
        s1, _ = _step1(raw, SEED + 1)
        s2, _, _ = _step2(raw, tw, SEED + 271829, SEED + 271832, n_cores=4)
        rows.append({"network": net_label, "rep": rep, "time": t,
                     "rho_gene": s1["rho"], "z_gene": s1["z"],
                     "rho_delta": s2["rho_delta"], "rho_delta_random": s2["rho_delta_random"],
                     "z_het": s2["z_het"], "z_reg": s2["z_div"]})
    print(f"  [{net_label} rep {rep}] {len(rows)} timepoints", flush=True)
    return rows


def series_fig(agg, specs, ylabel, title, stem, thr=None):
    n = len(specs)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.3 * ncol, 3.6 * nrow), squeeze=False, sharex=True)
    for ax, (col, name) in zip(axes.flat, specs):
        for net_label, _, cc in NETS:
            a = agg[agg.network == net_label].sort_values("time")
            m, s = a[f"{col}_mean"], a[f"{col}_std"]
            ax.plot(a.time, m, "-o", ms=3, color=cc, label=net_label)
            ax.fill_between(a.time, m - s, m + s, color=cc, alpha=0.13)
        ax.axhline(0, color="0.7", lw=0.8)
        if thr:
            for y in (thr, -thr):
                ax.axhline(y, ls="--", lw=1, color="0.5")
        ax.set_title(name, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("twin time after division (h)")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUT, stem)
    fig.savefig(p, dpi=170, bbox_inches="tight")
    print("saved", p)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    STEM = f"lowmid_{_a.tmax}h_step12"
    HRS = f"{_a.tmax} h"
    tasks = [(nl, f) for nl, tag, _ in NETS
             for f in sorted(glob.glob(os.path.join(SIM, f"df_rows_0_0_*_{tag}_*.csv")), key=_rep)]
    print(f"{len(tasks)} sim files, {len(TIMES)} timepoints ({TIMES[0]}..{TIMES[-1]} every {_a.every} h) -> {OUT}", flush=True)

    results = Parallel(n_jobs=min(_a.jobs, len(tasks)), backend="loky")(
        delayed(one_file)(nl, f) for nl, f in tasks)
    df = pd.DataFrame([r for sub in results for r in sub])
    df.to_csv(os.path.join(OUT, f"{STEM}_vs_time_perrep.csv"), index=False)

    vcols = [c for c in df.columns if c not in ("network", "rep", "time")]
    agg = df.groupby(["network", "time"])[vcols].agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(os.path.join(OUT, f"{STEM}_vs_time.csv"), index=False)

    series_fig(agg,
               [("z_gene", "Step 1  z  (gene-gene rho)"),
                ("z_het", "Step 2  z_het"),
                ("z_reg", "Step 2  z_regulation (divergence)")],
               "z", f"Low/mid drift (recover), {HRS}  -  Step 1/2 z-scores vs time  -  mean of 3 reps",
               f"{STEM}_zscores_vs_time.png", thr=2.33)
    series_fig(agg,
               [("rho_gene", "gene-gene rho(g1,g2)"),
                ("rho_delta", "twin rho_Delta"),
                ("rho_delta_random", "random-pair rho_Delta")],
               r"$\rho$", f"Low/mid drift (recover), {HRS}  -  Step 1/2 correlations vs time  -  mean of 3 reps",
               f"{STEM}_correlations_vs_time.png")
