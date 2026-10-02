"""
For the 2-state drift simulations, plot the ratio of the population-mean
expression as a function of time, for gene_1/gene_2 x mRNA/protein:

    up / down       : drift "up" clones   vs drift "down" clones   (pure drift sim)
    up / baseline    : drift "up" clones   vs the matching non-drift steady-state
                      population (yscher A_B or A_to_B) -- i.e. the "up / mix" half

Averaged over the first N_REPS replicates.

Output: analysis_data/drift_inference/state_ratios/state_ratios_vs_time.png / .csv
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

DRIFT = f'{TWINFER_PROJECT_ROOT}/simulation_data/drift_simulation'
YSCHER = "/projects/b1255/yscher/Transcriptomic Distance/simulation_data/figure_2_simulations_1000"
OUT = f'{TWINFER_PROJECT_ROOT}/analysis_data/drift_inference/state_ratios'
N_REPS = 5
VARS = ["gene_1_mRNA", "gene_2_mRNA", "gene_1_protein", "gene_2_protein"]

NETS = [
    ("A,B (unregulated)", "A_B_no_reg_2_states",
     f"{YSCHER}/A_B",    "df_rows_0_1_*_ncells_6000_A_B_rep_*.csv"),
    ("A->B (regulated)",  "A_to_B_2_states",
     f"{YSCHER}/A_to_B", "df_rows_0_1_*_ncells_6000_A_to_B_rep_*.csv"),
]


def _rep(f):
    m = re.search(r"df_rows_0_0_(\d+)_", f) or re.search(r"_rep_(\d+)", f)
    return int(m.group(1)) if m else None


def by_rep(files):
    out = {}
    for f in sorted(files):
        r = _rep(os.path.basename(f))
        if r is not None:
            out.setdefault(r, f)
    return out


def mean_by_state_time(path, states_col="state"):
    d = pd.read_csv(path)
    if states_col in d.columns:
        return d.groupby([states_col, "time_step"])[VARS].mean()
    return d.groupby("time_step")[VARS].mean()


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    rows = []

    for net_label, drift_tag, base_dir, base_pat in NETS:
        drift_by = by_rep(glob.glob(os.path.join(DRIFT, f"df_rows_0_0_*_{drift_tag}_*.csv")))
        base_by = by_rep(glob.glob(os.path.join(base_dir, base_pat)))
        drift_reps = sorted(drift_by)[:N_REPS]
        base_reps = sorted(base_by)[:N_REPS]

        up_acc, down_acc, base_acc = [], [], []
        for k, dr in enumerate(drift_reps):
            g = mean_by_state_time(drift_by[dr])
            up_acc.append(g.loc["up"])
            down_acc.append(g.loc["down"])
            if k < len(base_reps):
                b = pd.read_csv(base_by[base_reps[k]]).groupby("time_step")[VARS].mean()
                base_acc.append(b)

        up = pd.concat(up_acc).groupby(level=0).mean()
        down = pd.concat(down_acc).groupby(level=0).mean()
        base = pd.concat(base_acc).groupby(level=0).mean()
        times = sorted(set(up.index) & set(down.index) & set(base.index))
        up, down, base = up.loc[times], down.loc[times], base.loc[times]

        for v in VARS:
            for t in times:
                rows.append({"network": net_label, "var": v, "time": t,
                             "mean_up": up.loc[t, v], "mean_down": down.loc[t, v],
                             "mean_baseline": base.loc[t, v],
                             "ratio_up_down": up.loc[t, v] / down.loc[t, v],
                             "ratio_up_baseline": up.loc[t, v] / base.loc[t, v]})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "state_ratios_vs_time.csv"), index=False)
    print("wrote", os.path.join(OUT, "state_ratios_vs_time.csv"))

    import matplotlib.ticker as mticker

    # ---- figure 1: ratios vs time (hourly) ----
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=True)
    for ri, (net_label, *_ ) in enumerate(NETS):
        for ci, v in enumerate(VARS):
            ax = axes[ri][ci]
            s = df[(df.network == net_label) & (df["var"] == v)].sort_values("time")
            ax.plot(s.time, s.ratio_up_down, "-o", ms=3, color="#0072B2", label="up / down")
            ax.plot(s.time, s.ratio_up_baseline, "-s", ms=3, color="#D55E00", label="up / baseline (mix)")
            ax.axhline(1, color="0.7", lw=0.8)
            ax.set_title(f"{net_label}\n{v}", fontsize=10)
            ax.set_yscale("log")
            ax.xaxis.set_major_locator(mticker.MultipleLocator(6))
            ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
            ax.grid(alpha=0.30, which="major")
            ax.grid(alpha=0.12, which="minor")
            if ci == 0:
                ax.set_ylabel("mean ratio  (log)")
            if ri == 1:
                ax.set_xlabel("time after division (h)")
            if ri == 0 and ci == 0:
                ax.legend(fontsize=8)
    fig.suptitle(f"Drift 2-state: population-mean ratio vs time, every hour  (mean of {N_REPS} reps)  ·  "
                 "up/down = pure drift,  up/baseline = drift-up vs non-drift steady state", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUT, "state_ratios_vs_time.png")
    fig.savefig(p, dpi=170, bbox_inches="tight")
    print("saved", p)

    # ---- figure 2: absolute population means vs time (hourly) ----
    fig2, ax2 = plt.subplots(2, 4, figsize=(20, 9), sharex=True)
    for ri, (net_label, *_ ) in enumerate(NETS):
        for ci, v in enumerate(VARS):
            ax = ax2[ri][ci]
            s = df[(df.network == net_label) & (df["var"] == v)].sort_values("time")
            ax.plot(s.time, s.mean_up, "-o", ms=3, color="#0072B2", label="up (drift)")
            ax.plot(s.time, s.mean_down, "-o", ms=3, color="#009E73", label="down (drift)")
            ax.plot(s.time, s.mean_baseline, "-s", ms=3, color="#D55E00", label="baseline (non-drift)")
            ax.set_title(f"{net_label}\n{v}", fontsize=10)
            ax.xaxis.set_major_locator(mticker.MultipleLocator(6))
            ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
            ax.grid(alpha=0.30, which="major")
            ax.grid(alpha=0.12, which="minor")
            if ci == 0:
                ax.set_ylabel("population mean")
            if ri == 1:
                ax.set_xlabel("time after division (h)")
            if ri == 0 and ci == 0:
                ax.legend(fontsize=8)
    fig2.suptitle(f"Drift 2-state: population-mean expression vs time, every hour  (mean of {N_REPS} reps)", fontsize=12)
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    p2 = os.path.join(OUT, "state_means_vs_time.png")
    fig2.savefig(p2, dpi=170, bbox_inches="tight")
    print("saved", p2)

    # compact printout at key timepoints
    for t in [1, 10, 20, 48]:
        print(f"\n--- t = {t} h ---")
        sub = df[df.time == t].copy()
        print(sub[["network", "var", "ratio_up_down", "ratio_up_baseline"]]
              .to_string(index=False, float_format=lambda x: f"{x:6.2f}"))
