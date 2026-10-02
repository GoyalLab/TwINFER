"""
Full-population (3 reps) vs. half-population (3 reps x 5 random clone-subsample
draws each = 15 points) comparison, per scenario, for every z-score.

Purpose: isolate pure population-size noise (spread of the 15 half-draws)
from rep-to-rep simulation noise (spread of the 3 full-population reps) --
answers "how does subsampling affect noise" directly, on the same underlying
simulated data (no resimulation).

Within each scenario: left box = full population (solid), right box = half
population (hatched), same family color (reg=blue, unreg=orange).

Usage:
  python plot_half_subsample_comparison.py [out_dir]
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

DEFAULT_DIR = f'{TWINFER_PROJECT_ROOT}/analysis_data/drift_inference/eight_scenario/t1_10_t2_20'

_ap = argparse.ArgumentParser()
_ap.add_argument("out_dir", nargs="?", default=DEFAULT_DIR)
_args = _ap.parse_args()
OUT_DIR = _args.out_dir

full = pd.read_csv(os.path.join(DEFAULT_DIR, "six_scenario_zscores.csv"))
half = pd.read_csv(os.path.join(DEFAULT_DIR, "halfdata", "half_subsample_zscores.csv"))
half["scenario"] = half["scenario"].str.replace("_half", "", regex=False)
full["population"] = "full (n=6000, 3 reps)"
half["population"] = "half (n=3000, 3 reps x 5 draws)"
df = pd.concat([full, half], ignore_index=True)

SCEN = [
    ("no_regulation",     "A, B",          "unreg"),
    ("A_to_B",            "A→B",           "reg"),
    ("multistate_A_B",    "A, B\nmultistate\n(high/low)",  "unreg"),
    ("multistate_A_to_B", "A→B\nmultistate\n(high/low)",   "reg"),
    ("kramp_A_B",         "A, B\ndrift\nchanging K",   "unreg"),
    ("kramp_A_to_B",      "A→B\ndrift\nchanging K",    "reg"),
    ("kfrozen_A_B",       "A, B\ndrift\nfrozen K",     "unreg"),
    ("kfrozen_A_to_B",    "A→B\ndrift\nfrozen K",      "reg"),
]
COL = {"reg": "#0072B2", "unreg": "#D55E00"}
rng = np.random.default_rng(0)

T1 = int(df["t1"].dropna().iloc[0])
T2 = int(df["t2"].dropna().iloc[0])
THRESH = 2.33

Z_SPECS = [
    ("step1_z_t1",     r"Step 1 z  ($t_1$)",           (THRESH, -THRESH)),
    ("step1_z_t2",     r"Step 1 z  ($t_2$)",           (THRESH, -THRESH)),
    ("step3_z_d",      r"Step 3 z  $z_d$",             (THRESH, -THRESH)),
    ("step2_z_het_t1", r"Step 2 z$_{\rm het}$  ($t_1$)", (THRESH, -THRESH)),
    ("step2_z_het_t2", r"Step 2 z$_{\rm het}$  ($t_2$)", (THRESH, -THRESH)),
    ("step4_z_1to2",   r"Step 4 z  $g_1\!\to\!g_2$",   (THRESH, -THRESH)),
    ("step4_z_2to1",   r"Step 4 z  $g_2\!\to\!g_1$",   (THRESH, -THRESH)),
    ("z_gamma_1to2",   r"Step 4 z$_\gamma$", (THRESH, -THRESH)),
    ("z_div_t1",       r"z$_{\rm regulation}$  ($t_1$)", (THRESH, -THRESH)),
    ("z_div_t2",       r"z$_{\rm regulation}$  ($t_2$)", (THRESH, -THRESH)),
    ("z_reg_gated_t1", r"z$_{\rm reg,gated}$  ($t_1$)",  (THRESH, -THRESH)),
    ("z_reg_gated_t2", r"z$_{\rm reg,gated}$  ($t_2$)",  (THRESH, -THRESH)),
]


def shared_ylim(col, pad=0.10):
    v = df[col].replace([np.inf, -np.inf], np.nan).dropna()
    lo, hi = v.min(), v.max()
    m = (hi - lo) * pad or 0.1
    return lo - m, hi + m


def panel(ax, col, title, ylim):
    xt, xl = [], []
    for k, (scen, lbl, fam) in enumerate(SCEN):
        x0 = k * 2.2
        for j, (poplab, hatch) in enumerate([("full (n=6000, 3 reps)", None),
                                              ("half (n=3000, 3 reps x 5 draws)", "///")]):
            v = df.loc[(df.scenario == scen) & (df["population"] == poplab), col] \
                   .replace([np.inf, -np.inf], np.nan).dropna().values
            if not len(v):
                continue
            x = x0 + j * 0.9
            ax.boxplot(v, positions=[x], widths=0.75, patch_artist=True, showfliers=True,
                       medianprops=dict(color=COL[fam], lw=2.0),
                       boxprops=dict(facecolor=COL[fam], alpha=0.20 if j == 0 else 0.10,
                                     edgecolor=COL[fam], lw=1.2, hatch=hatch),
                       whiskerprops=dict(color=COL[fam], lw=1.1),
                       capprops=dict(color=COL[fam], lw=1.1), zorder=2)
        xt.append(x0 + 0.45)
        xl.append(lbl)
    ax.axhline(THRESH, ls=":", lw=1, color="0.5")
    ax.axhline(-THRESH, ls=":", lw=1, color="0.5")
    ax.axhline(0, color="0.75", lw=0.8, zorder=1)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=8)
    ax.set_xlim(-0.8, (len(SCEN) - 1) * 2.2 + 1.8)
    ax.set_ylim(*ylim[:2])
    ax.spines[["top", "right"]].set_visible(False)


def make_fig(specs, stem, ncol=3):
    n = len(specs)
    nrow = int(np.ceil(n / ncol))
    w = max(6.0, 1.3 * len(SCEN)) * ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(w, 3.9 * nrow), squeeze=False)
    for ax, (col, title, hl) in zip(axes.flat, specs):
        panel(ax, col, title, shared_ylim(col))
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.legend(handles=[
        Patch(facecolor=COL["reg"], label="regulated (A→B)"),
        Patch(facecolor=COL["unreg"], label="unregulated (A, B)"),
        Patch(facecolor="0.5", alpha=0.20, edgecolor="0.3", label="full (n=6000, 3 reps)"),
        Patch(facecolor="0.5", alpha=0.10, edgecolor="0.3", hatch="///", label="half (n=3000, 3 reps x 5 draws)"),
    ], loc="lower center", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Full vs. half-population subsampling  ·  gene_1-gene_2  ·  "
                 rf"$t_1$={T1}h, $t_2$={T2}h  ·  N=5000 shuffles", fontsize=11)
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.subplots_adjust(wspace=0.32, hspace=0.6)
    p = os.path.join(OUT_DIR, f"{stem}.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print("saved", p)


make_fig(Z_SPECS, "half_subsample_zscore_comparison", ncol=3)
