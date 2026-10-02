"""
Boxplot the NEW abs-drift z-scores (compute_abs_drift_zscores.py) across
replicates, one box per scenario, for all 8 scenarios currently on disk.

    z_abs_drift_twin                  twin rho_Delta,   pool_permute null
    z_abs_drift_random_pool_permute   random-pair rho_Delta, pool_permute null
    z_abs_drift_random_pool_relabel   random-pair rho_Delta, pool_relabel null

Usage:
    python plot_abs_drift_zscores.py [CSV]
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

DEFAULT_CSV = (f'{TWINFER_PROJECT_ROOT}/analysis_data/drift_inference/eight_scenario/t1_1_t2_20/abs_drift/abs_drift_zscores.csv')

ap = argparse.ArgumentParser()
ap.add_argument("csv", nargs="?", default=DEFAULT_CSV)
args = ap.parse_args()

OUT_DIR = os.path.dirname(args.csv)
df = pd.read_csv(args.csv)

SCEN = [
    ("no_regulation",     "no reg",         "unreg"),
    ("A_to_B",            "A→B",        "reg"),
    ("multistate_A_B",    "2-state\nA, B",   "unreg"),
    ("multistate_A_to_B", "2-state\nA→B", "reg"),
    ("kfrozen_A_B",       "drift(Kfz)\nA, B", "unreg"),
    ("kfrozen_A_to_B",    "drift(Kfz)\nA→B", "reg"),
    ("kramp_A_B",         "drift(Kr)\nA, B", "unreg"),
    ("kramp_A_to_B",      "drift(Kr)\nA→B", "reg"),
]
SCEN = [s for s in SCEN if s[0] in set(df["scenario"])]
COL = {"reg": "#0072B2", "unreg": "#D55E00"}
THRESH = 2.33
n_by_scen = df.groupby("scenario")["rep_id"].nunique().to_dict()

Z_SPECS = [
    ("z_abs_drift_twin", "twin  |Δρ|  z-score\n(null: pool_permute)"),
    ("z_abs_drift_random_pool_permute", "random-pair  |Δρ|  z-score\n(null: pool_permute)"),
    ("z_abs_drift_random_pool_relabel", "random-pair  |Δρ|  z-score\n(null: pool_relabel)"),
]
z_cols = [c for c, _ in Z_SPECS]


def shared_ylim(cols, pad=0.1, clip=(1, 99)):
    v = pd.concat([df[c] for c in cols if c in df]).replace([np.inf, -np.inf], np.nan).dropna()
    lo, hi = np.percentile(v, clip[0]), np.percentile(v, clip[1])
    lo, hi = min(lo, v.min()), max(hi, v.max())
    m = (hi - lo) * pad or 0.1
    return lo - m, hi + m


def panel(ax, col, title, ylim):
    for k, (scen, lbl, fam) in enumerate(SCEN):
        v = df.loc[df["scenario"] == scen, col].replace([np.inf, -np.inf], np.nan).dropna().values
        if not len(v):
            continue
        ax.boxplot(v, positions=[k], widths=0.6, patch_artist=True, showfliers=True,
                   medianprops=dict(color=COL[fam], lw=2.2),
                   boxprops=dict(facecolor=COL[fam], alpha=0.16, edgecolor=COL[fam], lw=1.3),
                   whiskerprops=dict(color=COL[fam], lw=1.3),
                   capprops=dict(color=COL[fam], lw=1.3), zorder=2)
    ax.axhline(THRESH, ls="--", lw=1, color="0.5")
    ax.axhline(0, color="0.75", lw=0.8, zorder=1)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(len(SCEN)))
    ax.set_xticklabels([f"{l}\n(n={n_by_scen.get(s, 0)})" for s, l, _ in SCEN], fontsize=7)
    ax.set_xlim(-0.6, len(SCEN) - 0.4)
    ax.set_ylim(*ylim)
    ax.spines[["top", "right"]].set_visible(False)


fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
ylim = shared_ylim(z_cols)
for ax, (col, title) in zip(axes, Z_SPECS):
    panel(ax, col, title, ylim)
fig.legend(handles=[Patch(facecolor=COL["reg"], label="regulated (A→B)"),
                    Patch(facecolor=COL["unreg"], label="unregulated (A, B)")],
           loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.05))
_t1 = int(df["scenario"].map(lambda s: np.nan).pipe(lambda s: np.nan))  # placeholder, overwritten below
fig.suptitle("|Δρ| z-scores  ·  gene_1-gene_2  ·  8 scenarios  ·  new abs-drift statistic", fontsize=11)
fig.tight_layout(rect=[0, 0.08, 1, 0.93])

png = os.path.join(OUT_DIR, "abs_drift_zscores.png")
fig.savefig(png, dpi=200, bbox_inches="tight")
print("saved", png)

print("\nmedian z by scenario:")
print(df.groupby("scenario")[z_cols].median().round(2).to_string())
print(f"\n|z| > {THRESH} fraction by scenario:")
frac = df.groupby("scenario")[z_cols].apply(lambda g: (g.abs() > THRESH).mean())
print(frac.round(2).to_string())
