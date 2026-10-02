"""
Boxplot EVERY z-score infer.py's Steps 1-4 produce for gene_1-gene_2, across
replicates, one panel per z-score, one box per scenario.

Source: six_scenario_zscores.csv (output of run_6scenario_zscores.py's
process_replicate(), which reproduces infer_with_twinfer's Steps 1-4 exactly
and additionally evaluates Step 1/2/z_div at BOTH t1 and t2 -- the live
pipeline itself only evaluates them at t1 for routing).

The 16 z-scores plotted (name -> what it tests):
  step1_z_t1/t2      gene-gene rho vs cell-scramble null (existence)
  step2_z_het_t1/t2  twin rho_Delta vs random-pair rho_Delta (heterogeneity)
  z_div_t1/t2        fixed-Delta divergence shuffle (Yuval), heterogeneity alt
  z_reg_gated_t1/t2  heterogeneity-gated regulation statistic
  step3_z_d          d = rho_Delta(t2) - rho_Delta(t1) vs its null
  step3_z_d_div      same d, divergence-flavored null variant
  step4_z_1to2/2to1  signed cross-correlation vs scramble, per direction
  z_gamma_1to2       |rho_cross(1->2)| - |rho_cross(2->1)| asymmetry
  z_abs_drift_twin                  new: |rho_Delta_twin(t2)-rho_Delta_twin(t1)|
                                     vs pair-delta-permutation null
                                     (compute_abs_drift_zscores.py)
  z_abs_drift_random_pool_permute   same |Delta|, random-pair reference,
                                     pair-delta-permutation null
  z_abs_drift_random_pool_relabel   same |Delta|, random-pair reference,
                                     cell-pool-and-relabel null

Scenarios currently on disk (`eight_scenario/t1_1_t2_20/six_scenario_zscores.csv`,
3 reps/scenario as of 2026-09-04): no_regulation, A_to_B, multistate_A_B,
multistate_A_to_B, kframe_A_B/kfrozen_A_to_B (K_frozen drift), kramp_A_B/
kramp_A_to_B (K_ramp drift) -- 8 total, replacing the old buggy drift_A_B/
drift_A_to_B pair. Both drift variants are shown; pass --drift kfrozen|kramp
to narrow to 6.

Usage:
    python plot_6scenario_all_zscores.py [CSV] [--drift kfrozen|kramp|both]
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

DEFAULT_CSV = (f'{TWINFER_PROJECT_ROOT}/analysis_data/drift_inference/eight_scenario/t1_1_t2_20/six_scenario_zscores.csv')

ap = argparse.ArgumentParser()
ap.add_argument("csv", nargs="?", default=DEFAULT_CSV)
ap.add_argument("--drift", choices=["kfrozen", "kramp", "both"], default="both")
args = ap.parse_args()

OUT_DIR = os.path.join(os.path.dirname(args.csv), "all_zscores")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(args.csv)

# ---- merge in the new abs-drift z-scores (compute_abs_drift_zscores.py) ----
# Outer join on scenario+rep_id: the abs-drift run has more replicates for
# some scenarios than the original 3-rep six_scenario_zscores.csv, so this
# is NOT a plain left-merge (that would silently drop the extra reps).
ABS_DRIFT_CSV = os.path.join(OUT_DIR, "..", "abs_drift", "abs_drift_zscores.csv")
ABS_DRIFT_COLS = ["z_abs_drift_twin", "z_abs_drift_random_pool_permute",
                   "z_abs_drift_random_pool_relabel"]
if os.path.exists(ABS_DRIFT_CSV):
    adf = pd.read_csv(ABS_DRIFT_CSV)[["scenario", "rep_id"] + ABS_DRIFT_COLS]
    df = df.merge(adf, on=["scenario", "rep_id"], how="outer")
else:
    print(f"WARNING: {ABS_DRIFT_CSV} not found -- abs-drift panels will be empty")

BASE_SCEN = [
    ("no_regulation",     "no reg",         "unreg"),
    ("A_to_B",            "A→B",        "reg"),
    ("multistate_A_B",    "2-state\nA, B",   "unreg"),
    ("multistate_A_to_B", "2-state\nA→B", "reg"),
]
DRIFT_SCEN = {
    "kfrozen": [("kfrozen_A_B", "drift(Kfz)\nA, B", "unreg"),
                ("kfrozen_A_to_B", "drift(Kfz)\nA→B", "reg")],
    "kramp":   [("kramp_A_B", "drift(Kr)\nA, B", "unreg"),
                ("kramp_A_to_B", "drift(Kr)\nA→B", "reg")],
}
if args.drift == "both":
    SCEN = BASE_SCEN + DRIFT_SCEN["kfrozen"] + DRIFT_SCEN["kramp"]
else:
    SCEN = BASE_SCEN + DRIFT_SCEN[args.drift]

SCEN = [s for s in SCEN if s[0] in set(df["scenario"])]
COL = {"reg": "#0072B2", "unreg": "#D55E00"}   # Okabe-Ito, CVD-safe
THRESH = 2.33

Z_SPECS = [
    ("step1_z_t1",     r"Step 1 z  ($t_1$)"),
    ("step1_z_t2",     r"Step 1 z  ($t_2$)"),
    ("step2_z_het_t1", r"Step 2 z$_{\rm het}$  ($t_1$)"),
    ("step2_z_het_t2", r"Step 2 z$_{\rm het}$  ($t_2$)"),
    ("z_div_t1",       r"z$_{\rm div}$  ($t_1$)"),
    ("z_div_t2",       r"z$_{\rm div}$  ($t_2$)"),
    ("z_reg_gated_t1", r"z$_{\rm reg,gated}$  ($t_1$)"),
    ("z_reg_gated_t2", r"z$_{\rm reg,gated}$  ($t_2$)"),
    ("step3_z_d",      r"Step 3 $z_d$"),
    ("step3_z_d_div",  r"Step 3 $z_{d,\rm div}$"),
    ("step4_z_1to2",   r"Step 4 z  $g_1\!\to\!g_2$"),
    ("step4_z_2to1",   r"Step 4 z  $g_2\!\to\!g_1$"),
    ("z_gamma_1to2",   r"z$_\gamma$  ($g_1\!\to\!g_2$ vs $g_2\!\to\!g_1$)"),
    # -- new: |Delta rho| abs-drift statistic (compute_abs_drift_zscores.py) --
    ("z_abs_drift_twin", "twin |Δρ| z\n(null: pool_permute)"),
    ("z_abs_drift_random_pool_permute", "random-pair |Δρ| z\n(null: pool_permute)"),
    ("z_abs_drift_random_pool_relabel", "random-pair |Δρ| z\n(null: pool_relabel)"),
]
z_cols = [c for c, _ in Z_SPECS]


def shared_ylim(cols, pad=0.08, clip=(1, 99)):
    v = pd.concat([df[c] for c in cols if c in df]).replace([np.inf, -np.inf], np.nan).dropna()
    lo, hi = np.percentile(v, clip[0]), np.percentile(v, clip[1])
    lo, hi = min(lo, v.min()), max(hi, v.max())
    m = (hi - lo) * pad or 0.1
    return lo - m, hi + m


def panel(ax, col, title, ylim):
    # n is computed per-panel (not from one shared dict): the abs-drift
    # columns have different replicate coverage per scenario than the
    # original 13 z-scores after the outer-merge above.
    n_this = {}
    for k, (scen, lbl, fam) in enumerate(SCEN):
        v = df.loc[df["scenario"] == scen, col].replace([np.inf, -np.inf], np.nan).dropna().values \
            if col in df else np.array([])
        n_this[scen] = len(v)
        if not len(v):
            continue
        ax.boxplot(v, positions=[k], widths=0.6, patch_artist=True, showfliers=True,
                   medianprops=dict(color=COL[fam], lw=2.2),
                   boxprops=dict(facecolor=COL[fam], alpha=0.16, edgecolor=COL[fam], lw=1.3),
                   whiskerprops=dict(color=COL[fam], lw=1.3),
                   capprops=dict(color=COL[fam], lw=1.3), zorder=2)
    for y in (THRESH, -THRESH):
        if ylim[0] <= y <= ylim[1]:
            ax.axhline(y, ls="--", lw=1, color="0.5")
    if ylim[0] <= 0 <= ylim[1]:
        ax.axhline(0, color="0.75", lw=0.8, zorder=1)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(len(SCEN)))
    ax.set_xticklabels([f"{l}\n(n={n_this.get(s, 0)})" for s, l, _ in SCEN], fontsize=6.5)
    ax.set_xlim(-0.6, len(SCEN) - 0.4)
    ax.set_ylim(*ylim)
    ax.spines[["top", "right"]].set_visible(False)


from matplotlib.patches import Patch

nrow, ncol = 6, 3


def make_fig(per_panel_ylim, suptitle_suffix):
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.6 * nrow), squeeze=False)
    for ax, (col, title) in zip(axes.flat, Z_SPECS):
        ylim = per_panel_ylim(col) if callable(per_panel_ylim) else per_panel_ylim
        panel(ax, col, title, ylim)
    for ax in axes.flat[len(Z_SPECS):]:
        ax.set_visible(False)
    fig.legend(handles=[Patch(facecolor=COL["reg"], label="regulated (A→B)"),
                        Patch(facecolor=COL["unreg"], label="unregulated (A, B)")],
               loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("TwINFER -- all Step 1-4 z-scores  ·  gene_1-gene_2  ·  "
                 f"{len(SCEN)} scenarios  ·  $t_1$={_t1} h, $t_2$={_t2} h{suptitle_suffix}",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0.035, 1, 0.95])
    return fig


_t1s = df["t1"].dropna().unique() if "t1" in df else []
_t2s = df["t2"].dropna().unique() if "t2" in df else []
_t1 = int(_t1s[0]) if len(_t1s) else "?"
_t2 = int(_t2s[0]) if len(_t2s) else "?"

# filename tag: "both" keeps the original (default) filenames; narrowing to
# one drift variant gets its own suffix so it doesn't clobber the full view.
TAG = "" if args.drift == "both" else f"_{args.drift}"

# ---- full-scale version: one shared y-axis across all 16 panels ----
shared = shared_ylim(z_cols)
fig = make_fig(shared, "")
png = os.path.join(OUT_DIR, f"six_scenario_all_zscores{TAG}.png")
fig.savefig(png, dpi=200, bbox_inches="tight")
pdf_path = os.path.join(OUT_DIR, f"six_scenario_all_zscores{TAG}.pdf")
with PdfPages(pdf_path) as pdf:
    pdf.savefig(fig, bbox_inches="tight")
print("saved", png)
print("saved", pdf_path)

# ---- zoomed version: each panel gets its OWN y-axis, scaled to its own
# data range (multistate's step1_z/step4_z reach ~35 and otherwise squash
# the smaller-scale panels -- z_reg_gated, the new abs-drift panels, etc --
# into an unreadable sliver near zero on the shared axis above). ----
per_panel = {c: shared_ylim([c], pad=0.15) for c in z_cols}
fig_z = make_fig(lambda c: per_panel[c], "  (zoomed: per-panel y-axis)")
png_z = os.path.join(OUT_DIR, f"six_scenario_all_zscores{TAG}_zoomed.png")
fig_z.savefig(png_z, dpi=200, bbox_inches="tight")
pdf_z_path = os.path.join(OUT_DIR, f"six_scenario_all_zscores{TAG}_zoomed.pdf")
with PdfPages(pdf_z_path) as pdf:
    pdf.savefig(fig_z, bbox_inches="tight")
print("saved", png_z)
print("saved", pdf_z_path)

print("\nmedian z by scenario:")
print(df[df["scenario"].isin([s[0] for s in SCEN])]
      .groupby("scenario")[z_cols].median().round(2).to_string())

frac = pd.DataFrame(index=[s[0] for s in SCEN], columns=z_cols, dtype=float)
for scen in frac.index:
    sub = df[df["scenario"] == scen]
    for col in z_cols:
        v = sub[col].abs().dropna()
        frac.loc[scen, col] = (v > THRESH).mean() if len(v) else np.nan
frac_csv = os.path.join(OUT_DIR, f"six_scenario_all_zscores{TAG}_threshold_fraction.csv")
frac.to_csv(frac_csv)
print(f"\n|z| > {THRESH} fraction by scenario:")
print(frac.round(2).to_string())
print("saved", frac_csv)
