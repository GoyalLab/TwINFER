"""
Two clean figures for the 6-scenario x 20-replicate TwINFER analysis
(gene_1-gene_2), from six_scenario_zscores.csv:

  six_scenario_correlations.png/pdf  - every correlation, one shared y-limit
  six_scenario_zscores.png/pdf       - every z-score,     one shared y-limit

Scenarios (x-axis order):
  no reg | A->B | 2-state A,B | 2-state A->B | drift A,B | drift A->B
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

DEFAULT_DIR = f'{TWINFER_PROJECT_ROOT}/analysis_data/drift_inference/six_scenario'

_ap = argparse.ArgumentParser()
_ap.add_argument("csv", nargs="?", default=os.path.join(DEFAULT_DIR, "six_scenario_zscores_v2.csv"))
_ap.add_argument("out_dir", nargs="?", default=None,
                 help="output dir (default: next to the input CSV)")
_ap.add_argument("--exclude", default="",
                 help="comma-separated scenario keys to drop from the plot")
_ap.add_argument("--suffix", default="",
                 help="appended to every output filename stem, e.g. '_noMS'")
_ap.add_argument("--ytight", action="store_true",
                 help="zoom the y-limit to the exact data extent of the kept scenarios")
_ap.add_argument("--free-y", action="store_true",
                 help="give every panel its own y-limit (so small-magnitude panels "
                      "like Step 2 are readable) instead of one shared scale")
_args = _ap.parse_args()

CSV = _args.csv
OUT_DIR = _args.out_dir or os.path.dirname(os.path.abspath(CSV))
EXCLUDE = {s.strip() for s in _args.exclude.split(",") if s.strip()}
SUF = _args.suffix
YTIGHT = _args.ytight
FREEY = _args.free_y

SCEN = [
    ("no_regulation",     "A, B",          "unreg"),
    ("A_to_B",            "A→B",           "reg"),
    ("multistate_A_B",    "A, B\nmultistate\n(high/low)",  "unreg"),
    ("multistate_A_to_B", "A→B\nmultistate\n(high/low)",   "reg"),
    ("multistate_lowbase_A_B",    "A, B\nmultistate\n(low/mid)",  "unreg"),
    ("multistate_lowbase_A_to_B", "A→B\nmultistate\n(low/mid)",   "reg"),
    ("kramp_A_B",         "A, B\ndrift\nchanging K",   "unreg"),
    ("kramp_A_to_B",      "A→B\ndrift\nchanging K",    "reg"),
    ("kfrozen_A_B",       "A, B\ndrift\nfrozen K",     "unreg"),
    ("kfrozen_A_to_B",    "A→B\ndrift\nfrozen K",      "reg"),
    # earlier scenarios, kept for reference (not in the current requested set):
    # ("multistate_lowbase_A_B",    "2-state\nlow/mid\nA, B",  "unreg"),
    # ("multistate_lowbase_A_to_B", "2-state\nlow/mid\nA→B",   "reg"),
    # ("drift_A_B",         "drift\nlow/high\nA, B",  "unreg"),
    # ("drift_A_to_B",      "drift\nlow/high\nA→B",   "reg"),
    # ("drift_A_B_recover",    "drift\nlow/base\nA, B",  "unreg"),
    # ("drift_A_to_B_recover", "drift\nlow/base\nA→B",   "reg"),
    # ("mix_upDrift_A_B",    "mix up-drift A, B",  "unreg"),
    # ("mix_upDrift_A_to_B", "mix up-drift A→B",   "reg"),
    # ("drift_A_B_fast5h",     "drift 5h A, B",   "unreg"),
    # ("drift_A_to_B_fast5h",  "drift 5h A→B",    "reg"),
]
COL = {"reg": "#0072B2", "unreg": "#D55E00"}   # Okabe-Ito, CVD-safe

df = pd.read_csv(CSV)
rng = np.random.default_rng(0)

# keep only the scenarios present in this CSV and not excluded
SCEN = [s for s in SCEN if (df["scenario"] == s[0]).any() and s[0] not in EXCLUDE]
# restrict the data to the plotted scenarios so shared y-limits / tables ignore the rest
df = df[df["scenario"].isin({s[0] for s in SCEN})].copy()

# timepoints for the figure headings: CSV columns first, then env, then default
T1 = int(df["t1"].dropna().iloc[0]) if "t1" in df and df["t1"].notna().any() \
    else int(os.environ.get("DRIFT_T1", "1"))
T2 = int(df["t2"].dropna().iloc[0]) if "t2" in df and df["t2"].notna().any() \
    else int(os.environ.get("DRIFT_T2", "20"))
TSTR = rf"$t_1$={T1} h, $t_2$={T2} h"


def shared_ylim(cols, pad=0.08, clip=(1, 99)):
    v = pd.concat([df[c] for c in cols if c in df]).replace([np.inf, -np.inf], np.nan).dropna()
    if YTIGHT:
        # zoom to the exact data extent (all points visible), minimal pad
        lo, hi = v.min(), v.max()
        m = (hi - lo) * 0.03 or 0.1
    else:
        lo, hi = np.percentile(v, clip[0]), np.percentile(v, clip[1])
        lo, hi = min(lo, v.min()), max(hi, v.max())  # keep true extremes visible
        m = (hi - lo) * pad or 0.1
    return lo - m, hi + m


def panel(ax, col, title, ylim, hlines=()):
    for k, (scen, lbl, fam) in enumerate(SCEN):
        v = df.loc[df["scenario"] == scen, col].replace([np.inf, -np.inf], np.nan).dropna().values \
            if col in df else np.array([])
        if not len(v):
            continue
        ax.boxplot(v, positions=[k], widths=0.6, patch_artist=True, showfliers=True,
                   medianprops=dict(color=COL[fam], lw=2.2),
                   boxprops=dict(facecolor=COL[fam], alpha=0.16, edgecolor=COL[fam], lw=1.3),
                   whiskerprops=dict(color=COL[fam], lw=1.3),
                   capprops=dict(color=COL[fam], lw=1.3), zorder=2)
        x = k + rng.uniform(-0.11, 0.11, size=v.shape)
        # ax.scatter(x, v, s=22, color=COL[fam], alpha=0.8, edgecolor="white",
        #            linewidth=0.5, zorder=3)
    for y in hlines:
        ax.axhline(y, ls="--", lw=1, color="0.5")
    ax.axhline(0, color="0.75", lw=0.8, zorder=1)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(len(SCEN)))
    ax.set_xticklabels([l for _, l, _ in SCEN], fontsize=8.5 if len(SCEN) <= 8 else 6.0)
    ax.set_xlim(-0.7, len(SCEN) - 0.3)
    ax.set_ylim(*ylim)
    ax.spines[["top", "right"]].set_visible(False)


def make_fig(specs, ylim, suptitle, stem, ncol=3):
    n = len(specs)
    nrow = int(np.ceil(n / ncol))
    w = max(4.7, 0.95 * len(SCEN)) * ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(w, 3.9 * nrow), squeeze=False)
    for ax, (col, title, hl) in zip(axes.flat, specs):
        # ylim=None (or --free-y) -> each panel scales to its own data
        yl = shared_ylim([col]) if ylim is None else ylim
        panel(ax, col, title, yl, hlines=hl)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=COL["reg"], label="regulated (A→B)"),
                        Patch(facecolor=COL["unreg"], label="unregulated (A, B)")],
               loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=[0, 0.045, 1, 0.94])
    fig.subplots_adjust(wspace=0.32, hspace=0.55)
    fig.savefig(os.path.join(OUT_DIR, f"{stem}.png"), dpi=200, bbox_inches="tight")
    print("saved", os.path.join(OUT_DIR, f"{stem}.png"))
    return fig


# Per-step pipeline defaults (kept for reference, not used in the plot):
#   zc1, zc4 = 2.576, 2.5
#   zcd = float(df["step3_z_critical"].dropna().iloc[0])   # one-sided ~2.33
# The user asked for a single |z| > 2.33 line on Steps 1, 2, 3 and 4.
THRESH = 2.33

Z_SPECS = [
    ("step1_z_t1",     r"Step 1 z  ($t_1$)",           (THRESH, -THRESH)),
    ("step1_z_t2",     r"Step 1 z  ($t_2$)",           (THRESH, -THRESH)),
    ("step3_z_d",      r"Step 3 z  $z_d$",             (THRESH, -THRESH)),
    ("step2_z_het_t1", r"Step 2 z$_{\rm het}$  ($t_1$)", (THRESH, -THRESH)),
    ("step2_z_het_t2", r"Step 2 z$_{\rm het}$  ($t_2$)", (THRESH, -THRESH)),
    ("step4_z_1to2",   r"Step 4 z  $g_1\!\to\!g_2$",   (THRESH, -THRESH)),
    ("step4_z_2to1",   r"Step 4 z  $g_2\!\to\!g_1$",   (THRESH, -THRESH)),
    ("z_gamma_1to2",   r"Step 4 z$_\gamma$  ($\gamma=|\rho_{1\to2}|-|\rho_{2\to1}|$)", (THRESH, -THRESH)),
    ("z_div_t1",       r"z$_{\rm regulation}$  ($t_1$)", (THRESH, -THRESH)),
    ("z_div_t2",       r"z$_{\rm regulation}$  ($t_2$)", (THRESH, -THRESH)),
    ("z_reg_gated_t1", r"z$_{\rm reg,gated}$  ($t_1$)",  (THRESH, -THRESH)),
    ("z_reg_gated_t2", r"z$_{\rm reg,gated}$  ($t_2$)",  (THRESH, -THRESH)),
]
z_cols = [s[0] for s in Z_SPECS]
ylz = shared_ylim(z_cols)

C_SPECS = [
    ("rho_t1",                r"gene-gene $\rho$  ($t_1$)", ()),
    ("rho_t2",                r"gene-gene $\rho$  ($t_2$)", ()),
    ("step4_rho_cross_1to2",  r"cross $\rho$  $g_1\!\to\!g_2$", ()),
    ("rho_delta_t1",          r"twin $\hat{\rho}_\Delta$  ($t_1$)", ()),
    ("rho_delta_t2",          r"twin $\hat{\rho}_\Delta$  ($t_2$)", ()),
    ("step4_rho_cross_2to1",  r"cross $\rho$  $g_2\!\to\!g_1$", ()),
    ("rho_delta_random_t1",   r"random-pair $\rho_\Delta$  ($t_1$)", ()),
    ("rho_delta_random_t2",   r"random-pair $\rho_\Delta$  ($t_2$)", ()),
]
c_cols = [s[0] for s in C_SPECS]
ylc = shared_ylim(c_cols)

figz = make_fig(Z_SPECS, None if FREEY else ylz, "TwINFER z-scores  ·  gene_1–gene_2  ·  "
                f"{len(SCEN)} scenarios × 20 replicates  ·  " + TSTR,
                f"six_scenario_zscores_v2{SUF}", ncol=3)
figc = make_fig(C_SPECS, None if FREEY else ylc, "TwINFER correlations  ·  gene_1–gene_2  ·  "
                f"{len(SCEN)} scenarios × 20 replicates  ·  " + TSTR,
                f"six_scenario_correlations_v2{SUF}", ncol=3)

# ---- extra derived contrasts ----
#   1. directional asymmetry of the cross-rho:  rho_cross(g1->g2) - rho_cross(g2->g1)
#   2. t1->t2 change in the random-pair (null) twin rho_Delta:  rho_Delta_rand(t2) - rho_Delta_rand(t1)
df["cross_rho_dir_diff"] = df["step4_rho_cross_1to2"] - df["step4_rho_cross_2to1"]
df["random_rho_time_diff"] = df["rho_delta_random_t2"] - df["rho_delta_random_t1"]
X_SPECS = [
    ("cross_rho_dir_diff",
     r"$\rho_{\rm cross}(g_1\!\to\!g_2) - \rho_{\rm cross}(g_2\!\to\!g_1)$", ()),
    ("random_rho_time_diff",
     r"random-pair $\rho_\Delta(t_2) - \rho_\Delta(t_1)$", ()),
]
figx = make_fig(X_SPECS, None if FREEY else shared_ylim([s[0] for s in X_SPECS]),
                "TwINFER derived contrasts  ·  gene_1–gene_2  ·  "
                f"{len(SCEN)} scenarios × 20 replicates  ·  " + TSTR,
                f"six_scenario_derived_contrasts_v2{SUF}", ncol=2)

# ---- threshold-crossing table: fraction of the 20 replicates with |z| > THRESH ----
scen_order = [s[0] for s in SCEN]
z_labels = {
    "step1_z_t1": "Step1 z (t1)", "step1_z_t2": "Step1 z (t2)",
    "step2_z_het_t1": "Step2 z_het (t1)", "step2_z_het_t2": "Step2 z_het (t2)",
    "z_div_t1": "z_reg (t1)", "z_div_t2": "z_reg (t2)",
    "z_reg_gated_t1": "z_reg_gated (t1)", "z_reg_gated_t2": "z_reg_gated (t2)",
    "step3_z_d": "Step3 z_d", "step4_z_1to2": "Step4 z (g1->g2)",
    "step4_z_2to1": "Step4 z (g2->g1)", "z_gamma_1to2": "Step4 z_gamma",
}
frac = pd.DataFrame(index=scen_order, columns=list(z_labels.values()), dtype=float)
cnt = frac.copy()
n_reps = pd.Series(index=scen_order, dtype=int)
for scen in scen_order:
    sub = df[df["scenario"] == scen]
    n_reps[scen] = len(sub)
    for col, lbl in z_labels.items():
        v = sub[col].abs().dropna()
        cnt.loc[scen, lbl] = int((v > THRESH).sum())
        frac.loc[scen, lbl] = (v > THRESH).mean()
frac.index = [l for _, l, _ in SCEN]
cnt.index = frac.index
n_reps.index = frac.index

frac.to_csv(os.path.join(OUT_DIR, f"six_scenario_threshold_crossing_fraction_v2{SUF}.csv"))
cnt.to_csv(os.path.join(OUT_DIR, f"six_scenario_threshold_crossing_count_v2{SUF}.csv"))

fig_t, ax_t = plt.subplots(figsize=(1.5 + 1.15 * len(z_labels), 0.55 * len(SCEN) + 1.6))
im = ax_t.imshow(frac.values.astype(float), cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax_t.set_xticks(range(len(z_labels)))
ax_t.set_xticklabels(list(z_labels.values()), rotation=40, ha="right", fontsize=8)
ax_t.set_yticks(range(len(SCEN)))
ax_t.set_yticklabels(frac.index, fontsize=9)
for i in range(len(SCEN)):
    for j in range(len(z_labels)):
        fr = frac.values[i, j]
        ax_t.text(j, i, f"{cnt.values[i, j]:.0f}/{n_reps.iloc[i]}", ha="center", va="center",
                  fontsize=8, color="white" if fr > 0.55 else "0.15")
ax_t.set_title(f"replicates with |z| > {THRESH}  (count / n_reps, varies by scenario)  ·  {TSTR}", fontsize=11, pad=10)
fig_t.colorbar(im, ax=ax_t, label="fraction", fraction=0.025, pad=0.02)
fig_t.tight_layout()
fig_t.savefig(os.path.join(OUT_DIR, f"six_scenario_threshold_crossing_v2{SUF}.png"), dpi=200, bbox_inches="tight")
print("saved", os.path.join(OUT_DIR, f"six_scenario_threshold_crossing_v2{SUF}.png"))

PDF_TAG = f"_t{T1}" if T1 != 1 else ""
combined = os.path.join(OUT_DIR, f"six_scenario_all_v2{SUF}{PDF_TAG}.pdf")
with PdfPages(combined) as pdf:
    pdf.savefig(figz, bbox_inches="tight")
    pdf.savefig(figc, bbox_inches="tight")
    pdf.savefig(figx, bbox_inches="tight")
    pdf.savefig(fig_t, bbox_inches="tight")
print("saved", combined)

print(df.groupby("scenario")[z_cols].median().round(2).to_string())
print("\n|z| > %.2f  (count/n_reps):" % THRESH)
print(cnt.to_string())
