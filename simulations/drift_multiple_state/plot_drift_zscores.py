"""
Plot TwINFER correlations and Step-1/2/3/4 z-scores for the 20 drift-simulation
replicates, regulated (A_to_B) vs unregulated (A_B_no_reg), gene_1-gene_2.

Figures (analysis_data/drift_inference/):
  drift_gene_correlations{TAG}.png/pdf  - all correlation panels, one shared y-limit
  drift_step_zscores{TAG}.png/pdf       - all z-score panels, one shared y-limit
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import mannwhitneyu

OUT_DIR = f'{TWINFER_PROJECT_ROOT}/analysis_data/drift_inference'
TAG = sys.argv[1] if len(sys.argv) > 1 else ""   # e.g. "_10k"
CSV = os.path.join(OUT_DIR, f"drift_step2_step3_zscores{TAG}.csv")

C = {"reg": "#0072B2", "no_reg": "#D55E00"}          # Okabe-Ito, CVD-safe
LABEL = {"reg": "regulated (A→B)", "no_reg": "unregulated (A, B)"}

df = pd.read_csv(CSV)
df = df[(df["error"].isna()) | (df["error"] == "")]
rng = np.random.default_rng(0)


def shared_ylim(cols, pad=0.08):
    vals = pd.concat([df[c].astype(float) for c in cols]).replace([np.inf, -np.inf], np.nan).dropna()
    lo, hi = vals.min(), vals.max()
    m = (hi - lo) * pad or 0.1
    return lo - m, hi + m


def panel(ax, col, title, formula, ylim, hlines=()):
    handles = []
    for k, cond in enumerate(["reg", "no_reg"]):
        v = df.loc[df["condition"] == cond, col].astype(float).replace([np.inf, -np.inf], np.nan).dropna().values
        if not len(v):
            continue
        ax.boxplot(
            v, positions=[k], widths=0.5, patch_artist=True, showfliers=False,
            medianprops=dict(color=C[cond], lw=2.4),
            boxprops=dict(facecolor=C[cond], alpha=0.16, edgecolor=C[cond], lw=1.4),
            whiskerprops=dict(color=C[cond], lw=1.4),
            capprops=dict(color=C[cond], lw=1.4), zorder=2,
        )
        x = k + rng.uniform(-0.10, 0.10, size=v.shape)
        handles.append(ax.scatter(x, v, s=32, color=C[cond], alpha=0.85, edgecolor="white",
                                  linewidth=0.6, zorder=3, label=f"{LABEL[cond]} (n={len(v)})"))
    for y, lbl in hlines:
        ax.axhline(y, ls="--", lw=1, color="0.5")
        if lbl:
            ax.text(0.5, y, lbl, ha="center", va="bottom", fontsize=7.5, color="0.4")
    ax.axhline(0, color="0.75", lw=0.8, zorder=1)

    a = df.loc[df["condition"] == "reg", col].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    b = df.loc[df["condition"] == "no_reg", col].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    sub = title if not (len(a) and len(b)) else f"{title}\n(M–W p = {mannwhitneyu(a, b).pvalue:.2g})"
    ax.set_title(sub, fontsize=9.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["regulated", "unregulated"], fontsize=8.5)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(*ylim)
    ax.set_xlabel(formula, fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    return handles


def finish(fig, handles, suptitle, stem):
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    p = os.path.join(OUT_DIR, f"{stem}{TAG}.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print("saved", p)


zc = float(df["step3_z_critical"].dropna().iloc[0]) if df["step3_z_critical"].notna().any() else 2.33
zc1 = 2.576   # Step-1 two-sided critical at alpha=0.01
zc4 = 2.5     # Step-4 cross-correlation z threshold

# ================================================================ correlations
corr_cols = ["rho_t1", "rho_t2", "rho_delta_t1", "rho_delta_t2",
             "step4_rho_1to2", "step4_rho_2to1"]
ylc = shared_ylim(corr_cols)
fig1, ax1 = plt.subplots(2, 3, figsize=(13, 9))
specs1 = [
    ("rho_t1",         r"gene–gene $\rho$  ($t_1$)",       r"$\rho(g_1,g_2)\,|\,t_1$"),
    ("rho_t2",         r"gene–gene $\rho$  ($t_2$)",       r"$\rho(g_1,g_2)\,|\,t_2$"),
    ("step4_rho_1to2", r"cross $\rho$  $g_1(t_1)\!\to\!g_2(t_2)$", r"$\rho_{\rm cross}(g_1\!\to\!g_2)$"),
    ("rho_delta_t1",   r"twin $\hat{\rho}_\Delta$  ($t_1$)",  r"$\hat{\rho}_\Delta\,|\,t_1$"),
    ("rho_delta_t2",   r"twin $\hat{\rho}_\Delta$  ($t_2$)",  r"$\hat{\rho}_\Delta\,|\,t_2$"),
    ("step4_rho_2to1", r"cross $\rho$  $g_2(t_1)\!\to\!g_1(t_2)$", r"$\rho_{\rm cross}(g_2\!\to\!g_1)$"),
]
H = None
for ax, (col, t, f) in zip(ax1.flat, specs1):
    H = panel(ax, col, t, f, ylc) or H
finish(fig1, H, f"TwINFER correlations  ·  gene_1–gene_2  ·  20 drift replicates  ·  "
                f"$t_1$=1 h, $t_2$=20 h{'  ·  '+TAG.strip('_')+' shuffles' if TAG else ''}",
       "drift_gene_correlations")

# ================================================================ z-scores
z_cols = ["step1_z_t1", "step1_z_t2", "step2_z_t1", "step2_z_t2",
          "step3_z_d", "step4_z_1to2", "step4_z_2to1"]
ylz = shared_ylim(z_cols)
fig2, ax2 = plt.subplots(2, 4, figsize=(17, 9))
specs2 = [
    ("step1_z_t1",   "Step 1 z  ($t_1$)",  r"gene–gene $\rho$ vs scramble", [(zc1, f"|z|>{zc1:.2f}"), (-zc1, "")]),
    ("step1_z_t2",   "Step 1 z  ($t_2$)",  r"gene–gene $\rho$ vs scramble", [(zc1, ""), (-zc1, "")]),
    ("step2_z_t1",   "Step 2 z  ($t_1$)",  r"$\hat{\rho}_\Delta(t_1)$ vs random", [(5, "multi-state |z|>5"), (-5, "")]),
    ("step2_z_t2",   "Step 2 z  ($t_2$)",  r"$\hat{\rho}_\Delta(t_2)$ vs random", [(5, ""), (-5, "")]),
    ("step3_z_d",    "Step 3 z  $z_d$",    r"$\Delta\hat{\rho}_\Delta$ ($t_1\!\to\!t_2$)", [(zc, f"reg. $z_d>{zc:.2f}$")]),
    ("step4_z_1to2", r"Step 4 z  $g_1\!\to\!g_2$", r"signed $\rho_{\rm cross}$ vs scramble", [(zc4, f"|z|>{zc4}"), (-zc4, "")]),
    ("step4_z_2to1", r"Step 4 z  $g_2\!\to\!g_1$", r"signed $\rho_{\rm cross}$ vs scramble", [(zc4, ""), (-zc4, "")]),
]
H = None
for ax, (col, t, f, hl) in zip(ax2.flat, specs2):
    H = panel(ax, col, t, f, ylz, hlines=hl) or H
ax2.flat[-1].set_visible(False)
finish(fig2, H, f"TwINFER Step-1/2/3/4 z-scores  ·  gene_1–gene_2  ·  20 drift replicates  ·  "
                f"$t_1$=1 h, $t_2$=20 h{'  ·  '+TAG.strip('_')+' shuffles' if TAG else ''}",
       "drift_step_zscores")

# ================================================ everything in one PDF
combined = os.path.join(OUT_DIR, f"drift_twinfer_all{TAG}.pdf")
with PdfPages(combined) as pdf:
    pdf.savefig(fig2, bbox_inches="tight")   # page 1: z-scores
    pdf.savefig(fig1, bbox_inches="tight")   # page 2: correlations
print("saved", combined)

# ================================================================ summary
print(df.groupby("condition")[z_cols + ["rho_t1", "rho_t2"]]
        .agg(["median", "mean", "std"]).round(3).to_string())
