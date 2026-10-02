"""
Plot all five TwINFER z-scores (Step 1, Step 2 z_het, z_divergence -- each at
t1 and t2 -- plus Step 3 z_d and Step 4 z both directions) for the regenerated
20 A_B (no-reg) + 20 A_to_B (reg) 2-state drift sims.

Reads drift_allsteps/six_scenario_zscores.csv (from run_6scenario_zscores.py).
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

D = f'{TWINFER_PROJECT_ROOT}/analysis_data/drift_inference/drift_allsteps'
df = pd.read_csv(os.path.join(D, "six_scenario_zscores.csv"))

SC = [("drift_A_B", "no reg", "#D55E00"), ("drift_A_to_B", "A→B", "#0072B2")]
THRESH = 2.33
rng = np.random.default_rng(0)

PANELS = [
    ("step1_z_t1",     r"Step 1 z  ($t_1$)"),
    ("step1_z_t2",     r"Step 1 z  ($t_2$)"),
    ("step3_z_d",      r"Step 3 z  $z_d$"),
    ("step2_z_het_t1", r"Step 2 z$_{\rm het}$  ($t_1$)"),
    ("step2_z_het_t2", r"Step 2 z$_{\rm het}$  ($t_2$)"),
    ("step4_z_1to2",   r"Step 4 z  $g_1\!\to\!g_2$"),
    ("z_div_t1",       r"z$_{\rm divergence}$  ($t_1$)"),
    ("z_div_t2",       r"z$_{\rm divergence}$  ($t_2$)"),
    ("step4_z_2to1",   r"Step 4 z  $g_2\!\to\!g_1$"),
]
cols = [c for c, _ in PANELS]
allv = pd.concat([df[c] for c in cols]).replace([np.inf, -np.inf], np.nan).dropna()
lo, hi = allv.min(), allv.max()
pad = (hi - lo) * 0.06
YL = (lo - pad, hi + pad)

fig, axes = plt.subplots(3, 3, figsize=(14, 12))
for ax, (col, title) in zip(axes.flat, PANELS):
    for k, (scen, lbl, color) in enumerate(SC):
        v = df.loc[df["scenario"] == scen, col].astype(float).replace([np.inf, -np.inf], np.nan).dropna().values
        if not len(v):
            continue
        ax.boxplot(v, positions=[k], widths=0.55, showfliers=False, patch_artist=True,
                   medianprops=dict(color=color, lw=2.2),
                   boxprops=dict(facecolor=color, alpha=0.15, edgecolor=color, lw=1.3),
                   whiskerprops=dict(color=color, lw=1.3), capprops=dict(color=color, lw=1.3))
        ax.scatter(k + rng.uniform(-0.1, 0.1, len(v)), v, s=26, color=color,
                   alpha=0.85, edgecolor="white", linewidth=0.5, zorder=3)
        n_cross = int((np.abs(v) > THRESH).sum())
        ax.text(k, YL[1], f"{n_cross}/{len(v)}", ha="center", va="top", fontsize=8, color=color)
    for s in (THRESH, -THRESH):
        ax.axhline(s, ls="--", lw=1, color="0.5")
    ax.axhline(0, color="0.8", lw=0.8)
    ax.set_xticks([0, 1]); ax.set_xticklabels([l for _, l, _ in SC], fontsize=9)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(*YL)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("z-score")
    ax.spines[["top", "right"]].set_visible(False)

from matplotlib.patches import Patch
fig.legend(handles=[Patch(facecolor=c, label=f"drift {l}") for _, l, c in SC],
           loc="lower center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.01))
fig.suptitle("TwINFER z-scores (all steps, forced through pair)  ·  gene_1–gene_2  ·  "
             "regenerated 2-state drift  ·  20 no-reg + 20 A→B  ·  $t_1$=1 h, $t_2$=20 h  ·  "
             f"annotations = |z|>{THRESH} count / 20", fontsize=10)
fig.tight_layout(rect=[0, 0.04, 1, 0.95])

for ext in ("png", "pdf"):
    p = os.path.join(D, f"drift_allsteps_zscores.{ext}")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print("saved", p)

# crossing table
tbl = pd.DataFrame({
    lbl: {t: int((df.loc[df.scenario == scen, c].abs() > THRESH).sum()) for c, t in PANELS}
    for scen, lbl, _ in SC
})
tbl.to_csv(os.path.join(D, "drift_allsteps_threshold_crossing.csv"))
print()
print(f"|z| > {THRESH}  (count / 20):")
print(tbl.to_string())
print()
print("medians:")
print(df.groupby("scenario")[cols].median().round(2).T.to_string())
