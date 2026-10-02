"""
Plot infer_with_twinfer results on the regenerated 2-state drift sims
(drift_infer_summary.csv): classification breakdown + the z-scores the
pipeline actually produced, drift_A_B (no reg) vs drift_A_to_B (reg).
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

D = f'{TWINFER_PROJECT_ROOT}/analysis_data/drift_inference/drift_infer_with_twinfer'
df = pd.read_csv(os.path.join(D, "drift_infer_summary.csv"))
df = df[(df["error"].isna()) | (df["error"] == "")]

SC = [("drift_A_B", "no reg", "#D55E00"), ("drift_A_to_B", "A→B", "#0072B2")]
CLS_ORDER = ["no_regulation", "single_state_regulation",
             "multiple_states_no_reg", "multiple_states_and_reg"]
CLS_COL = {"no_regulation": "#bdbdbd", "single_state_regulation": "#0072B2",
           "multiple_states_no_reg": "#E69F00", "multiple_states_and_reg": "#009E73"}
rng = np.random.default_rng(0)

fig, ax = plt.subplots(1, 4, figsize=(17, 4.6))

# ---- panel 0: classification breakdown ----
for k, (scen, lbl, _) in enumerate(SC):
    sub = df[df["scenario"] == scen]
    bottom = 0
    for cls in CLS_ORDER:
        n = int((sub["classification"] == cls).sum())
        if n:
            ax[0].bar(k, n, bottom=bottom, color=CLS_COL[cls], width=0.6,
                      label=cls if k == 0 or cls not in [t.get_label() for t in ax[0].containers] else None)
            ax[0].text(k, bottom + n / 2, str(n), ha="center", va="center", fontsize=9)
            bottom += n
ax[0].set_xticks(range(len(SC)))
ax[0].set_xticklabels([l for _, l, _ in SC])
ax[0].set_ylabel("replicates")
ax[0].set_title(f"TwINFER classification\n(t1=1 h, t2=20 h)", fontsize=10)
handles = [plt.Rectangle((0, 0), 1, 1, color=CLS_COL[c]) for c in CLS_ORDER]
ax[0].legend(handles, [c.replace("_", " ") for c in CLS_ORDER], fontsize=7, frameon=False, loc="upper center")
ax[0].spines[["top", "right"]].set_visible(False)

# ---- panels 1-3: z-scores ----
panels = [
    ("step1_z", "Step 1 z  (gene–gene ρ, t1)", 2.576),
    ("step2_z_het", "Step 2 z$_{\\rm het}$  (where computed)", 5.0),
    ("step4_z_1to2", "Step 4 z  g1→g2  (where computed)", 2.5),
]
for a, (col, title, thr) in zip(ax[1:], panels):
    for k, (scen, lbl, color) in enumerate(SC):
        v = df.loc[df["scenario"] == scen, col].astype(float).replace([np.inf, -np.inf], np.nan).dropna().values
        if len(v):
            a.boxplot(v, positions=[k], widths=0.5, showfliers=False, patch_artist=True,
                      medianprops=dict(color=color, lw=2), boxprops=dict(facecolor=color, alpha=0.15, edgecolor=color),
                      whiskerprops=dict(color=color), capprops=dict(color=color))
            a.scatter(k + rng.uniform(-0.09, 0.09, len(v)), v, s=28, color=color,
                      alpha=0.85, edgecolor="white", linewidth=0.5, zorder=3)
        a.text(k, a.get_ylim()[0], f"n={len(v)}", ha="center", va="bottom", fontsize=8, color="0.4")
    for s in ({thr, -thr} if col != "step4_z_1to2" else {thr, -thr}):
        a.axhline(s, ls="--", lw=1, color="0.5")
    a.axhline(0, color="0.8", lw=0.8)
    a.set_xticks(range(len(SC)))
    a.set_xticklabels([l for _, l, _ in SC])
    a.set_title(title, fontsize=10)
    a.set_ylabel("z-score")
    a.spines[["top", "right"]].set_visible(False)

fig.suptitle("infer_with_twinfer on regenerated 2-state drift sims  ·  gene_1–gene_2  ·  "
             f"{(df.scenario=='drift_A_B').sum()} no-reg + {(df.scenario=='drift_A_to_B').sum()} A→B replicates",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
for ext in ("png", "pdf"):
    p = os.path.join(D, f"drift_infer_plot.{ext}")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print("saved", p)

print()
print(df.groupby(["scenario", "classification"]).size().to_string())
