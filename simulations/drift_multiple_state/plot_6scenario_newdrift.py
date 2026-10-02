"""
Same two figures + threshold-crossing heatmap as plot_6scenario.py, but with the
`drift_A_to_B` scenario replaced by the freshly regenerated replicates.

  - 5 unchanged scenarios come from six_scenario_zscores_v2.csv
    (no_regulation, A_to_B, multistate_A_B, multistate_A_to_B, drift_A_B)
  - drift_A_to_B comes from drift_step2_step3_zscores.csv, the output of
    run_infer_drift_replicates.py --reg-only  (gene_1-gene_2 forced through
    Steps 1-4, 5000 null draws each).  Only the reps that exist are used
    (currently 10), so its box reflects n<=20.

Columns that run_infer_drift_replicates.py does not produce are left NaN for the
new drift_A_to_B rows and simply come up empty in those panels:
    z_div_t1 / z_div_t2            (divergence shuffle not computed)
    rho_delta_random_t1 / _t2      (random-pair rho_Delta not saved)

Output (six_scenario/new_drift_A_to_B/):
    six_scenario_zscores_newAtoB.png / _correlations_newAtoB.png
    six_scenario_threshold_crossing_newAtoB.png
    six_scenario_all_newAtoB.pdf
    six_scenario_zscores_merged_newAtoB.csv   (the merged long table)

Usage:
    python plot_6scenario_newdrift.py [NEW_DRIFT_CSV] [BASE_6SCENARIO_CSV]
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

BASE_DIR = f'{TWINFER_PROJECT_ROOT}/analysis_data/drift_inference'
NEW_DRIFT_CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    BASE_DIR, "drift_step2_step3_zscores.csv")
BASE_CSV = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    BASE_DIR, "six_scenario", "six_scenario_zscores_v2.csv")
OUT_DIR = os.path.join(BASE_DIR, "six_scenario", "new_drift_A_to_B")
os.makedirs(OUT_DIR, exist_ok=True)

SCEN = [
    ("no_regulation",     "no reg",         "unreg"),
    ("A_to_B",            "A→B",        "reg"),
    ("multistate_A_B",    "2-state\nA, B",   "unreg"),
    ("multistate_A_to_B", "2-state\nA→B", "reg"),
    ("drift_A_B",         "drift\nA, B",     "unreg"),
    ("drift_A_to_B",      "drift\nA→B", "reg"),
]
COL = {"reg": "#0072B2", "unreg": "#D55E00"}   # Okabe-Ito, CVD-safe
THRESH = 2.33

# ---------------------------------------------------------------- build the table
# run_infer_drift_replicates.py column  ->  six_scenario_zscores_v2.py column
NEW_TO_BASE = {
    "replicate":       "rep_id",
    "step1_z_t1":      "step1_z_t1",
    "step1_z_t2":      "step1_z_t2",
    "rho_t1":          "rho_t1",
    "rho_t2":          "rho_t2",
    "step2_z_t1":      "step2_z_het_t1",
    "step2_z_t2":      "step2_z_het_t2",
    "rho_delta_t1":    "rho_delta_t1",
    "rho_delta_t2":    "rho_delta_t2",
    "step3_z_d":       "step3_z_d",
    "step3_d":         "step3_d",
    "step3_z_critical": "step3_z_critical",
    "step4_z_1to2":    "step4_z_1to2",
    "step4_z_2to1":    "step4_z_2to1",
    "step4_rho_1to2":  "step4_rho_cross_1to2",
    "step4_rho_2to1":  "step4_rho_cross_2to1",
}

base = pd.read_csv(BASE_CSV)
base = base[base["scenario"] != "drift_A_to_B"].copy()

new = pd.read_csv(NEW_DRIFT_CSV)
if "error" in new:
    new = new[(new["error"].isna()) | (new["error"].astype(str).str.strip() == "")]
new = new.rename(columns=NEW_TO_BASE)[list(NEW_TO_BASE.values())].copy()
new.insert(0, "scenario", "drift_A_to_B")
new["step4_z"] = new["step4_z_1to2"]

df = pd.concat([base, new], ignore_index=True, sort=False)
merged_csv = os.path.join(OUT_DIR, "six_scenario_zscores_merged_newAtoB.csv")
df.to_csv(merged_csv, index=False)
print("wrote", merged_csv)

n_by_scen = df.groupby("scenario")["rep_id"].nunique().to_dict()
rng = np.random.default_rng(0)


# ---------------------------------------------------------------- plot helpers
def shared_ylim(cols, pad=0.08, clip=(1, 99)):
    v = pd.concat([df[c] for c in cols if c in df]).replace([np.inf, -np.inf], np.nan).dropna()
    lo, hi = np.percentile(v, clip[0]), np.percentile(v, clip[1])
    lo, hi = min(lo, v.min()), max(hi, v.max())
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
    for y in hlines:
        ax.axhline(y, ls="--", lw=1, color="0.5")
    ax.axhline(0, color="0.75", lw=0.8, zorder=1)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(len(SCEN)))
    ax.set_xticklabels([f"{l}\n(n={n_by_scen.get(s, 0)})" for s, l, _ in SCEN], fontsize=7)
    ax.set_xlim(-0.6, len(SCEN) - 0.4)
    ax.set_ylim(*ylim)
    ax.spines[["top", "right"]].set_visible(False)


def make_fig(specs, ylim, suptitle, stem, ncol=3):
    n = len(specs)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.7 * ncol, 3.9 * nrow), squeeze=False)
    for ax, (col, title, hl) in zip(axes.flat, specs):
        panel(ax, col, title, ylim, hlines=hl)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.legend(handles=[Patch(facecolor=COL["reg"], label="regulated (A→B)"),
                        Patch(facecolor=COL["unreg"], label="unregulated (A, B)")],
               loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    p = os.path.join(OUT_DIR, f"{stem}.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print("saved", p)
    return fig


Z_SPECS = [
    ("step1_z_t1",     r"Step 1 z  ($t_1$)",            (THRESH, -THRESH)),
    ("step1_z_t2",     r"Step 1 z  ($t_2$)",            (THRESH, -THRESH)),
    ("step3_z_d",      r"Step 3 z  $z_d$",              (THRESH, -THRESH)),
    ("step2_z_het_t1", r"Step 2 z$_{\rm het}$  ($t_1$)", (THRESH, -THRESH)),
    ("step2_z_het_t2", r"Step 2 z$_{\rm het}$  ($t_2$)", (THRESH, -THRESH)),
    ("step4_z_1to2",   r"Step 4 z  $g_1\!\to\!g_2$",    (THRESH, -THRESH)),
    ("z_div_t1",       r"z$_{\rm divergence}$  ($t_1$)", (THRESH, -THRESH)),
    ("z_div_t2",       r"z$_{\rm divergence}$  ($t_2$)", (THRESH, -THRESH)),
    ("step4_z_2to1",   r"Step 4 z  $g_2\!\to\!g_1$",    (THRESH, -THRESH)),
]
C_SPECS = [
    ("rho_t1",               r"gene-gene $\rho$  ($t_1$)", ()),
    ("rho_t2",               r"gene-gene $\rho$  ($t_2$)", ()),
    ("step4_rho_cross_1to2", r"cross $\rho$  $g_1\!\to\!g_2$", ()),
    ("rho_delta_t1",         r"twin $\hat{\rho}_\Delta$  ($t_1$)", ()),
    ("rho_delta_t2",         r"twin $\hat{\rho}_\Delta$  ($t_2$)", ()),
    ("step4_rho_cross_2to1", r"cross $\rho$  $g_2\!\to\!g_1$", ()),
    ("rho_delta_random_t1",  r"random-pair $\rho_\Delta$  ($t_1$)", ()),
    ("rho_delta_random_t2",  r"random-pair $\rho_\Delta$  ($t_2$)", ()),
]

z_cols = [s[0] for s in Z_SPECS]
c_cols = [s[0] for s in C_SPECS]

figz = make_fig(Z_SPECS, shared_ylim(z_cols),
                "TwINFER z-scores  ·  gene_1–gene_2  ·  6 scenarios  ·  "
                "drift A→B = regenerated reps  ·  $t_1$=1 h, $t_2$=20 h",
                "six_scenario_zscores_newAtoB", ncol=3)
figc = make_fig(C_SPECS, shared_ylim(c_cols),
                "TwINFER correlations  ·  gene_1–gene_2  ·  6 scenarios  ·  "
                "drift A→B = regenerated reps  ·  $t_1$=1 h, $t_2$=20 h",
                "six_scenario_correlations_newAtoB", ncol=3)

# ---------------------------------------------------------------- threshold table
scen_order = [s[0] for s in SCEN]
z_labels = {
    "step1_z_t1": "Step1 z (t1)", "step1_z_t2": "Step1 z (t2)",
    "step2_z_het_t1": "Step2 z_het (t1)", "step2_z_het_t2": "Step2 z_het (t2)",
    "z_div_t1": "z_div (t1)", "z_div_t2": "z_div (t2)",
    "step3_z_d": "Step3 z_d", "step4_z_1to2": "Step4 z (g1->g2)",
    "step4_z_2to1": "Step4 z (g2->g1)",
}
frac = pd.DataFrame(index=scen_order, columns=list(z_labels.values()), dtype=float)
cnt = frac.copy()
tot = pd.Series({s: n_by_scen.get(s, 0) for s in scen_order})
for scen in scen_order:
    sub = df[df["scenario"] == scen]
    for col, lbl in z_labels.items():
        v = sub[col].abs().dropna() if col in sub else pd.Series(dtype=float)
        cnt.loc[scen, lbl] = int((v > THRESH).sum())
        frac.loc[scen, lbl] = (v > THRESH).mean() if len(v) else np.nan
labels = [l for _, l, _ in SCEN]
frac.index = labels
cnt.index = labels
frac.to_csv(os.path.join(OUT_DIR, "six_scenario_threshold_crossing_fraction_newAtoB.csv"))
cnt.to_csv(os.path.join(OUT_DIR, "six_scenario_threshold_crossing_count_newAtoB.csv"))

fig_t, ax_t = plt.subplots(figsize=(1.5 + 1.15 * len(z_labels), 0.55 * len(SCEN) + 1.6))
im = ax_t.imshow(frac.values.astype(float), cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax_t.set_xticks(range(len(z_labels)))
ax_t.set_xticklabels(list(z_labels.values()), rotation=40, ha="right", fontsize=8)
ax_t.set_yticks(range(len(SCEN)))
ax_t.set_yticklabels(frac.index, fontsize=9)
for i, scen in enumerate(scen_order):
    for j in range(len(z_labels)):
        fr = frac.values[i, j]
        txt = "-" if np.isnan(fr) else f"{cnt.values[i, j]:.0f}/{tot[scen]}"
        ax_t.text(j, i, txt, ha="center", va="center",
                  fontsize=8, color="white" if (not np.isnan(fr) and fr > 0.55) else "0.15")
ax_t.set_title(f"replicates with |z| > {THRESH}  (count / n)", fontsize=11, pad=10)
fig_t.colorbar(im, ax=ax_t, label="fraction", fraction=0.025, pad=0.02)
fig_t.tight_layout()
p = os.path.join(OUT_DIR, "six_scenario_threshold_crossing_newAtoB.png")
fig_t.savefig(p, dpi=200, bbox_inches="tight")
print("saved", p)

with PdfPages(os.path.join(OUT_DIR, "six_scenario_all_newAtoB.pdf")) as pdf:
    pdf.savefig(figz, bbox_inches="tight")
    pdf.savefig(figc, bbox_inches="tight")
    pdf.savefig(fig_t, bbox_inches="tight")
print("saved", os.path.join(OUT_DIR, "six_scenario_all_newAtoB.pdf"))

print("\nmedian z by scenario:")
print(df.groupby("scenario")[[c for c in z_cols if c in df]].median().round(2).to_string())
print(f"\n|z| > {THRESH}  (count / n):")
print(cnt.to_string())
