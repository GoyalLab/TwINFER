"""
Single-state (k_on = 0.66 median) comparison: what does a wrong Hill K do to the
TwINFER z-scores when the population is NOT multi-state?

  A,B                 unregulated control (no K)                grey  solid
  A->B  K@0.66        Hill K calibrated at the true k_on        blue  solid
  A->B  K@0.12        Hill K calibrated too low -> edge sits    red   dashed
                      saturated (g1 >> K), g2 over-driven

From analysis_data/drift_lowmid/single_2K/<tag>_zscores_vs_time.csv .
Output: analysis_data/drift_lowmid/single2K_combined_{zscores,correlations}_vs_time.png
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = f'{TWINFER_PROJECT_ROOT}/analysis_data/drift_lowmid/single_2K'
OUT = f'{TWINFER_PROJECT_ROOT}/analysis_data/drift_lowmid'
SERIES = [  # (label, csv, color, linestyle)
    ("A,B  (unreg control)", f"{BASE}/A_B_no_reg_single_Kcalib_0p66_zscores_vs_time.csv", "0.45", "-"),
    ("A->B  K@0.66 (correct)", f"{BASE}/A_to_B_single_Kcalib_0p66_zscores_vs_time.csv",   "#0072B2", "-"),
    ("A->B  K@0.12 (too low)", f"{BASE}/A_to_B_single_Kcalib_0p12_zscores_vs_time.csv",   "#D55E00", "--"),
]
Z_SPECS = [("z_gene", "Step 1  z (gene-gene rho)"), ("z_het", "Step 2  z_het"),
           ("z_reg", "Step 2  z_regulation"), ("z_d", "Step 3  z_d"),
           ("z_1to2", "Step 4  z  g1->g2"), ("z_2to1", "Step 4  z  g2->g1"),
           ("z_gamma", "Step 4  z_gamma")]
C_SPECS = [("rho_gene", "gene-gene rho"), ("rho_delta", "twin rho_Delta"),
           ("rho_delta_random", "random-pair rho_Delta"), ("step3_d", "d = rho_D(t)-rho_D(t1)"),
           ("rho_cross_1to2", "cross rho g1->g2"), ("rho_cross_2to1", "cross rho g2->g1")]

DATA = []
for lab, csv, c, ls in SERIES:
    if os.path.exists(csv):
        DATA.append((lab, pd.read_csv(csv).sort_values("time"), c, ls))
    else:
        print("missing", csv)


def make(specs, ylab, title, stem, thr=None):
    n = len(specs); ncol = 3; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 3.6 * nrow), squeeze=False)
    for ax, (col, name) in zip(axes.flat, specs):
        for lab, d, c, ls in DATA:
            if f"{col}_mean" not in d:
                continue
            m, s = d[f"{col}_mean"], d[f"{col}_std"]
            ax.plot(d.time, m, ls, marker="o", ms=2.5, lw=1.6, color=c)
            ax.fill_between(d.time, m - s, m + s, color=c, alpha=0.10)
        ax.axhline(0, color="0.7", lw=0.8)
        if thr:
            for y in (thr, -thr):
                ax.axhline(y, ls=":", lw=1, color="0.5")
        ax.set_title(name, fontsize=10)
        ax.set_ylabel(ylab)
        ax.set_xlabel("twin time after division (h)")
        ax.tick_params(labelbottom=True)
        ax.grid(alpha=0.25)
    for ax in axes.flat[n:]:
        ax.set_visible(False)

    handles = [Line2D([0], [0], color=c, lw=2, ls=ls, label=lab) for lab, _, c, ls in DATA]
    fig.legend(handles=handles, ncol=3, loc="lower center", frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    p = os.path.join(OUT, stem)
    fig.savefig(p, dpi=170, bbox_inches="tight")
    print("saved", p)


if __name__ == "__main__":
    ttl = ("single-state k_on=0.66  -  effect of a mis-calibrated Hill K  -  3 reps, N=5000, ref t1=1h")
    make(Z_SPECS, "z", ttl + "  -  z-scores vs time",
         "single2K_combined_zscores_vs_time.png", thr=2.33)
    make(C_SPECS, r"$\rho$", ttl + "  -  correlations vs time",
         "single2K_combined_correlations_vs_time.png")
