"""
Overlay all four low->mid series on one figure:
   {K_frozen, K_ramp}  x  {A_B (unreg), A_to_B (reg)}
from the aggregated <tag>_zscores_vs_time.csv files.

  color     = network   (A_B orange, A_to_B blue)
  linestyle = kind       (K_frozen solid, K_ramp dashed)

K_ramp: mid half's k_on ramps 0.12->0.66 over tau AND its g1->g2 Hill K is
tracked live (per-timestep mean-field K from the population-average gene_1).

Output: analysis_data/drift_lowmid/lowmid_combined_{zscores,correlations}_vs_time.png
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

BASE = f'{TWINFER_PROJECT_ROOT}/analysis_data/drift_lowmid'
SERIES = [  # (label, csv, color, linestyle)
    ("A,B  K_frozen",  f"{BASE}/K_frozen/A_B_no_reg_lowmid_K_frozen_zscores_vs_time.csv",  "#D55E00", "-"),
    ("A,B  K_ramp",    f"{BASE}/K_ramp/A_B_no_reg_lowmid_K_ramp_zscores_vs_time.csv",      "#D55E00", "--"),
    ("A->B K_frozen",  f"{BASE}/K_frozen/A_to_B_lowmid_K_frozen_zscores_vs_time.csv",      "#0072B2", "-"),
    ("A->B K_ramp",    f"{BASE}/K_ramp/A_to_B_lowmid_K_ramp_zscores_vs_time.csv",          "#0072B2", "--"),
]
Z_SPECS = [("z_gene", "Step 1  z (gene-gene rho)"), ("z_het", "Step 2  z_het"),
           ("z_reg", "Step 2  z_regulation"), ("z_d", "Step 3  z_d"),
           ("z_1to2", "Step 4  z  g1->g2"), ("z_2to1", "Step 4  z  g2->g1"),
           ("z_gamma", "Step 4  z_gamma")]
C_SPECS = [("rho_gene", "gene-gene rho"), ("rho_delta", "twin rho_Delta"),
           ("rho_delta_random", "random-pair rho_Delta"), ("step3_d", "d = rho_D(t)-rho_D(t1)"),
           ("rho_cross_1to2", "cross rho g1->g2"), ("rho_cross_2to1", "cross rho g2->g1")]

DATA = [(lab, pd.read_csv(csv).sort_values("time"), c, ls) for lab, csv, c, ls in SERIES]


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

    handles = [
        Line2D([0], [0], color="#D55E00", lw=2, label="A,B  (unregulated)"),
        Line2D([0], [0], color="#0072B2", lw=2, label="A->B  (regulated)"),
        Line2D([0], [0], color="0.3", lw=2, ls="-",  label="K_frozen"),
        Line2D([0], [0], color="0.3", lw=2, ls="--", label="K_ramp"),
    ]
    fig.legend(handles=handles, ncol=4, loc="lower center", frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    p = os.path.join(BASE, stem)
    fig.savefig(p, dpi=170, bbox_inches="tight")
    print("saved", p)


if __name__ == "__main__":
    ttl = ("low->mid drift  -  K_frozen (solid) vs K_ramp (dashed)  x  "
           "A,B (orange) vs A->B (blue)  -  3 reps, N=5000, ref t1=1h")
    make(Z_SPECS, "z", ttl + "  -  z-scores vs time",
         "lowmid_combined_zscores_vs_time.png", thr=2.33)
    make(C_SPECS, r"$\rho$", ttl + "  -  correlations vs time",
         "lowmid_combined_correlations_vs_time.png")
