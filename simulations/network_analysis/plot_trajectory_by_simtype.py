#!/usr/bin/env python
"""
Compare per-gene mRNA / protein trajectories across the three simulation
variants of the real-network sims, to see what actually changed between them:

  type 1  "original"    -- the pre-multistate-work single-state sims
                           (GSD/HSC/VSC/mCAD in simulation_data/real_data/,
                            EMT in analysis_data/paper_analysis/EMT/simulate/20260825_224653/)
  type 2  "multistate"  -- IC-unset rerun at per-network tuned (Hill n, k_add)
                           (analysis_data/paper_analysis/<net>/simulate/multistate_*steps/)
  type 3  "seeded"      -- seeded-IC rerun (cells split across mean-field basins)
                           (analysis_data/paper_analysis/<net>/simulate/multistate_*steps_seeded/)

VSC and mCAD have no type-2 (those IC-unset jobs were cancelled as identical to
the originals), so they show 2 curves; GSD/HSC/EMT show 3.

Mean +/- std across the 6000 cells at each time_step, from rep-0's
simulation_before_division_df. One PDF, per network: one page for mRNA and one
for protein, one subplot per gene, the sim variants overlaid on a shared time
axis (they end at different steps -- 6000 vs 2000 vs 800/1500 -- which is itself
part of what changed).

Output: analysis_data/paper_analysis/real_networks/real_networks_trajectory_by_simtype.pdf
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, os.path.dirname(__file__))
from simulations.network_analysis.plot_real_network_trajectories import load_stats  # streamed mean/std reader

ROOT = f'{TWINFER_PROJECT_ROOT}'
REAL_DATA = f"{ROOT}/simulation_data/real_data"
PAPER = f"{ROOT}/analysis_data/paper_analysis"
OUT_PDF = f"{PAPER}/real_networks/real_networks_trajectory_by_simtype.pdf"

BAND_ALPHA = 0.18
VARIANT_COLORS = {"original": "#4C72B0", "multistate": "#DD8452", "seeded": "#55A868"}

# network -> {variant: glob for rep-0 simulation_before_division file}
NETWORKS = {
    "GSD": {
        "original":   f"{REAL_DATA}/simulation_before_division_df*_ncells_6000_GSD_0_0_*.csv",
        "multistate": f"{PAPER}/GSD/simulate/multistate_2000steps/simulation_before_division_df*_GSD_multistate_rep_0_*.csv",
        "seeded":     f"{PAPER}/GSD/simulate/multistate_2000steps_seeded/simulation_before_division_df*_GSD_seeded_rep_0_*.csv",
    },
    "HSC": {
        "original":   f"{REAL_DATA}/simulation_before_division_df*_ncells_6000_HSC_balanced_0_0_*.csv",
        "multistate": f"{PAPER}/HSC/simulate/multistate_2000steps/simulation_before_division_df*_HSC_multistate_rep_0_*.csv",
        "seeded":     f"{PAPER}/HSC/simulate/multistate_2000steps_seeded/simulation_before_division_df*_HSC_seeded_rep_0_*.csv",
    },
    "EMT": {
        "original":   f"{PAPER}/EMT/simulate/20260825_224653/simulation_before_division_df*_EMT_rep_0_*.csv",
        "multistate": f"{PAPER}/EMT/simulate/multistate_2000steps/simulation_before_division_df*_EMT_multistate_rep_0_*.csv",
        "seeded":     f"{PAPER}/EMT/simulate/multistate_2000steps_seeded/simulation_before_division_df*_EMT_seeded_rep_0_*.csv",
    },
    "VSC": {
        "original": f"{REAL_DATA}/simulation_before_division_df*_ncells_6000_VSC_2_0_*.csv",
        "seeded":   f"{PAPER}/VSC/simulate/multistate_6000steps_seeded/simulation_before_division_df*_VSC_seeded_rep_0_*.csv",
    },
    "mCAD": {
        "original": f"{REAL_DATA}/simulation_before_division_df*_ncells_6000_mCAD_1_0_*.csv",
        "seeded":   f"{PAPER}/mCAD/simulate/multistate_2000steps_seeded/simulation_before_division_df*_mCAD_seeded_rep_0_*.csv",
    },
}


def pick(pattern):
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None


def plot_page(pdf, network, species, per_variant):
    """per_variant: {variant: (time, genes, stats)}"""
    genes = next(iter(per_variant.values()))[1]
    n = len(genes)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    lo, hi = np.inf, -np.inf
    for _, (_, _, stats) in per_variant.items():
        for g in genes:
            m, s = stats[g][species]
            lo = min(lo, float(np.clip(m - s, 0, None).min()))
            hi = max(hi, float((m + s).max()))
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    ylim = (lo - pad, hi + pad)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.5 * nrows),
                             squeeze=False, sharex=True, sharey=True)
    axf = axes.ravel()
    for i, g in enumerate(genes):
        ax = axf[i]
        for variant, (time, _, stats) in per_variant.items():
            m, s = stats[g][species]
            c = VARIANT_COLORS[variant]
            ax.plot(time, m, color=c, lw=1.3, label=variant)
            ax.fill_between(time, np.clip(m - s, 0, None), m + s, color=c, alpha=BAND_ALPHA, lw=0)
        ax.set_title(g, fontsize=9)
        ax.set_ylim(ylim)
        ax.margins(x=0)
    for j in range(n, len(axf)):
        axf[j].set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("time step")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"{species} count")
    axf[0].legend(fontsize=8, frameon=False, loc="best")

    variants_str = ", ".join(per_variant)
    fig.suptitle(f"{network} -- {species}  (rep 0, {n} genes; mean +/- std across cells; variants: {variants_str})",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.01, 1, 0.97))
    pdf.savefig(fig)
    plt.close(fig)


def main():
    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:
        for network, variants in NETWORKS.items():
            per_variant = {}
            for variant, pattern in variants.items():
                path = pick(pattern)
                if path is None:
                    print(f"[{network}/{variant}] NOT FOUND: {pattern}")
                    continue
                print(f"[{network}/{variant}] {os.path.relpath(path, ROOT)}", flush=True)
                time, genes, stats = load_stats(path)
                per_variant[variant] = (time, genes, stats)
                print(f"    genes={len(genes)}  timesteps={time.min()}..{time.max()}", flush=True)
            if not per_variant:
                continue
            for species in ("mRNA", "protein"):
                plot_page(pdf, network, species, per_variant)
    print(f"\nwrote {OUT_PDF}")


if __name__ == "__main__":
    main()
