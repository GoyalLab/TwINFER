#!/usr/bin/env python
"""
Plot mean gene-expression trajectories (mRNA and protein) across time for each
real-network simulation, using the rep-0 `simulation_before_division_df`.

One PDF, per network: one page for mRNA and one page for protein.
  - n_genes < 5  -> all genes on a single axes (one coloured line + std band each)
  - n_genes >= 5 -> one subplot per gene, shared y-scale across the page

Mean +/- std is computed across the 6000 cells at each time_step.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

REAL_DATA = f'{TWINFER_PROJECT_ROOT}/simulation_data/real_data'
PAPER = f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis'
OUT_PDF = os.path.join(
    PAPER, "real_networks", "real_networks_gene_expression_trajectories.pdf"
)

# network -> glob for the simulation_before_division rep files (rep 0 = first sorted)
NETWORKS = {
    "Circadian_cycle": f"{PAPER}/Circadian_cycle/simulate/*/simulation_before_division_df*_Circadian_rep_0_*.csv",
    "mCAD": f"{REAL_DATA}/simulation_before_division_df*_ncells_6000_mCAD_1_0_*.csv",
    "VSC": f"{REAL_DATA}/simulation_before_division_df*_ncells_6000_VSC_2_0_*.csv",
    "B_cell_activation": f"{PAPER}/B_cell_activation/simulate/*/simulation_before_division_df*_B_cell_activation_rep_0_*.csv",
    "HSC_balanced": f"{REAL_DATA}/simulation_before_division_df*_ncells_6000_HSC_balanced_0_0_*.csv",
    "EMT": f"{PAPER}/EMT/simulate/*/simulation_before_division_df*_EMT_rep_0_*.csv",
    "GSD": f"{REAL_DATA}/simulation_before_division_df*_ncells_6000_GSD_0_0_*.csv",
    "Pluripotent": f"{PAPER}/Pluripotent/simulate/*/simulation_before_division_df*_Pluripotent_rep_0_*.csv",
}

BAND_ALPHA = 0.25


def pick_file(pattern):
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]


def load_stats(path, chunksize=500_000):
    """Return (time, genes, {gene: {'mRNA': (mean,std), 'protein': (mean,std)}}).

    Streamed in chunks so peak memory stays small even for the 36-gene /
    6000-timestep networks. mean/std are across all cells at each time_step.
    """
    header = pd.read_csv(path, nrows=0).columns.tolist()
    genes = sorted(
        {c.rsplit("_", 1)[0] for c in header if c.endswith("_mRNA")},
        key=lambda g: int(g.split("_")[1]),
    )
    value_cols = [f"{g}_{sp}" for g in genes for sp in ("mRNA", "protein")]
    usecols = ["time_step"] + value_cols
    dtypes = {c: "float64" for c in value_cols}

    acc_n = acc_sum = acc_sumsq = None
    tmax = -1
    for chunk in pd.read_csv(
        path, usecols=usecols, dtype=dtypes, chunksize=chunksize
    ):
        ts = chunk["time_step"].to_numpy().astype(np.int64)
        cmax = int(ts.max())
        if cmax > tmax:
            grow = cmax + 1 if acc_sum is None else cmax + 1 - acc_sum.shape[0]
            if acc_sum is None:
                acc_n = np.zeros(cmax + 1)
                acc_sum = np.zeros((cmax + 1, len(value_cols)))
                acc_sumsq = np.zeros((cmax + 1, len(value_cols)))
            else:
                acc_n = np.concatenate([acc_n, np.zeros(grow)])
                acc_sum = np.vstack([acc_sum, np.zeros((grow, len(value_cols)))])
                acc_sumsq = np.vstack([acc_sumsq, np.zeros((grow, len(value_cols)))])
            tmax = cmax
        vals = chunk[value_cols].to_numpy()
        np.add.at(acc_n, ts, 1.0)
        np.add.at(acc_sum, ts, vals)
        np.add.at(acc_sumsq, ts, vals * vals)

    n = acc_n
    mean = acc_sum / n[:, None]
    var = acc_sumsq / n[:, None] - mean ** 2
    # sample std (ddof=1) to match pandas .std()
    var = var * (n / np.clip(n - 1, 1, None))[:, None]
    std = np.sqrt(np.clip(var, 0, None))
    time = np.arange(mean.shape[0])

    stats = {}
    for gi, g in enumerate(genes):
        mi, pi = 2 * gi, 2 * gi + 1
        stats[g] = {
            "mRNA": (mean[:, mi], std[:, mi]),
            "protein": (mean[:, pi], std[:, pi]),
        }
    return time, genes, stats


def plot_single_axes(pdf, network, time, genes, stats, species, src):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    cmap = plt.get_cmap("tab10")
    for i, g in enumerate(genes):
        m, s = stats[g][species]
        c = cmap(i % 10)
        ax.plot(time, m, color=c, lw=1.5, label=g)
        ax.fill_between(
            time, np.clip(m - s, 0, None), m + s, color=c, alpha=BAND_ALPHA, lw=0
        )
    ax.set_xlabel("time step")
    ax.set_ylabel(f"{species} count  (mean ± std across cells)")
    ax.set_title(f"{network} — {species}  (rep 0, {len(genes)} genes)")
    ax.legend(ncol=2, fontsize=8, frameon=False)
    ax.margins(x=0)
    fig.text(0.01, 0.01, src, fontsize=5, color="grey")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_grid(pdf, network, time, genes, stats, species, src):
    n = len(genes)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    # shared y-scale across the page
    lo, hi = np.inf, -np.inf
    for g in genes:
        m, s = stats[g][species]
        lo = min(lo, float(np.clip(m - s, 0, None).min()))
        hi = max(hi, float((m + s).max()))
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    ylim = (lo - pad, hi + pad)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.0 * ncols, 2.4 * nrows), squeeze=False,
        sharex=True, sharey=True,
    )
    axes_flat = axes.ravel()
    for i, g in enumerate(genes):
        ax = axes_flat[i]
        m, s = stats[g][species]
        ax.plot(time, m, color="C0", lw=1.3)
        ax.fill_between(
            time, np.clip(m - s, 0, None), m + s, color="C0", alpha=BAND_ALPHA, lw=0
        )
        ax.set_title(g, fontsize=9)
        ax.set_ylim(ylim)
        ax.margins(x=0)
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    for ax in axes[-1, :]:
        ax.set_xlabel("time step")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"{species} count")

    fig.suptitle(
        f"{network} — {species}  (rep 0, {n} genes; shared y-scale, mean ± std across cells)",
        fontsize=11,
    )
    fig.text(0.01, 0.005, src, fontsize=5, color="grey")
    fig.tight_layout(rect=(0, 0.01, 1, 0.97))
    pdf.savefig(fig)
    plt.close(fig)


def main():
    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:
        for network, pattern in NETWORKS.items():
            path = pick_file(pattern)
            src = os.path.relpath(path, f'{TWINFER_PROJECT_ROOT}')
            print(f"[{network}] {src}")
            time, genes, stats = load_stats(path)
            print(f"    genes={len(genes)}  timesteps={len(time)} ({time.min()}..{time.max()})")
            for species in ("mRNA", "protein"):
                if len(genes) < 5:
                    plot_single_axes(pdf, network, time, genes, stats, species, src)
                else:
                    plot_grid(pdf, network, time, genes, stats, species, src)
    print(f"\nwrote {OUT_PDF}")


if __name__ == "__main__":
    main()
