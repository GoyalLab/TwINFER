#!/usr/bin/env python
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_807d072d_2026-09-20_real_network_sim; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""
Mean +/- std gene-expression trajectories (mRNA and protein) for the SEEDED
real-network sims, one line per steady-state basin the cells were seeded into.

Unlike the final-timepoint GMM clustering used for the IC-unset runs
(plot_real_network_final_clusters.py), the seeded runs assign each cell's
basin DETERMINISTICALLY at construction time: real_network_seeded_sim.py
splits cell_id into k contiguous blocks (np.linspace(0, n_cells, k+1)), one
per stable mean-field fixed point (ordered largest-mean-expression first),
so the state label is exact -- no re-clustering needed, just replicate that
same block assignment.

Output: analysis_data/paper_analysis/real_networks/real_networks_seeded_trajectories_by_state.pdf
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
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/network_analysis")
from simulations.network_analysis.predict_multistability import MeanField, load_params

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/package")
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/work_in_progress/benchmark")
from benchmarks.network_benchmarks.simulate.real_network_multistate_sim import NETWORKS as SIM_SPEC, build_matrices

ROOT = f'{TWINFER_PROJECT_ROOT}'
PAPER = f"{ROOT}/analysis_data/paper_analysis"
INPUT_ROOT = f"{ROOT}/input_data"
OUT_PDF = f"{PAPER}/real_networks/real_networks_seeded_trajectories_by_state.pdf"

N_CELLS = 6000
BAND_ALPHA = 0.18

SEEDED_PATTERN = {
    "VSC":  f"{PAPER}/VSC/simulate/multistate_6000steps_seeded/simulation_before_division_df*_VSC_seeded_rep_0_*.csv",
    "mCAD": f"{PAPER}/mCAD/simulate/multistate_2000steps_seeded/simulation_before_division_df*_mCAD_seeded_rep_0_*.csv",
    "GSD":  f"{PAPER}/GSD/simulate/multistate_2000steps_seeded/simulation_before_division_df*_GSD_seeded_rep_0_*.csv",
    "HSC":  f"{PAPER}/HSC/simulate/multistate_2000steps_seeded/simulation_before_division_df*_HSC_seeded_rep_0_*.csv",
    "EMT":  f"{PAPER}/EMT/simulate/multistate_2000steps_seeded/simulation_before_division_df*_EMT_seeded_rep_0_*.csv",
}


def read_input_matrix(path):
    M = np.loadtxt(path, dtype=int, delimiter=",")
    if M.ndim == 0:
        M = M.reshape((1, 1))
    return M


def pick(pattern):
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]


def n_seed_states(net):
    """Same computation as real_network_seeded_sim.seeded_states(): number of
    stable mean-field fixed points -> number of seed blocks, and the block
    boundaries over N_CELLS cell_ids (largest-mean-expression state first)."""
    spec = SIM_SPEC[net]
    matrix_path = f"{INPUT_ROOT}/real_world_networks/{net}.txt"
    M = read_input_matrix(matrix_path)
    P = load_params()
    mf = MeanField(M, P, kadd_scale=spec["kadd_scale"], n_hill=spec["n_hill"])
    fps = [f for f in mf.fixed_points(n_starts=250) if f["stable"]]
    k = len(fps)
    bounds = np.linspace(0, N_CELLS, k + 1).astype(int)
    return k, bounds


def load_by_state(path, k, bounds, chunksize=500_000):
    header = pd.read_csv(path, nrows=0).columns.tolist()
    genes = sorted(
        {c.rsplit("_", 1)[0] for c in header if c.endswith("_mRNA")},
        key=lambda g: int(g.split("_")[1]),
    )
    value_cols = [f"{g}_{sp}" for g in genes for sp in ("mRNA", "protein")]
    usecols = ["cell_id", "time_step"] + value_cols
    dtypes = {"cell_id": "int32", "time_step": "int32", **{c: "float64" for c in value_cols}}

    # map cell_id -> state block once
    state_of_cell = np.zeros(N_CELLS, dtype=np.int64)
    for s in range(k):
        state_of_cell[bounds[s]:bounds[s + 1]] = s

    acc_n = acc_sum = acc_sumsq = None
    tmax = -1
    for chunk in pd.read_csv(path, usecols=usecols, dtype=dtypes, chunksize=chunksize):
        ts = chunk["time_step"].to_numpy()
        cmax = int(ts.max())
        if cmax > tmax:
            if acc_sum is None:
                acc_n = np.zeros((k, cmax + 1))
                acc_sum = np.zeros((k, cmax + 1, len(value_cols)))
                acc_sumsq = np.zeros((k, cmax + 1, len(value_cols)))
            else:
                grow = cmax + 1 - acc_sum.shape[1]
                acc_n = np.concatenate([acc_n, np.zeros((k, grow))], axis=1)
                acc_sum = np.concatenate([acc_sum, np.zeros((k, grow, len(value_cols)))], axis=1)
                acc_sumsq = np.concatenate([acc_sumsq, np.zeros((k, grow, len(value_cols)))], axis=1)
            tmax = cmax
        state = state_of_cell[chunk["cell_id"].to_numpy()]
        vals = chunk[value_cols].to_numpy()
        flat = state * acc_sum.shape[1] + ts
        np.add.at(acc_n.reshape(-1), flat, 1.0)
        np.add.at(acc_sum.reshape(k * acc_sum.shape[1], len(value_cols)), flat, vals)
        np.add.at(acc_sumsq.reshape(k * acc_sum.shape[1], len(value_cols)), flat, vals * vals)

    n = acc_n[:, :, None]
    mean = acc_sum / np.clip(n, 1, None)
    var = acc_sumsq / np.clip(n, 1, None) - mean ** 2
    var = var * np.where(n > 1, n / np.clip(n - 1, 1, None), 1.0)
    std = np.sqrt(np.clip(var, 0, None))
    time = np.arange(mean.shape[1])

    stats = {}
    for gi, g in enumerate(genes):
        mi, pi = 2 * gi, 2 * gi + 1
        stats[g] = {
            "mRNA": (mean[:, :, mi], std[:, :, mi]),      # (k, T)
            "protein": (mean[:, :, pi], std[:, :, pi]),
        }
    return time, genes, stats, acc_n[:, 0].astype(int)  # cells per state


def plot_page(pdf, network, species, time, genes, stats, sizes, src, k):
    n = len(genes)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    cmap = plt.get_cmap("tab10")

    lo, hi = np.inf, -np.inf
    for g in genes:
        m, s = stats[g][species]
        lo = min(lo, float(np.clip(m - s, 0, None).min()))
        hi = max(hi, float((m + s).max()))
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    ylim = (lo - pad, hi + pad)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.5 * nrows),
                              squeeze=False, sharex=True, sharey=True)
    af = axes.ravel()
    for gi, g in enumerate(genes):
        ax = af[gi]
        m, s = stats[g][species]  # (k, T)
        for state in range(k):
            c = cmap(state % 10)
            ax.plot(time, m[state], color=c, lw=1.2)
            ax.fill_between(time, np.clip(m[state] - s[state], 0, None), m[state] + s[state],
                             color=c, alpha=BAND_ALPHA, lw=0)
        ax.set_title(g, fontsize=9)
        ax.set_ylim(ylim)
        ax.margins(x=0)
    for j in range(n, len(af)):
        af[j].set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("time step")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"{species} count")

    total = int(sizes.sum())
    handles = [plt.Line2D([], [], color=cmap(s % 10), lw=2,
                           label=f"state {s}: n={sizes[s]} ({100*sizes[s]/total:.0f}%)")
               for s in range(k)]
    fig.legend(handles=handles, loc="lower center", ncol=min(k, 5), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"{network} (seeded) -- {species}  (k={k} seed states, rep 0, "
                 f"mean +/- std within state; shared y-scale)", fontsize=11)
    fig.text(0.01, 0.005, src, fontsize=5, color="grey")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:
        for network, pattern in SEEDED_PATTERN.items():
            path = pick(pattern)
            src = os.path.relpath(path, ROOT)
            k, bounds = n_seed_states(network)
            print(f"[{network}] k={k} seed states, bounds={bounds.tolist()}", flush=True)
            print(f"    {os.path.basename(path)}", flush=True)
            time, genes, stats, sizes = load_by_state(path, k, bounds)
            print(f"    genes={len(genes)}  t_end={time.max()}  sizes={sizes.tolist()}", flush=True)
            for species in ("mRNA", "protein"):
                plot_page(pdf, network, species, time, genes, stats, sizes, src, k)
    print(f"\nwrote {OUT_PDF}")


if __name__ == "__main__":
    main()
