#!/usr/bin/env python
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_807d072d_2026-09-20_real_network_sim; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""
Independent, model-free relaxation-time / steady-state-window estimate for
each real network -- NOT the codebase's own is_steady_state()/exponential-fit
machinery.

Why not fit an exponential (v1 attempt): some genes (e.g. mCAD gene_4/gene_5)
show a transient OVERSHOOT -- the population-mean protein spikes early
(before repressors build up) then relaxes DOWN to a lower plateau. A
monotonic single-exponential can't represent that, so instead of assuming
any functional form we detect settling directly from the data:

Method ("last exit from the noise band"):
  1. P_inf(gene) = mean of the population-mean trajectory over the final
     `tail_frac` of time steps (the plateau level).
  2. Statistical noise band: at each time step the population mean is an
     average over n_cells (~6000) i.i.d. cells, so its own sampling
     uncertainty is SEM(t) = std_across_cells(t) / sqrt(n_cells). We take
     the reference noise level as the SEM averaged over that same tail
     window, and call the trajectory "settled" once it's within k*SEM_tail
     of P_inf -- i.e. statistically indistinguishable from the endpoint,
     scale-free across genes regardless of absolute expression level (fixes
     the earlier bug where a %-of-range band blew up for near-zero /
     near-constant genes).
  3. Settling time t* = the LAST time index at which |mean(t) - P_inf| >
     k*SEM_tail, plus one (so re-entries into the band from a later bump
     don't fool it -- must stay inside forever after).
  4. Also record the transient overshoot: the largest |mean(t) - P_inf|
     reached before t*, and whether it's an overshoot (above) or undershoot
     (below), so the "bump" is quantified, not just implied.
  5. Network-level relaxation time = max t* over all genes (mRNA + protein);
     the slowest gene/species sets the pace for the whole network.
"""
import glob
import math
import os
import sys

import numpy as np
import pandas as pd

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/network_analysis")
from simulations.network_analysis.plot_real_network_trajectories import NETWORKS, pick_file

K_SEM = 5.0        # "settled" = within 5 standard errors of the plateau mean
TAIL_FRAC = 0.10   # last 10% of time steps define the plateau + reference noise


def load_stats_with_n(path, chunksize=500_000):
    """Same streaming accumulation as plot_real_network_trajectories.load_stats,
    but also returns n(t) (cells contributing at each time_step) for SEM."""
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
    for chunk in pd.read_csv(path, usecols=usecols, dtype=dtypes, chunksize=chunksize):
        ts = chunk["time_step"].to_numpy().astype(np.int64)
        cmax = int(ts.max())
        if cmax > tmax:
            if acc_sum is None:
                acc_n = np.zeros(cmax + 1)
                acc_sum = np.zeros((cmax + 1, len(value_cols)))
                acc_sumsq = np.zeros((cmax + 1, len(value_cols)))
            else:
                grow = cmax + 1 - acc_sum.shape[0]
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
    var = var * (n / np.clip(n - 1, 1, None))[:, None]
    std = np.sqrt(np.clip(var, 0, None))
    time = np.arange(mean.shape[0])

    stats = {}
    for gi, g in enumerate(genes):
        mi, pi = 2 * gi, 2 * gi + 1
        stats[g] = {
            "mRNA": (mean[:, mi], std[:, mi], n),
            "protein": (mean[:, pi], std[:, pi], n),
        }
    return time, genes, stats


def settle(mean_traj, std_traj, n_traj):
    tail_n = max(3, int(TAIL_FRAC * len(mean_traj)))
    P_inf = mean_traj[-tail_n:].mean()
    sem_traj = std_traj / np.sqrt(np.clip(n_traj, 1, None))
    sem_ref = sem_traj[-tail_n:].mean()
    band = max(K_SEM * sem_ref, 1e-9)

    dev = mean_traj - P_inf
    outside = np.abs(dev) > band
    outside_idx = np.where(outside)[0]
    t_star = 0 if outside_idx.size == 0 else int(outside_idx[-1]) + 1

    # transient extremum before settling
    if t_star > 0:
        pre = dev[:t_star]
        peak_i = int(np.argmax(np.abs(pre)))
        peak_val = pre[peak_i]
        peak_t = peak_i
        kind = "overshoot" if peak_val > 0 else "undershoot"
    else:
        peak_t, peak_val, kind = None, 0.0, "none"

    return t_star, P_inf, band, peak_t, peak_val, kind


def main():
    rows = []
    for network, pattern in NETWORKS.items():
        try:
            path = pick_file(pattern)
        except FileNotFoundError:
            print(f"[{network}] NOT FOUND")
            continue
        print(f"[{network}] loading {os.path.basename(path)}", flush=True)
        time, genes, stats = load_stats_with_n(path)
        t_end = int(time.max())

        per = {}
        for g in genes:
            for species in ("mRNA", "protein"):
                m, s, n = stats[g][species]
                t_star, P_inf, band, peak_t, peak_val, kind = settle(m, s, n)
                per[(g, species)] = t_star
                tag = f"{kind} of {peak_val:+.3g} at t={peak_t}" if kind != "none" else "monotonic/flat"
                print(f"    {g:<10}{species:<9} P_inf={P_inf:12.3g}  band=+/-{band:9.3g}  "
                      f"settle_t={t_star:6d}  {tag}", flush=True)

        onset = max(per.values())
        worst = max(per, key=per.get)
        print(f"    -> network settle_t={onset} / t_end={t_end} ({onset/t_end:.1%}), slowest={worst}\n", flush=True)
        rows.append((network, len(genes), t_end, onset, onset / t_end if t_end else float("nan"), worst))

    print("\n=== summary: model-free settling time (k=%.0f SEM, tail=%.0f%%) ===" % (K_SEM, TAIL_FRAC * 100))
    print(f"{'network':<20}{'n_genes':>8}{'t_end':>8}{'settle_t':>10}{'frac':>8}   slowest gene/species")
    for network, ng, t_end, onset, frac, worst in rows:
        print(f"{network:<20}{ng:>8}{t_end:>8}{onset:>10}{frac:>8.1%}   {worst}")


if __name__ == "__main__":
    main()
