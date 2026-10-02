#!/usr/bin/env python
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_807d072d_2026-09-20_real_network_sim; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""
For each real-network sim, estimate the time window during which the system
has reached steady state.

Method ("settling time", analogous to control-theory step-response settling):
  1. For each gene x species (mRNA, protein), take the reference level x_final
     as the mean of the mean-trajectory over the LAST `tail_frac` fraction of
     time steps (assumes the sim runs long enough to actually equilibrate by
     the end).
  2. Tolerance band = tol * range(trajectory), where range = max-min of the
     mean trajectory over all time (so it's scale-free per gene).
  3. t*_gene = the earliest time such that |mean(t) - x_final| <= band for
     EVERY t >= t*_gene (i.e. the last time it exits the band, plus one) --
     this is the standard "last crossing" settling-time definition, robust to
     transient re-entries.
  4. Network-level onset = max over all genes & both species of t*_gene
     (the whole system isn't "settled" until every gene is).
  5. Steady-state window reported as [onset, T_end] where T_end is the last
     simulated time step.

Reuses the streamed load_stats() from plot_real_network_trajectories.py so
memory stays flat even for the multi-GB CSVs.
"""
import glob
import os
import sys

import numpy as np

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/network_analysis")
from simulations.network_analysis.plot_real_network_trajectories import load_stats, NETWORKS, pick_file

TOL = 0.02        # 2% of the trajectory's own range
TAIL_FRAC = 0.10  # last 10% of time steps define the "final" reference level


def settling_time(mean_traj):
    x_final = mean_traj[-max(1, int(TAIL_FRAC * len(mean_traj))):].mean()
    rng = mean_traj.max() - mean_traj.min()
    band = TOL * rng if rng > 0 else 1e-9
    within = np.abs(mean_traj - x_final) <= band
    # last index where it's OUTSIDE the band; settle = next step after that
    outside_idx = np.where(~within)[0]
    if outside_idx.size == 0:
        return 0
    return int(outside_idx[-1]) + 1


def main():
    results = []
    for network, pattern in NETWORKS.items():
        try:
            path = pick_file(pattern)
        except FileNotFoundError:
            print(f"[{network}] NOT FOUND: {pattern}")
            continue
        print(f"[{network}] loading {os.path.basename(path)}", flush=True)
        time, genes, stats = load_stats(path)
        t_end = int(time.max())

        per_gene_onset = {}
        for g in genes:
            for species in ("mRNA", "protein"):
                m, s = stats[g][species]
                per_gene_onset[(g, species)] = settling_time(m)

        onset = max(per_gene_onset.values())
        worst = max(per_gene_onset, key=per_gene_onset.get)
        frac = onset / t_end if t_end else float("nan")
        print(f"    n_genes={len(genes)}  t_end={t_end}  onset={onset} "
              f"({frac:.1%} of run)  slowest={worst}", flush=True)
        results.append((network, len(genes), t_end, onset, frac, worst))

    print("\n=== summary (tol=%.0f%% of range, tail=%.0f%%) ===" % (TOL * 100, TAIL_FRAC * 100))
    print(f"{'network':<20}{'n_genes':>8}{'t_end':>8}{'onset':>8}{'frac':>8}   slowest gene/species")
    for network, ng, t_end, onset, frac, worst in results:
        print(f"{network:<20}{ng:>8}{t_end:>8}{onset:>8}{frac:>8.1%}   {worst}")


if __name__ == "__main__":
    main()
