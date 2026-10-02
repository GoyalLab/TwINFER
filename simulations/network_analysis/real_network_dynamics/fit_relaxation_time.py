#!/usr/bin/env python
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_807d072d_2026-09-20_real_network_sim; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""
Independent estimate of each real network's relaxation time to steady state,
computed directly from the population-mean trajectory -- NOT via the
codebase's own is_steady_state()/relaxation-fit machinery.

Method: for each gene's mean-protein trajectory P(t) (population mean across
all cells at each time_step, from rep-0's simulation_before_division_df),
fit a single-exponential relaxation

    P(t) = P_inf + (P0 - P_inf) * exp(-t / tau)

by nonlinear least squares (scipy.optimize.curve_fit). tau is the e-folding
relaxation time. We report:
  - tau_gene for every gene (mRNA relaxes fast/noisy; protein is the
    meaningful slow variable, so protein tau is what's used for the
    network-level number)
  - t_63  = tau                       (63% of the way to steady state)
  - t_95  = 3*tau                     (95% of the way)
  - t_99  = 5*tau                     (99% of the way)
  - network relaxation time = max protein tau across genes (slowest gene
    sets the pace for the whole network)

Fit quality (R^2) is reported so a bad fit (e.g. oscillatory Circadian gene,
or a trajectory that's already flat at t=0) is visible rather than silently
trusted.
"""
import glob
import os
import sys
import warnings

import numpy as np
from scipy.optimize import curve_fit

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/network_analysis")
from simulations.network_analysis.plot_real_network_trajectories import load_stats, NETWORKS, pick_file

warnings.filterwarnings("ignore")


def exp_relax(t, P0, Pinf, tau):
    return Pinf + (P0 - Pinf) * np.exp(-t / tau)


def fit_tau(time, mean_traj):
    """Return (tau, r2, ok). ok=False if trajectory is ~flat (no relaxation
    to measure) or the fit fails / is unstable."""
    y = mean_traj
    rng = y.max() - y.min()
    if rng < 1e-6 * max(abs(y.mean()), 1.0):
        return 0.0, 1.0, False  # already flat -- "instantaneous" relaxation, not a real fit

    P0_guess, Pinf_guess = y[0], y[-max(1, len(y) // 20):].mean()
    tau_guess = max(time[-1] / 10, 1.0)
    try:
        popt, _ = curve_fit(
            exp_relax, time, y,
            p0=[P0_guess, Pinf_guess, tau_guess],
            maxfev=20000,
            bounds=([-np.inf, -np.inf, 1e-3], [np.inf, np.inf, 10 * time[-1] + 10]),
        )
        P0f, Pinff, tau = popt
        yhat = exp_relax(time, *popt)
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
        return abs(tau), r2, True
    except Exception:
        return np.nan, np.nan, False


def main():
    rows = []
    for network, pattern in NETWORKS.items():
        try:
            path = pick_file(pattern)
        except FileNotFoundError:
            print(f"[{network}] NOT FOUND")
            continue
        print(f"[{network}] loading {os.path.basename(path)}", flush=True)
        time, genes, stats = load_stats(path)
        t_end = int(time.max())
        tf = time.astype(float)

        gene_taus = {}
        for g in genes:
            m, _ = stats[g]["protein"]
            tau, r2, ok = fit_tau(tf, m)
            gene_taus[g] = (tau, r2, ok)
            print(f"    {g:<10} protein tau={tau:8.1f}  R2={r2:5.2f}  ok={ok}", flush=True)

        valid = {g: v for g, v in gene_taus.items() if v[2] and np.isfinite(v[0])}
        if valid:
            slow_gene = max(valid, key=lambda g: valid[g][0])
            tau_net = valid[slow_gene][0]
            r2_net = valid[slow_gene][1]
        else:
            slow_gene, tau_net, r2_net = None, float("nan"), float("nan")

        print(f"    -> network tau={tau_net:.1f} (t_end={t_end}), "
              f"slowest gene={slow_gene}, t_95%={3*tau_net:.0f}, t_99%={5*tau_net:.0f}\n", flush=True)
        rows.append((network, len(genes), t_end, slow_gene, tau_net, r2_net, 3 * tau_net, 5 * tau_net))

    print("\n=== summary: relaxation time (protein, single-exponential fit) ===")
    print(f"{'network':<20}{'n_genes':>8}{'t_end':>8}{'slow_gene':>12}{'tau':>10}{'R2':>7}{'t_95%':>9}{'t_99%':>9}")
    for network, ng, t_end, slow_gene, tau, r2, t95, t99 in rows:
        print(f"{network:<20}{ng:>8}{t_end:>8}{str(slow_gene):>12}{tau:>10.1f}{r2:>7.2f}{t95:>9.0f}{t99:>9.0f}")


if __name__ == "__main__":
    main()
