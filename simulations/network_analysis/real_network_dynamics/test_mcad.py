# [UNREVIEWED: rescued 2026-09-30 from scratchpad_807d072d_2026-09-20_real_network_sim; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import sys
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, ".")
from simulations.network_analysis.real_network_dynamics.settling_time_v2 import load_stats_with_n, settle
from simulations.network_analysis.plot_real_network_trajectories import NETWORKS, pick_file

path = pick_file(NETWORKS["mCAD"])
time, genes, stats = load_stats_with_n(path)
for g in genes:
    for species in ("mRNA", "protein"):
        m, s, n = stats[g][species]
        t_star, P_inf, band, peak_t, peak_val, kind = settle(m, s, n)
        tag = f"{kind} {peak_val:+.3g} @t={peak_t}" if kind != "none" else "monotonic"
        print(f"{g:<8}{species:<8} P_inf={P_inf:10.3g} band=+/-{band:8.3g} settle_t={t_star:5d}  {tag}")
