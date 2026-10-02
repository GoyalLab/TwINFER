# [UNREVIEWED: rescued 2026-09-30 from scratchpad_807d072d_2026-09-20_real_network_sim; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import sys
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, ".")
from benchmarks.network_benchmarks.analysis_plots.plot_seeded_trajectories import n_seed_states, load_by_state, pick, SEEDED_PATTERN

k, bounds = n_seed_states("mCAD")
print("mCAD k=", k, "bounds=", bounds)
path = pick(SEEDED_PATTERN["mCAD"])
print("path:", path)
time, genes, stats, sizes = load_by_state(path, k, bounds)
print("genes", genes, "t_end", time.max(), "sizes", sizes)
for g in genes[:2]:
    m, s = stats[g]["protein"]
    print(g, "protein means at t=0, mid, end per state:")
    for st in range(k):
        print("  state", st, m[st,0], m[st, len(time)//2], m[st,-1])
