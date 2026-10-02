# [UNREVIEWED: rescued 2026-09-30 from scratchpad_807d072d_2026-09-20_real_network_sim; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import sys
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/network_analysis")
from simulations.network_analysis.plot_real_network_trajectories import load_stats, NETWORKS, pick_file
import numpy as np

network = "mCAD"
path = pick_file(NETWORKS[network])
print("loading", path)
time, genes, stats = load_stats(path)
print("genes:", genes, "t_end:", time.max())
for g in genes:
    for species in ("mRNA", "protein"):
        m, s = stats[g][species]
        idx = np.linspace(0, len(m)-1, 25).astype(int)
        vals = " ".join(f"{m[i]:.1f}" for i in idx)
        print(f"{g} {species}: {vals}")
