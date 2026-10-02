from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import pandas as pd, numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

genes = [l.strip() for l in open(f'{TWINFER_PROJECT_ROOT}/simulation_data/twinfer_format/GSD/gene_order.txt') if l.strip()]
M = np.loadtxt(f'{TWINFER_PROJECT_ROOT}/simulation_data/twinfer_format/GSD/interaction_matrix.txt', delimiter=",")
gi = {g: i for i, g in enumerate(genes)}

d = pd.read_csv(f'{TWINFER_PROJECT_ROOT}/clean_data/simulations/network_analysis/gillespie_checks/gsd_harvest_test/GSD/twin_final_states.csv')
d = d[d.source.isin(["twin_A", "twin_B"])]
div = d[d.step == d.branch_tp]
divA = div[div.source == "twin_A"].set_index("pair_id")
lastA = d[d.source == "twin_A"].sort_values("step").groupby("pair_id").tail(1).set_index("pair_id")
lastB = d[d.source == "twin_B"].sort_values("step").groupby("pair_id").tail(1).set_index("pair_id")
common = divA.index.intersection(lastA.index).intersection(lastB.index)
divA, lastA, lastB = divA.loc[common], lastA.loc[common], lastB.loc[common]

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
for ax, target in zip(axes, ["CBX2", "NR5A1"]):
    rows = []
    for reg in genes:
        is_edge = M[gi[reg], gi[target]] != 0
        r_self = spearmanr(divA[reg], lastA[target])[0]
        r_sis = spearmanr(divA[reg], lastB[target])[0]
        rows.append((reg, is_edge, r_self, r_sis))
    rows.sort(key=lambda x: -abs(x[3]))
    labels = [r[0] + (" (self)" if r[0] == target else "") for r in rows]
    sister_vals = [r[3] for r in rows]
    colors = ["tab:orange" if r[1] else "tab:blue" for r in rows]
    y = np.arange(len(rows))
    ax.barh(y, sister_vals, color=colors)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="gray", lw=0.7)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman corr (reg @ division vs target @ harvest, sister cells)")
    ax.set_title(f"GSD, staggered harvest: target = {target}")

from matplotlib.patches import Patch
fig.legend(handles=[Patch(color="tab:orange", label="true edge"), Patch(color="tab:blue", label="non-edge")],
           loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
fig.suptitle("Staggered-harvest GSD: true edges don't rank above non-edges (indirect/pathway confound)", y=1.08, fontsize=11)
fig.tight_layout()
# [2026-10-01 commented out: ephemeral Claude scratchpad; the data was copied to clean_data/_rescued_scratchpad] fig.savefig("/gpfs/home/gzu5140/.claude-tmp/claude-2000104/-gpfs-projects-b1255-hzhang-TwINFER-KA/b160c36f-c13b-490a-af32-6242da3a3aed/scratchpad/gsd_harvest_edges.png", dpi=140, bbox_inches="tight")
fig.savefig(f"{TWINFER_PROJECT_ROOT}/clean_data/_rescued_scratchpad/b160c36f/gsd_harvest_edges.png", dpi=140, bbox_inches="tight")
print("saved")
