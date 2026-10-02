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
# [2026-10-01 commented out: ephemeral Claude scratchpad; the data was copied to clean_data/_rescued_scratchpad] SC = "/gpfs/home/gzu5140/.claude-tmp/claude-2000104/-gpfs-projects-b1255-hzhang-TwINFER-KA/b160c36f-c13b-490a-af32-6242da3a3aed/scratchpad"
SC = f"{TWINFER_PROJECT_ROOT}/clean_data/_rescued_scratchpad/b160c36f"

d = pd.read_csv(f"{SC}/gsd_harvest_test/GSD/twin_final_states.csv")
d = d[d.source.isin(["twin_A", "twin_B"])]
div = d[d.step == d.branch_tp]
divA = div[div.source == "twin_A"].set_index("pair_id")
lastA = d[d.source == "twin_A"].sort_values("step").groupby("pair_id").tail(1).set_index("pair_id")
lastB = d[d.source == "twin_B"].sort_values("step").groupby("pair_id").tail(1).set_index("pair_id")
common = divA.index.intersection(lastA.index).intersection(lastB.index)
divA, lastA, lastB = divA.loc[common], lastA.loc[common], lastB.loc[common]

N_BINS = 8
harvest_t = lastA["t"]
bin_edges = np.quantile(harvest_t, np.linspace(0, 1, N_BINS + 1))
bin_id = np.digitize(harvest_t, bin_edges[1:-1])

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
target_regs = {
    "CBX2": [("CBX2", "self"), ("UGR", "edge"), ("RSPO1", "non-edge"), ("NR0B1", "edge"), ("CTNNB1", "edge"), ("DMRT1", "non-edge")],
    "NR5A1": [("NR5A1", "self"), ("RSPO1", "non-edge"), ("CBX2", "edge"), ("UGR", "edge"), ("CTNNB1", "non-edge"), ("NR0B1", "edge")],
}
colors = {"self": "tab:blue", "edge": "tab:orange", "non-edge": "tab:red"}
styles = {"self": "o-", "edge": "s--", "non-edge": "^:"}

for ax, target in zip(axes, ["CBX2", "NR5A1"]):
    bin_t = [harvest_t[bin_id == b].mean() for b in range(N_BINS)]
    for reg, kind in target_regs[target]:
        corr_by_bin = []
        for b in range(N_BINS):
            mask = bin_id == b
            r = spearmanr(divA[reg][mask], lastB[target][mask])[0]
            corr_by_bin.append(r)
        label = f"{reg} ({kind})" + (f" -> {target}" if reg != target else "")
        ax.plot(bin_t, corr_by_bin, styles[kind], color=colors[kind], label=label, alpha=0.85)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("harvest time t (binned across staggered pairs)")
    ax.set_ylabel("Spearman corr(reg @ division, target @ harvest), sister cells")
    ax.set_title(f"GSD: target = {target}")
    ax.legend(fontsize=8, loc="best")
fig.suptitle("GSD staggered-harvest: correlation vs harvest time, reconstructed by binning pairs\n(each pair = one (division, harvest) pair; binning across pairs approximates a trajectory)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(f"{SC}/gsd_harvest_trajectory.png", dpi=140, bbox_inches="tight")
print("saved")
