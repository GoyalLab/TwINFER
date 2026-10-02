from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import pandas as pd, numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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
print(f"n_pairs={len(common)}")

N_BINS = 8
harvest_t = lastA["t"]
bin_edges = np.quantile(harvest_t, np.linspace(0, 1, N_BINS + 1))
bin_id = np.digitize(harvest_t, bin_edges[1:-1])
bin_t = [harvest_t[bin_id == b].mean() for b in range(N_BINS)]
bin_n = [int((bin_id == b).sum()) for b in range(N_BINS)]
print("bin sizes:", bin_n)

with PdfPages(f"{SC}/gsd_harvest_all_edges.pdf") as pdf:
    for target in ["CBX2", "NR5A1"]:
        regs = [g for g in genes if g != target]
        ncols = 3; nrows = int(np.ceil(len(regs) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(13, 2.6 * nrows), squeeze=False)
        axes = axes.ravel()
        # self trajectory (same-cell and sister), shared across all panels for this target
        self_sc = [spearmanr(divA[target][bin_id == b], lastA[target][bin_id == b])[0] for b in range(N_BINS)]
        self_ss = [spearmanr(divA[target][bin_id == b], lastB[target][bin_id == b])[0] for b in range(N_BINS)]
        for ax, reg in zip(axes, regs):
            is_edge = M[gi[reg], gi[target]] != 0
            reg_sc = [spearmanr(divA[reg][bin_id == b], lastA[target][bin_id == b])[0] for b in range(N_BINS)]
            reg_ss = [spearmanr(divA[reg][bin_id == b], lastB[target][bin_id == b])[0] for b in range(N_BINS)]
            ax.plot(bin_t, self_sc, "o-", color="tab:blue", ms=3, lw=1, label="self, same-cell")
            ax.plot(bin_t, self_ss, "o-", color="tab:cyan", ms=3, lw=1, alpha=0.6, label="self, sister")
            col = "tab:orange" if is_edge else "tab:red"
            ax.plot(bin_t, reg_sc, "s--", color=col, ms=3, lw=1, label="reg, same-cell")
            ax.plot(bin_t, reg_ss, "^:", color=col, ms=3, lw=1, alpha=0.6, label="reg, sister")
            ax.axhline(0, color="gray", lw=0.4)
            ax.set_ylim(-1.05, 1.05)
            tag = "EDGE" if is_edge else "non-edge"
            ax.set_title(f"{reg} -> {target}  [{tag}]", fontsize=8.5)
            ax.tick_params(labelsize=7)
        for ax in axes[len(regs):]:
            ax.axis("off")
        axes[0].legend(fontsize=6, loc="upper left")
        fig.suptitle(f"GSD staggered-harvest, ALL candidate regulators, target={target} "
                     f"(n_pairs={len(common)}, {N_BINS} bins, sizes {bin_n})", fontsize=10)
        fig.supxlabel("harvest time t (binned)"); fig.supylabel("Spearman corr")
        fig.tight_layout(rect=[0.01, 0.01, 1, 0.94])
        pdf.savefig(fig); plt.close(fig)
print("saved PDF")
