"""Network-sweep AUPRC heatmap (same layout as plot_heatmap_mixed_sweep.py), same layout as the old heatmap_mixed_sweep.pdf: four blocks
('6 genes, 6 edges', '6 genes, 12 edges', '10 genes, 10 edges', '10 genes, 20 edges'), each with
'Cycle a' / 'Cycle b' sub-blocks, and replicate ticks 1-3 under every a-level.
Input:  clean_data/benchmarks/network_benchmarks/score_results/heatmap_data_network_sweep.csv
Output: final_plots/benchmark/synthetic_network_simulations/heatmap_network_sweep_<variant>.pdf

Date:   2026-10-04
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib import font_manager as fm
ARIAL_DIR = "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/package/twinfer/plotting/assets/fonts"
for f in ["Arial.ttf", "Arial Bold.ttf", "Arial Italic.ttf", "Arial Bold Italic.ttf"]:
    fm.fontManager.addfont(os.path.join(ARIAL_DIR, f))
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.sans-serif'] = ["Arial"]
plt.rcParams['font.family'] = "sans-serif"

CSV = "/home/gzu5140/TwINFER_KA/clean_data/benchmarks/network_benchmarks/score_results/heatmap_data_network_sweep.csv"
OUT = "/gpfs/projects/b1255/hzhang/TwINFER_KA/final_plots/benchmark/synthetic_network_simulations"
os.makedirs(OUT, exist_ok=True)

# (block heading, [(cycle label, [a-level group names in order])])
BLOCKS = [
    ("Network density", [("5 edges", ["5 edges"]), ("9 edges", ["9 edges"]), ("13 edges", ["13 edges"])]),
    ("Fraction of negative edges", [("All negative", ["all negative"]), ("Balanced", ["balanced"])]),
]
VARIANT_TITLE = {"signed_directed": "Signed directed", "directed_unsigned": "Unsigned directed",
                 "undirected": "Unsigned undirected"}
METHOD_LABEL = {"TODO4v2": "TwINFER", "GRNBOOST2": "GRNBoost2"}


def main():
    d = pd.read_csv(CSV)
    for variant, title in VARIANT_TITLE.items():
        v = d[d.variant == variant]
        cols = []  # (block, cycle, a-level label, group, rep)
        for block, cycles in BLOCKS:
            for cyc, groups in cycles:
                for g in groups:
                    for rep in (1, 2, 3):
                        cols.append((block, cyc, "", g, rep))
        grid = pd.DataFrame(index=sorted(v.method.unique()), columns=range(len(cols)), dtype=float)
        for j, (_, _, _, g, rep) in enumerate(cols):
            sub = v[(v.group == g) & (v.rep == rep - 1)]
            for _, r in sub.iterrows():
                grid.loc[r.method, j] = r.auprc
        order = grid.mean(axis=1).sort_values(ascending=False).index.tolist()
        grid = grid.loc[order]
        avg = grid.mean(axis=1)

        fig_w = 0.2 * len(cols) + 3.2
        fig, ax = plt.subplots(figsize=(fig_w, 0.42 * len(order) + 3.0))
        im = ax.imshow(grid.to_numpy(float), cmap="GnBu", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([str(c[4]) for c in cols], fontsize=7)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([METHOD_LABEL.get(m, m) for m in order], fontsize=9)
        for i, m in enumerate(order):
            if m == "TODO4v2":
                ax.get_yticklabels()[i].set_fontweight("bold")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                val = grid.iloc[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5.5,
                            color="white" if val > 0.6 else "black")
        # block and cycle headers above the grid
        j0 = 0
        for block, cycles in BLOCKS:
            n_block = sum(len(groups) * 3 for _, groups in cycles)
            ax.text(j0 + n_block / 2 - 0.5, -2.0, block, ha="center", va="bottom", fontsize=9, fontweight="bold")
            jc = j0
            for cyc, groups in cycles:
                n_cyc = len(groups) * 3
                ax.text(jc + n_cyc / 2 - 0.5, -1.0, cyc, ha="center", va="bottom", fontsize=8)
                # a-level labels under the replicate ticks
                ax.plot([jc - 0.4, jc + n_cyc - 0.6], [-0.55, -0.55], color="#555555", lw=0.6, clip_on=False)
                jc += n_cyc
            ax.plot([j0 - 0.4, j0 + n_block - 0.6], [-1.6, -1.6], color="#222222", lw=0.8, clip_on=False)
            j0 += n_block
        for x in np.cumsum([sum(len(g) * 3 for _, g in cyc) for _, cyc in BLOCKS])[:-1]:
            ax.axvline(x - 0.5, color="white", lw=3)
        ax.axhline(0.5, color="white", lw=0)
        ax.set_title(f"AUPRC – {title}", fontsize=11, fontweight="bold", pad=34)
        cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
        cb.set_label("Score", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        ax.set_xlim(-0.6, len(cols) - 0.4)
        ax.spines[:].set_visible(False)
        fig.savefig(f"{OUT}/heatmap_network_sweep_{variant}.pdf", format="pdf", bbox_inches="tight")
        plt.close(fig)
        print("wrote", variant, "methods:", order, "avg:", avg.round(3).to_dict())


if __name__ == "__main__":
    main()
