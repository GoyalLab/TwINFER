#!/usr/bin/env python
"""One-figure benchmark grid rendered with the notebook's OWN plotting function.

Execs the melt + heatmap cells straight out of benchmark_network_sweep.ipynb
(so `plot_heatmap_by_method_dataset` etc. are the notebook's existing code,
not a reimplementation), then composes a 3x3 grid in a single figure:
rows = AUPRC / top-k F1 / natural F1, columns = the three scoring schemes
(signed_directed / directed_unsigned / undirected). TwINFER appears with all
of its pool/stage variants as separate heatmap rows, exactly as in the notebook.
"""
import json
import re  # noqa: F401  (notebook cells use re)
import numpy as np  # noqa: F401
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns  # noqa: F401
from matplotlib.colors import Normalize  # noqa: F401
from pathlib import Path

NB_PATH = "/home/gzu5140/TwINFER_KA/code/TwINFER/synthetic_network_analysis/benchmark_network_sweep.ipynb"
TW_CSV = "/home/gzu5140/TwINFER_KA/analysis_data/synthetic_network_benchmark_13082026/twinfer_analysis_output.csv"
BE_CSV = "/home/gzu5140/TwINFER_KA/analysis_data/synthetic_network_benchmark_06082026/beeline_gmm_analysis_output.csv"
OUT_DIR = Path("/home/gzu5140/TwINFER_KA/analysis_data/synthetic_network_benchmark_13082026/summary_plot_24082026")

nb = json.load(open(NB_PATH))


def cell_src(marker):
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            s = "".join(c["source"])
            if marker in s:
                return s
    raise SystemExit(f"FATAL: no notebook cell contains marker {marker!r}")


twinfer_results_df = pd.read_csv(TW_CSV)
beeline_results_df = pd.read_csv(BE_CSV)

G = globals()
exec(cell_src("TOPK_POOL_NAMES = ["), G)              # pool/stage constants
exec(cell_src("def melt_beeline_wide"), G)             # builds scores_long
exec(cell_src("def plot_heatmap_by_method_dataset"), G)  # notebook's plot fn + style

ROWS = [("auprc", "auprc"), ("topk", "f1"), ("natural", "f1")]
COLS = ["signed_directed", "directed_unsigned", "undirected"]

fig, axes = plt.subplots(3, 3, figsize=(38, 18))
fig.patch.set_facecolor(G.get("HEATMAP_SURFACE", "#fcfcfb"))
for i, (fam, sn) in enumerate(ROWS):
    for j, variant in enumerate(COLS):
        subset = scores_long[
            (scores_long["metric_family"] == fam)
            & (scores_long["score_name"] == sn)
            & (scores_long["variant"] == variant)
            & ((scores_long["scheme"] == "twin_paired") | (scores_long["method"] == "TwINFER"))
        ]
        ax = axes[i][j]
        if subset.empty:
            ax.axis("off")
            continue
        im = plot_heatmap_by_method_dataset(
            subset, ax=ax, title=f"{fam} -- {sn} -- {variant.replace('_', ' ')}"
        )
        if im is not None:
            fig.colorbar(im, ax=ax, shrink=0.75, pad=0.01)
fig.suptitle(
    "Network sweep benchmark -- rendered with the notebook's plot_heatmap_by_method_dataset "
    "(rows: AUPRC / top-k F1 / natural F1; columns: scoring schemes; BEELINE = twin_paired runs)",
    fontsize=18, y=0.995,
)
fig.tight_layout(rect=(0, 0, 1, 0.985))
OUT_DIR.mkdir(parents=True, exist_ok=True)
png = OUT_DIR / "benchmark_sweep_heatmap_grid.png"
fig.savefig(png, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
fig.savefig(str(png).replace(".png", ".pdf"), dpi=400, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print("WROTE:", png)
