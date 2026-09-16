#!/usr/bin/env python
"""Combined network-sweep benchmark summary -- one figure, three metric panels.

Source: cached score tables produced by benchmark_network_sweep.ipynb
(TwINFER_KA, alphabetically first notebook in code/TwINFER/synthetic_network_analysis):
  TwINFER : analysis_data/synthetic_network_benchmark_13082026/twinfer_analysis_output.csv
  BEELINE : analysis_data/synthetic_network_benchmark_06082026/beeline_gmm_analysis_output.csv

Conventions (matching the notebook):
  - scheme suffix ''            -> directed + unsigned
  - scheme suffix '_signed'     -> directed + signed
  - scheme suffix '_undirected' -> undirected + unsigned
  - BEELINE rows: scheme == 'twin_paired' runs only (the notebook's SCHEMES choice)
  - TwINFER top-k pool = fan_out_removed ; natural stage = after_fan_out
    (the pipeline's actual hard decisions)
  - per (method, topology): mean across sim reps; summary = mean +/- SEM across topologies
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

TW_CSV = "/home/gzu5140/TwINFER_KA/analysis_data/synthetic_network_benchmark_13082026/twinfer_analysis_output.csv"
BE_CSV = "/home/gzu5140/TwINFER_KA/analysis_data/synthetic_network_benchmark_06082026/beeline_gmm_analysis_output.csv"
OUT_DIR = Path("/home/gzu5140/TwINFER_KA/analysis_data/synthetic_network_benchmark_13082026/summary_plot_24082026")

SCHEMES = [("_signed", "Directed + signed"),
           ("", "Directed + unsigned"),
           ("_undirected", "Undirected + unsigned")]
METRICS = [("auprc", "AUPRC"),
           ("f1_topk", "F1 (top-k)"),
           ("f1_natural", "F1 (natural threshold)")]
TW_COLS = {"auprc": "auprc",
           "f1_topk": "f1_topk_fan_out_removed",
           "f1_natural": "f1_natural_after_fan_out"}
BE_COLS = {"auprc": "auprc", "f1_topk": "f1_topk", "f1_natural": "f1_natural"}

SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; BASE = "#c3c2b7"
SCHEME_COLORS = {"Directed + signed": "#2a78d6",
                 "Directed + unsigned": "#eb6834",
                 "Undirected + unsigned": "#1baf7a"}


def build_long():
    t = pd.read_csv(TW_CSV)
    b = pd.read_csv(BE_CSV)
    b = b[b["scheme"] == "twin_paired"].copy()
    rec = []
    for mkey, _ in METRICS:
        for suf, slabel in SCHEMES:
            col = TW_COLS[mkey] + suf
            for ds, v in t.groupby("dataset_id")[col].mean().items():
                rec.append(("TwINFER", mkey, slabel, ds, v))
    for algo, g in b.groupby("algorithm"):
        for mkey, _ in METRICS:
            for suf, slabel in SCHEMES:
                col = BE_COLS[mkey] + suf
                for ds, v in g.groupby("dataset_id")[col].mean().items():
                    rec.append((algo, mkey, slabel, ds, v))
    return pd.DataFrame(rec, columns=["method", "metric", "scheme", "dataset_id", "value"])


def summarize(long_df):
    def sem(x):
        x = x.dropna()
        return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    s = (long_df.groupby(["method", "metric", "scheme"])["value"]
         .agg(mean="mean", sem=sem, n="count").reset_index())
    return s


def plot(summ, method_order, out_png):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    x = np.arange(len(method_order))
    w = 0.26
    for ax, (mkey, mtitle) in zip(axes, METRICS):
        ax.set_facecolor(SURFACE)
        for k, (suf, slabel) in enumerate(SCHEMES):
            sub = (summ[(summ.metric == mkey) & (summ.scheme == slabel)]
                   .set_index("method").reindex(method_order))
            xs = x + (k - 1) * w
            ax.bar(xs, sub["mean"].values, width=w - 0.03,
                   color=SCHEME_COLORS[slabel], edgecolor=SURFACE, linewidth=0.6,
                   zorder=3, label=slabel)
            ax.errorbar(xs, sub["mean"].values, yerr=sub["sem"].values, fmt="none",
                        ecolor=INK2, elinewidth=1.0, capsize=2.2, capthick=1.0, zorder=4)
            for xi, (m, e) in zip(xs, zip(sub["mean"].values, sub["sem"].values)):
                if np.isfinite(m):
                    ax.text(xi, m + (e if np.isfinite(e) else 0) + 0.012, f"{m:.2f}",
                            ha="center", va="bottom", fontsize=6.4, rotation=90,
                            color=INK2, zorder=5)
        ax.set_title(mtitle, fontsize=13, color=INK, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(method_order, rotation=38, ha="right", fontsize=9.5, color=INK)
        ax.set_ylim(0, 1.02)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(BASE)
        ax.tick_params(colors=MUTED, length=0)
        for lab in ax.get_xticklabels():
            lab.set_color(INK)
    axes[0].set_ylabel("Score (mean +/- SEM across topologies)", fontsize=10.5, color=INK2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.005), fontsize=11)
    fig.suptitle("Network sweep benchmark -- TwINFER vs BEELINE methods "
                 "(15 topologies, mean of sim reps per topology)",
                 fontsize=14.5, color=INK, y=1.075)
    fig.text(0.5, -0.045,
             "TwINFER: top-k pool = fan_out_removed, natural = after_fan_out. "
             "BEELINE methods: twin_paired runs, natural = 2-component GMM threshold. "
             "k = number of true edges (tie-aware).",
             ha="center", fontsize=8.6, color=MUTED)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(str(out_png).replace(".png", ".pdf"), bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    long_df = build_long()
    long_df.to_csv(OUT_DIR / "per_topology_means.csv", index=False)
    summ = summarize(long_df)
    summ.to_csv(OUT_DIR / "benchmark_summary_mean_sem.csv", index=False)
    order = (summ[(summ.metric == "auprc") & (summ.scheme == "Directed + unsigned")]
             .sort_values("mean", ascending=False)["method"].tolist())
    plot(summ, order, OUT_DIR / "benchmark_sweep_metrics_by_scheme.png")
    print("ORDER:", ",".join(order))
    print("BEGIN_SUMMARY")
    for _, r in summ.iterrows():
        print(f"{r['method']};{r['metric']};{r['scheme']};{r['mean']:.6f};{r['sem']:.6f};{int(r['n'])}")
    print("END_SUMMARY")
    print("WROTE:", OUT_DIR)


if __name__ == "__main__":
    main()
