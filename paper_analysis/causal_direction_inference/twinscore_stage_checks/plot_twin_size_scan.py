"""One PDF per statistic (z_rho, z_rho_s = z_C, z_rho_delta = z_U, two-term |z_rho| - |z_C|) vs the number of twin pairs (250 ... 3000), box plots over 100 replicates, one colour per scenario / condition.
figure 2: the 6 two-gene scenarios; figure 3: the 4 regulation conditions + the figure 2 single-state A,B and A->B references.  No titles, no descriptions on the plot (see README_twin_size_scan.md in the output folder).
Follows the Illustrator-editable recipe: Arial text, Latin Modern (cm) math, fonttype 42, guide font sizes, no clips/masks, transparent PDF.   usage: python plot_twin_size_scan.py"""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Patch
for f in ["Arial.ttf", "Arial Bold.ttf", "Arial Italic.ttf"]: fm.fontManager.addfont("/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/package/twinfer/plotting/assets/fonts/" + f)
plt.rcParams["pdf.fonttype"] = 42; plt.rcParams["ps.fonttype"] = 42; plt.rcParams["font.sans-serif"] = ["Arial"]; plt.rcParams["font.family"] = "sans-serif"; plt.rcParams["svg.fonttype"] = "none"; plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["axes.labelsize"] = 18; plt.rcParams["axes.titlesize"] = 20; plt.rcParams["xtick.labelsize"] = 12; plt.rcParams["ytick.labelsize"] = 12; plt.rcParams["legend.fontsize"] = 12; plt.rcParams["legend.title_fontsize"] = 12
A = "/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/"; O = A + "causal_direction_inference/zrho_pdf_network_auprc/"; OUT = A + "final_plots/plot_z_Scores_2_gene_3_gene/new_metrics_designs/"; PREVIEW = os.environ.get("PREVIEW_DIR")
NS = [250, 500, 1000, 1500, 2000, 2500, 3000]; COLS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]      # validated categorical slots 1-6
STATS = [("z_rho", r"$z_\rho$", "z_rho", True), ("z_C", r"$z_{\rho_s}$", "z_rho_s", True), ("z_U", r"$z_{\hat\rho_\Delta}$", "z_rho_delta", True), ("two", r"$|z_\rho| - |z_{\rho_s}|$", "two_term", False), ("two_exact", r"$|z_\rho| - |C/\sigma_\rho|$", "two_term_exact", False)]      # last: exact version, C divided by sigma_rho (z_rho = C/sigma_rho + U/sigma_rho)
def make(D, key, order, labels, prefix):
    D = D.copy(); D["two"] = D.z_rho.abs() - D.z_C.abs(); D["two_exact"] = D.z_rho.abs() - D.A_s.abs()
    for col, ylab, fname, thr in STATS:
        fig, ax = plt.subplots(figsize=(13, 6)); ax.set_facecolor("none"); fig.patch.set_alpha(0); w = 0.125; k = len(order)
        for i, N in enumerate(NS):
            for j, sc in enumerate(order):
                v = D[(D[key] == sc) & (D.N == N)][col].dropna().values; c = COLS[j]
                b = ax.boxplot(v, positions=[i + (j - (k - 1) / 2) * w], widths=w * 0.85, patch_artist=True, showfliers=True, medianprops=dict(color="#0b0b0b", lw=1.3), whiskerprops=dict(color=c, lw=1.0), capprops=dict(color=c, lw=1.0),
                               boxprops=dict(color=c, lw=1.0), flierprops=dict(marker="o", ms=2.2, mfc="none", mec=c, mew=0.7))
                b["boxes"][0].set_facecolor(c)
        if thr:
            for s_ in (1.96, -1.96): ax.axhline(s_, color="#52514e", lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.axhline(0, color="#898781", lw=0.7, ls=":", zorder=1); ax.set_xticks(range(len(NS))); ax.set_xticklabels([str(n) for n in NS]); ax.set_xlim(-0.55, len(NS) - 0.45)
        ax.set_xlabel("number of twin pairs"); ax.set_ylabel(ylab); ax.grid(axis="y", color="#e1e0d9", lw=0.6); ax.set_axisbelow(True)
        for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"): ax.spines[s_].set_color("#c3c2b7")
        ax.legend(handles=[Patch(fc=COLS[j], ec=COLS[j], label=labels[j]) for j in range(k)], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
        for artist in fig.findobj():                                           # remove every clip/mask (Illustrator-editable)
            if hasattr(artist, "set_clip_on"): artist.set_clip_on(False)
        fig.savefig(f"{OUT}{prefix}_twinscan_{fname}.pdf", format="pdf", bbox_inches="tight", facecolor="none", edgecolor="none", transparent=True)
        if PREVIEW: fig.savefig(f"{PREVIEW}/{prefix}_twinscan_{fname}.png", dpi=110, bbox_inches="tight", facecolor="white")
        plt.close(fig)
F2 = ["A_B (single state)", "A_to_B (single state)", "A_B (two states)", "A_to_B (two states)", "A_B (drift)", "A_to_B (drift)"]
L2 = ["A, B", "A→B", "A, B (2 states)", "A→B (2 states)", "A, B (drift)", "A→B (drift)"]
F3 = ["A_B (single state)", "A_to_B (single state)", "A_rep_B", "A_rep_B_B_rep_A", "A_rep_B_B_to_A", "A_to_B_B_to_A"]
L3 = ["A, B (fig 2)", "A→B (fig 2)", "A rep. B", "A rep. B, B rep. A", "A rep. B, B → A", "A → B, B → A"]

RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#256abf", "#184f95", "#0d366b"]      # sequential single-hue blue ramp (reference palette steps 250-700), light = few twins, dark = many
def make_by_network(D, key, order, tick_labels, prefix):
    """Group by scenario/network on x; the 7 twin counts are the boxes inside each group (sequential ramp)."""
    D = D.copy(); D["two"] = D.z_rho.abs() - D.z_C.abs(); D["two_exact"] = D.z_rho.abs() - D.A_s.abs()
    for col, ylab, fname, thr in STATS:
        fig, ax = plt.subplots(figsize=(13, 6)); ax.set_facecolor("none"); fig.patch.set_alpha(0); w = 0.112; k = len(NS)
        for i, sc in enumerate(order):
            for j, N in enumerate(NS):
                v = D[(D[key] == sc) & (D.N == N)][col].dropna().values; c = RAMP[j]
                b = ax.boxplot(v, positions=[i + (j - (k - 1) / 2) * w], widths=w * 0.85, patch_artist=True, showfliers=True, medianprops=dict(color="#0b0b0b", lw=1.2), whiskerprops=dict(color=c, lw=1.0), capprops=dict(color=c, lw=1.0),
                               boxprops=dict(color=c, lw=1.0), flierprops=dict(marker="o", ms=2.2, mfc="none", mec=c, mew=0.7))
                b["boxes"][0].set_facecolor(c)
        if thr:
            for s_ in (1.96, -1.96): ax.axhline(s_, color="#52514e", lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.axhline(0, color="#898781", lw=0.7, ls=":", zorder=1); ax.set_xticks(range(len(order))); ax.set_xticklabels(tick_labels); ax.set_xlim(-0.55, len(order) - 0.45)
        ax.set_ylabel(ylab); ax.grid(axis="y", color="#e1e0d9", lw=0.6); ax.set_axisbelow(True); ax.tick_params(axis="x", length=0)
        for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"): ax.spines[s_].set_color("#c3c2b7")
        ax.legend(handles=[Patch(fc=RAMP[j], ec=RAMP[j], label=str(NS[j])) for j in range(len(NS))], title="number of twin pairs", loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=7, frameon=False, columnspacing=1.2, handlelength=1.4)
        for artist in fig.findobj():
            if hasattr(artist, "set_clip_on"): artist.set_clip_on(False)
        fig.savefig(f"{OUT}{prefix}_by_network_{fname}.pdf", format="pdf", bbox_inches="tight", facecolor="none", edgecolor="none", transparent=True)
        if PREVIEW: fig.savefig(f"{PREVIEW}/{prefix}_by_network_{fname}.png", dpi=110, bbox_inches="tight", facecolor="white")
        plt.close(fig)
T2 = ["A, B", "A→B", "A, B\n2 states", "A→B\n2 states", "A, B\ndrift", "A→B\ndrift"]
T3 = ["A, B\n(fig 2)", "A→B\n(fig 2)", "A rep. B", "A rep. B,\nB rep. A", "A rep. B,\nB → A", "A → B,\nB → A"]

def two_vs_three(D, key, order, tick_labels, prefix):
    """(1) two panels side by side (two-term | three-term, shared y axis), grouped by network, 7 twin counts per group;  (2) 3,000 twin pairs only: two-term and three-term boxes next to each other per network.  Original z-scores."""
    D = D.copy(); D["two"] = D.z_rho.abs() - D.z_C.abs(); D["three"] = D["two"] + D.z_U.abs()
    fig, axs = plt.subplots(1, 3, figsize=(27, 6), sharey=True); fig.patch.set_alpha(0); w = 0.112; k = len(NS)      # [2026-10-17 third panel: z_rho_delta alone]
    for ax, col, ylab in ((axs[0], "two", r"$|z_\rho| - |z_{\rho_s}|$"), (axs[1], "three", r"$|z_\rho| - |z_{\rho_s}| + |z_{\hat\rho_\Delta}|$"), (axs[2], "z_U", r"$z_{\hat\rho_\Delta}$")):
        ax.set_facecolor("none")
        for i, sc in enumerate(order):
            for j, N in enumerate(NS):
                v = D[(D[key] == sc) & (D.N == N)][col].dropna().values; c = RAMP[j]
                b = ax.boxplot(v, positions=[i + (j - (k - 1) / 2) * w], widths=w * 0.85, patch_artist=True, showfliers=True, medianprops=dict(color="#0b0b0b", lw=1.2), whiskerprops=dict(color=c, lw=1.0), capprops=dict(color=c, lw=1.0),
                               boxprops=dict(color=c, lw=1.0), flierprops=dict(marker="o", ms=2.2, mfc="none", mec=c, mew=0.7)); b["boxes"][0].set_facecolor(c)
        ax.axhline(0, color="#898781", lw=0.7, ls=":", zorder=1); ax.set_xticks(range(len(order))); ax.set_xticklabels(tick_labels); ax.set_xlim(-0.55, len(order) - 0.45); ax.set_ylabel(ylab)
        if col == "z_U":
            for s_ in (1.96, -1.96): ax.axhline(s_, color="#52514e", lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.grid(axis="y", color="#e1e0d9", lw=0.6); ax.set_axisbelow(True); ax.tick_params(axis="x", length=0); ax.yaxis.set_tick_params(labelleft=True)
        for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"): ax.spines[s_].set_color("#c3c2b7")
    fig.legend(handles=[Patch(fc=RAMP[j], ec=RAMP[j], label=str(NS[j])) for j in range(k)], title="number of twin pairs", loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=7, frameon=False, columnspacing=1.2, handlelength=1.4)
    fig.tight_layout(w_pad=3)
    for artist in fig.findobj():
        if hasattr(artist, "set_clip_on"): artist.set_clip_on(False)
    fig.savefig(f"{OUT}{prefix}_by_network_score_comparison.pdf", format="pdf", bbox_inches="tight", facecolor="none", edgecolor="none", transparent=True)
    if PREVIEW: fig.savefig(f"{PREVIEW}/{prefix}_by_network_score_comparison.png", dpi=80, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(13, 6)); ax.set_facecolor("none"); fig.patch.set_alpha(0); w = 0.22; cols = ["#2a78d6", "#eb6834", "#1baf7a"]; labs = [r"$|z_\rho| - |z_{\rho_s}|$", r"$|z_\rho| - |z_{\rho_s}| + |z_{\hat\rho_\Delta}|$", r"$z_{\hat\rho_\Delta}$"]
    for i, sc in enumerate(order):
        for j, col in enumerate(("two", "three", "z_U")):
            v = D[(D[key] == sc) & (D.N == 3000)][col].dropna().values; c = cols[j]
            b = ax.boxplot(v, positions=[i + (j - 1) * w * 1.15], widths=w, patch_artist=True, showfliers=True, medianprops=dict(color="#0b0b0b", lw=1.4), whiskerprops=dict(color=c, lw=1.1), capprops=dict(color=c, lw=1.1),
                           boxprops=dict(color=c, lw=1.1), flierprops=dict(marker="o", ms=2.6, mfc="none", mec=c, mew=0.8)); b["boxes"][0].set_facecolor(c)
    ax.axhline(0, color="#898781", lw=0.7, ls=":", zorder=1); ax.set_xticks(range(len(order))); ax.set_xticklabels(tick_labels); ax.set_xlim(-0.55, len(order) - 0.45); ax.set_ylabel("score"); ax.grid(axis="y", color="#e1e0d9", lw=0.6); ax.set_axisbelow(True); ax.tick_params(axis="x", length=0)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"): ax.spines[s_].set_color("#c3c2b7")
    for s_ in (1.96, -1.96): ax.axhline(s_, color="#52514e", lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.legend(handles=[Patch(fc=cols[j], ec=cols[j], label=labs[j]) for j in range(3)], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    for artist in fig.findobj():
        if hasattr(artist, "set_clip_on"): artist.set_clip_on(False)
    fig.savefig(f"{OUT}{prefix}_by_network_score_comparison_3000_pairs.pdf", format="pdf", bbox_inches="tight", facecolor="none", edgecolor="none", transparent=True)
    if PREVIEW: fig.savefig(f"{PREVIEW}/{prefix}_by_network_score_comparison_3000_pairs.png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def single_n_comparison(D, key, order, tick_labels, prefix, N):
    """Two-term, three-term and z_rho_delta boxes next to each other per network for ONE twin count N."""
    D = D.copy(); D["two"] = D.z_rho.abs() - D.z_C.abs(); D["three"] = D["two"] + D.z_U.abs()
    fig, ax = plt.subplots(figsize=(13, 6)); ax.set_facecolor("none"); fig.patch.set_alpha(0); w = 0.22; cols = ["#2a78d6", "#eb6834", "#1baf7a"]; labs = [r"$|z_\rho| - |z_{\rho_s}|$", r"$|z_\rho| - |z_{\rho_s}| + |z_{\hat\rho_\Delta}|$", r"$z_{\hat\rho_\Delta}$"]
    for i, sc in enumerate(order):
        for j, col in enumerate(("two", "three", "z_U")):
            v = D[(D[key] == sc) & (D.N == N)][col].dropna().values; c = cols[j]
            b = ax.boxplot(v, positions=[i + (j - 1) * w * 1.15], widths=w, patch_artist=True, showfliers=True, medianprops=dict(color="#0b0b0b", lw=1.4), whiskerprops=dict(color=c, lw=1.1), capprops=dict(color=c, lw=1.1),
                           boxprops=dict(color=c, lw=1.1), flierprops=dict(marker="o", ms=2.6, mfc="none", mec=c, mew=0.8)); b["boxes"][0].set_facecolor(c)
    ax.axhline(0, color="#898781", lw=0.7, ls=":", zorder=1); ax.set_xticks(range(len(order))); ax.set_xticklabels(tick_labels); ax.set_xlim(-0.55, len(order) - 0.45); ax.set_ylabel("score"); ax.grid(axis="y", color="#e1e0d9", lw=0.6); ax.set_axisbelow(True); ax.tick_params(axis="x", length=0)
    for s_ in (1.96, -1.96): ax.axhline(s_, color="#52514e", lw=0.9, ls=(0, (4, 3)), zorder=1)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"): ax.spines[s_].set_color("#c3c2b7")
    ax.legend(handles=[Patch(fc=cols[j], ec=cols[j], label=labs[j]) for j in range(3)], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    for artist in fig.findobj():
        if hasattr(artist, "set_clip_on"): artist.set_clip_on(False)
    fig.savefig(f"{OUT}{prefix}_by_network_score_comparison_{N}_pairs.pdf", format="pdf", bbox_inches="tight", facecolor="none", edgecolor="none", transparent=True)
    if PREVIEW: fig.savefig(f"{PREVIEW}/{prefix}_by_network_score_comparison_{N}_pairs.png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
if __name__ == "__main__":
    d2 = pd.read_csv(O + "twin_size_scan_figure2.csv"); make_by_network(d2, "scenario", F2, T2, "figure2"); two_vs_three(d2, "scenario", F2, T2, "figure2")      # [2026-10-17 grouped by network; the by-twin-number version make() is kept above]
    d3 = pd.read_csv(O + "twin_size_scan_figure3.csv").rename(columns={"condition": "scenario"}); d3 = pd.concat([d3, d2[d2.scenario.isin(F3[:2])]], ignore_index=True); make_by_network(d3, "scenario", F3, T3, "figure3")
    print("done")
    p6 = O + "twin_size_scan_figure2_6000.csv"
    if os.path.exists(p6): single_n_comparison(pd.read_csv(p6), "scenario", F2, T2, "figure2", 6000)
