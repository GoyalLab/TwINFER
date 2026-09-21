"""3 GRN plots for grn_n6_e9_pos100_center_rep0 (rep0 instance, sim replicate 0, t1=1):
ground truth, TODO4v2(nogate), and its best competitor for this topology family (GRNBOOST2,
auprc 0.580 vs TODO4v2's 0.634 on summary_metrics_table_network_sweep_t1_1.csv). Edge opacity in
the two prediction plots is proportional to that method's own rank over all scoreable ordered
pairs (rank 1 = fully opaque, worst rank -> near-transparent floor); ground truth draws only the
true edges, at full opacity.

Uses grn_plot_spread_v2.py's plot_grn(), extended this session with an edge_alpha parameter.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/work_in_progress/benchmark")
sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/larry_hematopoiesis_validation")
from grn_plot_spread_v2 import plot_grn
from analytic_zscores import DEFAULT_SD, z_signed
import todo4v2_sim_scoring as T

ROOT = T.ROOT
OUT_DIR = "/gpfs/projects/b1255/hzhang/TwINFER_KA/network_figures/n6_e9_rank_opacity"
import os
os.makedirs(OUT_DIR, exist_ok=True)

DATASET_ID = "grn_n6_e9_pos100_center_rep0"
JSON_PATH = f"{ROOT}/analysis_data/network_sweep_final/twinfer_inference_allpairs/{DATASET_ID}_rep0_all_results.json"
GT_PATH = f"{ROOT}/input_data/network_sweep_final/{DATASET_ID}.txt"
RANKED_EDGES_PATH = f"{ROOT}/analysis_data/network_sweep_final/beeline_inference/{DATASET_ID}/simrep0_twin_paired/GRNBOOST2/rankedEdges.csv"

MIN_ALPHA = 0.05


def rank_to_alpha(ranks, n_total):
    """rank 1 (best) -> alpha 1.0, rank n_total (worst) -> alpha MIN_ALPHA, linear in between."""
    ranks = np.asarray(ranks, float)
    if n_total <= 1:
        return np.ones_like(ranks)
    return 1.0 - (ranks - 1) / (n_total - 1) * (1.0 - MIN_ALPHA)


def main():
    d = json.load(open(JSON_PATH))
    gene_names = d["gene_names"]
    n = len(gene_names)
    gene_index = {g: i for i, g in enumerate(gene_names)}

    gt_matrix = np.loadtxt(GT_PATH, delimiter=",")
    assert gt_matrix.shape == (n, n)

    all_pairs = [(a, b) for a in gene_names for b in gene_names if a != b]
    n_pairs_total = len(all_pairs)

    # --- ground truth plot ---
    fig, ax = plot_grn(
        gt_matrix, node_labels=[g.replace("gene_", "g") for g in gene_names],
        title=f"Ground truth — {DATASET_ID}",
    )
    fig.savefig(f"{OUT_DIR}/ground_truth.pdf", format="pdf", bbox_inches="tight"); fig.savefig(f"{OUT_DIR}/ground_truth.png", format="png", dpi=150, bbox_inches="tight")
    print(f"wrote {OUT_DIR}/ground_truth.pdf")

    # --- TODO4v2(nogate) ---
    tsi = d["twin_score_inputs"]
    dd = pd.DataFrame(tsi["data"], columns=tsi["columns"])
    dd = dd[dd.gene_1 != dd.gene_2].reset_index(drop=True)
    dd = dd[dd.rho_cross_xy.notna()].reset_index(drop=True)
    z_reg_map = d["gated_regulation"]["z_reg_gated"]
    U = list(zip(dd.gene_1, dd.gene_2))
    sc, _ = T.todo4v2_score(dd, U, z_reg_map, "nogate")
    sign = np.sign(dd.rho_cross_xy.to_numpy())
    sign[sign == 0] = 1.0

    score_map = dict(zip(U, sc))
    sign_map = dict(zip(U, sign))
    # rank over the FULL universe: present pairs ranked by score (desc); missing pairs tied last
    present_sorted = sorted(U, key=lambda p: -score_map[p])
    rank_map = {p: i + 1 for i, p in enumerate(present_sorted)}
    missing = [p for p in all_pairs if p not in rank_map]
    for p in missing:
        rank_map[p] = len(present_sorted) + 1  # tied at worst rank

    todo_matrix = np.zeros((n, n))
    todo_alpha = {}
    for a, b in all_pairs:
        i, j = gene_index[a], gene_index[b]
        s = sign_map.get((a, b), 1.0)
        todo_matrix[i, j] = s
        rank = rank_map[(a, b)]
        todo_alpha[(i, j)] = float(rank_to_alpha([rank], n_pairs_total)[0])

    fig, ax = plot_grn(
        todo_matrix, node_labels=[g.replace("gene_", "g") for g in gene_names],
        title=f"TODO4v2 (rank-faded) — {DATASET_ID}", edge_alpha=todo_alpha,
    )
    fig.savefig(f"{OUT_DIR}/todo4v2.pdf", format="pdf", bbox_inches="tight"); fig.savefig(f"{OUT_DIR}/todo4v2.png", format="png", dpi=150, bbox_inches="tight")
    print(f"wrote {OUT_DIR}/todo4v2.pdf")

    # --- TODO4v2, unsigned (matches the 'directed_unsigned' scoring variant used everywhere
    # else this session -- existence + rank only, sign dropped, every edge drawn positive/blue) ---
    todo_matrix_unsigned = np.where(todo_matrix != 0, 1.0, 0.0)
    fig, ax = plot_grn(
        todo_matrix_unsigned, node_labels=[g.replace("gene_", "g") for g in gene_names],
        title=f"TODO4v2, unsigned (rank-faded) — {DATASET_ID}", edge_alpha=todo_alpha,
    )
    fig.savefig(f"{OUT_DIR}/todo4v2_unsigned.pdf", format="pdf", bbox_inches="tight"); fig.savefig(f"{OUT_DIR}/todo4v2_unsigned.png", format="png", dpi=150, bbox_inches="tight")
    print(f"wrote {OUT_DIR}/todo4v2_unsigned.pdf")

    # --- GRNBOOST2 (best competitor for this topology family) ---
    re = pd.read_csv(RANKED_EDGES_PATH, sep="\t", header=0)
    re = re[re.Gene1 != re.Gene2].reset_index(drop=True)
    re["_abs"] = re.EdgeWeight.fillna(0.0).abs()
    re = re.sort_values("_abs", ascending=False).drop_duplicates(subset=["Gene1", "Gene2"]).reset_index(drop=True)
    comp_score = dict(zip(zip(re.Gene1, re.Gene2), re.EdgeWeight))
    comp_present_sorted = sorted(comp_score.keys(), key=lambda p: -comp_score[p])
    comp_rank_map = {p: i + 1 for i, p in enumerate(comp_present_sorted)}
    for p in all_pairs:
        if p not in comp_rank_map:
            comp_rank_map[p] = len(comp_present_sorted) + 1

    comp_matrix = np.zeros((n, n))
    comp_alpha = {}
    for a, b in all_pairs:
        i, j = gene_index[a], gene_index[b]
        comp_matrix[i, j] = 1.0  # GRNBOOST2 importance is never negative -- default-positive sign
        rank = comp_rank_map[(a, b)]
        comp_alpha[(i, j)] = float(rank_to_alpha([rank], n_pairs_total)[0])

    fig, ax = plot_grn(
        comp_matrix, node_labels=[g.replace("gene_", "g") for g in gene_names],
        title=f"GRNBOOST2 (best competitor, rank-faded) — {DATASET_ID}", edge_alpha=comp_alpha,
    )
    fig.savefig(f"{OUT_DIR}/grnboost2_best_competitor.pdf", format="pdf", bbox_inches="tight"); fig.savefig(f"{OUT_DIR}/grnboost2_best_competitor.png", format="png", dpi=150, bbox_inches="tight")
    print(f"wrote {OUT_DIR}/grnboost2_best_competitor.pdf")

    # --- combined 1x3 figure, same node layout (plot_grn's circular fallback is deterministic
    # given n_nodes, so all 3 panels already place nodes identically) ---
    labels = [g.replace("gene_", "g") for g in gene_names]
    combined_fig, combined_axes = plt.subplots(1, 3, figsize=(21, 7))

    plot_grn(gt_matrix, node_labels=labels, title="Ground truth", ax=combined_axes[0])
    plot_grn(todo_matrix, node_labels=labels, title="TODO4v2 (rank-faded)",
              edge_alpha=todo_alpha, ax=combined_axes[1])
    plot_grn(comp_matrix, node_labels=labels, title="GRNBOOST2 — best competitor (rank-faded)",
              edge_alpha=comp_alpha, ax=combined_axes[2])

    combined_fig.suptitle(f"{DATASET_ID} — rep0, simrep0, t1=1", fontsize=13, fontweight="bold", y=1.02)
    combined_fig.tight_layout()
    combined_fig.savefig(f"{OUT_DIR}/combined.pdf", format="pdf", bbox_inches="tight")
    combined_fig.savefig(f"{OUT_DIR}/combined.png", format="png", dpi=150, bbox_inches="tight")
    print(f"wrote {OUT_DIR}/combined.pdf")

    # --- combined unsigned 1x3 figure (GRNBOOST2 was already effectively unsigned/positive-only) ---
    combined_fig_u, combined_axes_u = plt.subplots(1, 3, figsize=(21, 7))
    plot_grn(gt_matrix, node_labels=labels, title="Ground truth", ax=combined_axes_u[0])
    plot_grn(todo_matrix_unsigned, node_labels=labels, title="TODO4v2, unsigned (rank-faded)",
              edge_alpha=todo_alpha, ax=combined_axes_u[1])
    plot_grn(comp_matrix, node_labels=labels, title="GRNBOOST2 — best competitor (rank-faded)",
              edge_alpha=comp_alpha, ax=combined_axes_u[2])
    combined_fig_u.suptitle(f"{DATASET_ID} — rep0, simrep0, t1=1 (unsigned)", fontsize=13, fontweight="bold", y=1.02)
    combined_fig_u.tight_layout()
    combined_fig_u.savefig(f"{OUT_DIR}/combined_unsigned.pdf", format="pdf", bbox_inches="tight")
    combined_fig_u.savefig(f"{OUT_DIR}/combined_unsigned.png", format="png", dpi=150, bbox_inches="tight")
    print(f"wrote {OUT_DIR}/combined_unsigned.pdf")


if __name__ == "__main__":
    main()
