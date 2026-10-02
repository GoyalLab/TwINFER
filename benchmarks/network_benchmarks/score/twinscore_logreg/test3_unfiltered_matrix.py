# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""
Direct, correct test: use TwINFER's full unfiltered n x n direction matrix
(direction.unfiltered_matrix -- covers every directed pair, no fan-out, no
ranked_edges candidate-panel restriction) vs PEARSON's rankedEdges.csv on the
EXACT SAME simulated replicate, and compute AUPRC for both against the same
ground truth -- matching the actual notebook methodology
(load_twinfer_crosscorr_ranked_edges).
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

CASES = {
    "HSC": dict(
        json_path="analysis_data/paper_analysis/real_networks/twinfer_inference/HSC_rep_0_0_6bff4482_all_results.json",
        pearson_csv="benchmarking_analysis/twinfer_real/outputs/HSC/simrep6bff4482_twin_paired/PEARSON/rankedEdges.csv",
        topo="input_data/real_world_networks/HSC.txt",
    ),
    "EMT": dict(
        json_path="analysis_data/paper_analysis/real_networks/twinfer_inference/EMT_rep_0_839e1156_all_results.json",
        pearson_csv=None,
        topo="input_data/real_world_networks/EMT.txt",
    ),
    "VSC": dict(
        json_path="analysis_data/paper_analysis/real_networks/twinfer_inference/VSC_rep_2_0_2befd3f9_all_results.json",
        pearson_csv="benchmarking_analysis/twinfer_real/outputs/VSC/simrep2befd3f9_twin_paired/PEARSON/rankedEdges.csv",
        topo="input_data/real_world_networks/VSC.txt",
    ),
}


def true_edges_from_topo(topo_path):
    M = np.loadtxt(topo_path, delimiter=",", dtype=int)
    n = M.shape[0]
    genes = [f"gene_{i+1}" for i in range(n)]
    true = {(genes[i], genes[j]) for i in range(n) for j in range(n) if M[i, j] != 0}
    possible = {(genes[i], genes[j]) for i in range(n) for j in range(n) if i != j}
    return genes, true, possible


def auprc_from_scored_pairs(pair_scores: dict, true_edges: set, possible_edges: set):
    """pair_scores: (g1,g2) -> score. Missing pairs get the minimum score (worst rank)."""
    if pair_scores:
        floor = min(pair_scores.values()) - 1.0
    else:
        floor = -1.0
    y_true, y_score = [], []
    for p in possible_edges:
        y_true.append(1 if p in true_edges else 0)
        y_score.append(pair_scores.get(p, floor))
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    return auc(rec, prec)


for label, cfg in CASES.items():
    genes, true_edges, possible_edges = true_edges_from_topo(cfg["topo"])
    n = len(genes)

    d = json.load(open(cfg["json_path"]))
    um = d["direction"]["unfiltered_matrix"]
    mat = pd.DataFrame(um["data"], index=um["index"], columns=um["columns"])
    twinfer_scores = {}
    for g1 in genes:
        for g2 in genes:
            if g1 == g2:
                continue
            v = mat.loc[g1, g2]
            if pd.notna(v):
                twinfer_scores[(g1, g2)] = abs(float(v))

    n_finite = len(twinfer_scores)
    n_true_covered = sum(1 for p in true_edges if p in twinfer_scores)
    twinfer_auprc = auprc_from_scored_pairs(twinfer_scores, true_edges, possible_edges)

    print(f"\n===== {label} =====")
    print(f"  unfiltered_matrix: {n_finite}/{len(possible_edges)} directed pairs have a finite score "
          f"({n_finite/len(possible_edges):.1%} coverage)")
    print(f"  true edges covered by a finite score: {n_true_covered}/{len(true_edges)} "
          f"({n_true_covered/len(true_edges):.1%})")
    print(f"  TwINFER (unfiltered cross-corr magnitude) AUPRC = {twinfer_auprc:.4f}")

    if cfg["pearson_csv"]:
        pear = pd.read_csv(cfg["pearson_csv"], sep="\t")
        pear_scores = {(r.Gene1, r.Gene2): abs(r.EdgeWeight) for r in pear.itertuples()}
        pear_auprc = auprc_from_scored_pairs(pear_scores, true_edges, possible_edges)
        print(f"  PEARSON AUPRC (same replicate) = {pear_auprc:.4f}   "
              f"(pearson covers {len(pear_scores)}/{len(possible_edges)} pairs)")

        # where's the gap coming from? true edges TwINFER scores near-zero / misses vs PEARSON catches
        missed_by_twinfer = [p for p in true_edges if p not in twinfer_scores]
        print(f"  true edges TwINFER has NO score for at all: {len(missed_by_twinfer)} "
              f"e.g. {missed_by_twinfer[:8]}")
