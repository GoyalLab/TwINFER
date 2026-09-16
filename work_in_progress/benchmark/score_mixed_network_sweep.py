"""
Scores TwINFER and BEELINE results for the mixed_network_sweep benchmark
(cycle-core + linear + sprinkled-autoregulation networks, n=6/n=10 genes).

The core precision/recall/AUPRC/top-k-tie-aware machinery below is copied
verbatim from score_cyclic_g3456.py (itself extracted from
benchmark_network_sweep.ipynb) -- only two things are new:

1. Dataset/ground-truth lookup: dataset_id/label/n_genes are already explicit
   fields in each *_all_results.json (infer_mixed_network_sweep.py's output),
   so no filename stem-matching is needed; ground truth is
   input_data/mixed_network_sweep/<dataset_id>.txt directly.

2. adapt_result_schema(): the installed twinfer package's infer_with_twinfer
   has been refactored since score_cyclic_g3456.py/score_twinfer_20260824.py
   were written (see memory: reference_twinfer_infer_kwargs_stale) -- the
   *_all_results.json produced today has a different top-level shape. This
   adapter reconstructs the OLD shape score_twinfer_dataset() expects, so the
   scoring math itself is untouched:
       unfiltered_direction_matrix <- direction.unfiltered_matrix (same DataFrame shape)
       direction_matrix            <- built from ranked_edges: for each edge in
                                       directed_edges, its signed "gamma" value
                                       (falls back to the unfiltered value if a
                                       final edge has no matching ranked_edges row)
       final_directed_edges        <- directed_edges, SELF-PAIRS EXCLUDED (see below)
       fan_out_log                 <- fan_out (same shape: gene_a/gene_b/decision)
       gene_lists                  <- classification (renamed; already pairwise-only)
       potential_regulation        <- classification's non-"no_regulation" pairs

   IMPORTANT: directed_edges includes a (gene_i, gene_i) self-loop entry for
   EVERY gene on EVERY dataset, regardless of whether that network actually
   has autoregulation (verified: grn_n10_e10_c5_a0_pos50_rep0, whose ground
   truth has an all-zero diagonal, still gets all 10 self-entries). None of
   the pairwise machinery (classification/ranked_edges/direction matrices)
   ever considers a (gene_i, gene_i) pair -- self-loops appear to be a fixed
   structural placeholder, not an actual autoregulation call. TwINFER as
   currently built therefore cannot be scored for autoregulation detection;
   this script scores directed non-self edges only (same convention as the
   network_sweep_final scoring), and reports how many true self-loop edges
   each ground truth has (n_true_self_loops) so autoregulation coverage is
   at least visible, not silently dropped.

Usage:
    score_twinfer()  -- scores every *_all_results.json under TWINFER_JSON_DIR.
    score_beeline()  -- only meaningful once the BEELINE SLURM job
                        (run_mixed_network_sweep_benchmark.sh /
                        run_mixed_network_sweep_genie3_grnboost2.sh) has
                        produced rankedEdges.csv files under BEELINE_OUTPUT_ROOT.
"""

import glob
import json
import os
import re
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from sklearn.mixture import GaussianMixture
from sklearn.metrics import auc, precision_recall_curve

PROJECT_ROOT = Path("/home/gzu5140/TwINFER_KA")
GROUND_TRUTH_DIR = PROJECT_ROOT / "input_data" / "mixed_network_sweep"
TWINFER_JSON_DIR = PROJECT_ROOT / "analysis_data" / "mixed_network_sweep" / "twinfer_inference"
BEELINE_INPUT_ROOT = PROJECT_ROOT / "code" / "Beeline" / "inputs" / "mixed_network_sweep"
BEELINE_OUTPUT_ROOT = PROJECT_ROOT / "analysis_data" / "mixed_network_sweep" / "beeline_inference"

TWINFER_SCORES_CSV = PROJECT_ROOT / "analysis_data" / "mixed_network_sweep" / "twinfer_analysis_output.csv"
BEELINE_SCORES_CSV = PROJECT_ROOT / "analysis_data" / "mixed_network_sweep" / "beeline_analysis_output.csv"

RUN_ID_RE = re.compile(r"^simrep([0-9a-f]+)_(spread|twin_paired)$")


# ===================================================================================
# Shared helpers (verbatim from score_cyclic_g3456.py / benchmark_network_sweep.ipynb)
# ===================================================================================

def reconstruct_matrix(raw):
    return pd.DataFrame(raw["data"], index=raw["index"], columns=raw["columns"])


def precision_recall_f1(selected_edges, true_edges):
    if not selected_edges:
        return 0.0, 0.0, 0.0
    tp = len(selected_edges & true_edges)
    precision = tp / len(selected_edges)
    recall = tp / len(true_edges) if true_edges else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def build_directed_true_edges(ground_truth_matrix, gene_names):
    return {
        (gi, gj) for gi in gene_names for gj in gene_names
        if gi != gj and ground_truth_matrix.loc[gi, gj] != 0
    }


def build_directed_true_edges_signed(ground_truth_matrix, gene_names):
    true_edges = set()
    for gi in gene_names:
        for gj in gene_names:
            if gi == gj:
                continue
            val = ground_truth_matrix.loc[gi, gj]
            if val != 0:
                true_edges.add((gi, gj, int(np.sign(val))))
    return true_edges


def build_true_pairs_undirected(ground_truth_matrix, gene_names):
    true_pairs = set()
    for gi, gj in combinations(gene_names, 2):
        if ground_truth_matrix.loc[gi, gj] != 0 or ground_truth_matrix.loc[gj, gi] != 0:
            true_pairs.add(frozenset((gi, gj)))
    return true_pairs


def find_confounded_pairs(ground_truth_matrix, gene_names):
    confounded_pairs = set()
    n = len(gene_names)
    for idx_i in range(n):
        for idx_j in range(idx_i + 1, n):
            gi, gj = gene_names[idx_i], gene_names[idx_j]
            if ground_truth_matrix.loc[gi, gj] != 0 and ground_truth_matrix.loc[gj, gi] != 0:
                continue
            shared_regulator = any(
                ground_truth_matrix.loc[gk, gi] != 0 and ground_truth_matrix.loc[gk, gj] != 0
                for gk in gene_names if gk not in (gi, gj)
            )
            if shared_regulator:
                confounded_pairs.add(frozenset([gi, gj]))
    return confounded_pairs


def score_directed_edges(ground_truth_matrix, gene_names, final_directed_edges,
                          multiple_states, direction_matrix, wrong_sign_counts_as=("FP", "FN")):
    predicted_edges = set(tuple(e) for e in final_directed_edges)
    confounded_pairs = find_confounded_pairs(ground_truth_matrix, gene_names)

    TP = FP = FN = TN = 0
    TP_signed = FP_signed = FN_signed = WRONG_SIGN = 0
    confounded_correct = 0

    for gi in gene_names:
        for gj in gene_names:
            if gi == gj:
                continue
            gt_sign_val = ground_truth_matrix.loc[gi, gj]
            gt_edge = gt_sign_val != 0
            gt_sign = int(np.sign(gt_sign_val))
            pred_edge = (gi, gj) in predicted_edges

            if gt_edge and pred_edge:
                TP += 1
                pred_sign_val = direction_matrix.loc[gi, gj]
                pred_sign = int(np.sign(pred_sign_val)) if pred_sign_val != 0 else 0
                if pred_sign == gt_sign:
                    TP_signed += 1
                else:
                    WRONG_SIGN += 1
                    if "FP" in wrong_sign_counts_as:
                        FP_signed += 1
                    if "FN" in wrong_sign_counts_as:
                        FN_signed += 1
            elif gt_edge and not pred_edge:
                FN += 1
                FN_signed += 1
            elif not gt_edge and pred_edge:
                FP += 1
                FP_signed += 1
            else:
                TN += 1

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    precision_signed = TP_signed / (TP_signed + FP_signed) if (TP_signed + FP_signed) else 0.0
    recall_signed = TP_signed / (TP_signed + FN_signed) if (TP_signed + FN_signed) else 0.0
    f1_signed = (
        2 * precision_signed * recall_signed / (precision_signed + recall_signed)
        if (precision_signed + recall_signed) else 0.0
    )

    return {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "precision": precision, "recall": recall, "f1": f1,
        "TP_signed": TP_signed, "FP_signed": FP_signed, "FN_signed": FN_signed,
        "WRONG_SIGN": WRONG_SIGN,
        "precision_signed": precision_signed, "recall_signed": recall_signed, "f1_signed": f1_signed,
        "confounded_pairs": confounded_pairs, "confounded_pairs_correctly_found": confounded_correct,
    }


def compute_twinfer_natural_undirected(ground_truth_matrix, gene_names, final_directed_edges):
    predicted_edges = set(tuple(e) for e in final_directed_edges)
    true_pairs = build_true_pairs_undirected(ground_truth_matrix, gene_names)
    pred_pairs = set()
    for gi, gj in combinations(gene_names, 2):
        if (gi, gj) in predicted_edges or (gj, gi) in predicted_edges:
            pred_pairs.add(frozenset((gi, gj)))
    return precision_recall_f1(pred_pairs, true_pairs)


def build_master_ranked_table(unfiltered_direction_matrix, gene_names):
    rows = []
    for gi in gene_names:
        for gj in gene_names:
            if gi == gj:
                continue
            corr = unfiltered_direction_matrix.loc[gi, gj]
            # NaN means this pair's statistic couldn't be computed (e.g. a
            # gated subset had zero variance) -- score-cyclic_g3456's
            # network_sweep_final data never hit this; the newer schema's
            # unfiltered_matrix legitimately can. Treat as "no evidence",
            # same convention already used for edges missing from a lookup
            # (score_lookup.get(e, 0.0)) elsewhere in this scoring code.
            if pd.isna(corr):
                corr = 0.0
            rows.append({"gene_1": gi, "gene_2": gj, "directional_correlation": corr})
    return pd.DataFrame(rows)


def reconstruct_pre_fanout_edges(final_directed_edges_post, fan_out_log):
    pre = set(tuple(e) for e in final_directed_edges_post)
    for entry in (fan_out_log or []):
        if entry.get("decision") == "fan_out":
            a, b = entry["gene_a"], entry["gene_b"]
            pre.add((a, b))
            pre.add((b, a))
    return pre


def restrict_ranked_list(ranked_edge_list, edge_set):
    if ranked_edge_list is None or ranked_edge_list.empty:
        return ranked_edge_list
    mask = [(g1, g2) in edge_set for g1, g2 in zip(ranked_edge_list["gene_1"], ranked_edge_list["gene_2"])]
    return ranked_edge_list[mask].reset_index(drop=True)


def restrict_to_single_state_pairs(ranked_edge_list, single_state_regulation_pairs):
    if ranked_edge_list is None or ranked_edge_list.empty:
        return ranked_edge_list
    single_state_fs = {frozenset(p) for p in single_state_regulation_pairs}
    mask = [
        frozenset((g1, g2)) in single_state_fs
        for g1, g2 in zip(ranked_edge_list["gene_1"], ranked_edge_list["gene_2"])
    ]
    return ranked_edge_list[mask].reset_index(drop=True)


def build_potential_regulation_bidirectional_pairs(potential_regulation):
    pairs = [tuple(p) for p in potential_regulation]
    bidirectional = {(a, b) for a, b in pairs} | {(b, a) for a, b in pairs}
    genes = {g for pair in pairs for g in pair}
    self_pairs = {(g, g) for g in genes}
    return bidirectional | self_pairs


def _no_ranking_signal(pred_scores):
    """True if every candidate score is identical (e.g. a method that called
    zero edges, so every pair scores 0). sklearn's precision_recall_curve
    collapses ties into a degenerate 2-point curve whose auc() is
    0.5*(1+prevalence) -- a plausible-looking number that actually encodes
    NO ranking information (confirmed: reproduces exactly the constant
    values seen contaminating e13_pos100's TwINFER AUPRC column, where the
    method found literally zero correlated pairs in 25/30 runs). Treat as
    not-computable (NaN) rather than silently reporting that artifact."""
    return len(set(pred_scores)) <= 1


def compute_twinfer_auprc(ranked_edge_list, ground_truth_matrix, gene_names):
    possible_edges = list(permutations(gene_names, 2))
    true_labels = [1 if ground_truth_matrix.loc[gi, gj] != 0 else 0 for gi, gj in possible_edges]
    if sum(true_labels) == 0 or ranked_edge_list is None or ranked_edge_list.empty:
        return float("nan")
    score_lookup = {(row.gene_1, row.gene_2): abs(row.directional_correlation) for row in ranked_edge_list.itertuples()}
    pred_scores = [score_lookup.get(e, 0.0) for e in possible_edges]
    if _no_ranking_signal(pred_scores):
        return float("nan")
    precision, recall, _ = precision_recall_curve(true_labels, pred_scores)
    return float(auc(recall, precision))


def compute_twinfer_auprc_signed(ranked_edge_list, ground_truth_matrix, gene_names):
    possible_edges = list(permutations(gene_names, 2))
    possible_signed_positions = [(gi, gj, s) for gi, gj in possible_edges for s in (1, -1)]
    true_labels = [1 if int(np.sign(ground_truth_matrix.loc[gi, gj])) == s else 0 for gi, gj, s in possible_signed_positions]
    if sum(true_labels) == 0 or ranked_edge_list is None or ranked_edge_list.empty:
        return float("nan")
    score_lookup = {
        (row.gene_1, row.gene_2): (abs(row.directional_correlation), int(np.sign(row.directional_correlation)))
        for row in ranked_edge_list.itertuples()
    }
    pred_scores = []
    for gi, gj, s in possible_signed_positions:
        entry = score_lookup.get((gi, gj))
        pred_scores.append(entry[0] if entry is not None and entry[1] == s else 0.0)
    if _no_ranking_signal(pred_scores):
        return float("nan")
    precision, recall, _ = precision_recall_curve(true_labels, pred_scores)
    return float(auc(recall, precision))


def compute_twinfer_auprc_undirected(ranked_edge_list, ground_truth_matrix, gene_names):
    unordered_pairs = list(combinations(gene_names, 2))
    true_labels = [1 if (ground_truth_matrix.loc[gi, gj] != 0 or ground_truth_matrix.loc[gj, gi] != 0) else 0 for gi, gj in unordered_pairs]
    if sum(true_labels) == 0 or ranked_edge_list is None or ranked_edge_list.empty:
        return float("nan")
    score_lookup = {}
    for row in ranked_edge_list.itertuples():
        score = abs(row.directional_correlation)
        pair = frozenset((row.gene_1, row.gene_2))
        if pair not in score_lookup or score > score_lookup[pair]:
            score_lookup[pair] = score
    pred_scores = [score_lookup.get(frozenset(p), 0.0) for p in unordered_pairs]
    if _no_ranking_signal(pred_scores):
        return float("nan")
    precision, recall, _ = precision_recall_curve(true_labels, pred_scores)
    return float(auc(recall, precision))


def top_k_tie_aware_selection_twinfer(ranked_edge_list, num_true_edges):
    if ranked_edge_list is None or ranked_edge_list.empty or num_true_edges == 0:
        return set(), float("nan")
    scored = ranked_edge_list.copy()
    scored["_score"] = scored["directional_correlation"].abs()
    scored = scored.sort_values("_score", ascending=False).reset_index(drop=True)
    maxk = min(len(scored), num_true_edges)
    edge_score_topk = float(scored.iloc[maxk - 1]["_score"])
    nonzero = scored.loc[scored["_score"] > 0, "_score"]
    non_zero_min = float(nonzero.min()) if not nonzero.empty else 0.0
    best_val = max(non_zero_min, edge_score_topk)
    selected = scored[scored["_score"] >= best_val]
    return set(zip(selected["gene_1"], selected["gene_2"])), best_val


def top_k_tie_aware_selection_twinfer_signed(ranked_edge_list, num_true_edges_signed):
    if ranked_edge_list is None or ranked_edge_list.empty or num_true_edges_signed == 0:
        return set(), float("nan")
    scored = ranked_edge_list.copy()
    scored["_score"] = scored["directional_correlation"].abs()
    scored["_sign"] = np.sign(scored["directional_correlation"]).astype(int)
    scored = scored.sort_values("_score", ascending=False).reset_index(drop=True)
    maxk = min(len(scored), num_true_edges_signed)
    edge_score_topk = float(scored.iloc[maxk - 1]["_score"])
    nonzero = scored.loc[scored["_score"] > 0, "_score"]
    non_zero_min = float(nonzero.min()) if not nonzero.empty else 0.0
    best_val = max(non_zero_min, edge_score_topk)
    selected = scored[scored["_score"] >= best_val]
    return set(zip(selected["gene_1"], selected["gene_2"], selected["_sign"])), best_val


def top_k_tie_aware_selection_twinfer_undirected(ranked_edge_list, num_true_pairs_undirected):
    if ranked_edge_list is None or ranked_edge_list.empty or num_true_pairs_undirected == 0:
        return set(), float("nan")
    scored = ranked_edge_list.copy()
    scored["_score"] = scored["directional_correlation"].abs()
    scored["_pair"] = [frozenset(p) for p in zip(scored["gene_1"], scored["gene_2"])]
    scored = scored.sort_values("_score", ascending=False).drop_duplicates(subset=["_pair"]).reset_index(drop=True)
    maxk = min(len(scored), num_true_pairs_undirected)
    edge_score_topk = float(scored.iloc[maxk - 1]["_score"])
    nonzero = scored.loc[scored["_score"] > 0, "_score"]
    non_zero_min = float(nonzero.min()) if not nonzero.empty else 0.0
    best_val = max(non_zero_min, edge_score_topk)
    selected = scored[scored["_score"] >= best_val]
    return set(selected["_pair"]), best_val


TOPK_POOL_NAMES = ["all_edges", "pass_first_test", "not_heterogeneous", "fan_out_removed"]


def score_twinfer_dataset(results, gene_names, ground_truth_matrix):
    unfiltered_direction_matrix = reconstruct_matrix(results["unfiltered_direction_matrix"])
    direction_matrix = reconstruct_matrix(results["direction_matrix"])
    final_directed_edges_post = set(tuple(e) for e in results["final_directed_edges"])
    fan_out_log = results.get("fan_out_log")
    single_state_pairs = [tuple(p) for p in results["gene_lists"]["single_state_regulation"]]
    mult_list = results["gene_lists"]["multiple_states_and_reg"] + results["gene_lists"]["multiple_states_no_reg"]

    master_table = build_master_ranked_table(unfiltered_direction_matrix, gene_names)
    pre_fanout_edges = reconstruct_pre_fanout_edges(final_directed_edges_post, fan_out_log)

    potential_regulation_pairs = build_potential_regulation_bidirectional_pairs(results["potential_regulation"])
    pass_first_test_edges = pre_fanout_edges & potential_regulation_pairs
    pass_first_test_df = restrict_ranked_list(master_table, pass_first_test_edges)

    not_heterogeneous_df = restrict_to_single_state_pairs(pass_first_test_df, single_state_pairs)
    if not_heterogeneous_df is None or not_heterogeneous_df.empty:
        not_heterogeneous_edges = set()
    else:
        not_heterogeneous_edges = set(zip(not_heterogeneous_df["gene_1"], not_heterogeneous_df["gene_2"]))

    fan_out_pairs_fs = {
        frozenset((entry["gene_a"], entry["gene_b"]))
        for entry in (fan_out_log or []) if entry.get("decision") == "fan_out"
    }
    fan_out_removed_edges_nested = {e for e in not_heterogeneous_edges if frozenset(e) not in fan_out_pairs_fs}

    pools = {
        "all_edges": master_table,
        "pass_first_test": pass_first_test_df,
        "not_heterogeneous": not_heterogeneous_df,
        "fan_out_removed": restrict_ranked_list(master_table, fan_out_removed_edges_nested),
    }

    true_edges = build_directed_true_edges(ground_truth_matrix, gene_names)
    true_edges_signed = build_directed_true_edges_signed(ground_truth_matrix, gene_names)
    true_pairs_undirected = build_true_pairs_undirected(ground_truth_matrix, gene_names)
    n_true, n_true_signed, n_true_undirected = len(true_edges), len(true_edges_signed), len(true_pairs_undirected)

    row = {
        "sim_type": results.get("sim_type"), "rep_id": results.get("rep_id"), "n_genes": results.get("n_genes"),
        "n_fan_out_pairs": sum(1 for e in (fan_out_log or []) if e.get("decision") == "fan_out"),
        "n_mutual_regulation_pairs": sum(1 for e in (fan_out_log or []) if e.get("decision") == "mutual_regulation"),
    }

    row["auprc"] = compute_twinfer_auprc(master_table, ground_truth_matrix, gene_names)
    row["auprc_signed"] = compute_twinfer_auprc_signed(master_table, ground_truth_matrix, gene_names)
    row["auprc_undirected"] = compute_twinfer_auprc_undirected(master_table, ground_truth_matrix, gene_names)

    natural_edge_sets = {"before_fan_out": pre_fanout_edges, "after_fan_out": final_directed_edges_post}
    for stage_name, edges in natural_edge_sets.items():
        natural_metrics = score_directed_edges(ground_truth_matrix, gene_names, edges, mult_list, direction_matrix)
        row[f"precision_natural_{stage_name}"] = natural_metrics["precision"]
        row[f"recall_natural_{stage_name}"] = natural_metrics["recall"]
        row[f"f1_natural_{stage_name}"] = natural_metrics["f1"]
        row[f"precision_natural_{stage_name}_signed"] = natural_metrics["precision_signed"]
        row[f"recall_natural_{stage_name}_signed"] = natural_metrics["recall_signed"]
        row[f"f1_natural_{stage_name}_signed"] = natural_metrics["f1_signed"]
        p_u, r_u, f1_u = compute_twinfer_natural_undirected(ground_truth_matrix, gene_names, edges)
        row[f"precision_natural_{stage_name}_undirected"] = p_u
        row[f"recall_natural_{stage_name}_undirected"] = r_u
        row[f"f1_natural_{stage_name}_undirected"] = f1_u

    for pool_name in TOPK_POOL_NAMES:
        pool_df = pools[pool_name]
        selected, _ = top_k_tie_aware_selection_twinfer(pool_df, n_true)
        p, r, f1 = precision_recall_f1(selected, true_edges)
        row[f"precision_topk_{pool_name}"] = p
        row[f"recall_topk_{pool_name}"] = r
        row[f"f1_topk_{pool_name}"] = f1

        selected_s, _ = top_k_tie_aware_selection_twinfer_signed(pool_df, n_true_signed)
        p_s, r_s, f1_s = precision_recall_f1(selected_s, true_edges_signed)
        row[f"precision_topk_{pool_name}_signed"] = p_s
        row[f"recall_topk_{pool_name}_signed"] = r_s
        row[f"f1_topk_{pool_name}_signed"] = f1_s

        selected_u, _ = top_k_tie_aware_selection_twinfer_undirected(pool_df, n_true_undirected)
        p_u2, r_u2, f1_u2 = precision_recall_f1(selected_u, true_pairs_undirected)
        row[f"precision_topk_{pool_name}_undirected"] = p_u2
        row[f"recall_topk_{pool_name}_undirected"] = r_u2
        row[f"f1_topk_{pool_name}_undirected"] = f1_u2

    return row


def load_ground_truth_matrix(path_to_matrix, gene_names):
    matrix = np.loadtxt(path_to_matrix, dtype=int, delimiter=",")
    return pd.DataFrame(matrix, index=gene_names, columns=gene_names)


# ===================================================================================
# New: adapter from the current infer_with_twinfer output schema to the shape
# score_twinfer_dataset() expects (see module docstring).
# ===================================================================================

def adapt_result_schema(record, gene_names):
    unfiltered_direction_matrix = record["direction"]["unfiltered_matrix"]

    ranked_cols = record["ranked_edges"]["columns"]
    gamma_by_pair = {}
    if record["ranked_edges"]["data"]:
        i_g1, i_g2, i_gamma = ranked_cols.index("gene_1"), ranked_cols.index("gene_2"), ranked_cols.index("gamma")
        for r in record["ranked_edges"]["data"]:
            gamma_by_pair[(r[i_g1], r[i_g2])] = r[i_gamma]

    final_directed_edges = [tuple(e) for e in record["directed_edges"] if e[0] != e[1]]

    unfiltered_df = reconstruct_matrix(unfiltered_direction_matrix)
    direction_df = pd.DataFrame(0.0, index=gene_names, columns=gene_names)
    for gi, gj in final_directed_edges:
        # gamma is the post-processing signed direction statistic (what decided
        # "gi -> gj dominates"); fall back to the raw unfiltered correlation's
        # sign if this edge has no matching ranked_edges row. NaN -> 0 (sign
        # unknown; score_directed_edges treats a 0 predicted sign as neither
        # matching nor mismatching a nonzero ground-truth sign here since
        # np.sign(0)=0 never equals a +/-1 gt_sign, i.e. counted as wrong-sign).
        val = gamma_by_pair.get((gi, gj), unfiltered_df.loc[gi, gj])
        direction_df.loc[gi, gj] = 0.0 if pd.isna(val) else val

    cl = record["classification"]
    gene_lists = {
        "single_state_regulation": cl.get("single_state_regulation", []),
        "multiple_states_and_reg": cl.get("multiple_states_and_reg", []),
        "multiple_states_no_reg": cl.get("multiple_states_no_reg", []),
    }
    potential_regulation = (
        gene_lists["single_state_regulation"]
        + gene_lists["multiple_states_and_reg"]
        + gene_lists["multiple_states_no_reg"]
    )

    return {
        "sim_type": record["dataset_id"],
        "rep_id": record["label"],
        "n_genes": record["n_genes"],
        "unfiltered_direction_matrix": unfiltered_direction_matrix,
        "direction_matrix": {"data": direction_df.values.tolist(), "index": gene_names, "columns": gene_names},
        "final_directed_edges": final_directed_edges,
        "fan_out_log": record.get("fan_out", []),
        "gene_lists": gene_lists,
        "potential_regulation": potential_regulation,
    }


def score_twinfer():
    """Scores every mixed_network_sweep *_all_results.json against its network's
    ground truth (input_data/mixed_network_sweep/<dataset_id>.txt)."""
    records = []
    for jf in sorted(glob.glob(str(TWINFER_JSON_DIR / "*_all_results.json"))):
        with open(jf) as f:
            record = json.load(f)
        dataset_id = record["dataset_id"]
        n_genes = record["n_genes"]
        gene_names = record.get("gene_names") or [f"gene_{i+1}" for i in range(n_genes)]
        gt_path = GROUND_TRUTH_DIR / f"{dataset_id}.txt"
        if not gt_path.exists():
            print(f"[skip] no ground truth for {dataset_id} ({gt_path})")
            continue
        gt = load_ground_truth_matrix(gt_path, gene_names)

        adapted = adapt_result_schema(record, gene_names)
        row = score_twinfer_dataset(adapted, gene_names, gt)
        row["dataset_id"] = dataset_id
        row["json_file"] = os.path.basename(jf)
        row["n_true_self_loops"] = int(np.count_nonzero(np.diag(gt.values)))
        records.append(row)

    df = pd.DataFrame(records)
    TWINFER_SCORES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TWINFER_SCORES_CSV, index=False)
    print(f"Scored {len(df)} TwINFER (dataset, rep) result(s) -> {TWINFER_SCORES_CSV}")
    return df


# ===================================================================================
# BEELINE-side scoring (verbatim from score_cyclic_g3456.py / benchmark_network_sweep.ipynb)
# ===================================================================================

def load_beeline_ground_truth(dataset_id: str) -> pd.DataFrame:
    gt_path = BEELINE_INPUT_ROOT / dataset_id / "GroundTruthNetwork.csv"
    return pd.read_csv(gt_path, header=0)


def build_edge_universe(gt_df: pd.DataFrame):
    gt_no_self = gt_df[gt_df["Gene1"] != gt_df["Gene2"]].drop_duplicates()
    unique_nodes = sorted(set(gt_df["Gene1"]).union(set(gt_df["Gene2"])))
    possible_edges = set(permutations(unique_nodes, 2))
    true_edges = set(zip(gt_no_self["Gene1"], gt_no_self["Gene2"])) & possible_edges
    return possible_edges, true_edges


def dedupe_predictions(ranked_edges: pd.DataFrame) -> pd.DataFrame:
    pred = ranked_edges[ranked_edges["Gene1"] != ranked_edges["Gene2"]].copy()
    pred["_abs"] = pred["EdgeWeight"].fillna(0.0).abs()  # PPCOR occasionally emits NaN (e.g. zero-variance gene column); treat as no evidence
    return pred.sort_values("_abs", ascending=False).drop_duplicates(subset=["Gene1", "Gene2"]).reset_index(drop=True)


def top_k_tie_aware_selection(predicted: pd.DataFrame, num_true_edges: int):
    if predicted.empty or num_true_edges == 0:
        return set(), float("nan")
    maxk = min(len(predicted), num_true_edges)
    edge_weight_topk = float(predicted.iloc[maxk - 1]["_abs"])
    nonzero = predicted.loc[predicted["_abs"] > 0, "_abs"]
    non_zero_min = float(nonzero.min()) if not nonzero.empty else 0.0
    best_val = max(non_zero_min, edge_weight_topk)
    selected = predicted[predicted["_abs"] >= best_val]
    return set(zip(selected["Gene1"], selected["Gene2"])), best_val


def compute_auprc(ranked_edges: pd.DataFrame, gt_df: pd.DataFrame) -> float:
    gt_genes = sorted(set(gt_df["Gene1"]).union(set(gt_df["Gene2"])))
    possible_edges = list(permutations(gt_genes, 2))
    true_edge_set = set(zip(gt_df["Gene1"], gt_df["Gene2"]))
    pred = dedupe_predictions(ranked_edges)
    pred_lookup = dict(zip(zip(pred["Gene1"], pred["Gene2"]), pred["_abs"].astype(float)))
    true_labels = [1 if e in true_edge_set else 0 for e in possible_edges]
    pred_scores = [pred_lookup.get(e, 0.0) for e in possible_edges]
    if sum(true_labels) == 0:
        return float("nan")
    precision, recall, _ = precision_recall_curve(true_labels, pred_scores)
    return float(auc(recall, precision))


def build_signed_edge_universe(gt_df: pd.DataFrame):
    gt_no_self = gt_df[gt_df["Gene1"] != gt_df["Gene2"]].drop_duplicates()
    return set(zip(gt_no_self["Gene1"], gt_no_self["Gene2"], gt_no_self["Type"]))


def build_undirected_edge_universe(gt_df: pd.DataFrame):
    gt_no_self = gt_df[gt_df["Gene1"] != gt_df["Gene2"]].drop_duplicates()
    unique_nodes = sorted(set(gt_df["Gene1"]).union(set(gt_df["Gene2"])))
    possible_pairs = {frozenset(p) for p in combinations(unique_nodes, 2)}
    true_pairs = {frozenset((g1, g2)) for g1, g2 in zip(gt_no_self["Gene1"], gt_no_self["Gene2"])}
    return possible_pairs, true_pairs


def add_predicted_sign(predicted: pd.DataFrame) -> pd.DataFrame:
    pred = predicted.copy()
    pred["_sign"] = np.where(pred["EdgeWeight"] >= 0, "+", "-")
    return pred


def top_k_tie_aware_selection_signed(predicted_signed: pd.DataFrame, num_true_edges: int):
    if predicted_signed.empty or num_true_edges == 0:
        return set(), float("nan")
    maxk = min(len(predicted_signed), num_true_edges)
    edge_weight_topk = float(predicted_signed.iloc[maxk - 1]["_abs"])
    nonzero = predicted_signed.loc[predicted_signed["_abs"] > 0, "_abs"]
    non_zero_min = float(nonzero.min()) if not nonzero.empty else 0.0
    best_val = max(non_zero_min, edge_weight_topk)
    selected = predicted_signed[predicted_signed["_abs"] >= best_val]
    return set(zip(selected["Gene1"], selected["Gene2"], selected["_sign"])), best_val


def compute_auprc_signed(ranked_edges: pd.DataFrame, gt_df: pd.DataFrame) -> float:
    gt_genes = sorted(set(gt_df["Gene1"]).union(set(gt_df["Gene2"])))
    possible_edges = list(permutations(gt_genes, 2))
    possible_signed_positions = [(g1, g2, s) for (g1, g2) in possible_edges for s in ("+", "-")]
    true_signed_edges = build_signed_edge_universe(gt_df)
    pred = add_predicted_sign(dedupe_predictions(ranked_edges))
    pred_score_by_pair = dict(zip(zip(pred["Gene1"], pred["Gene2"]), pred["_abs"].astype(float)))
    pred_sign_by_pair = dict(zip(zip(pred["Gene1"], pred["Gene2"]), pred["_sign"]))
    true_labels, pred_scores = [], []
    for (g1, g2, s) in possible_signed_positions:
        true_labels.append(1 if (g1, g2, s) in true_signed_edges else 0)
        predicted_sign_here = pred_sign_by_pair.get((g1, g2))
        score = pred_score_by_pair.get((g1, g2), 0.0) if predicted_sign_here == s else 0.0
        pred_scores.append(score)
    if sum(true_labels) == 0:
        return float("nan")
    precision, recall, _ = precision_recall_curve(true_labels, pred_scores)
    return float(auc(recall, precision))


def dedupe_predictions_undirected(ranked_edges: pd.DataFrame) -> pd.DataFrame:
    pred = ranked_edges[ranked_edges["Gene1"] != ranked_edges["Gene2"]].copy()
    pred["_abs"] = pred["EdgeWeight"].fillna(0.0).abs()  # PPCOR occasionally emits NaN (e.g. zero-variance gene column); treat as no evidence
    pred["_pair"] = [frozenset(p) for p in zip(pred["Gene1"], pred["Gene2"])]
    return pred.sort_values("_abs", ascending=False).drop_duplicates(subset=["_pair"]).reset_index(drop=True)


def top_k_tie_aware_selection_undirected(predicted_undirected: pd.DataFrame, num_true_edges: int):
    if predicted_undirected.empty or num_true_edges == 0:
        return set(), float("nan")
    maxk = min(len(predicted_undirected), num_true_edges)
    edge_weight_topk = float(predicted_undirected.iloc[maxk - 1]["_abs"])
    nonzero = predicted_undirected.loc[predicted_undirected["_abs"] > 0, "_abs"]
    non_zero_min = float(nonzero.min()) if not nonzero.empty else 0.0
    best_val = max(non_zero_min, edge_weight_topk)
    selected = predicted_undirected[predicted_undirected["_abs"] >= best_val]
    return set(selected["_pair"]), best_val


def compute_auprc_undirected(ranked_edges: pd.DataFrame, gt_df: pd.DataFrame) -> float:
    gt_genes = sorted(set(gt_df["Gene1"]).union(set(gt_df["Gene2"])))
    possible_pairs = [frozenset(p) for p in combinations(gt_genes, 2)]
    true_pairs = {frozenset((g1, g2)) for g1, g2 in zip(gt_df["Gene1"], gt_df["Gene2"])}
    pred = dedupe_predictions_undirected(ranked_edges)
    pred_lookup = dict(zip(pred["_pair"], pred["_abs"].astype(float)))
    true_labels = [1 if p in true_pairs else 0 for p in possible_pairs]
    pred_scores = [pred_lookup.get(p, 0.0) for p in possible_pairs]
    if sum(true_labels) == 0:
        return float("nan")
    precision, recall, _ = precision_recall_curve(true_labels, pred_scores)
    return float(auc(recall, precision))


def fit_gmm_threshold(scores, min_points=2, random_state=0):
    x = np.asarray(scores).reshape(-1, 1)
    if len(x) < min_points or len(np.unique(x)) < 2:
        return None, np.inf, "GMM_too_few_points"
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=random_state)
    gmm.fit(x)
    weights, means, stds = gmm.weights_, gmm.means_.flatten(), np.sqrt(gmm.covariances_.flatten())
    order = np.argsort(means)
    w1, w2 = weights[order]
    m1, m2 = means[order]
    s1, s2 = stds[order]
    f = lambda z: w1 * norm.pdf(z, m1, s1) - w2 * norm.pdf(z, m2, s2)
    try:
        threshold = brentq(f, m1, m2)
        return gmm, threshold, "GMM_intersection"
    except ValueError:
        return gmm, np.inf, "GMM_no_intersection"


def gmm_natural_threshold_selection(predicted, threshold):
    if predicted.empty:
        return set()
    return set(zip(*[predicted.loc[predicted["_abs"] >= threshold, c] for c in ("Gene1", "Gene2")]))


def gmm_natural_threshold_selection_signed(predicted_signed, threshold):
    if predicted_signed.empty:
        return set()
    sel = predicted_signed[predicted_signed["_abs"] >= threshold]
    return set(zip(sel["Gene1"], sel["Gene2"], sel["_sign"]))


def gmm_natural_threshold_selection_undirected(predicted_undirected, threshold):
    if predicted_undirected.empty:
        return set()
    return set(predicted_undirected.loc[predicted_undirected["_abs"] >= threshold, "_pair"])


def _pvalue_file_threshold_selection(algorithm, algo_dir, predicted_signed):
    if algorithm == "PPCOR":
        pfile, alpha, pcol = algo_dir / "working_dir" / "outFile.txt", 0.01, "pValue"
    else:
        pfile, alpha, pcol = algo_dir / "working_dir" / "pvalues.txt", 0.01, "PValue"
    if not pfile.exists():
        return None
    pdf = pd.read_csv(pfile, sep="\t", header=0)
    pdf = pdf[pdf["Gene1"] != pdf["Gene2"]]
    selected_nat = set(zip(pdf.loc[pdf[pcol] < alpha, "Gene1"], pdf.loc[pdf[pcol] < alpha, "Gene2"]))
    sign_lookup = dict(zip(zip(predicted_signed["Gene1"], predicted_signed["Gene2"]), predicted_signed["_sign"]))
    selected_nat_signed = {(g1, g2, sign_lookup[(g1, g2)]) for (g1, g2) in selected_nat if (g1, g2) in sign_lookup}
    selected_nat_undirected = {frozenset((g1, g2)) for (g1, g2) in selected_nat}
    return selected_nat, selected_nat_signed, selected_nat_undirected, alpha, f"native_pvalue_{algorithm.lower()}"


def native_threshold_for_algorithm(algorithm, algo_dir, dataset_id, run_id, predicted_signed):
    if algorithm in ("PPCOR", "PEARSON"):
        return _pvalue_file_threshold_selection(algorithm, algo_dir, predicted_signed)
    return None


def score_beeline():
    """Scores every (dataset, run, algorithm) rankedEdges.csv under BEELINE_OUTPUT_ROOT.
    Run this AFTER run_mixed_network_sweep_benchmark.sh /
    run_mixed_network_sweep_genie3_grnboost2.sh have completed."""
    if not BEELINE_OUTPUT_ROOT.exists():
        raise FileNotFoundError(
            f"{BEELINE_OUTPUT_ROOT} doesn't exist yet -- run the BEELINE SLURM jobs first."
        )
    rows = []
    for dataset_dir in sorted(BEELINE_OUTPUT_ROOT.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name == "logs":
            continue
        dataset_id = dataset_dir.name
        if not (BEELINE_INPUT_ROOT / dataset_id / "GroundTruthNetwork.csv").exists():
            continue  # not a real dataset dir (e.g. a stray non-BEELINE folder)
        gt_df = load_beeline_ground_truth(dataset_id)
        possible_edges, true_edges = build_edge_universe(gt_df)
        true_signed_edges = build_signed_edge_universe(gt_df)
        possible_pairs_undirected, true_pairs_undirected = build_undirected_edge_universe(gt_df)
        num_true_edges = len(true_edges)
        num_true_edges_signed = len(true_signed_edges)
        num_true_edges_undirected = len(true_pairs_undirected)

        for run_dir in sorted(dataset_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            run_id = run_dir.name
            m = RUN_ID_RE.match(run_id)
            if not m:
                continue
            sim_rep, scheme = m.group(1), m.group(2)

            for algo_dir in sorted(run_dir.iterdir()):
                ranked_path = algo_dir / "rankedEdges.csv"
                if not ranked_path.exists():
                    continue
                algorithm = algo_dir.name
                ranked_edges = pd.read_csv(ranked_path, sep="\t", header=0)
                predicted = dedupe_predictions(ranked_edges)
                predicted_signed = add_predicted_sign(predicted)
                predicted_undirected = dedupe_predictions_undirected(ranked_edges)

                auprc = compute_auprc(ranked_edges, gt_df)
                auprc_signed = compute_auprc_signed(ranked_edges, gt_df)
                auprc_undirected = compute_auprc_undirected(ranked_edges, gt_df)

                selected_topk, _ = top_k_tie_aware_selection(predicted, num_true_edges)
                p_topk, r_topk, f1_topk = precision_recall_f1(selected_topk, true_edges)
                selected_topk_signed, _ = top_k_tie_aware_selection_signed(predicted_signed, num_true_edges_signed)
                p_topk_s, r_topk_s, f1_topk_s = precision_recall_f1(selected_topk_signed, true_signed_edges)
                selected_topk_und, _ = top_k_tie_aware_selection_undirected(predicted_undirected, num_true_edges_undirected)
                p_topk_u, r_topk_u, f1_topk_u = precision_recall_f1(selected_topk_und, true_pairs_undirected)

                native = native_threshold_for_algorithm(algorithm, algo_dir, dataset_id, run_id, predicted_signed)
                if native is not None:
                    selected_nat, selected_nat_signed, selected_nat_und, gmm_threshold, gmm_mode = native
                else:
                    gmm, gmm_threshold, gmm_mode = fit_gmm_threshold(predicted["_abs"].values)
                    selected_nat = gmm_natural_threshold_selection(predicted, gmm_threshold)
                    selected_nat_signed = gmm_natural_threshold_selection_signed(predicted_signed, gmm_threshold)
                    selected_nat_und = gmm_natural_threshold_selection_undirected(predicted_undirected, gmm_threshold)

                p_nat, r_nat, f1_nat = precision_recall_f1(selected_nat, true_edges)
                p_nat_s, r_nat_s, f1_nat_s = precision_recall_f1(selected_nat_signed, true_signed_edges)
                p_nat_u, r_nat_u, f1_nat_u = precision_recall_f1(selected_nat_und, true_pairs_undirected)

                rows.append({
                    "dataset_id": dataset_id, "run_id": run_id, "sim_rep": sim_rep, "scheme": scheme,
                    "algorithm": algorithm, "n_true_edges": num_true_edges,
                    "auprc": auprc, "auprc_signed": auprc_signed, "auprc_undirected": auprc_undirected,
                    "precision_topk": p_topk, "recall_topk": r_topk, "f1_topk": f1_topk,
                    "precision_topk_signed": p_topk_s, "recall_topk_signed": r_topk_s, "f1_topk_signed": f1_topk_s,
                    "precision_topk_undirected": p_topk_u, "recall_topk_undirected": r_topk_u, "f1_topk_undirected": f1_topk_u,
                    "gmm_threshold": gmm_threshold, "gmm_mode": gmm_mode,
                    "precision_natural": p_nat, "recall_natural": r_nat, "f1_natural": f1_nat,
                    "precision_natural_signed": p_nat_s, "recall_natural_signed": r_nat_s, "f1_natural_signed": f1_nat_s,
                    "precision_natural_undirected": p_nat_u, "recall_natural_undirected": r_nat_u, "f1_natural_undirected": f1_nat_u,
                })

    df = pd.DataFrame(rows)
    BEELINE_SCORES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(BEELINE_SCORES_CSV, index=False)
    print(f"Scored {len(df)} (dataset, run, algorithm) combination(s) -> {BEELINE_SCORES_CSV}")
    return df


if __name__ == "__main__":
    score_twinfer()
