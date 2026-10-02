# RECONSTRUCTED 2026-09-30 (AST extraction from benchmark_network_sweep.ipynb; cell number above each definition). The original /tmp/twinfer_scoring_extract.py is lost.
# UNREVIEWED: verify against analysis_data/synthetic_network_benchmark_20260824/ tables before relying on it. Where a name was defined in several cells the LAST definition is used.
from itertools import combinations
from itertools import permutations
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
import glob
import json
import numpy as np
import os
import pandas as pd
import re

# --- notebook cell 10 ---
def build_network_stem_lookup(network_dir, prefix="grn_n", suffix=".txt"):
    """Builds {topology_stem: full_ground_truth_path} by stripping prefix/suffix from
    each .txt filename in network_dir, e.g. 'grn_n6_e5_pos50_density_rep0.txt' ->
    stem 'e5_pos50_density_rep0'."""
    lookup = {}
    for f in glob.glob(os.path.join(network_dir, f"*{suffix}")):
        stem = os.path.basename(f)
        stem = re.sub(rf"^{prefix}\d+_", "", stem)
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
        lookup[stem] = f
    return lookup

# --- notebook cell 10 ---
def get_correct_gene_names(results):
    """Reconstructs gene names using TwINFER's actual internal naming convention
    ('gene_1', 'gene_2', ...), ignoring results['gene_names'] if it used the old
    'g1', 'g2', ... convention that doesn't match infer_with_twinfer's internal naming.
    """
    stored = results.get("gene_names")
    if stored and all(g.startswith("gene_") for g in stored):
        return stored
    n_genes = results["n_genes"]
    return [f"gene_{i+1}" for i in range(n_genes)]

# --- notebook cell 10 ---
def load_ground_truth_matrix(path_to_matrix, gene_names=None):
    """Loads a connectivity matrix and labels it with gene names for lookup."""
    matrix = np.loadtxt(path_to_matrix, dtype=int, delimiter=',')
    n = matrix.shape[0]
    if gene_names is None:
        gene_names = [f"gene_{i+1}" for i in range(n)]
    return pd.DataFrame(matrix, index=gene_names, columns=gene_names)

# --- notebook cell 10 ---
def match_topology_file(json_filename, stem_lookup):
    """Finds which topology stem appears as a substring within a result JSON's filename.
    Returns (matched_file_path or None, list_of_all_matching_stems); the longest
    (most specific) match wins if more than one stem matches."""
    matches = [stem for stem in stem_lookup if stem in json_filename]
    if not matches:
        return None, []
    best = max(matches, key=len)
    return stem_lookup[best], matches

# --- notebook cell 10 ---
def reconstruct_matrix(raw):
    """Reconstructs a labeled DataFrame from the same wrapper format for square matrices."""
    return pd.DataFrame(raw["data"], index=raw["index"], columns=raw["columns"])

# --- notebook cell 11 ---
def build_directed_true_edges(ground_truth_matrix, gene_names):
    """Set of directed (gi, gj) pairs with a nonzero (sign-blind) ground-truth edge."""
    return {
        (gi, gj)
        for gi in gene_names for gj in gene_names
        if gi != gj and ground_truth_matrix.loc[gi, gj] != 0
    }

# --- notebook cell 11 ---
def build_directed_true_edges_signed(ground_truth_matrix, gene_names):
    """Set of directed (gi, gj, sign) triples for every real ground-truth edge."""
    true_edges = set()
    for gi in gene_names:
        for gj in gene_names:
            if gi == gj:
                continue
            val = ground_truth_matrix.loc[gi, gj]
            if val != 0:
                true_edges.add((gi, gj, int(np.sign(val))))
    return true_edges

# --- notebook cell 11 ---
def build_true_pairs_undirected(ground_truth_matrix, gene_names):
    """Set of undirected (frozenset) gene pairs with a real edge in either direction --
    the true-edge side of the undirected/unsigned ('neither') variant."""
    true_pairs = set()
    for gi, gj in combinations(gene_names, 2):
        if ground_truth_matrix.loc[gi, gj] != 0 or ground_truth_matrix.loc[gj, gi] != 0:
            true_pairs.add(frozenset((gi, gj)))
    return true_pairs

# --- notebook cell 11 ---
def precision_recall_f1(selected_edges, true_edges):
    """Precision/recall/F1 of a selected edge set against a true edge set (both as sets
    of directed tuples, signed triples, or undirected frozensets -- generic over all
    three variant representations)."""
    if not selected_edges:
        return 0.0, 0.0, 0.0
    tp = len(selected_edges & true_edges)
    precision = tp / len(selected_edges)
    recall = tp / len(true_edges) if true_edges else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

# --- notebook cell 12 ---
def compute_twinfer_natural_undirected(ground_truth_matrix, gene_names, final_directed_edges):
    """
    Hard-decision precision/recall/F1 for TwINFER's natural threshold, undirected/unsigned
    ('neither') variant -- the natural-threshold counterpart to compute_twinfer_auprc_undirected.
    Collapses (gi, gj)/(gj, gi) into one frozenset position: predicted present if EITHER
    direction is in final_directed_edges, true if EITHER direction is nonzero in
    ground_truth_matrix.
    """
    predicted_edges = set(tuple(e) for e in final_directed_edges)
    true_pairs = build_true_pairs_undirected(ground_truth_matrix, gene_names)
    pred_pairs = set()
    for gi, gj in combinations(gene_names, 2):
        if (gi, gj) in predicted_edges or (gj, gi) in predicted_edges:
            pred_pairs.add(frozenset((gi, gj)))
    return precision_recall_f1(pred_pairs, true_pairs)

# --- notebook cell 12 ---
def find_confounded_pairs(ground_truth_matrix, gene_names):
    """
    Identifies gene pairs with no direct edge between them but sharing at least one
    common regulator (fan-out / feed-forward fork: gk->gi and gk->gj). Such pairs are
    expected to show correlation from shared upstream regulation even though they are
    not truly regulatory edges, so they are scored as an expected multi-state case
    rather than a straightforward false positive/negative.

    Pairs with a real direct edge between them -- including true mutual (bidirectional)
    regulation -- are excluded, since genuine regulation should be scored normally, not
    treated as a confound.
    """
    confounded_pairs = set()
    n = len(gene_names)
    for idx_i in range(n):
        for idx_j in range(idx_i + 1, n):
            gi, gj = gene_names[idx_i], gene_names[idx_j]
            if ground_truth_matrix.loc[gi, gj] != 0 and ground_truth_matrix.loc[gj, gi] != 0:
                continue  # real direct edge (incl. mutual) -- not a confound case
            shared_regulator = any(
                ground_truth_matrix.loc[gk, gi] != 0 and ground_truth_matrix.loc[gk, gj] != 0
                for gk in gene_names if gk not in (gi, gj)
            )
            if shared_regulator:
                confounded_pairs.add(frozenset([gi, gj]))
    return confounded_pairs

# --- notebook cell 12 ---
def score_directed_edges(ground_truth_matrix, gene_names, final_directed_edges,
                          multiple_states, direction_matrix, wrong_sign_counts_as=("FP", "FN")):
    """
    TwINFER's natural-threshold precision/recall/F1 against the ground truth, at two
    strictness levels computed side by side: sign-blind (directed+unsigned) and directed
    AND signed (strictest). Confounded (fan-out-in-the-ground-truth) pairs are scored
    leniently as TN if flagged multi-state, matching find_confounded_pairs.
    """
    predicted_edges = set(tuple(e) for e in final_directed_edges)
    confounded_pairs = find_confounded_pairs(ground_truth_matrix, gene_names)
    scored_confounded = set()

    TP = FP = FN = TN = 0
    TP_signed = FP_signed = FN_signed = WRONG_SIGN = 0
    confounded_correct = 0

    def _sign_match(a, b):
        gt_sign = int(np.sign(ground_truth_matrix.loc[a, b]))
        pred_sign_val = direction_matrix.loc[a, b]
        pred_sign = int(np.sign(pred_sign_val)) if pred_sign_val != 0 else 0
        return pred_sign == gt_sign

    for gi in gene_names:
        for gj in gene_names:
            if gi == gj:
                continue

            pair_fs = frozenset([gi, gj])

            # if pair_fs in confounded_pairs:
            #     continue
                # if pair_fs in scored_confounded:
                #     continue
                # scored_confounded.add(pair_fs)

                # is_flagged_multistate = ([gi, gj] in multiple_states or [gj, gi] in multiple_states)
                # if is_flagged_multistate:
                #     TN += 2
                #     confounded_correct += 1
                #     continue

                # has_gi_gj = (gi, gj) in predicted_edges
                # has_gj_gi = (gj, gi) in predicted_edges
                # gt_gi_gj = ground_truth_matrix.loc[gi, gj] != 0
                # gt_gj_gi = ground_truth_matrix.loc[gj, gi] != 0

                # if not gt_gi_gj and not gt_gj_gi:
                #     if has_gi_gj and has_gj_gi:
                #         FP += 2
                #         FP_signed += 2
                #     elif has_gi_gj or has_gj_gi:
                #         FP += 1
                #         FP_signed += 1
                #         TN += 1
                #     else:
                #         TN += 2
                # else:
                #     real_a, real_b = (gi, gj) if gt_gi_gj else (gj, gi)
                #     real_pred = (real_a, real_b) in predicted_edges
                #     fake_pred = (real_b, real_a) in predicted_edges

                #     if real_pred:
                #         TP += 1
                #     else:
                #         FN += 1

                #     if real_pred and _sign_match(real_a, real_b):
                #         TP_signed += 1
                #     elif real_pred:
                #         WRONG_SIGN += 1
                #         if "FP" in wrong_sign_counts_as:
                #             FP_signed += 1
                #         if "FN" in wrong_sign_counts_as:
                #             FN_signed += 1
                #     else:
                #         FN_signed += 1

                #     if fake_pred:
                #         FP += 1
                #         FP_signed += 1
                #     else:
                #         TN += 1
                # continue

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
        "confounded_pairs": confounded_pairs,
        "confounded_pairs_correctly_found": confounded_correct,
    }

# --- notebook cell 13 ---
def build_master_ranked_table(unfiltered_direction_matrix, gene_names):
    """
    Full n*(n-1) directed-pair table built directly from unfiltered_direction_matrix (raw
    correlation magnitude, always available for every pair, never mutated by fan-out
    removal). This table is the "all_edges" universe -- the loosest of TwINFER's 4 nested
    top-k candidate pools -- and is also what AUPRC is computed against, for a fair,
    threshold-free comparison against the other methods.
    """
    rows = []
    for gi in gene_names:
        for gj in gene_names:
            if gi == gj:
                continue
            corr = unfiltered_direction_matrix.loc[gi, gj]
            rows.append({"gene_1": gi, "gene_2": gj, "directional_correlation": corr})
    return pd.DataFrame(rows)

# --- notebook cell 13 ---
def build_potential_regulation_bidirectional_pairs(potential_regulation):
    """
    Reconstructs TwINFER's "all-potential-regulation" mode candidate-pair set
    (bidirectional_pairs | self_pairs, see infer_with_twinfer.py lines 432-442) from the
    JSON's potential_regulation list. Needed because the rerun used "all-edges" mode
    (testing every possible pair for directed significance, unconditional on Step 2's
    own gene-gene correlation threshold), which is a strictly looser test than
    "all-potential-regulation" mode. TwINFER's actual "first test" (per clarification)
    means: directionally significant AND part of potential_regulation -- so
    pass_first_test = pre_fanout_edges & this set, not pre_fanout_edges alone.
    """
    pairs = [tuple(p) for p in potential_regulation]
    bidirectional = {(a, b) for a, b in pairs} | {(b, a) for a, b in pairs}
    genes = {g for pair in pairs for g in pair}
    self_pairs = {(g, g) for g in genes}
    return bidirectional | self_pairs

# --- notebook cell 13 ---
def reconstruct_pre_fanout_edges(final_directed_edges_post, fan_out_log):
    """
    Reconstructs the pre-fan-out final_directed_edges set (i.e. TwINFER's "pass first
    test" pool -- the shuffled-null significance test result, before fan-out separation
    ran) by adding back both directions for every gene pair fan_out_log recorded as a
    "fan_out" decision. separate_fan_outs_from_mutual_regulation only ever discards
    edges (never adds any), so pre-fan-out is exactly post-fan-out plus these.
    """
    pre = set(tuple(e) for e in final_directed_edges_post)
    for entry in (fan_out_log or []):
        if entry.get("decision") == "fan_out":
            a, b = entry["gene_a"], entry["gene_b"]
            pre.add((a, b))
            pre.add((b, a))
    return pre

# --- notebook cell 13 ---
def restrict_ranked_list(ranked_edge_list, edge_set):
    """Restricts a gene_1/gene_2/directional_correlation table to rows whose
    (gene_1, gene_2) is in edge_set."""
    if ranked_edge_list is None or ranked_edge_list.empty:
        return ranked_edge_list
    mask = [(g1, g2) in edge_set for g1, g2 in zip(ranked_edge_list["gene_1"], ranked_edge_list["gene_2"])]
    return ranked_edge_list[mask].reset_index(drop=True)

# --- notebook cell 13 ---
def restrict_to_single_state_pairs(ranked_edge_list, single_state_regulation_pairs):
    """Further restricts to rows whose unordered gene pair is in single_state_regulation_pairs
    (gene_lists.single_state_regulation, i.e. NOT flagged heterogeneous/multi-state,
    z < 10 -- see differentiate_single_state_reg_and_multiple_states)."""
    if ranked_edge_list is None or ranked_edge_list.empty:
        return ranked_edge_list
    single_state_fs = {frozenset(p) for p in single_state_regulation_pairs}
    mask = [
        frozenset((g1, g2)) in single_state_fs
        for g1, g2 in zip(ranked_edge_list["gene_1"], ranked_edge_list["gene_2"])
    ]
    return ranked_edge_list[mask].reset_index(drop=True)

# --- notebook cell 14 ---
def compute_twinfer_auprc(ranked_edge_list, ground_truth_matrix, gene_names):
    """
    AUPRC for TwINFER's ranked table against a ground-truth connectivity matrix. Score =
    |directional_correlation|: cross-correlation magnitude is the
    primary ranking criterion. Always called with the full
    "all_edges" master table (build_master_ranked_table) here, so every possible pair has
    a row -- no fallback-to-zero scoring needed for absent pairs.
    """
    possible_edges = list(permutations(gene_names, 2))
    true_labels = [1 if ground_truth_matrix.loc[gi, gj] != 0 else 0 for gi, gj in possible_edges]
    if sum(true_labels) == 0:
        return float("nan")
    if ranked_edge_list is None or ranked_edge_list.empty:
        return float("nan")

    score_lookup = {
        (row.gene_1, row.gene_2): abs(row.directional_correlation)
        for row in ranked_edge_list.itertuples()
    }
    pred_scores = [score_lookup.get(e, 0.0) for e in possible_edges]
    precision, recall, _ = precision_recall_curve(true_labels, pred_scores)
    return float(auc(recall, precision))

# --- notebook cell 14 ---
def compute_twinfer_auprc_signed(ranked_edge_list, ground_truth_matrix, gene_names):
    """AUPRC over directed AND signed edges -- ground_truth_matrix is signed (+1/-1/0);
    directional_correlation's sign is the predicted sign. A wrong-signed prediction
    scores as an explicit false positive at that sign-position; the true signed position
    gets zero credit."""
    possible_edges = list(permutations(gene_names, 2))
    possible_signed_positions = [(gi, gj, s) for gi, gj in possible_edges for s in (1, -1)]
    true_labels = [
        1 if int(np.sign(ground_truth_matrix.loc[gi, gj])) == s else 0
        for gi, gj, s in possible_signed_positions
    ]
    if sum(true_labels) == 0:
        return float("nan")
    if ranked_edge_list is None or ranked_edge_list.empty:
        return float("nan")

    score_lookup = {
        (row.gene_1, row.gene_2): (
            abs(row.directional_correlation),
            int(np.sign(row.directional_correlation)),
        )
        for row in ranked_edge_list.itertuples()
    }
    pred_scores = []
    for gi, gj, s in possible_signed_positions:
        entry = score_lookup.get((gi, gj))
        pred_scores.append(entry[0] if entry is not None and entry[1] == s else 0.0)

    precision, recall, _ = precision_recall_curve(true_labels, pred_scores)
    return float(auc(recall, precision))

# --- notebook cell 14 ---
def compute_twinfer_auprc_undirected(ranked_edge_list, ground_truth_matrix, gene_names):
    """AUPRC over undirected, unsigned edges: (gi, gj)/(gj, gi) collapse into one
    frozenset position, true if EITHER direction is nonzero in ground truth, credited
    with whichever of the two predicted directions scored higher."""
    unordered_pairs = list(combinations(gene_names, 2))
    true_labels = [
        1 if (ground_truth_matrix.loc[gi, gj] != 0 or ground_truth_matrix.loc[gj, gi] != 0) else 0
        for gi, gj in unordered_pairs
    ]
    if sum(true_labels) == 0:
        return float("nan")
    if ranked_edge_list is None or ranked_edge_list.empty:
        return float("nan")

    score_lookup = {}
    for row in ranked_edge_list.itertuples():
        score = abs(row.directional_correlation)
        pair = frozenset((row.gene_1, row.gene_2))
        if pair not in score_lookup or score > score_lookup[pair]:
            score_lookup[pair] = score

    pred_scores = [score_lookup.get(frozenset(p), 0.0) for p in unordered_pairs]
    precision, recall, _ = precision_recall_curve(true_labels, pred_scores)
    return float(auc(recall, precision))

# --- notebook cell 15 ---
def top_k_tie_aware_selection_twinfer(ranked_edge_list, num_true_edges):
    """
    Selects the top-k predicted directed edges (k = num_true_edges) from a
    gene_1/gene_2/directional_correlation, expanded to include all ties at
    the boundary score. Works on any of the 4 nested candidate pools (all_edges,
    pass_first_test, not_heterogeneous, fan_out_removed) -- the pool restriction happens
    before this is called, not inside it.
    """
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

# --- notebook cell 15 ---
def top_k_tie_aware_selection_twinfer_signed(ranked_edge_list, num_true_edges_signed):
    """Signed counterpart: same magnitude-only ranking/threshold, but returns
    (gene_1, gene_2, predicted_sign) triples."""
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

# --- notebook cell 15 ---
def top_k_tie_aware_selection_twinfer_undirected(ranked_edge_list, num_true_pairs_undirected):
    """
    Undirected counterpart (NEW -- no equivalent existed on the TwINFER side before):
    collapses (gi, gj)/(gj, gi) into one frozenset position, keeping the higher-scoring
    direction, then applies the same tie-aware top-k boundary logic.
    """
    if ranked_edge_list is None or ranked_edge_list.empty or num_true_pairs_undirected == 0:
        return set(), float("nan")

    scored = ranked_edge_list.copy()
    scored["_score"] = scored["directional_correlation"].abs()
    scored["_pair"] = [frozenset(p) for p in zip(scored["gene_1"], scored["gene_2"])]
    scored = (
        scored.sort_values("_score", ascending=False)
        .drop_duplicates(subset=["_pair"])
        .reset_index(drop=True)
    )

    maxk = min(len(scored), num_true_pairs_undirected)
    edge_score_topk = float(scored.iloc[maxk - 1]["_score"])
    nonzero = scored.loc[scored["_score"] > 0, "_score"]
    non_zero_min = float(nonzero.min()) if not nonzero.empty else 0.0

    best_val = max(non_zero_min, edge_score_topk)
    selected = scored[scored["_score"] >= best_val]
    return set(selected["_pair"]), best_val

# --- notebook cell 16 ---
TOPK_POOL_NAMES = ["all_edges", "pass_first_test", "not_heterogeneous", "fan_out_removed"]

# --- notebook cell 16 ---
def score_twinfer_dataset(results, gene_names, ground_truth_matrix):
    """
    Scores one TwINFER fan-out-rerun result dict against its matching ground-truth
    matrix: AUPRC (3 variants, all_edges pool), natural threshold (3 variants x 2
    fan-out stages -- before_fan_out/after_fan_out, TwINFER's own hard decision with
    the fan-out flag off vs on), and top-k (3 variants x 4 nested candidate pools,
    strictly nested: all_edges superset pass_first_test superset not_heterogeneous
    superset fan_out_removed). Returns a single flat dict (one row).
    """
    # [2026-09-30 added: accept the current nested infer_with_twinfer output as well as the old flat one; identity for old-schema results]
    from benchmarks.network_benchmarks.score.twinfer_result_adapter import ensure_old_schema
    results = ensure_old_schema(results, gene_names)
    unfiltered_direction_matrix = reconstruct_matrix(results["unfiltered_direction_matrix"])
    direction_matrix = reconstruct_matrix(results["direction_matrix"])
    final_directed_edges_post = set(tuple(e) for e in results["final_directed_edges"])
    fan_out_log = results.get("fan_out_log")
    single_state_pairs = [tuple(p) for p in results["gene_lists"]["single_state_regulation"]]
    mult_list = results["gene_lists"]["multiple_states_and_reg"] + results["gene_lists"]["multiple_states_no_reg"]

    master_table = build_master_ranked_table(unfiltered_direction_matrix, gene_names)
    pre_fanout_edges = reconstruct_pre_fanout_edges(final_directed_edges_post, fan_out_log)

    # --- Top-k's 4 nested candidate pools (each a strict subset of the previous) ---
    # "pass_first_test" is TwINFER's "all-potential-regulation" mode result: directionally
    # significant AND part of potential_regulation (Step 2's own gene-gene correlation
    # threshold) -- NOT simply "significant under all-edges mode" (which tests every
    # possible pair regardless of Step 2's threshold, and is a looser criterion that
    # can include pairs never evaluated for heterogeneity at all).
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
        "sim_type": results.get("sim_type"),
        "rep_id": results.get("rep_id"),
        "n_genes": results.get("n_genes"),
        "n_fan_out_pairs": sum(1 for e in (fan_out_log or []) if e.get("decision") == "fan_out"),
        "n_mutual_regulation_pairs": sum(1 for e in (fan_out_log or []) if e.get("decision") == "mutual_regulation"),
    }

    # --- AUPRC (rank-based, all_edges pool only) ---
    row["auprc"] = compute_twinfer_auprc(master_table, ground_truth_matrix, gene_names)
    row["auprc_signed"] = compute_twinfer_auprc_signed(master_table, ground_truth_matrix, gene_names)
    row["auprc_undirected"] = compute_twinfer_auprc_undirected(master_table, ground_truth_matrix, gene_names)

    # --- Natural threshold: TwINFER's own hard decision, before vs after fan-out removal.
    # "before_fan_out" = pre_fanout_edges (what TwINFER decides with the fan-out flag off);
    # "after_fan_out" = final_directed_edges_post (what it actually decided with the flag
    # on -- the JSON's own final answer). This directly answers "how does TwINFER do with
    # vs without fan-out removal" at the level of its actual hard decision, independent of
    # the top-k pools' extra potential_regulation/heterogeneity restriction above.
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

    # --- Top-k, 4 nested pools x 3 variants ---
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

# --- notebook cell 17 ---
_RETRY_RE = re.compile(r"^(?P<stem>.+?)_rep(?P<techrep>\d+)_(?P<ts>\d{8}_\d{6})_rep_None_all_results\.json$")

# --- notebook cell 17 ---
def _dedupe_reruns(files):
    """Collapses repeated inference reruns of the same (topology, technical-replicate)
    slot down to one file each -- the most recent by timestamp.

    Result filenames encode a technical-replicate index and a generation timestamp
    (e.g. "e17_pos100_density_rep1_rep3_19072026_034125_..."), and some slots were
    re-inferred multiple times across several days without the earlier attempt's
    output ever being removed. The "skip if output JSON already exists" resume logic
    never caught this, because each rerun's filename embeds a fresh timestamp rather
    than reusing the same name. Left undeduplicated, a handful of retried slots
    silently dominate the mean -- grn_n6_e17_pos100_density_rep1 alone had 46 files
    behind 10 true technical replicates, 44% of the full 250-file sweep in one topology.
    Files that don't match the expected naming are passed through unchanged (kept, not
    deduplicated) rather than silently dropped.
    """
    latest = {}
    passthrough = []
    for jf in files:
        m = _RETRY_RE.match(os.path.basename(jf))
        if not m:
            passthrough.append(jf)
            continue
        slot = (m["stem"], m["techrep"])
        if slot not in latest or m["ts"] > _RETRY_RE.match(os.path.basename(latest[slot]))["ts"]:
            latest[slot] = jf
    return sorted(latest.values()) + passthrough

# --- notebook cell 17 ---
def score_twinfer_sweep_folder(json_dir, network_dir, json_pattern="*_all_results.json", verbose=False):
    """Scores every fan-out-rerun result JSON in json_dir against its matching
    ground-truth topology file in network_dir, after de-duplicating reruns (see
    _dedupe_reruns) so every dataset_id gets equal weight -- up to 10 technical
    replicates each, the sweep's intended design. One dataset_id
    (grn_n6_e17_pos100_density_rep2) tops out at 8 regardless: techreps 5 and 6 have
    no result file at all under any timestamp, not just a retry-vs-original question.
    """
    stem_lookup = build_network_stem_lookup(network_dir)
    if verbose:
        print(f"Found {len(stem_lookup)} ground-truth topology file(s) in {network_dir}")

    all_files = sorted(glob.glob(os.path.join(json_dir, json_pattern)))
    files_to_score = _dedupe_reruns(all_files)
    n_dropped = len(all_files) - len(files_to_score)
    if n_dropped:
        print(f"De-duplicated {len(all_files)} result file(s) down to {len(files_to_score)} "
              f"(dropped {n_dropped} older rerun(s) of an already-inferred (topology, technical-replicate) slot).")

    records = []
    unmatched, ambiguous = [], []
    for jf in files_to_score:
        json_name = os.path.basename(jf)
        matrix_path, matches = match_topology_file(json_name, stem_lookup)
        if matrix_path is None:
            unmatched.append(json_name)
            continue
        if len(matches) > 1:
            ambiguous.append((json_name, matches))

        with open(jf) as f:
            results = json.load(f)
        gene_names = get_correct_gene_names(results)
        gt = load_ground_truth_matrix(matrix_path, gene_names=gene_names)

        row = score_twinfer_dataset(results, gene_names, gt)
        row["json_file"] = json_name
        row["matched_topology_file"] = os.path.basename(matrix_path)
        row["dataset_id"] = os.path.basename(matrix_path).replace(".txt", "")
        records.append(row)

        if verbose:
            print(f"  {json_name}  ->  {os.path.basename(matrix_path)}")

    if ambiguous:
        print(f"WARNING: {len(ambiguous)} file(s) matched MORE THAN ONE topology (used longest match -- verify):")
        for name, matches in ambiguous:
            print(f"   {name}: {matches}")
    if unmatched:
        print(f"WARNING: {len(unmatched)} file(s) had NO matching topology file (skipped):")
        for name in unmatched:
            print(f"   {name}")

    result_df = pd.DataFrame(records)
    if not result_df.empty:
        rep_counts = result_df["dataset_id"].value_counts()
        short = rep_counts[rep_counts < rep_counts.max()]
        if not short.empty:
            print(f"NOTE: not every dataset_id reaches {rep_counts.max()} technical replicates even "
                  f"after de-duplication -- these are missing result files, not just retries:")
            print(short.to_string())

    return result_df
