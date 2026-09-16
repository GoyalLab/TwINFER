"""
Metrics summary table for the 8 real networks (GSD, HSC, VSC, mCAD,
B_cell_activation, Circadian_cycle, EMT, Pluripotent) -- TwINFER vs. the 7
BEELINE methods (GENIE3, GRNBOOST2, PPCOR, SCODE, SCSGL, PEARSON, PIDC).

Scoring machinery (score_twinfer_dataset and its full dependency chain,
score_beeline_sweep_folder and its full dependency chain, melt_twinfer_wide/
melt_beeline_wide, build_summary_table) is ported VERBATIM from
benchmark_network_sweep.ipynb -- same convention explicitly requested, to avoid
silently diverging from its already-debugged scoring logic. Only the file-
discovery layer below (REAL_NETWORKS, the two BEELINE input/output root pairs)
is new, since the network_sweep notebook's own file-discovery is built around
a different dataset (topology x replicate sweep, name-matched by substring)
that doesn't apply here -- these 8 real-network JSON/rankedEdges filenames are
already explicit.

Two BEELINE input/output root pairs are needed since the 8 networks split into
two on-disk conventions:
  - GSD/HSC/VSC/mCAD            : inputs under code/Beeline/inputs/<net>/,
                                   results under analysis_data/real_networks/beeline_inference/
  - B_cell_activation/Circadian_cycle/EMT/Pluripotent :
                                   inputs under code/Beeline/inputs/real_networks/<net>/,
                                   results under analysis_data/paper_analysis/real_networks/beeline_inference/

TwINFER's fan-out-on JSONs (produced by infer_real_networks_extra.py) are read
directly from /scratch/gzu5140/twinfer_nb/fanout_on/ for all 8 networks --
"after_fan_out" is TwINFER's actual final hard decision (fan-out flag on),
matching melt_twinfer_wide's convention in the source notebook.

Run under the twinfer-code conda env (needs numba, via SCSGL's native
permutation-threshold import) -- see run_real_networks_summary_table.sh.
"""
import glob
import json
import os
import re
import sys
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm
from sklearn.metrics import auc, precision_recall_curve
from sklearn.mixture import GaussianMixture

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer_real_networks_extra import NETWORKS  # noqa: E402

PROJECT_ROOT = Path("/home/gzu5140/TwINFER_KA")
GROUND_TRUTH_TOPOLOGY_DIR = PROJECT_ROOT / "input_data" / "real_world_networks"
FANOUT_ON_DIR = Path("/scratch/gzu5140/twinfer_nb/fanout_on")

BEELINE_ROOT = PROJECT_ROOT / "code" / "Beeline"
_SCSGL_PYSRC_DIR = BEELINE_ROOT / "Algorithms" / "SCSGL" / "scSGL"
if str(_SCSGL_PYSRC_DIR) not in sys.path:
    sys.path.append(str(_SCSGL_PYSRC_DIR))
from pysrc.associations.correlation import permutations as scsgl_correlation_permutations  # noqa: E402

# Two (BEELINE_INPUT_ROOT, BEELINE_OUTPUT_ROOT, [nets]) groups -- see module docstring.
BEELINE_GROUPS = [
    (
        BEELINE_ROOT / "inputs",
        PROJECT_ROOT / "analysis_data" / "real_networks" / "beeline_inference",
        ["GSD", "HSC", "VSC", "mCAD"],
    ),
    (
        BEELINE_ROOT / "inputs" / "real_networks",
        PROJECT_ROOT / "analysis_data" / "paper_analysis" / "real_networks" / "beeline_inference",
        ["B_cell_activation", "Circadian_cycle", "EMT", "Pluripotent"],
    ),
]

OUT_DIR = PROJECT_ROOT / "analysis_data" / "paper_analysis" / "real_networks"
TWINFER_SCORES_CSV = OUT_DIR / "twinfer_scores.csv"
BEELINE_SCORES_CSV = OUT_DIR / "beeline_scores.csv"
SUMMARY_TABLE_CSV = OUT_DIR / "summary_metrics_table.csv"


# ============================================================================
# Part 2 (ported verbatim from benchmark_network_sweep.ipynb) -- TwINFER scoring
# ============================================================================
def reconstruct_ranked_edge_list(raw):
    if isinstance(raw, dict) and raw.get("__type__") == "DataFrame":
        return pd.DataFrame(raw["data"], columns=raw["columns"])
    return None


def reconstruct_matrix(raw):
    return pd.DataFrame(raw["data"], index=raw["index"], columns=raw["columns"])


def load_ground_truth_matrix(path_to_matrix, gene_names=None):
    matrix = np.loadtxt(path_to_matrix, dtype=int, delimiter=',')
    n = matrix.shape[0]
    if gene_names is None:
        gene_names = [f"gene_{i+1}" for i in range(n)]
    return pd.DataFrame(matrix, index=gene_names, columns=gene_names)


def get_correct_gene_names(results):
    stored = results.get("gene_names")
    if stored and all(g.startswith("gene_") for g in stored):
        return stored
    n_genes = results["n_genes"]
    return [f"gene_{i+1}" for i in range(n_genes)]


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
        (gi, gj)
        for gi in gene_names for gj in gene_names
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
    confounded_correct = 0

    TP = FP = FN = TN = 0
    TP_signed = FP_signed = FN_signed = WRONG_SIGN = 0

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
        "confounded_pairs": confounded_pairs,
        "confounded_pairs_correctly_found": confounded_correct,
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
    """NaN entries (observed on HSC: gene_6/gene_7 have exactly zero variance in the raw
    simulation, making their Pearson correlation with every other gene undefined) are
    filled with 0 -- "no evidence of an edge". This matches how TwINFER's own
    ranked_edge_list handles the same pairs: it drops them entirely, and every AUPRC/
    top-k score lookup elsewhere in this pipeline already defaults an absent pair to
    0.0 (see e.g. ka_metrics.py's load_twinfer_ranked_edges + compute_auprc). Kept as
    an explicit fillna (not a dropped row) because build_master_ranked_table must stay
    the full n*(n-1) table for the pool-membership logic (restrict_ranked_list /
    restrict_to_single_state_pairs) to work -- unlike ranked_edge_list, which the
    source notebook deliberately avoids here since it's also missing rows for
    fan-out-removed pairs, which would corrupt those same pools."""
    rows = []
    for gi in gene_names:
        for gj in gene_names:
            if gi == gj:
                continue
            corr = unfiltered_direction_matrix.loc[gi, gj]
            rows.append({"gene_1": gi, "gene_2": gj, "directional_correlation": corr})
    df = pd.DataFrame(rows)
    df["directional_correlation"] = df["directional_correlation"].fillna(0.0)
    return df


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


def compute_twinfer_auprc(ranked_edge_list, ground_truth_matrix, gene_names):
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


def compute_twinfer_auprc_signed(ranked_edge_list, ground_truth_matrix, gene_names):
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


def compute_twinfer_auprc_undirected(ranked_edge_list, ground_truth_matrix, gene_names):
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


TOPK_POOL_NAMES = ["all_edges", "pass_first_test", "not_heterogeneous", "fan_out_removed"]
NATURAL_FANOUT_STAGES = ["before_fan_out", "after_fan_out"]


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
        "sim_type": results.get("sim_type"),
        "rep_id": results.get("rep_id"),
        "n_genes": results.get("n_genes") or len(gene_names),
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


# ============================================================================
# Real-networks file discovery (new -- network_sweep's substring-matching
# discovery doesn't apply to these 8 explicitly-named datasets)
# ============================================================================
def score_real_network_twinfer():
    records = []
    for net, cfg in NETWORKS.items():
        gt_path = GROUND_TRUTH_TOPOLOGY_DIR / cfg["topology"]
        json_files = sorted(glob.glob(str(FANOUT_ON_DIR / f"{net}_rep_*_all_results.json")))
        for jf in json_files:
            with open(jf) as f:
                results = json.load(f)
            gene_names = get_correct_gene_names(results)
            gt = load_ground_truth_matrix(gt_path, gene_names=gene_names)
            row = score_twinfer_dataset(results, gene_names, gt)
            row["dataset_id"] = net
            row["json_file"] = os.path.basename(jf)
            records.append(row)
        print(f"[twinfer] {net}: scored {len(json_files)} replicate(s)")
    return pd.DataFrame(records)


# ============================================================================
# Part 3 (ported verbatim from benchmark_network_sweep.ipynb) -- BEELINE scoring
# ============================================================================
RUN_ID_RE = re.compile(r"^simrep([0-9a-f]+)_(spread|twin_paired)$")


def load_beeline_ground_truth(beeline_input_root: Path, dataset_id: str) -> pd.DataFrame:
    gt_path = beeline_input_root / dataset_id / "GroundTruthNetwork.csv"
    if not gt_path.exists():
        gt_path = beeline_input_root / "GroundTruthNetwork.csv"
    return pd.read_csv(gt_path, header=0)


def build_edge_universe(gt_df: pd.DataFrame):
    gt_no_self = gt_df[gt_df["Gene1"] != gt_df["Gene2"]].drop_duplicates()
    unique_nodes = sorted(set(gt_df["Gene1"]).union(set(gt_df["Gene2"])))
    possible_edges = set(permutations(unique_nodes, 2))
    true_edges = set(zip(gt_no_self["Gene1"], gt_no_self["Gene2"])) & possible_edges
    return possible_edges, true_edges


def dedupe_predictions(ranked_edges: pd.DataFrame) -> pd.DataFrame:
    """NaN EdgeWeight (observed on Pluripotent: PPCOR's partial-correlation matrix
    inversion produces NaN for every pair touching gene_22, which has exactly zero
    variance -- same root cause as HSC's gene_6/gene_7 in build_master_ranked_table
    above) is filled with 0 -- "no evidence of an edge" -- same reasoning as there."""
    pred = ranked_edges[ranked_edges["Gene1"] != ranked_edges["Gene2"]].copy()
    pred["_abs"] = pred["EdgeWeight"].abs().fillna(0.0)
    return (
        pred.sort_values("_abs", ascending=False)
        .drop_duplicates(subset=["Gene1", "Gene2"])
        .reset_index(drop=True)
    )


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

    pred = dedupe_predictions(ranked_edges)
    pred = add_predicted_sign(pred)
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
    """See dedupe_predictions above -- same NaN -> 0 fill."""
    pred = ranked_edges[ranked_edges["Gene1"] != ranked_edges["Gene2"]].copy()
    pred["_abs"] = pred["EdgeWeight"].abs().fillna(0.0)
    pred["_pair"] = [frozenset(p) for p in zip(pred["Gene1"], pred["Gene2"])]
    return (
        pred.sort_values("_abs", ascending=False)
        .drop_duplicates(subset=["_pair"])
        .reset_index(drop=True)
    )


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

    weights = gmm.weights_
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())

    order = np.argsort(means)
    w1, w2 = weights[order]
    m1, m2 = means[order]
    s1, s2 = stds[order]

    f = lambda z: w1 * norm.pdf(z, m1, s1) - w2 * norm.pdf(z, m2, s2)  # noqa: E731
    try:
        threshold = brentq(f, m1, m2)
        return gmm, threshold, "GMM_intersection"
    except ValueError:
        return gmm, np.inf, "GMM_no_intersection"


def gmm_natural_threshold_selection(predicted: pd.DataFrame, threshold: float):
    if predicted.empty:
        return set()
    selected = predicted[predicted["_abs"] >= threshold]
    return set(zip(selected["Gene1"], selected["Gene2"]))


def gmm_natural_threshold_selection_signed(predicted_signed: pd.DataFrame, threshold: float):
    if predicted_signed.empty:
        return set()
    selected = predicted_signed[predicted_signed["_abs"] >= threshold]
    return set(zip(selected["Gene1"], selected["Gene2"], selected["_sign"]))


def gmm_natural_threshold_selection_undirected(predicted_undirected: pd.DataFrame, threshold: float):
    if predicted_undirected.empty:
        return set()
    selected = predicted_undirected[predicted_undirected["_abs"] >= threshold]
    return set(selected["_pair"])


def _signed_undirected_from_pairs(selected_nat, predicted_signed):
    sign_lookup = dict(zip(zip(predicted_signed["Gene1"], predicted_signed["Gene2"]), predicted_signed["_sign"]))
    selected_nat_signed = {
        (g1, g2, sign_lookup[(g1, g2)]) for (g1, g2) in selected_nat if (g1, g2) in sign_lookup
    }
    selected_nat_undirected = {frozenset((g1, g2)) for (g1, g2) in selected_nat}
    return selected_nat_signed, selected_nat_undirected


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

    selected_nat_signed, selected_nat_undirected = _signed_undirected_from_pairs(selected_nat, predicted_signed)
    mode = f"native_pvalue_{algorithm.lower()}"
    return selected_nat, selected_nat_signed, selected_nat_undirected, alpha, mode


def _scsgl_permutation_selection(beeline_input_root, dataset_id, run_id, predicted_signed, alpha=0.05, k=100):
    expr_path = beeline_input_root / dataset_id / run_id / "ExpressionData.csv"
    if not expr_path.exists():
        return None
    expr_df = pd.read_csv(expr_path, header=0, index_col=0)
    genes = expr_df.index.tolist()
    counts = expr_df.to_numpy()

    thresholds = scsgl_correlation_permutations(
        counts, k=k, tau_neg=[100 * alpha / 2], tau_pos=[100 * (1 - alpha / 2)]
    )
    lower, upper = thresholds[0]

    corr = np.corrcoef(counts)
    n = len(genes)
    selected_nat = {
        (genes[i], genes[j])
        for i in range(n) for j in range(n)
        if i != j and (corr[i, j] <= lower or corr[i, j] >= upper)
    }
    selected_nat_signed, selected_nat_undirected = _signed_undirected_from_pairs(selected_nat, predicted_signed)
    return selected_nat, selected_nat_signed, selected_nat_undirected, (float(lower), float(upper)), "native_permutation_scsgl"


def native_threshold_for_algorithm(beeline_input_root, algorithm, algo_dir, dataset_id, run_id, predicted_signed):
    if algorithm in ("PPCOR", "PEARSON"):
        return _pvalue_file_threshold_selection(algorithm, algo_dir, predicted_signed)
    if algorithm == "SCSGL":
        return _scsgl_permutation_selection(beeline_input_root, dataset_id, run_id, predicted_signed)
    return None


def score_beeline_sweep_folder(beeline_input_root: Path, output_root: Path, nets=None, verbose=False):
    """Same as benchmark_network_sweep.ipynb's score_beeline_sweep_folder, parametrized
    over beeline_input_root/output_root instead of module-level globals (needed here
    since the 8 real networks split across two different root pairs -- see module
    docstring). `nets`, if given, restricts to those dataset_ids (skips stray dirs
    like beeline_inference/logs/)."""
    rows = []
    skipped_gt = []

    for dataset_dir in sorted(output_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset_id = dataset_dir.name
        if nets is not None and dataset_id not in nets:
            continue

        gt_path = beeline_input_root / dataset_id / "GroundTruthNetwork.csv"
        if not gt_path.exists():
            skipped_gt.append(dataset_id)
            continue

        gt_df = load_beeline_ground_truth(beeline_input_root, dataset_id)
        possible_edges, true_edges = build_edge_universe(gt_df)
        num_true_edges = len(true_edges)

        true_signed_edges = build_signed_edge_universe(gt_df)
        num_true_edges_signed = len(true_signed_edges)

        possible_pairs_undirected, true_pairs_undirected = build_undirected_edge_universe(gt_df)
        num_true_edges_undirected = len(true_pairs_undirected)

        n_runs_scored = 0
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

                native = native_threshold_for_algorithm(beeline_input_root, algorithm, algo_dir, dataset_id, run_id, predicted_signed)
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
                    "algorithm": algorithm,
                    "n_true_edges": num_true_edges,
                    "auprc": auprc, "auprc_signed": auprc_signed, "auprc_undirected": auprc_undirected,
                    "precision_topk": p_topk, "recall_topk": r_topk, "f1_topk": f1_topk,
                    "precision_topk_signed": p_topk_s, "recall_topk_signed": r_topk_s, "f1_topk_signed": f1_topk_s,
                    "precision_topk_undirected": p_topk_u, "recall_topk_undirected": r_topk_u, "f1_topk_undirected": f1_topk_u,
                    "gmm_threshold": gmm_threshold, "gmm_mode": gmm_mode,
                    "precision_natural": p_nat, "recall_natural": r_nat, "f1_natural": f1_nat,
                    "precision_natural_signed": p_nat_s, "recall_natural_signed": r_nat_s, "f1_natural_signed": f1_nat_s,
                    "precision_natural_undirected": p_nat_u, "recall_natural_undirected": r_nat_u, "f1_natural_undirected": f1_nat_u,
                })
                n_runs_scored += 1

                if verbose:
                    print(f"  {dataset_id}/{run_id}/{algorithm}")

        print(f"[beeline] {dataset_id}: scored {n_runs_scored} (run, algorithm) combination(s)")

    if skipped_gt:
        print(f"Skipped {len(skipped_gt)} non-dataset director(y/ies) under {output_root} (no matching GroundTruthNetwork.csv): {skipped_gt}")

    return pd.DataFrame(rows)


# ============================================================================
# Part 5 (ported verbatim) -- melt into long form, then build the summary table
# ============================================================================
VARIANT_SUFFIX = {"": "directed_unsigned", "_signed": "signed_directed", "_undirected": "undirected"}


def melt_beeline_wide(df):
    rows = []
    for _, r in df.iterrows():
        base = {"method": r["algorithm"], "dataset_id": r["dataset_id"], "scheme": r["scheme"], "pool": "n/a"}
        for suffix, variant in VARIANT_SUFFIX.items():
            rows.append({**base, "metric_family": "auprc", "variant": variant,
                         "score_name": "auprc", "value": r[f"auprc{suffix}"]})
            for score_name in ("precision", "recall", "f1"):
                rows.append({**base, "metric_family": "natural", "variant": variant,
                             "score_name": score_name, "value": r[f"{score_name}_natural{suffix}"]})
                rows.append({**base, "metric_family": "topk", "variant": variant,
                             "score_name": score_name, "value": r[f"{score_name}_topk{suffix}"]})
    return pd.DataFrame(rows)


def melt_twinfer_wide(df):
    NATURAL_STAGE_PLOTTED = "after_fan_out"
    TOPK_POOL_PLOTTED = "all_edges"
    rows = []
    for _, r in df.iterrows():
        base = {"method": "TwINFER", "dataset_id": r["dataset_id"], "scheme": "twinfer", "pool": "n/a"}
        for suffix, variant in VARIANT_SUFFIX.items():
            rows.append({**base, "metric_family": "auprc", "variant": variant,
                         "score_name": "auprc", "value": r[f"auprc{suffix}"]})
            for score_name in ("precision", "recall", "f1"):
                rows.append({**base, "metric_family": "natural", "variant": variant,
                             "score_name": score_name,
                             "value": r[f"{score_name}_natural_{NATURAL_STAGE_PLOTTED}{suffix}"]})
                rows.append({**base, "metric_family": "topk", "variant": variant,
                             "score_name": score_name,
                             "value": r[f"{score_name}_topk_{TOPK_POOL_PLOTTED}{suffix}"]})
    return pd.DataFrame(rows)


SUMMARY_METRICS = [
    ("f1_natural", "natural", "f1"),
    ("precision_natural", "natural", "precision"),
    ("f1_topk", "topk", "f1"),
    ("precision_topk", "topk", "precision"),
    ("auprc", "auprc", "auprc"),
]


def build_summary_table(sub_scores):
    rows = []
    for method in sorted(sub_scores["method"].unique()):
        sub = sub_scores[sub_scores["method"] == method]
        row = {"method": method}
        for col_name, metric_family, score_name in SUMMARY_METRICS:
            mask = (sub["metric_family"] == metric_family) & (sub["score_name"] == score_name)
            row[col_name] = sub.loc[mask, "value"].mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("method")


# ============================================================================
# Driver
# ============================================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Scoring TwINFER (fan-out ON, 'after_fan_out' natural / 'all_edges' topk) ===")
    twinfer_results_df = score_real_network_twinfer()
    twinfer_results_df.to_csv(TWINFER_SCORES_CSV, index=False)
    print(f"Saved {len(twinfer_results_df)} TwINFER row(s) -> {TWINFER_SCORES_CSV}")

    print("\n=== Scoring BEELINE (7 methods) ===")
    beeline_dfs = []
    for beeline_input_root, output_root, nets in BEELINE_GROUPS:
        beeline_dfs.append(score_beeline_sweep_folder(beeline_input_root, output_root, nets=nets))
    beeline_results_df = pd.concat(beeline_dfs, ignore_index=True)
    beeline_results_df.to_csv(BEELINE_SCORES_CSV, index=False)
    print(f"Saved {len(beeline_results_df)} BEELINE row(s) across "
          f"{beeline_results_df['algorithm'].nunique()} algorithm(s) -> {BEELINE_SCORES_CSV}")

    print("\n=== Building summary table ===")
    scores_long = pd.concat(
        [melt_twinfer_wide(twinfer_results_df), melt_beeline_wide(beeline_results_df)],
        ignore_index=True,
    )

    _summary_blocks = []
    for variant in scores_long["variant"].unique():
        v_scores = scores_long[scores_long["variant"] == variant]
        for dataset_id in sorted(v_scores["dataset_id"].unique()):
            block = build_summary_table(v_scores[v_scores["dataset_id"] == dataset_id]).round(4).reset_index()
            block.insert(0, "dataset_id", dataset_id)
            block.insert(0, "variant", variant)
            _summary_blocks.append(block)
        pooled = build_summary_table(v_scores).round(4).reset_index()
        pooled.insert(0, "dataset_id", "ALL")
        pooled.insert(0, "variant", variant)
        _summary_blocks.append(pooled)

    summary_table_all = pd.concat(_summary_blocks, ignore_index=True)
    summary_table_all.to_csv(SUMMARY_TABLE_CSV, index=False)
    print(f"Saved summary metrics table ({len(summary_table_all)} rows, "
          f"variants = {sorted(scores_long['variant'].unique())}, "
          f"{summary_table_all['dataset_id'].nunique()} dataset_id group(s) incl. ALL) -> {SUMMARY_TABLE_CSV}")

    print("\nALL / directed_unsigned block:")
    print(summary_table_all[
        (summary_table_all["variant"] == "directed_unsigned") & (summary_table_all["dataset_id"] == "ALL")
    ].to_string(index=False))


if __name__ == "__main__":
    main()
