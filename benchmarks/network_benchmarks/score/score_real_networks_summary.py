"""
Metrics summary tables for the real-network benchmark (GSD, HSC, VSC, mCAD,
B_cell_activation, Circadian_cycle, EMT, Pluripotent) -- BEELINE algorithms'
rankedEdges.csv plus TwINFER's own *_all_results.json, scored against each
network's GroundTruthNetwork.csv.

Scoring logic (edge universes, top-k tie-aware selection, natural-threshold
selection, AUPRC, and the signed/undirected variants) is reused verbatim from
evaluate_real_networks_boolode.ipynb, generalized here to:
  - all 8 real networks instead of just the 4 BoolODE-curated ones (GSD/HSC/
    VSC/mCAD live under a legacy path, the other 4 under the current
    paper_analysis/real_networks layout -- REAL_NETWORK_DATASETS below
    resolves both).
  - both TwINFER fanout variants (fanout_on / fanout_off), each scored as
    its own method row, rather than a single "TwINFER" row.

Summary table format (SUMMARY_TABLE_CSV) mirrors benchmark_network_sweep.ipynb's
build_summary_table: one row per (variant, dataset_id, method), block-appended
per variant with a pooled dataset_id == "ALL" block averaged over every
(dataset, run) for that variant.

Outputs
-------
REAL_NETWORKS_SCORES_CSV : full per-(dataset, run, algorithm) metrics, all
    variants in wide form (mirrors evaluate_real_networks_boolode.ipynb's
    RESULTS_CSV).
REAL_NETWORKS_SUMMARY_CSV : the method x metric summary table described above.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]

import json
import re
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

PROJECT_ROOT = Path(f'{TWINFER_PROJECT_ROOT}')

# (input_root, output_root) pairs -- input_root/<dataset_id>/GroundTruthNetwork.csv,
# output_root/<dataset_id>/<run_id>/<algorithm>/rankedEdges.csv
#
# GSD/HSC/VSC/mCAD have TWO separate, self-consistent tracks in this codebase:
#   - TwINFER-simulation track (LEGACY_TWINFER_SIM_IO): raw data from TwINFER's own
#     Gillespie simulator, converted to BEELINE format by (an earlier invocation of)
#     twinfer_to_boolode_format.py -- gene_N labels throughout (that script's
#     convert_ground_truth() always emits generic gene_N names), matching the
#     TwINFER JSON's own labeling, so no name translation is needed. Ground truth +
#     a matching BEELINE rerun (GENIE3/PIDC/etc. on that same TwINFER-simulated data)
#     both live under /scratch/gzu5140/twinfer_real/.
#   - BoolODE-simulation track (unused here): real gene symbols, ground truth +
#     BEELINE under code/Beeline/inputs/<net> (LEGACY_ROOT), TwINFER-on-BoolODE-data
#     JSONs scored separately at analysis_data/boolode_sims/twinfer_inference/.
# TwINFER's own JSON output (TWINFER_JSON_ROOTS below) is always from the
# TwINFER-simulation track, so BEELINE must be compared against the matching track,
# not the BoolODE one -- mixing them silently compares TwINFER and BEELINE on two
# different underlying simulated datasets.
# [2026-10-01 commented out: /scratch/gzu5140 is purged scratch; see REPOINT_LOG.tsv] LEGACY_TWINFER_SIM_IO = (Path("/scratch/gzu5140/twinfer_real/inputs"),
LEGACY_TWINFER_SIM_IO = (Path(f"{TWINFER_PROJECT_ROOT}/benchmarking_analysis/twinfer_real/inputs"),
                          # [2026-10-01 commented out: /scratch/gzu5140 is purged scratch; see REPOINT_LOG.tsv] Path("/scratch/gzu5140/twinfer_real/outputs"))
                          Path(f"{TWINFER_PROJECT_ROOT}/benchmarking_analysis/twinfer_real/outputs"))
CURRENT_IO = (PROJECT_ROOT / "code" / "Beeline" / "inputs" / "real_networks",
              PROJECT_ROOT / "analysis_data" / "paper_analysis" / "real_networks" / "beeline_inference")

REAL_NETWORK_DATASETS = {
    "GSD": LEGACY_TWINFER_SIM_IO, "HSC": LEGACY_TWINFER_SIM_IO,
    "VSC": LEGACY_TWINFER_SIM_IO, "mCAD": LEGACY_TWINFER_SIM_IO,
    "B_cell_activation": CURRENT_IO, "Circadian_cycle": CURRENT_IO,
    "EMT": CURRENT_IO, "Pluripotent": CURRENT_IO,
}

TWINFER_JSON_ROOTS = {
    # [2026-10-01 commented out: /scratch/gzu5140 is purged scratch; see REPOINT_LOG.tsv] "fanout_on": Path("/scratch/gzu5140/twinfer_nb/fanout_on"),
    "fanout_on": Path(f"{TWINFER_PROJECT_ROOT}/benchmarking_analysis/twinfer_nb/fanout_on"),
    # [2026-10-01 commented out: /scratch/gzu5140 is purged scratch; see REPOINT_LOG.tsv] "fanout_off": Path("/scratch/gzu5140/twinfer_nb/fanout_off"),
    "fanout_off": Path(f"{TWINFER_PROJECT_ROOT}/benchmarking_analysis/twinfer_nb/fanout_off"),
}

OUT_DIR = PROJECT_ROOT / "analysis_data" / "paper_analysis" / "real_networks"
REAL_NETWORKS_SCORES_CSV = OUT_DIR / "real_networks_evaluation_metrics.csv"
REAL_NETWORKS_SUMMARY_CSV = OUT_DIR / "real_networks_summary_metrics_table.csv"

MAX_REPS_PER_DATASET = 100  # effectively no cap -- datasets' rep counts already vary intentionally

# Matches e.g. "simrep0_spread" -> sim_rep="0", scheme="spread"
RUN_ID_RE = re.compile(r"^simrep([0-9a-fA-F]+)_(spread|twin_paired)$")


# --------------------------------------------------------------------------
# Scoring functions -- verbatim from evaluate_real_networks_boolode.ipynb
# --------------------------------------------------------------------------
def load_ground_truth(input_root: Path, dataset_id: str) -> pd.DataFrame:
    gt_path = input_root / dataset_id / "GroundTruthNetwork.csv"
    if not gt_path.exists():
        gt_path = input_root / "GroundTruthNetwork.csv"
    return pd.read_csv(gt_path, header=0)


def build_edge_universe(gt_df: pd.DataFrame):
    gt_no_self = gt_df[gt_df["Gene1"] != gt_df["Gene2"]].drop_duplicates()
    unique_nodes = sorted(set(gt_df["Gene1"]).union(set(gt_df["Gene2"])))
    possible_edges = set(permutations(unique_nodes, 2))
    true_edges = set(zip(gt_no_self["Gene1"], gt_no_self["Gene2"])) & possible_edges
    return possible_edges, true_edges


def dedupe_predictions(ranked_edges: pd.DataFrame) -> pd.DataFrame:
    pred = ranked_edges[ranked_edges["Gene1"] != ranked_edges["Gene2"]].copy()
    # NaN EdgeWeight (e.g. PPCOR's partial correlation is undefined for a pair
    # involving a zero-variance gene) is treated as no evidence of an edge, not
    # a crash -- same convention as a natural-threshold selection excluding it.
    pred["_abs"] = pred["EdgeWeight"].fillna(0.0).abs()
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


def natural_threshold_selection(predicted: pd.DataFrame):
    if predicted.empty:
        return set()
    selected = predicted[predicted["_abs"] > 0]
    return set(zip(selected["Gene1"], selected["Gene2"]))


def precision_recall_f1(selected_edges: set, true_edges: set):
    if not selected_edges:
        return 0.0, 0.0, 0.0
    tp = len(selected_edges & true_edges)
    precision = tp / len(selected_edges)
    recall = tp / len(true_edges) if true_edges else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


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


def natural_threshold_selection_signed(predicted_signed: pd.DataFrame):
    if predicted_signed.empty:
        return set()
    selected = predicted_signed[predicted_signed["_abs"] > 0]
    return set(zip(selected["Gene1"], selected["Gene2"], selected["_sign"]))


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
    pred = ranked_edges[ranked_edges["Gene1"] != ranked_edges["Gene2"]].copy()
    pred["_abs"] = pred["EdgeWeight"].fillna(0.0).abs()
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


def natural_threshold_selection_undirected(predicted_undirected: pd.DataFrame):
    if predicted_undirected.empty:
        return set()
    selected = predicted_undirected[predicted_undirected["_abs"] > 0]
    return set(selected["_pair"])


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


def score_one(ranked_edges: pd.DataFrame, gt_df: pd.DataFrame, gt_cache: dict) -> dict:
    """Score one rankedEdges-shaped frame against one dataset's cached ground-truth
    universes (all 3 variants: unsigned-directed, signed-directed, undirected)."""
    predicted = dedupe_predictions(ranked_edges)

    selected_topk, threshold = top_k_tie_aware_selection(predicted, gt_cache["num_true_edges"])
    p_topk, r_topk, f1_topk = precision_recall_f1(selected_topk, gt_cache["true_edges"])
    epr = (p_topk / gt_cache["random_precision"]) if gt_cache["random_precision"] else float("nan")

    selected_nat = natural_threshold_selection(predicted)
    p_nat, r_nat, f1_nat = precision_recall_f1(selected_nat, gt_cache["true_edges"])

    auprc = compute_auprc(ranked_edges, gt_df)

    predicted_signed = add_predicted_sign(predicted)
    selected_topk_signed, threshold_signed = top_k_tie_aware_selection_signed(
        predicted_signed, gt_cache["num_true_edges_signed"]
    )
    p_topk_s, r_topk_s, f1_topk_s = precision_recall_f1(selected_topk_signed, gt_cache["true_signed_edges"])
    epr_signed = (p_topk_s / gt_cache["random_precision_signed"]) if gt_cache["random_precision_signed"] else float("nan")

    selected_nat_signed = natural_threshold_selection_signed(predicted_signed)
    p_nat_s, r_nat_s, f1_nat_s = precision_recall_f1(selected_nat_signed, gt_cache["true_signed_edges"])

    auprc_signed = compute_auprc_signed(ranked_edges, gt_df)

    predicted_undirected = dedupe_predictions_undirected(ranked_edges)
    selected_topk_und, threshold_und = top_k_tie_aware_selection_undirected(
        predicted_undirected, gt_cache["num_true_edges_undirected"]
    )
    p_topk_u, r_topk_u, f1_topk_u = precision_recall_f1(selected_topk_und, gt_cache["true_pairs_undirected"])
    epr_undirected = (p_topk_u / gt_cache["random_precision_undirected"]) if gt_cache["random_precision_undirected"] else float("nan")

    selected_nat_und = natural_threshold_selection_undirected(predicted_undirected)
    p_nat_u, r_nat_u, f1_nat_u = precision_recall_f1(selected_nat_und, gt_cache["true_pairs_undirected"])

    auprc_undirected = compute_auprc_undirected(ranked_edges, gt_df)

    return {
        "n_true_edges": gt_cache["num_true_edges"],
        "n_selected_topk": len(selected_topk), "threshold_weight_topk": threshold,
        "precision_topk": p_topk, "recall_topk": r_topk, "f1_topk": f1_topk, "epr_topk": epr,
        "n_selected_natural": len(selected_nat),
        "precision_natural": p_nat, "recall_natural": r_nat, "f1_natural": f1_nat,
        "auprc": auprc,
        "n_true_edges_signed": gt_cache["num_true_edges_signed"],
        "n_selected_topk_signed": len(selected_topk_signed), "threshold_weight_topk_signed": threshold_signed,
        "precision_topk_signed": p_topk_s, "recall_topk_signed": r_topk_s, "f1_topk_signed": f1_topk_s,
        "epr_topk_signed": epr_signed,
        "n_selected_natural_signed": len(selected_nat_signed),
        "precision_natural_signed": p_nat_s, "recall_natural_signed": r_nat_s, "f1_natural_signed": f1_nat_s,
        "auprc_signed": auprc_signed,
        "n_true_edges_undirected": gt_cache["num_true_edges_undirected"],
        "n_selected_topk_undirected": len(selected_topk_und), "threshold_weight_topk_undirected": threshold_und,
        "precision_topk_undirected": p_topk_u, "recall_topk_undirected": r_topk_u, "f1_topk_undirected": f1_topk_u,
        "epr_topk_undirected": epr_undirected,
        "n_selected_natural_undirected": len(selected_nat_und),
        "precision_natural_undirected": p_nat_u, "recall_natural_undirected": r_nat_u, "f1_natural_undirected": f1_nat_u,
        "auprc_undirected": auprc_undirected,
    }


def make_gt_cache(gt_df: pd.DataFrame) -> dict:
    possible_edges, true_edges = build_edge_universe(gt_df)
    num_true_edges = len(true_edges)
    random_precision = num_true_edges / len(possible_edges) if possible_edges else float("nan")

    true_signed_edges = build_signed_edge_universe(gt_df)
    num_true_edges_signed = len(true_signed_edges)
    random_precision_signed = num_true_edges_signed / (len(possible_edges) * 2) if possible_edges else float("nan")

    possible_pairs_undirected, true_pairs_undirected = build_undirected_edge_universe(gt_df)
    num_true_edges_undirected = len(true_pairs_undirected)
    random_precision_undirected = (
        num_true_edges_undirected / len(possible_pairs_undirected) if possible_pairs_undirected else float("nan")
    )
    return dict(
        true_edges=true_edges, num_true_edges=num_true_edges, random_precision=random_precision,
        true_signed_edges=true_signed_edges, num_true_edges_signed=num_true_edges_signed,
        random_precision_signed=random_precision_signed,
        true_pairs_undirected=true_pairs_undirected, num_true_edges_undirected=num_true_edges_undirected,
        random_precision_undirected=random_precision_undirected,
    )


def load_twinfer_ranked_edges(json_path: Path) -> pd.DataFrame:
    with open(json_path) as fh:
        record = json.load(fh)
    red = record["ranked_edge_list"]
    df = pd.DataFrame(red["data"], columns=red["columns"])
    return df.rename(columns={
        "gene_1": "Gene1", "gene_2": "Gene2", "directional_correlation": "EdgeWeight",
    })[["Gene1", "Gene2", "EdgeWeight"]]


# --------------------------------------------------------------------------
# Score every BEELINE rankedEdges.csv
# --------------------------------------------------------------------------
def score_beeline():
    rows = []
    for dataset_id, (input_root, output_root) in REAL_NETWORK_DATASETS.items():
        dataset_dir = output_root / dataset_id
        if not dataset_dir.is_dir():
            print(f"[skip] no BEELINE output dir for {dataset_id}: {dataset_dir}")
            continue
        gt_df = load_ground_truth(input_root, dataset_id)
        gt_cache = make_gt_cache(gt_df)

        for run_dir in sorted(dataset_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            m = RUN_ID_RE.match(run_dir.name)
            if not m:
                continue
            sim_rep, scheme = m.group(1), m.group(2)

            for algo_dir in sorted(run_dir.iterdir()):
                ranked_path = algo_dir / "rankedEdges.csv"
                if not ranked_path.exists():
                    continue
                ranked_edges = pd.read_csv(ranked_path, sep="\t", header=0)
                metrics = score_one(ranked_edges, gt_df, gt_cache)
                rows.append({
                    "dataset_id": dataset_id, "run_id": run_dir.name, "sim_rep": sim_rep,
                    "scheme": scheme, "algorithm": algo_dir.name, **metrics,
                })
    return pd.DataFrame(rows)


def score_twinfer():
    rows = []
    for fanout_label, json_root in TWINFER_JSON_ROOTS.items():
        for dataset_id, (input_root, _) in REAL_NETWORK_DATASETS.items():
            gt_path = input_root / dataset_id / "GroundTruthNetwork.csv"
            if not gt_path.exists():
                continue
            gt_df = load_ground_truth(input_root, dataset_id)
            gt_cache = make_gt_cache(gt_df)

            pattern = re.compile(rf"^{re.escape(dataset_id)}_rep_(?P<rep>.+)_all_results\.json$")
            for json_path in sorted(json_root.glob(f"{dataset_id}_rep_*_all_results.json")):
                m = pattern.match(json_path.name)
                if not m:
                    continue
                sim_rep = m["rep"]
                ranked_edges = load_twinfer_ranked_edges(json_path)
                metrics = score_one(ranked_edges, gt_df, gt_cache)
                rows.append({
                    "dataset_id": dataset_id, "run_id": f"simrep{sim_rep}_twinfer_{fanout_label}",
                    "sim_rep": sim_rep, "scheme": f"twinfer_{fanout_label}",
                    "algorithm": f"TwINFER_{fanout_label}", **metrics,
                })
    return pd.DataFrame(rows)


def select_common_reps(df: pd.DataFrame, max_reps: int = MAX_REPS_PER_DATASET):
    """Restrict every (dataset_id, algorithm) group to the same up-to-max_reps sim_reps --
    the intersection of sim_reps available across every algorithm for that dataset."""
    kept_frames, summary_rows = [], []
    for dataset_id, ds_df in df.groupby("dataset_id"):
        rep_sets = [set(g["sim_rep"].astype(str)) for _, g in ds_df.groupby("algorithm")]
        common = set.intersection(*rep_sets) if rep_sets else set()
        try:
            common_sorted = sorted(common, key=int)
        except ValueError:
            common_sorted = sorted(common)
        selected = common_sorted[:max_reps]
        summary_rows.append({"dataset_id": dataset_id, "n_common_reps": len(common_sorted), "n_selected_reps": len(selected)})
        kept_frames.append(ds_df[ds_df["sim_rep"].astype(str).isin(selected)])
    return pd.concat(kept_frames, ignore_index=True), pd.DataFrame(summary_rows)


# --------------------------------------------------------------------------
# Summary table -- method x metric, blocked per (variant, dataset_id incl. "ALL")
# same shape as benchmark_network_sweep.ipynb's build_summary_table / SUMMARY_TABLE_CSV
# --------------------------------------------------------------------------
VARIANT_METRIC_COLUMNS = {
    "directed_unsigned": {
        "f1_natural": "f1_natural", "precision_natural": "precision_natural",
        "f1_topk": "f1_topk", "precision_topk": "precision_topk", "auprc": "auprc",
    },
    "signed_directed": {
        "f1_natural": "f1_natural_signed", "precision_natural": "precision_natural_signed",
        "f1_topk": "f1_topk_signed", "precision_topk": "precision_topk_signed", "auprc": "auprc_signed",
    },
    "undirected": {
        "f1_natural": "f1_natural_undirected", "precision_natural": "precision_natural_undirected",
        "f1_topk": "f1_topk_undirected", "precision_topk": "precision_topk_undirected", "auprc": "auprc_undirected",
    },
}


def method_label(row) -> str:
    if row["algorithm"].startswith("TwINFER_"):
        return row["algorithm"].replace("TwINFER_", "TwINFER (") + ")"
    return f"{row['algorithm']} ({row['scheme']})"


def build_summary_table(scored: pd.DataFrame) -> pd.DataFrame:
    scored = scored.assign(method=scored.apply(method_label, axis=1))
    blocks = []
    for variant, colmap in VARIANT_METRIC_COLUMNS.items():
        for dataset_id in sorted(scored["dataset_id"].unique()) + ["ALL"]:
            sub = scored if dataset_id == "ALL" else scored[scored["dataset_id"] == dataset_id]
            rows = []
            for method in sorted(sub["method"].unique()):
                m_sub = sub[sub["method"] == method]
                row = {"method": method}
                for out_col, src_col in colmap.items():
                    row[out_col] = m_sub[src_col].mean()
                rows.append(row)
            block = pd.DataFrame(rows).round(4)
            block.insert(0, "dataset_id", dataset_id)
            block.insert(0, "variant", variant)
            blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    beeline_scores = score_beeline()
    print(f"Scored {len(beeline_scores)} BEELINE (dataset, run, algorithm) combinations.")
    twinfer_scores = score_twinfer()
    print(f"Scored {len(twinfer_scores)} TwINFER (dataset, rep, fanout variant) combinations.")

    # Common-rep intersection is applied separately within BEELINE and within
    # TwINFER, not across the two: TwINFER's sim_rep identifiers (derived from
    # the simulation file's hash suffix for the 4 legacy networks) don't share
    # a naming convention with BEELINE's simrepN run directories, so
    # intersecting them together would always be empty for those datasets and
    # silently drop them entirely.
    beeline_scores, beeline_rep_summary = select_common_reps(beeline_scores)
    twinfer_scores, twinfer_rep_summary = select_common_reps(twinfer_scores)
    all_scores = pd.concat([beeline_scores, twinfer_scores], ignore_index=True)
    rep_selection_summary = pd.concat(
        [beeline_rep_summary.assign(track="beeline"), twinfer_rep_summary.assign(track="twinfer")],
        ignore_index=True,
    )

    short = rep_selection_summary[rep_selection_summary["n_selected_reps"] < MAX_REPS_PER_DATASET]
    if not short.empty:
        print(f"{len(short)} dataset(s) capped by common-rep intersection (not all algorithms share every rep):")
        print(short.to_string(index=False))

    all_scores.to_csv(REAL_NETWORKS_SCORES_CSV, index=False)
    print(f"Saved {len(all_scores)} rows -> {REAL_NETWORKS_SCORES_CSV}")

    summary_table = build_summary_table(all_scores)
    summary_table.to_csv(REAL_NETWORKS_SUMMARY_CSV, index=False)
    print(f"Saved summary table ({len(summary_table)} rows, "
          f"{summary_table['dataset_id'].nunique()} dataset_id group(s) incl. ALL) -> {REAL_NETWORKS_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
