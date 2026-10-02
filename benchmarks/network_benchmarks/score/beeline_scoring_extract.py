from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# RECONSTRUCTED 2026-09-30 (AST extraction from benchmark_network_sweep.ipynb; cell number above each definition). The original /tmp/beeline_scoring_extract.py is lost.
# UNREVIEWED: verify against analysis_data/synthetic_network_benchmark_20260824/ tables before relying on it. Where a name was defined in several cells the LAST definition is used.
from itertools import combinations
from itertools import permutations
from pathlib import Path
from scipy.optimize import brentq
from scipy.stats import norm
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.mixture import GaussianMixture
from tqdm.auto import tqdm
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import re
import os, sys
_SCSGL_DIR = os.environ.get('BEELINE_SCSGL_DIR', os.path.join(os.environ.get('TWINFER_BEELINE_PATH', f'{TWINFER_PROJECT_ROOT}/code/Beeline'), 'Algorithms/SCSGL/scSGL'))
if _SCSGL_DIR not in sys.path:
    sys.path.append(_SCSGL_DIR)
try:
    from pysrc.associations.correlation import permutations as scsgl_correlation_permutations  # SCSGL source (third-party, not vendored here)
except ImportError:
    scsgl_correlation_permutations = None

BEELINE_INPUT_ROOT = None  # set by the caller before use, e.g. bse.BEELINE_INPUT_ROOT = Path(...)

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

# --- notebook cell 21 ---
RUN_ID_RE = re.compile(r"^simrep([0-9a-f]+)_(spread|twin_paired)$")

# --- notebook cell 21 ---
def build_edge_universe(gt_df: pd.DataFrame):
    """All directed non-self-loop gene pairs among GT genes, and the true-edge subset."""
    gt_no_self = gt_df[gt_df["Gene1"] != gt_df["Gene2"]].drop_duplicates()
    unique_nodes = sorted(set(gt_df["Gene1"]).union(set(gt_df["Gene2"])))
    possible_edges = set(permutations(unique_nodes, 2))
    true_edges = set(zip(gt_no_self["Gene1"], gt_no_self["Gene2"])) & possible_edges
    return possible_edges, true_edges

# --- notebook cell 21 ---
def compute_auprc(ranked_edges: pd.DataFrame, gt_df: pd.DataFrame) -> float:
    """Fully threshold-free complementary metric -- mirrors BLEval.AUPRC._compute_auprc."""
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

# --- notebook cell 21 ---
def dedupe_predictions(ranked_edges: pd.DataFrame) -> pd.DataFrame:
    """Drop self-loops; keep the highest |EdgeWeight| per (Gene1, Gene2)."""
    pred = ranked_edges[ranked_edges["Gene1"] != ranked_edges["Gene2"]].copy()
    pred["_abs"] = pred["EdgeWeight"].abs()
    return (
        pred.sort_values("_abs", ascending=False)
        .drop_duplicates(subset=["Gene1", "Gene2"])
        .reset_index(drop=True)
    )

# --- notebook cell 21 ---
def load_beeline_ground_truth(dataset_id: str) -> pd.DataFrame:
    gt_path = BEELINE_INPUT_ROOT / dataset_id / "GroundTruthNetwork.csv"
    if not gt_path.exists():
        gt_path = BEELINE_INPUT_ROOT / "GroundTruthNetwork.csv"
    return pd.read_csv(gt_path, header=0)

# --- notebook cell 21 ---
def top_k_tie_aware_selection(predicted: pd.DataFrame, num_true_edges: int):
    """Select the top-k predictions (k = num_true_edges), expanded to include all ties
    at the boundary weight -- same logic as BLEval.EarlyPrecision._compute_early_precision."""
    if predicted.empty or num_true_edges == 0:
        return set(), float("nan")
    maxk = min(len(predicted), num_true_edges)
    edge_weight_topk = float(predicted.iloc[maxk - 1]["_abs"])
    nonzero = predicted.loc[predicted["_abs"] > 0, "_abs"]
    non_zero_min = float(nonzero.min()) if not nonzero.empty else 0.0
    best_val = max(non_zero_min, edge_weight_topk)
    selected = predicted[predicted["_abs"] >= best_val]
    return set(zip(selected["Gene1"], selected["Gene2"])), best_val

# --- notebook cell 22 ---
def add_predicted_sign(predicted: pd.DataFrame) -> pd.DataFrame:
    """Attach a '+'/'-' predicted sign column derived from EdgeWeight's sign."""
    pred = predicted.copy()
    pred["_sign"] = np.where(pred["EdgeWeight"] >= 0, "+", "-")
    return pred

# --- notebook cell 22 ---
def build_signed_edge_universe(gt_df: pd.DataFrame):
    """Directed non-self-loop (Gene1, Gene2, Type) true-edge triples."""
    gt_no_self = gt_df[gt_df["Gene1"] != gt_df["Gene2"]].drop_duplicates()
    return set(zip(gt_no_self["Gene1"], gt_no_self["Gene2"], gt_no_self["Type"]))

# --- notebook cell 22 ---
def build_undirected_edge_universe(gt_df: pd.DataFrame):
    """All undirected non-self-loop gene pairs among GT genes (as frozensets), and the
    true subset."""
    gt_no_self = gt_df[gt_df["Gene1"] != gt_df["Gene2"]].drop_duplicates()
    unique_nodes = sorted(set(gt_df["Gene1"]).union(set(gt_df["Gene2"])))
    possible_pairs = {frozenset(p) for p in combinations(unique_nodes, 2)}
    true_pairs = {frozenset((g1, g2)) for g1, g2 in zip(gt_no_self["Gene1"], gt_no_self["Gene2"])}
    return possible_pairs, true_pairs

# --- notebook cell 22 ---
def compute_auprc_signed(ranked_edges: pd.DataFrame, gt_df: pd.DataFrame) -> float:
    """AUPRC over directed AND signed edges. A wrong-signed prediction is scored as an
    explicit false positive (full magnitude credited to the wrong sign-position)."""
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

# --- notebook cell 22 ---
def compute_auprc_undirected(ranked_edges: pd.DataFrame, gt_df: pd.DataFrame) -> float:
    """AUPRC over undirected, unsigned edges -- only presence of a relationship is scored."""
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

# --- notebook cell 22 ---
def dedupe_predictions_undirected(ranked_edges: pd.DataFrame) -> pd.DataFrame:
    """Drop self-loops; collapse (Gene1,Gene2) and (Gene2,Gene1) into one undirected edge
    (frozenset pair), keeping the highest |EdgeWeight| between the two directions."""
    pred = ranked_edges[ranked_edges["Gene1"] != ranked_edges["Gene2"]].copy()
    pred["_abs"] = pred["EdgeWeight"].abs()
    pred["_pair"] = [frozenset(p) for p in zip(pred["Gene1"], pred["Gene2"])]
    return (
        pred.sort_values("_abs", ascending=False)
        .drop_duplicates(subset=["_pair"])
        .reset_index(drop=True)
    )

# --- notebook cell 22 ---
def top_k_tie_aware_selection_signed(predicted_signed: pd.DataFrame, num_true_edges: int):
    """Same top-k tie-aware boundary logic as top_k_tie_aware_selection, but selects and
    returns (Gene1, Gene2, predicted_sign) triples instead of (Gene1, Gene2) pairs."""
    if predicted_signed.empty or num_true_edges == 0:
        return set(), float("nan")
    maxk = min(len(predicted_signed), num_true_edges)
    edge_weight_topk = float(predicted_signed.iloc[maxk - 1]["_abs"])
    nonzero = predicted_signed.loc[predicted_signed["_abs"] > 0, "_abs"]
    non_zero_min = float(nonzero.min()) if not nonzero.empty else 0.0
    best_val = max(non_zero_min, edge_weight_topk)
    selected = predicted_signed[predicted_signed["_abs"] >= best_val]
    return set(zip(selected["Gene1"], selected["Gene2"], selected["_sign"])), best_val

# --- notebook cell 22 ---
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

# --- notebook cell 23 ---
def _pvalue_file_threshold_selection(algorithm, algo_dir, predicted_signed):
    """PPCOR/PEARSON: threshold on a genuine per-pair p-value file already on disk.
    PPCOR's outFile.txt (Gene1, Gene2, corVal, pValue) is the raw, unfiltered ppcor::pcor
    output; PEARSON's pvalues.txt (Gene1, Gene2, PValue) is the scipy.stats.pearsonr
    output added to pearsonRunner.py. alpha=0.01 for PPCOR matches its own existing
    config convention (params.pVal); alpha=0.01 for PEARSON is a standard cutoff, since
    PEARSON has no pre-existing convention of its own to match."""
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

# --- notebook cell 23 ---
def _scsgl_permutation_selection(dataset_id, run_id, predicted_signed, alpha=0.05, k=100):
    """SCSGL: recomputes scSGL's own (never-invoked) permutation-null threshold on the
    raw correlation matrix from the same ExpressionData.csv that run's SCSGL invocation
    used, then thresholds the raw correlation (not SCSGL's ADMM-reweighted EdgeWeight)
    against the resulting null bounds. See the module-level comment above for why this
    measures something slightly different from SCSGL's actual graph output."""
    expr_path = BEELINE_INPUT_ROOT / dataset_id / run_id / "ExpressionData.csv"
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

# --- notebook cell 23 ---
def _signed_undirected_from_pairs(selected_nat, predicted_signed):
    """Shared helper: given a set of (Gene1, Gene2) natural-threshold pairs, build the
    matching signed and undirected selections the same way the GMM path does, by
    looking up each pair's predicted sign / undirected-pair identity."""
    sign_lookup = dict(zip(zip(predicted_signed["Gene1"], predicted_signed["Gene2"]), predicted_signed["_sign"]))
    selected_nat_signed = {
        (g1, g2, sign_lookup[(g1, g2)]) for (g1, g2) in selected_nat if (g1, g2) in sign_lookup
    }
    selected_nat_undirected = {frozenset((g1, g2)) for (g1, g2) in selected_nat}
    return selected_nat_signed, selected_nat_undirected

# --- notebook cell 23 ---
def fit_gmm_threshold(scores, min_points=2, random_state=0):
    """
    Fits a 2-component GaussianMixture on `scores` (1D array of raw edge weights/
    importances) and returns (gmm, threshold, threshold_type) -- ported from
    figure_4_f_score copy.ipynb, where this was previously only applied to GRNBoost2.
    threshold is the intersection of the two components' weighted PDFs (found via
    brentq between the two component means): the point where the "low" and "high"
    score populations cross over, i.e. a natural, data-driven cutoff rather than an
    externally chosen rank (top-k) or an arbitrary fixed value.

    Returns (None, np.inf, "GMM_too_few_points") if there aren't enough distinct
    values to fit two components; (gmm, np.inf, "GMM_no_intersection") if the fit
    succeeds but the two components don't cross within their means.

    Fallback path only -- used for methods with no native significance metric
    (PIDC, GENIE3, GRNBoost2, SINCERITIES). PPCOR, SCODE, SCSGL, PEARSON instead use
    a real method-native metric; see native_threshold_for_algorithm below.
    """
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

    f = lambda z: w1 * norm.pdf(z, m1, s1) - w2 * norm.pdf(z, m2, s2)
    try:
        threshold = brentq(f, m1, m2)
        return gmm, threshold, "GMM_intersection"
    except ValueError:
        return gmm, np.inf, "GMM_no_intersection"

# --- notebook cell 23 ---
def gmm_natural_threshold_selection(predicted: pd.DataFrame, threshold: float):
    """GMM-thresholded natural-threshold edge set (directed, unsigned) -- the GMM-based
    counterpart to the crude nonzero-weight natural_threshold_selection."""
    if predicted.empty:
        return set()
    selected = predicted[predicted["_abs"] >= threshold]
    return set(zip(selected["Gene1"], selected["Gene2"]))

# --- notebook cell 23 ---
def gmm_natural_threshold_selection_signed(predicted_signed: pd.DataFrame, threshold: float):
    if predicted_signed.empty:
        return set()
    selected = predicted_signed[predicted_signed["_abs"] >= threshold]
    return set(zip(selected["Gene1"], selected["Gene2"], selected["_sign"]))

# --- notebook cell 23 ---
def gmm_natural_threshold_selection_undirected(predicted_undirected: pd.DataFrame, threshold: float):
    if predicted_undirected.empty:
        return set()
    selected = predicted_undirected[predicted_undirected["_abs"] >= threshold]
    return set(selected["_pair"])

# --- notebook cell 23 ---
def native_threshold_for_algorithm(algorithm, algo_dir, dataset_id, run_id, predicted_signed):
    """Dispatches to the method-native significance metric for PPCOR/PEARSON/SCSGL.
    Returns None for every other algorithm -- including SCODE (see the comment above
    _scode_restart_ttest_selection's definition for why it's kept but not dispatched to)
    -- so the caller falls back to the generic GMM path -- same (selected,
    selected_signed, selected_undirected, threshold, mode) contract as the GMM functions
    above either way."""
    if algorithm in ("PPCOR", "PEARSON"):
        return _pvalue_file_threshold_selection(algorithm, algo_dir, predicted_signed)
    if algorithm == "SCSGL":
        return _scsgl_permutation_selection(dataset_id, run_id, predicted_signed)
    return None

# --- notebook cell 24 ---
def score_beeline_sweep_folder(output_root: Path, verbose=False):
    """
    Scores every (dataset, run, algorithm) rankedEdges.csv under output_root: AUPRC (3
    variants, threshold-free), top-k (3 variants, tie-aware, k = true edge count), and
    natural threshold (3 variants) -- a real method-native significance metric for
    PPCOR/SCSGL/PEARSON (see native_threshold_for_algorithm), GMM for everything else.
    PIDC/GENIE3/GRNBoost2/SINCERITIES have no native significance concept in how they're
    actually invoked here (confirmed by reading both the BLRun wrapper and the
    underlying package source, and by checking completed runs' working_dir contents
    directly); SCODE has a candidate (_scode_restart_ttest_selection) but it's kept
    un-dispatched since it isn't a method the SCODE paper/package itself designed as a
    significance test and it measurably underperforms GMM -- see the comment above that
    function's definition.
    """
    rows = []
    skipped_gt = []

    for dataset_dir in tqdm(sorted(output_root.iterdir())):
        if not dataset_dir.is_dir():
            continue
        dataset_id = dataset_dir.name

        gt_path = BEELINE_INPUT_ROOT / dataset_id / "GroundTruthNetwork.csv"
        if not gt_path.exists():
            skipped_gt.append(dataset_id)
            continue

        gt_df = load_beeline_ground_truth(dataset_id)
        possible_edges, true_edges = build_edge_universe(gt_df)
        num_true_edges = len(true_edges)

        true_signed_edges = build_signed_edge_universe(gt_df)
        num_true_edges_signed = len(true_signed_edges)

        possible_pairs_undirected, true_pairs_undirected = build_undirected_edge_universe(gt_df)
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

                # --- AUPRC (threshold-free) ---
                auprc = compute_auprc(ranked_edges, gt_df)
                auprc_signed = compute_auprc_signed(ranked_edges, gt_df)
                auprc_undirected = compute_auprc_undirected(ranked_edges, gt_df)

                # --- top-k (tie-aware) ---
                selected_topk, _ = top_k_tie_aware_selection(predicted, num_true_edges)
                p_topk, r_topk, f1_topk = precision_recall_f1(selected_topk, true_edges)

                selected_topk_signed, _ = top_k_tie_aware_selection_signed(predicted_signed, num_true_edges_signed)
                p_topk_s, r_topk_s, f1_topk_s = precision_recall_f1(selected_topk_signed, true_signed_edges)

                selected_topk_und, _ = top_k_tie_aware_selection_undirected(predicted_undirected, num_true_edges_undirected)
                p_topk_u, r_topk_u, f1_topk_u = precision_recall_f1(selected_topk_und, true_pairs_undirected)

                # --- Natural threshold: method-native metric when available, else GMM ---
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

                if verbose:
                    print(f"  {dataset_id}/{run_id}/{algorithm}")

    if skipped_gt:
        print(f"Skipped {len(skipped_gt)} non-dataset director(y/ies) under {output_root} (no matching GroundTruthNetwork.csv).")

    return pd.DataFrame(rows)
