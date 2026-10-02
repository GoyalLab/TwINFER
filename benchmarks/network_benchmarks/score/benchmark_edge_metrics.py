"""Edge-recovery metrics for network-inference benchmarks (directed / signed / undirected).

Evaluation layer only: takes a ranked edge list (Gene1, Gene2, EdgeWeight) from ANY method (TwINFER, BEELINE algorithms)
plus a ground-truth network and scores it. It does not decide how TwINFER ranks edges; that is the scoring formula
(todo4v2, twinfer.scoring.todo4v2) which produces EdgeWeight. See docs/SCORING_VERSIONS.md.

[Renamed 2026-10-01 from ka_metrics.py (/home/gzu5140/motif_sims/ka_metrics.py, 30 Jul 2026; the functions there were extracted
verbatim from evaluate_real_networks_boolode.ipynb). Differences from that file, each marked in the code:
  1. NaN EdgeWeight -> 0 in dedupe_predictions / dedupe_predictions_undirected (newest notebook version; "no evidence of an edge").
  2. load_twinfer_ranked_edges reads the NEW result schema (ranked_edges, twinScore as EdgeWeight); the old schema
     (ranked_edge_list, directional_correlation) is still accepted.
  3. `import json` added (load_twinfer_ranked_edges used it without importing).
  4. INPUT_ROOT / MAX_REPS_PER_DATASET are taken from the caller's namespace if preset (the notebooks exec this file into a
     namespace after setting them), otherwise defaults.]

Basis of scoring (all methods, same pipeline)
  Ranking : score = |EdgeWeight| per directed pair; self-loops dropped; duplicate pairs keep the largest |weight|.
            Undirected variants collapse (A,B) and (B,A) to one pair, keeping the larger |weight|.
  Top-k F1: take the top k pairs, k = number of true edges. All ties at the cut-off weight are included (as in BEELINE's
            early-precision metric). The cut-off is never below the smallest non-zero weight. Precision, recall and F1
            are computed on that set (functions top_k_tie_aware_selection*, precision_recall_f1).
  Natural-threshold F1: keep every edge the method gave a non-zero weight (no truncation to k) (natural_threshold_selection*).
  AUPRC   : every possible pair among the ground-truth genes is scored (missing pairs get 0) with
            sklearn precision_recall_curve + auc, matching BEELINE's AUPRC (compute_auprc*). This is RAW AUPRC; the
            project headline is AUPRC x base rate (AUPRC / fraction of true pairs), which todo4v2.full_report reports.
  Variants: directed (right direction), signed (right direction AND sign; a wrong sign is a false positive),
            undirected (only whether the two genes are linked).
  TwINFER's EdgeWeight is the twinScore (new result schema, `ranked_edges`).

Contents: ground-truth loading, edge universes, deduplication, top-k / natural-threshold selection, precision/recall/F1,
AUPRC (directed/signed/undirected), load_twinfer_ranked_edges, and notebook plotting helpers (plot_* functions use
globals the notebook defines: results, SCHEME_COLORS, INK_*, ...; they are not usable standalone).
"""
import numpy as np, pandas as pd, itertools, re
import json  # [2026-10-01 added: load_twinfer_ranked_edges used json without importing it]
from itertools import permutations, combinations, product
from pathlib import Path
from sklearn.metrics import precision_recall_curve, auc

# [2026-10-01 added: keep values the caller (a notebook exec-ing this file) already set]
INPUT_ROOT = globals().get('INPUT_ROOT')
MAX_REPS_PER_DATASET = globals().get('MAX_REPS_PER_DATASET', 100)

def load_ground_truth(dataset_id: str) -> pd.DataFrame:
    # cycle_6_node uses one fixed topology for the whole sweep, so
    # GroundTruthNetwork.csv lives once directly under INPUT_ROOT rather than
    # being duplicated per dataset_id -- fall back to that shared copy.
    gt_path = INPUT_ROOT / dataset_id / "GroundTruthNetwork.csv"
    if not gt_path.exists():
        gt_path = INPUT_ROOT / "GroundTruthNetwork.csv"
    return pd.read_csv(gt_path, header=0)

def build_edge_universe(gt_df: pd.DataFrame):
    """All directed non-self-loop gene pairs among GT genes, and the true-edge subset."""
    gt_no_self = gt_df[gt_df["Gene1"] != gt_df["Gene2"]].drop_duplicates()
    unique_nodes = sorted(set(gt_df["Gene1"]).union(set(gt_df["Gene2"])))
    possible_edges = set(permutations(unique_nodes, 2))
    true_edges = set(zip(gt_no_self["Gene1"], gt_no_self["Gene2"])) & possible_edges
    return possible_edges, true_edges

def dedupe_predictions(ranked_edges: pd.DataFrame) -> pd.DataFrame:
    """Drop self-loops; keep the highest |EdgeWeight| per (Gene1, Gene2)."""
    pred = ranked_edges[ranked_edges["Gene1"] != ranked_edges["Gene2"]].copy()
    pred["_abs"] = pred["EdgeWeight"].fillna(0.0).abs()  # [2026-10-01 NaN -> 0 ("no evidence of an edge"), as in the newest notebook version; was .abs()]
    return (
        pred.sort_values("_abs", ascending=False)
        .drop_duplicates(subset=["Gene1", "Gene2"])
        .reset_index(drop=True)
    )

def top_k_tie_aware_selection(predicted: pd.DataFrame, num_true_edges: int):
    """
    Select the top-k predictions (k = num_true_edges), expanded to include all
    ties at the boundary weight -- same logic as
    BLEval.EarlyPrecision._compute_early_precision. Returns (selected_edge_set,
    threshold_weight_used).
    """
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
    """
    Use whatever edges the algorithm itself assigned a nonzero weight to --
    no truncation or padding to match the ground truth's edge count. Fair to
    algorithms that make their own hard include/exclude decision (e.g. a
    significance test, or a fixed-size candidate-regulator list) rather than
    emitting a full continuous ranking meant to be externally truncated.
    """
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
    """Fully threshold-free complementary metric -- mirrors BLEval.AUPRC._compute_auprc exactly."""
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
    """Directed non-self-loop (Gene1, Gene2, Type) true-edge triples."""
    gt_no_self = gt_df[gt_df["Gene1"] != gt_df["Gene2"]].drop_duplicates()
    true_signed_edges = set(zip(gt_no_self["Gene1"], gt_no_self["Gene2"], gt_no_self["Type"]))
    return true_signed_edges

def build_undirected_edge_universe(gt_df: pd.DataFrame):
    """All undirected non-self-loop gene pairs among GT genes (as frozensets), and the true subset."""
    gt_no_self = gt_df[gt_df["Gene1"] != gt_df["Gene2"]].drop_duplicates()
    unique_nodes = sorted(set(gt_df["Gene1"]).union(set(gt_df["Gene2"])))
    possible_pairs = {frozenset(p) for p in combinations(unique_nodes, 2)}
    true_pairs = {frozenset((g1, g2)) for g1, g2 in zip(gt_no_self["Gene1"], gt_no_self["Gene2"])}
    return possible_pairs, true_pairs

def add_predicted_sign(predicted: pd.DataFrame) -> pd.DataFrame:
    """Attach a '+'/'-' predicted sign column derived from EdgeWeight's sign."""
    pred = predicted.copy()
    pred["_sign"] = np.where(pred["EdgeWeight"] >= 0, "+", "-")
    return pred

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

def natural_threshold_selection_signed(predicted_signed: pd.DataFrame):
    if predicted_signed.empty:
        return set()
    selected = predicted_signed[predicted_signed["_abs"] > 0]
    return set(zip(selected["Gene1"], selected["Gene2"], selected["_sign"]))

def compute_auprc_signed(ranked_edges: pd.DataFrame, gt_df: pd.DataFrame) -> float:
    """
    AUPRC over directed AND signed edges. Scoring universe is every directed pair x both
    signs, so a wrong-signed prediction is scored as an explicit false positive (full
    magnitude credited to the *wrong* sign-position) rather than merely withheld -- see the
    markdown cell above for why.
    """
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
    """Drop self-loops; collapse (Gene1,Gene2) and (Gene2,Gene1) into one undirected edge
    (frozenset pair), keeping the highest |EdgeWeight| between the two directions."""
    pred = ranked_edges[ranked_edges["Gene1"] != ranked_edges["Gene2"]].copy()
    pred["_abs"] = pred["EdgeWeight"].fillna(0.0).abs()  # [2026-10-01 NaN -> 0 ("no evidence of an edge"), as in the newest notebook version; was .abs()]
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

def load_twinfer_ranked_edges(json_path: Path) -> pd.DataFrame:
    """Reconstruct a rankedEdges.csv-shaped frame (Gene1, Gene2, EdgeWeight) from a TwINFER
    result JSON's ranked_edges -- treated identically to a BEELINE algorithm's continuous
    ranking (twinScore as EdgeWeight, sign and all) so it goes through the exact same scoring
    pipeline as every other method.
    [2026-10-01: new schema (ranked_edges / twinScore) is primary; the old schema (ranked_edge_list / directional_correlation)
    of the pre-Sept results is still accepted.]"""
    with open(json_path) as fh:
        record = json.load(fh)
    if "ranked_edges" in record:
        red, wcol = record["ranked_edges"], "twinScore"
    else:
        red, wcol = record["ranked_edge_list"], "directional_correlation"
    df = pd.DataFrame(red["data"], columns=red["columns"])
    return df.rename(columns={
        "gene_1": "Gene1", "gene_2": "Gene2", wcol: "EdgeWeight",
    })[["Gene1", "Gene2", "EdgeWeight"]]

def _select_common_reps(df: pd.DataFrame, max_reps: int = MAX_REPS_PER_DATASET):
    """Restrict every (dataset_id, algorithm) group to the same `max_reps` sim_reps --
    the intersection of sim_reps available across every algorithm for that dataset,
    sorted numerically and capped at max_reps. Returns (filtered_df, per-dataset summary)."""
    kept_frames = []
    summary_rows = []
    for dataset_id, ds_df in df.groupby("dataset_id"):
        rep_sets = [set(g["sim_rep"].astype(str)) for _, g in ds_df.groupby("algorithm")]
        common = set.intersection(*rep_sets) if rep_sets else set()
        try:
            common_sorted = sorted(common, key=int)
        except ValueError:
            common_sorted = sorted(common)
        selected = common_sorted[:max_reps]
        summary_rows.append({
            "dataset_id": dataset_id,
            "n_common_reps": len(common_sorted),
            "n_selected_reps": len(selected),
        })
        kept_frames.append(ds_df[ds_df["sim_rep"].astype(str).isin(selected)])
    summary_df = pd.DataFrame(summary_rows)
    return pd.concat(kept_frames, ignore_index=True), summary_df

def plot_metric_by_algorithm(metric: str, ax=None, title=None):
    pivot = results.groupby(["algorithm", "scheme"])[metric].mean().unstack("scheme")
    pivot = pivot.reindex(columns=[c for c in SCHEME_ORDER if c in pivot.columns])
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    own_ax = ax is None
    if own_ax:
        fig, ax = plt.subplots(figsize=(9, 4.5))

    x = range(len(pivot))
    width = 0.8 / max(len(pivot.columns), 1)
    for i, scheme in enumerate(pivot.columns):
        offsets = [xi + (i - (len(pivot.columns) - 1) / 2) * width for xi in x]
        ax.bar(offsets, pivot[scheme], width=width, color=SCHEME_COLORS.get(scheme, INK_MUTED),
               label=scheme, zorder=3)

    ax.set_xticks(list(x))
    ax.set_xticklabels(pivot.index, rotation=45, ha="right", color=INK_PRIMARY, fontsize=11)
    ax.set_ylabel(metric, color=INK_PRIMARY)
    ax.set_title(title or f"Mean {metric} by algorithm", color=INK_PRIMARY)
    ax.grid(axis="y", color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED)
    # Older matplotlib (3.0.x, installed in the BEELINE env) doesn't support
    # legend(labelcolor=...) -- added in 3.3. Set text color on the returned
    # Legend's text objects directly instead, which works on any version.
    legend = ax.legend(frameon=False)
    for text in legend.get_texts():
        text.set_color(INK_PRIMARY)

    if own_ax:
        fig.tight_layout()
        return fig

def plot_box_dot_by_dataset(metric: str, ax=None, title=None):
    dataset_ids = sorted(results["dataset_id"].unique())
    grouped = [
        results.loc[results["dataset_id"] == d, metric].dropna().values
        for d in dataset_ids
    ]

    own_ax = ax is None
    if own_ax:
        fig, ax = plt.subplots(figsize=(12, 5))

    positions = range(1, len(dataset_ids) + 1)
    bp = ax.boxplot(
        grouped, positions=list(positions), widths=0.5, showfliers=False,
        patch_artist=True,
    )
    for box in bp["boxes"]:
        box.set(facecolor="none", edgecolor=INK_MUTED, linewidth=1.2, zorder=2)
    for element in ["whiskers", "caps", "medians"]:
        for artist in bp[element]:
            artist.set(color=INK_MUTED, linewidth=1.2)

    # Jittered dots on top, colored by scheme (same mapping as the bar charts above)
    for pos, dataset_id in zip(positions, dataset_ids):
        subset = results[results["dataset_id"] == dataset_id].dropna(subset=[metric])
        jitter = _rng.uniform(-0.15, 0.15, size=len(subset))
        colors = subset["scheme"].map(SCHEME_COLORS).fillna(INK_MUTED)
        ax.scatter(
            pos + jitter, subset[metric],
            color=colors, s=18, alpha=0.75, zorder=3, linewidths=0,
        )

    ax.set_xticks(list(positions))
    ax.set_xticklabels(dataset_ids, rotation=60, ha="right", color=INK_PRIMARY, fontsize=11)
    ax.set_ylabel(metric, color=INK_PRIMARY)
    ax.set_title(title or f"{metric} distribution by dataset", color=INK_PRIMARY)
    ax.grid(axis="y", color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED)

    # Legend for scheme colors (dots aren't from ax.bar/scatter-with-label, so build handles manually)
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=c, label=s)
        for s, c in SCHEME_COLORS.items()
        if s in results["scheme"].unique()
    ]
    legend = ax.legend(handles=handles, frameon=False, loc="upper right")
    for text in legend.get_texts():
        text.set_color(INK_PRIMARY)

    if own_ax:
        fig.tight_layout()
        return fig

def prettify_dataset_id(dataset_id: str) -> str:
    m = DATASET_LABEL_RE.match(dataset_id)
    if not m:
        return dataset_id
    pos_label = f"pos{m['pos']} " if m["variant"] == "sign_ratio" else ""
    return f"E{m['e']} {pos_label}{m['variant'].replace('_', ' ')} \u00b7 r{m['rep']}"

def _label_color_for(val: float, norm, cmap) -> str:
    # Pick white vs. ink by the actual rendered cell color's luminance rather
    # than assuming which end of the ramp is dark -- keeps this correct
    # regardless of which colormap HEATMAP_CMAP is set to.
    r, g, b, _ = cmap(norm(val))
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "#ffffff" if luminance < 0.6 else INK_PRIMARY

def plot_heatmap_by_algorithm_dataset(metric: str, scheme: str = None, ax=None, title=None, norm=None):
    # TwINFER isn't run through either sampling scheme (it's its own method
    # operating on the raw twin-simulation data directly), so its row is the
    # same regardless of scheme -- fold it in for direct row-by-row
    # comparison against the BEELINE algorithms instead of segregating it.
    scheme = scheme or HEATMAP_SCHEME
    subset = results[(results["scheme"] == scheme) | (results["algorithm"] == "TwINFER")]
    pivot = subset.groupby(["algorithm", "dataset_id"])[metric].mean().unstack("dataset_id")

    # Average across datasets and each row's rank by that average (1 = best)
    # -- computed before sorting so the rank doesn't just restate row order.
    row_avg = pivot.mean(axis=1)
    row_rank = row_avg.rank(ascending=False, method="min").astype(int)
    pivot = pivot.loc[row_avg.sort_values(ascending=False).index]
    row_avg = row_avg.loc[pivot.index]
    row_rank = row_rank.loc[pivot.index]

    n_dataset_cols = pivot.shape[1]

    own_ax = ax is None
    if own_ax:
        fig, ax = plt.subplots(figsize=(0.6 * (n_dataset_cols + 2) + 3, 0.4 * len(pivot.index) + 4))

    # Anchored at true zero (norm.vmin == 0), plain linear scaling to vmax --
    # no gamma stretch, so cell color is directly proportional to the metric.
    # vmax is taken from the dataset cells only; Average can't exceed that.
    if norm is None:
        vmax = float(pivot.values[~pd.isna(pivot.values)].max())
        norm = Normalize(vmin=0.0, vmax=vmax)

    # Append Average (colored, same scale as the metric) and Rank (left as
    # NaN so it renders as a flat, unfilled column -- rank position isn't a
    # magnitude and shouldn't be pulled from the metric's color scale).
    grid = np.hstack([
        pivot.values,
        row_avg.values.reshape(-1, 1),
        np.full((len(pivot), 1), np.nan),
    ])

    display_cmap = HEATMAP_CMAP.copy()
    display_cmap.set_bad(HEATMAP_SURFACE)

    im = ax.imshow(grid, cmap=display_cmap, aspect="auto", norm=norm)

    for i in range(grid.shape[0]):
        for j in range(n_dataset_cols + 1):  # dataset columns + Average
            val = grid[i, j]
            if pd.isna(val):
                continue
            label_color = _label_color_for(val, norm, HEATMAP_CMAP)
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=label_color)
        # Rank column: plain text on the unfilled background, no magnitude coloring.
        ax.text(n_dataset_cols + 1, i, f"#{row_rank.iloc[i]}", ha="center", va="center",
                 fontsize=9, color=INK_PRIMARY)

    # Separator between the per-dataset cells and the Average/Rank summary columns.
    ax.axvline(n_dataset_cols - 0.5, color=GRIDLINE, linewidth=1.5)

    ax.set_xticks(range(grid.shape[1]))
    ax.set_xticklabels(
        [prettify_dataset_id(c) for c in pivot.columns] + ["Average", "Rank"],
        rotation=45, ha="right", color=INK_PRIMARY, fontsize=11,
    )
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, color=INK_PRIMARY, fontsize=9)
    ax.set_title(title or f"{metric} -- {scheme}", color=INK_PRIMARY)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    if own_ax:
        fig.colorbar(im, ax=ax, shrink=0.8, label=metric)
        fig.tight_layout()
        return fig
    return im