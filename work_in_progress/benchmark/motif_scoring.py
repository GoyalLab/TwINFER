"""
Per-motif GRN-inference accuracy scoring.

Two independent parts:
  1. Detection -- pure functions of a ground-truth network (TwINFER's native
     signed-adjacency-matrix .txt format: headerless, comma-separated,
     matrix[i,j] = sign of the i->j edge, 0 = no edge). No prediction data
     involved. Four motif types:
       - fan_out:          Z->A, Z->B, no direct A-B edge either direction.
       - feed_forward:     Z->A, Z->B, and exactly one of A->B / B->A.
       - regulated_mutual: Z->A, Z->B, and both A->B and B->A.
       - auto_regulation:  external regulators X->G of a self-looping gene G
                            (G->G real) -- NOT the G->G edge itself, which is
                            never predicted by any method's pipeline (self-pairs
                            are always dropped before scoring) and is trivially
                            ~1.0 self-correlated regardless of real auto-regulation.
                            The informative question is whether G's own
                            auto-regulation confounds inference of its real
                            external regulators.
     `detect_motifs()` is the dispatcher; adding a new motif type later is one
     more `detect_*` function plus one dict entry.

  2. Scoring -- given the detected motifs and a method's predicted edge set
     (however the caller turned a ranking into a hard decision -- top-k,
     natural threshold, whatever), compute per-motif recall on the motif's
     defining edges, plus (fan_out only) the confound false-positive rate: how
     often the method hallucinates the direct A-B edge that fan-out's shared
     upstream regulation classically produces spurious correlation for.
     Recall (not a network-style precision) is the primary metric for
     fan_out/feed_forward/regulated_mutual/auto_regulation, since "false
     positive" is only unambiguous for fan-out's confound pairs -- a method
     predicting some edge that isn't part of ANY detected motif instance isn't
     a "motif-specific" false positive for the other three types.

This module is dataset-agnostic and has no dependency on any particular
notebook's prediction-loading conventions -- see motif_scoring_analysis.ipynb
for a driver that points it at a specific dataset's ground truth + predictions
(both the 7 BEELINE methods' rankedEdges.csv outputs, and TwINFER's own
fanout_on/fanout_off all_results.json pair).
"""
from itertools import combinations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------
def load_ground_truth(path, gene_names=None) -> pd.DataFrame:
    """
    Load a TwINFER-format signed adjacency matrix (.txt, headerless,
    comma-separated). Returns a gene-indexed square pd.DataFrame; entries are
    the raw signed values from the file (0 = no edge, sign = direction).

    gene_names defaults to gene_1..gene_n (TwINFER's own convention), matching
    convert_ground_truth() in twinfer_to_boolode_format.py.
    """
    matrix = np.loadtxt(path, delimiter=",")
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    n = matrix.shape[0]
    if gene_names is None:
        gene_names = [f"gene_{i + 1}" for i in range(n)]
    return pd.DataFrame(matrix, index=gene_names, columns=gene_names)


def _sign(gt: pd.DataFrame, a, b) -> int:
    return 1 if gt.loc[a, b] > 0 else -1


# ---------------------------------------------------------------------------
# Part 1 -- motif detection
# ---------------------------------------------------------------------------
def detect_fan_out(gt: pd.DataFrame) -> list:
    """Z->A, Z->B, no direct A-B edge in either direction."""
    genes = list(gt.index)
    instances = []
    for z in genes:
        targets = [g for g in genes if g != z and gt.loc[z, g] != 0]
        for a, b in combinations(targets, 2):
            if gt.loc[a, b] == 0 and gt.loc[b, a] == 0:
                instances.append({
                    "regulator": z, "targets": (a, b),
                    "edges": [(z, a), (z, b)],
                    "edges_signed": [(z, a, _sign(gt, z, a)), (z, b, _sign(gt, z, b))],
                    "confound_edges": [(a, b), (b, a)],
                })
    return instances


def detect_feed_forward(gt: pd.DataFrame) -> list:
    """Z->A, Z->B, and exactly one of A->B / B->A."""
    genes = list(gt.index)
    instances = []
    for z in genes:
        targets = [g for g in genes if g != z and gt.loc[z, g] != 0]
        for a, b in combinations(targets, 2):
            ab, ba = gt.loc[a, b] != 0, gt.loc[b, a] != 0
            if ab and not ba:
                instances.append({
                    "regulator": z, "intermediate": a, "target": b,
                    "edges": [(z, a), (z, b), (a, b)],
                    "edges_signed": [(z, a, _sign(gt, z, a)), (z, b, _sign(gt, z, b)), (a, b, _sign(gt, a, b))],
                })
            elif ba and not ab:
                instances.append({
                    "regulator": z, "intermediate": b, "target": a,
                    "edges": [(z, b), (z, a), (b, a)],
                    "edges_signed": [(z, b, _sign(gt, z, b)), (z, a, _sign(gt, z, a)), (b, a, _sign(gt, b, a))],
                })
    return instances


def detect_regulated_mutual(gt: pd.DataFrame) -> list:
    """Z->A, Z->B, and both A->B and B->A."""
    genes = list(gt.index)
    instances = []
    for z in genes:
        targets = [g for g in genes if g != z and gt.loc[z, g] != 0]
        for a, b in combinations(targets, 2):
            if gt.loc[a, b] != 0 and gt.loc[b, a] != 0:
                instances.append({
                    "regulator": z, "pair": (a, b),
                    "edges": [(z, a), (z, b), (a, b), (b, a)],
                    "edges_signed": [
                        (z, a, _sign(gt, z, a)), (z, b, _sign(gt, z, b)),
                        (a, b, _sign(gt, a, b)), (b, a, _sign(gt, b, a)),
                    ],
                })
    return instances


def detect_auto_regulation(gt: pd.DataFrame) -> list:
    """
    External regulators X->G of a self-looping gene G (G->G real) -- one
    instance per self-looping gene, `edges` is every real external regulator
    of that gene. Networks with no self-loops (e.g. every
    triplet_motif_discrimination topology -- TwINFER's own simulator forbids
    self-loops) simply produce no instances.
    """
    genes = list(gt.index)
    instances = []
    for g in genes:
        if gt.loc[g, g] == 0:
            continue
        external = [x for x in genes if x != g and gt.loc[x, g] != 0]
        if not external:
            continue
        instances.append({
            "target": g, "self_edge": (g, g),
            "edges": [(x, g) for x in external],
            "edges_signed": [(x, g, _sign(gt, x, g)) for x in external],
        })
    return instances


def detect_motifs(gt: pd.DataFrame) -> dict:
    """Dispatcher -- run all four detectors. Add a new motif type by adding one
    more detect_* function and one entry here."""
    return {
        "fan_out": detect_fan_out(gt),
        "feed_forward": detect_feed_forward(gt),
        "regulated_mutual": detect_regulated_mutual(gt),
        "auto_regulation": detect_auto_regulation(gt),
    }


# ---------------------------------------------------------------------------
# Part 2 -- scoring
# ---------------------------------------------------------------------------
def score_motif_instance(instance: dict, predicted_edges: set, predicted_signed_edges: set = None) -> dict:
    """Recovery of one detected motif instance's defining edges (TP/FN), plus
    (fan_out instances only) whether either confound edge was wrongly
    predicted present."""
    edges = instance["edges"]
    tp = sum(1 for e in edges if e in predicted_edges)
    result = {"n_edges": len(edges), "tp": tp, "fn": len(edges) - tp}

    if predicted_signed_edges is not None:
        tp_s = sum(1 for e in instance["edges_signed"] if e in predicted_signed_edges)
        result["tp_signed"] = tp_s
        result["fn_signed"] = len(edges) - tp_s

    if "confound_edges" in instance:
        result["n_confound_pairs"] = len(instance["confound_edges"])
        result["confound_fp"] = sum(1 for e in instance["confound_edges"] if e in predicted_edges)

    return result


def score_by_motif(motifs: dict, predicted_edges: set, predicted_signed_edges: set = None) -> pd.DataFrame:
    """One row per motif type: n_instances, n_defining_edges, tp, fn, recall
    (+ recall_signed if predicted_signed_edges given; + confound_fp_rate for
    fan_out)."""
    has_signed = predicted_signed_edges is not None
    rows = []
    for motif_name, instances in motifs.items():
        n_edges = tp = fn = tp_s = fn_s = n_confound = confound_fp = 0
        has_confound = False
        for inst in instances:
            r = score_motif_instance(inst, predicted_edges, predicted_signed_edges)
            n_edges += r["n_edges"]; tp += r["tp"]; fn += r["fn"]
            if has_signed:
                tp_s += r["tp_signed"]; fn_s += r["fn_signed"]
            if "confound_fp" in r:
                has_confound = True
                n_confound += r["n_confound_pairs"]; confound_fp += r["confound_fp"]

        row = {
            "motif": motif_name,
            "n_instances": len(instances),
            "n_defining_edges": n_edges,
            "tp": tp, "fn": fn,
            "recall": tp / n_edges if n_edges else float("nan"),
        }
        if has_signed:
            row["recall_signed"] = tp_s / n_edges if n_edges else float("nan")
        if has_confound:
            row["n_confound_pairs"] = n_confound
            row["confound_fp"] = confound_fp
            row["confound_fp_rate"] = confound_fp / n_confound if n_confound else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def score_methods_by_motif(motifs: dict, method_predictions: dict, method_predictions_signed: dict = None) -> pd.DataFrame:
    """
    Long-form (method x motif) table -- the main "compare methods on this
    motif" output.

    method_predictions: {method_name: set of (Gene1, Gene2) predicted-present
        directed edges}.
    method_predictions_signed: optional {method_name: set of
        (Gene1, Gene2, sign) predicted-present signed edges}, sign in {1, -1}.
    """
    method_predictions_signed = method_predictions_signed or {}
    frames = []
    for method, preds in method_predictions.items():
        df = score_by_motif(motifs, preds, method_predictions_signed.get(method))
        df.insert(0, "method", method)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Small, general convenience: rankedEdges.csv (Gene1,Gene2,EdgeWeight) -> a
# top-k tie-aware predicted edge set. Kept here since it's a genuinely
# dataset-agnostic utility; per-dataset file discovery/paths belong in the
# calling notebook, not this module.
# ---------------------------------------------------------------------------
def dedupe_predictions(ranked_edges: pd.DataFrame) -> pd.DataFrame:
    """Drop self-loops; keep the highest |EdgeWeight| per (Gene1, Gene2)."""
    pred = ranked_edges[ranked_edges["Gene1"] != ranked_edges["Gene2"]].copy()
    pred["_abs"] = pred["EdgeWeight"].abs()
    return (
        pred.sort_values("_abs", ascending=False)
        .drop_duplicates(subset=["Gene1", "Gene2"])
        .reset_index(drop=True)
    )


def top_k_tie_aware_selection(predicted: pd.DataFrame, k: int) -> set:
    """Top-k predictions (expanded to include ties at the boundary weight)."""
    if predicted.empty or k == 0:
        return set()
    maxk = min(len(predicted), k)
    edge_weight_topk = float(predicted.iloc[maxk - 1]["_abs"])
    nonzero = predicted.loc[predicted["_abs"] > 0, "_abs"]
    non_zero_min = float(nonzero.min()) if not nonzero.empty else 0.0
    best_val = max(non_zero_min, edge_weight_topk)
    selected = predicted[predicted["_abs"] >= best_val]
    return set(zip(selected["Gene1"], selected["Gene2"]))


def predicted_edges_from_ranked(ranked_edges: pd.DataFrame, k: int) -> set:
    """Convenience: rankedEdges.csv-style DataFrame -> top-k tie-aware predicted edge set."""
    return top_k_tie_aware_selection(dedupe_predictions(ranked_edges), k)


def predicted_signed_edges_from_ranked(ranked_edges: pd.DataFrame, k: int) -> set:
    """Signed counterpart: (Gene1, Gene2, sign) triples for the same top-k selection."""
    predicted = dedupe_predictions(ranked_edges)
    selected_pairs = top_k_tie_aware_selection(predicted, k)
    sign_lookup = dict(zip(zip(predicted["Gene1"], predicted["Gene2"]),
                           np.sign(predicted["EdgeWeight"]).astype(int)))
    return {(g1, g2, sign_lookup[(g1, g2)]) for (g1, g2) in selected_pairs}


# ---------------------------------------------------------------------------
# TwINFER's own predictions: replay the fan-out-removal decision on its raw
# ranked correlation output (all_results.json, fanout_on/fanout_off pair).
# Symmetric to the rankedEdges.csv helpers above -- given a method's raw
# output (however that method structures it), produce a predicted edge set
# ready for score_methods_by_motif. threshold/flipped/eliminate_false_arms
# select which fan-out regime to replay (e.g. z>8 eliminate for TwINFER's own
# simulator, z<137 flipped for BoolODE-simulated data -- see
# benchmark_twinfer_vs_boolode.ipynb, this is the same logic centralized here
# instead of copied into every notebook that needs it).
# ---------------------------------------------------------------------------
def unwrap_matrix(x) -> pd.DataFrame:
    """Reconstructs a labeled DataFrame from either this project's
    {"__type__":"DataFrame","data":...,"index":...,"columns":...} JSON
    wrapper, or a plain dict-of-columns."""
    if isinstance(x, dict) and "data" in x:
        return pd.DataFrame(x["data"], index=x["index"], columns=x["columns"])
    return pd.DataFrame(x)


def fanout_decision(z: float, threshold: float, flipped: bool) -> bool:
    return (abs(z) < threshold) if flipped else (abs(z) > threshold)


def apply_fanout(ranked: pd.DataFrame, fan_out_log, dmat: pd.DataFrame, final_edges: set,
                  threshold: float, flipped: bool, eliminate_false_arms: bool, gate: float = 0.049) -> pd.DataFrame:
    """Replay the fan-out stage on a ranked edge list (Gene1,Gene2,EdgeWeight)."""
    drop = set()
    for e in (fan_out_log or []):
        A, B = e["gene_a"], e["gene_b"]
        if A not in dmat.index or B not in dmat.columns:
            continue
        ca, cb = float(dmat.loc[A, B]), float(dmat.loc[B, A])
        if abs(ca) <= gate or abs(cb) <= gate:
            continue
        if ca < 0 and cb < 0:
            continue
        if eliminate_false_arms:
            common = [
                z for z in (e.get("common_regulators") or [])
                if z in dmat.index
                and abs(float(dmat.loc[z, A])) > gate and abs(float(dmat.loc[z, B])) > gate
                and (A, z) not in final_edges and (B, z) not in final_edges
            ]
            if not common:
                continue
        if fanout_decision(float(e["z_score"]), threshold, flipped):
            drop.add(frozenset((A, B)))
    return ranked[[frozenset((r.Gene1, r.Gene2)) not in drop for r in ranked.itertuples()]]


def twinfer_predicted_edges(fanout_on_json: dict, fanout_off_json: dict,
                             threshold: float, flipped: bool, eliminate_false_arms: bool,
                             k: int, gate: float = 0.049):
    """Given the loaded fanout_on/fanout_off all_results.json dicts for one
    TwINFER run, replay the fan-out decision and select the top-k tie-aware
    subset of what survives. Returns (predicted_edges, predicted_signed_edges)
    in the same format the rankedEdges.csv helpers above produce.

    The top-k step matters: `ranked_edge_list` includes every gene pair with a
    nonzero correlation score (near-total coverage in practice), and
    apply_fanout() only *removes* specific fan-out-flagged pairs -- it applies
    no overall magnitude/significance cutoff. Without the same k budget the
    BEELINE methods get, `keep` is effectively "predict almost every possible
    edge", which trivially maximizes recall while saying nothing about
    selectivity -- not a fair comparison. Mirrors how the notebook's own
    f1_signed(keep, gt)/f1_unsigned(keep, gt) already top-k-restrict `keep`."""
    rel = unwrap_matrix(fanout_off_json["ranked_edge_list"])
    ranked = pd.DataFrame({
        "Gene1": rel["gene_1"], "Gene2": rel["gene_2"],
        "EdgeWeight": pd.Series(rel["directional_correlation"]).astype(float).fillna(0.0),
    })
    dmat = unwrap_matrix(fanout_on_json["unfiltered_direction_matrix"]).astype(float).fillna(0.0)
    final_edges = {tuple(x) for x in fanout_on_json["final_directed_edges"] if len(x) == 2}
    keep = apply_fanout(ranked, fanout_on_json.get("fan_out_log"), dmat, final_edges,
                         threshold=threshold, flipped=flipped,
                         eliminate_false_arms=eliminate_false_arms, gate=gate)
    predicted = dedupe_predictions(keep)
    predicted_edges = top_k_tie_aware_selection(predicted, k)
    sign_lookup = dict(zip(zip(predicted["Gene1"], predicted["Gene2"]),
                           np.sign(predicted["EdgeWeight"]).astype(int)))
    predicted_signed_edges = {(g1, g2, sign_lookup[(g1, g2)]) for (g1, g2) in predicted_edges}
    return predicted_edges, predicted_signed_edges
