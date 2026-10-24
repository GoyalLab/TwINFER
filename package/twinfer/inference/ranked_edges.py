"""Ranked gene-pair lists from the output of infer_with_twinfer (twinfer.inference.covariance_decomposition_z_scores).

One list per score, by default
  z_rho_s       twin-shared part of the correlation (heterogeneity: a shared cell state)
  z_rho_Delta   twin-difference correlation (regulation)
  z_dag         only when the result has two time points: the cross-time correlation of sisters, ranked by the larger of
                |z_dag_xy| (gene_x earlier with gene_y later) and |z_dag_yx| (the reverse); each pair is listed as source -> target
                in the direction of the larger one, with delta and p_delta (the test of the direction)
Each list ranks the gene pairs by |z| (the sign does not change the rank), largest first. By default only pairs whose
expression correlation z_rho is significant are ranked: q_rho <= fdr when the result has a q_rho column (Benjamini-Hochberg,
the default for real data), otherwise |z_rho| > z_rho_threshold. This is how the top lists of the TwinScore procedure are made
(pairs passing Stage I, ranked by |z|).

Example:
    from twinfer.inference.ranked_edges import ranked_edge_lists
    lists = ranked_edge_lists(result, top=100)
    lists["z_rho_Delta"]          # rank, gene_x, gene_y, z_rho_Delta, z_rho, q_rho, inference_scenario, ...
    lists["z_dag"]                # two time points only: rank, source, target, z_dag, z_dag_xy, z_dag_yx, delta, p_delta, ...
"""
# [2026-10-24 renamed: z_rho_delta, q_rho_delta, rho_delta and q_values_of_z_rho_delta to z_rho_Delta, q_rho_Delta, rho_Delta, q_values_of_z_rho_Delta]
import numpy as np
import pandas as pd


# [2026-10-23 replaced: scores default None adds the cross-correlation list (z_dag) for several time points]
# def ranked_edge_lists(result, scores=("z_rho_s", "z_rho_delta"), time="tb", significant_rho_only=True, fdr=0.05,
#                       z_rho_threshold=1.96, top=None):
def ranked_edge_lists(result, scores=None, time="tb", significant_rho_only=True, fdr=0.05, z_rho_threshold=1.96, top=None):
    """Ranked lists of gene pairs, one per score.

    Parameters
    ----------
    result : pandas.DataFrame
        Returned by infer_with_twinfer (one row per gene pair; with several time points, one (t_a, t_b) pair).
    scores : list of str, optional
        Score columns to rank by (without the _ta / _tb suffix), and/or "z_dag" (cross-time correlation, several time
        points only). None (default): ("z_rho_s", "z_rho_Delta"), plus "z_dag" when the result has two time points.
    time : {"ta", "tb"}
        Which twin group to use when the result has several time points (columns with that suffix).
    significant_rho_only, fdr, z_rho_threshold : bool, float, float
        Rank only pairs whose z_rho is significant (q_rho <= fdr if a q_rho column exists, else ``|z_rho|`` > z_rho_threshold).
    top : int, optional
        Keep the first `top` pairs of each list (None = all).

    Returns
    -------
    dict
        {score: DataFrame} with the columns rank (1 = largest ``|score|``), gene_x, gene_y, the score, z_rho, and q_rho, q_rho_Delta,
        inference_scenario where present. The z_dag list has source, target (the direction of the larger ``|z_dag|``), z_dag, z_dag_xy,
        z_dag_yx, delta, p_delta instead of gene_x, gene_y. Pairs with a missing score are left out; tied pairs keep their order in `result`.
    """
    has_cross = {"z_dag_xy", "z_dag_yx"} <= set(result.columns)
    if scores is None:
        scores = ("z_rho_s", "z_rho_Delta") + (("z_dag",) if has_cross else ())
    suffix = "" if "z_rho" in result.columns else f"_{time}"
    if f"z_rho{suffix}" not in result.columns:
        raise ValueError(f"result has no column z_rho{suffix}")
    table = result.rename(columns={c: c[: -len(suffix)] for c in result.columns if suffix and c.endswith(suffix)})
    if significant_rho_only:
        if "q_rho" in table.columns:
            table = table[table["q_rho"] <= fdr]
        else:
            table = table[np.abs(table["z_rho"]) > z_rho_threshold]
    extra = [c for c in ("z_rho", "q_rho", "q_rho_Delta", "inference_scenario") if c in table.columns]
    lists = {}
    if "z_dag" in scores:
        if not has_cross:
            raise ValueError("z_dag needs a result with two time points (columns z_dag_xy, z_dag_yx)")
        forward = np.abs(table["z_dag_xy"].to_numpy()) >= np.abs(table["z_dag_yx"].to_numpy())
        table = table.assign(source=np.where(forward, table["gene_x"], table["gene_y"]),
                             target=np.where(forward, table["gene_y"], table["gene_x"]),
                             z_dag=np.where(forward, table["z_dag_xy"], table["z_dag_yx"]))
    for score in scores:
        if score not in table.columns:
            raise ValueError(f"result has no column {score}{suffix}")
        ranked = table.dropna(subset=[score])
        ranked = ranked.iloc[np.argsort(-np.abs(ranked[score].to_numpy()), kind="stable")]
        if top is not None:
            ranked = ranked.head(top)
        # [2026-10-23 replaced: own columns for the cross-correlation list]
        # columns = ["gene_x", "gene_y", score] + [c for c in extra if c != score]
        if score == "z_dag":
            columns = ["source", "target", "z_dag", "z_dag_xy", "z_dag_yx", "delta", "p_delta"] + extra
        else:
            columns = ["gene_x", "gene_y", score] + [c for c in extra if c != score]
        ranked = ranked[columns].reset_index(drop=True)
        ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
        lists[score] = ranked
    return lists


# [2026-10-23 replaced: the earlier pipeline's twinfer_to_ranked_edges (ranking by directed permutation p-values of infer.py) is kept
#  below as comments; the module now ranks the output of covariance_decomposition_z_scores.infer_with_twinfer by z_rho_s and z_rho_delta]
# """
# Convert infer_with_twinfer()'s output into a ranked edge list, for benchmarking or
# downstream analysis.
#
# Ranking is done by the directed significance p-value (result['directed_p_values']),
# not raw correlation magnitude. Raw magnitude alone is not a fair ranking criterion here:
# statistical significance depends on both magnitude and that specific gene pair's null
# distribution width (which varies pair to pair), so a noisy, non-significant edge can have
# a larger raw correlation than a genuinely significant one. Ranking by p-value first lets
# non-significant edges sink to the bottom on their own merits; magnitude is used only as a
# tie-breaker, since a fixed number of shuffles gives p-values a resolution floor (e.g. many
# strong true edges can all report p=0.0 at n_shuffles=10000) where magnitude is the only
# thing left to differentiate them.
#
# This is agnostic to which infer_direction_for_which_edges mode produced `result`:
#     - "all-edges": directed_p_values covers every gene pair -- intended for benchmarking
#       against a full ground-truth network (unbiased, no pre-filtering by TwINFER's own
#       earlier screening steps).
#     - "single-state" (the default) / "all-potential-regulation": directed_p_values only
#       covers pairs that survived the earlier undirected screen and single/multi-state
#       classification -- for real analysis, the ranked list is expected to only include
#       those pairs; that's TwINFER's actual staged inference, not a limitation to work around.
# """
# import numpy as np
# import pandas as pd
#
#
# def twinfer_to_ranked_edges(result, magnitude_matrix='unfiltered'):
#     """
#     Parameters
#     ----------
#     result : dict
#         Output of infer_with_twinfer(). Must contain 'directed_p_values' (added alongside
#         'final_directed_edges' -- requires calling identify_actual_directed_edges /
#         infer_with_twinfer from a version of this codebase that returns it).
#     magnitude_matrix : {'unfiltered', 'filtered'}
#         Which correlation matrix to pull the tie-breaking magnitude from.
#         'unfiltered' -> result['unfiltered_direction_matrix'] (recommended: the raw value,
#             not zeroed out by the significance threshold).
#         'filtered' -> result['direction_matrix'].
#
#     Returns
#     -------
#     pd.DataFrame with columns ['regulator', 'target', 'p_value', 'score'], sorted by
#     p_value ascending with |score| descending as the tie-break.
#     """
#     p_values = result.get('directed_p_values')
#     if not p_values:
#         raise ValueError(
#             "result['directed_p_values'] is empty or missing -- either no directed pairs "
#             "were tested (check infer_direction_for_which_edges and the classification "
#             "results in result['gene_lists']), or this result came from a version of "
#             "infer_with_twinfer that doesn't return directed_p_values yet."
#         )
#
#     if magnitude_matrix not in ('unfiltered', 'filtered'):
#         raise ValueError("magnitude_matrix must be 'unfiltered' or 'filtered'")
#     matrix = result['unfiltered_direction_matrix'] if magnitude_matrix == 'unfiltered' else result['direction_matrix']
#
#     rows = []
#     for (gi, gj), p in p_values.items():
#         score = matrix.loc[gi, gj] if (gi in matrix.index and gj in matrix.columns) else np.nan
#         rows.append({'regulator': gi, 'target': gj, 'p_value': p, 'score': score})
#
#     edge_df = pd.DataFrame(rows)
#     edge_df['abs_score'] = edge_df['score'].abs()
#     edge_df = (
#         edge_df.sort_values(['p_value', 'abs_score'], ascending=[True, False])
#         .drop(columns='abs_score')
#         .reset_index(drop=True)
#     )
#     return edge_df
