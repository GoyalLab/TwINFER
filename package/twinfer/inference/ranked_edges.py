"""Ranked gene-pair lists from the output of infer_with_twinfer (twinfer.inference.covariance_decomposition_z_scores).

One list per score, by default
  z_rho_s       twin-shared part of the correlation (heterogeneity: a shared cell state)
  z_rho_Delta   twin-difference correlation (regulation)
  z_dag         only when the result has two time points: the cross-time correlation of sisters (z_dag_xy: gene_x earlier with
                gene_y later; z_dag_yx the reverse), one row per direction found by `pair_directions` (the rule network_from_scenarios
                draws), source -> target, ranked by |z_dag| of that direction:
                  scenario 2 (regulation)  each direction on its own: |z_dag| > z_dag_threshold; both directions can be listed
                                           (mutual regulation)
                  other scenarios          delta = |rho_dag_xy| - |rho_dag_yx| and its permutation p-value p_delta: one direction
                                           (the sign of delta) when p_delta < direction_alpha. In scenario 3 a shared state
                                           correlates both directions, so only the difference is used.
                  no direction found       one row, the larger |z_dag|, directed False
Each list ranks the gene pairs by |z| (the sign does not change the rank), largest first. By default only pairs whose
expression correlation z_rho is significant are ranked: q_rho <= fdr when the result has a q_rho column (Benjamini-Hochberg,
the default for real data), otherwise |z_rho| > z_rho_threshold. This is how the top lists of the TwinScore procedure are made
(pairs passing Stage I, ranked by |z|).

Example:
    from twinfer.inference.ranked_edges import ranked_edge_lists
    lists = ranked_edge_lists(result, top=100)
    lists["z_rho_Delta"]          # rank, gene_x, gene_y, z_rho_Delta, z_rho, q_rho, inference_scenario, ...
    lists["z_dag"]                # two time points only: rank, source, target, directed, z_dag, z_dag_xy, z_dag_yx, delta, p_delta, ...
    pair_directions(z_dag_xy, z_dag_yx, delta, p_delta, scenario)   # [(forward, z_dag, directed), ...] of one pair
"""
# [2026-10-24 renamed: z_rho_delta, q_rho_delta, rho_delta and q_values_of_z_rho_delta to z_rho_Delta, q_rho_Delta, rho_Delta, q_values_of_z_rho_Delta]
import numpy as np
import pandas as pd


# [2026-10-24 added: one direction rule for the ranked list and the network]
def pair_directions(z_dag_xy, z_dag_yx, delta, p_delta, scenario, direction_alpha=0.05, z_dag_threshold=1.96, two_way_scenarios=(2,)):
    """Directions of one gene pair from its cross-time statistics.

    Returns a list of (forward, z_dag, directed): forward True = gene_x -> gene_y, False = gene_y -> gene_x; z_dag of that direction.
    scenario in `two_way_scenarios` (default 2, regulation): each direction on its own, directed when ``|z_dag|`` > z_dag_threshold;
    both can be returned (mutual regulation). Other scenarios (3: regulation + a shared state, which correlates both directions):
    one direction, the sign of delta, when p_delta < direction_alpha. When no direction is found: [(larger ``|z_dag|``, z_dag, False)].
    Without cross-time statistics (NaN): [(True, NaN, False)].
    """
    if pd.isna(z_dag_xy) or pd.isna(z_dag_yx):
        return [(True, np.nan, False)]
    if scenario in two_way_scenarios:
        found = [(forward, z, True) for forward, z in ((True, z_dag_xy), (False, z_dag_yx)) if abs(z) > z_dag_threshold]
    elif p_delta < direction_alpha:
        found = [(True, z_dag_xy, True)] if delta >= 0 else [(False, z_dag_yx, True)]
    else:
        found = []
    if not found:
        forward = abs(z_dag_xy) >= abs(z_dag_yx)
        found = [(forward, z_dag_xy if forward else z_dag_yx, False)]
    return found


# [2026-10-23 replaced: scores default None adds the cross-correlation list (z_dag) for several time points]
# def ranked_edge_lists(result, scores=("z_rho_s", "z_rho_delta"), time="tb", significant_rho_only=True, fdr=0.05,
#                       z_rho_threshold=1.96, top=None):
# [2026-10-24 replaced: direction_alpha added (column directed of the z_dag list)]
# def ranked_edge_lists(result, scores=None, time="tb", significant_rho_only=True, fdr=0.05, z_rho_threshold=1.96, top=None):
# [2026-10-24 replaced: z_dag_threshold, two_way_scenarios added (pair_directions)]
# def ranked_edge_lists(result, scores=None, time="tb", significant_rho_only=True, fdr=0.05, z_rho_threshold=1.96, top=None,
#                       direction_alpha=0.05):
def ranked_edge_lists(result, scores=None, time="tb", significant_rho_only=True, fdr=0.05, z_rho_threshold=1.96, top=None,
                      direction_alpha=0.05, z_dag_threshold=1.96, two_way_scenarios=(2,)):
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
    direction_alpha, z_dag_threshold, two_way_scenarios : float, float, tuple of int
        z_dag list: the direction rule of `pair_directions` (as network_from_scenarios draws it).

    Returns
    -------
    dict
        {score: DataFrame} with the columns rank (1 = largest ``|score|``), gene_x, gene_y, the score, z_rho, and q_rho, q_rho_Delta,
        inference_scenario where present. The z_dag list has one row per direction (`pair_directions`): source, target, directed,
        z_dag (of that direction), z_dag_xy, z_dag_yx, delta, p_delta instead of gene_x, gene_y. Pairs with a missing score are left
        out; tied rows keep their order in `result`.
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
        # [2026-10-24 replaced: one row per direction of pair_directions (both directions in scenario 2)]
        # forward = np.abs(table["z_dag_xy"].to_numpy()) >= np.abs(table["z_dag_yx"].to_numpy())
        # table = table.assign(source=np.where(forward, table["gene_x"], table["gene_y"]),
        #                      target=np.where(forward, table["gene_y"], table["gene_x"]),
        #                      z_dag=np.where(forward, table["z_dag_xy"], table["z_dag_yx"]),
        #                      # [2026-10-24 added: directed]
        #                      directed=(table["p_delta"] < direction_alpha).to_numpy())
        scenario = table["inference_scenario"] if "inference_scenario" in table.columns else pd.Series(np.nan, index=table.index)
        cross_rows = []
        for (_, row), s in zip(table.iterrows(), scenario):
            for forward, z, directed in pair_directions(row["z_dag_xy"], row["z_dag_yx"], row["delta"], row["p_delta"], s,
                                                        direction_alpha, z_dag_threshold, two_way_scenarios):
                cross_rows.append(row.to_dict() | dict(source=row["gene_x"] if forward else row["gene_y"],
                                                       target=row["gene_y"] if forward else row["gene_x"], z_dag=z, directed=directed))
        cross_table = pd.DataFrame(cross_rows, columns=list(table.columns) + ["source", "target", "z_dag", "directed"])
    for score in scores:
        source_table = cross_table if score == "z_dag" else table
        if score not in source_table.columns:
            raise ValueError(f"result has no column {score}{suffix}")
        ranked = source_table.dropna(subset=[score])
        ranked = ranked.iloc[np.argsort(-np.abs(ranked[score].to_numpy()), kind="stable")]
        if top is not None:
            ranked = ranked.head(top)
        # [2026-10-23 replaced: own columns for the cross-correlation list]
        # columns = ["gene_x", "gene_y", score] + [c for c in extra if c != score]
        if score == "z_dag":
            # [2026-10-24 replaced: column directed added]
            # columns = ["source", "target", "z_dag", "z_dag_xy", "z_dag_yx", "delta", "p_delta"] + extra
            columns = ["source", "target", "directed", "z_dag", "z_dag_xy", "z_dag_yx", "delta", "p_delta"] + extra
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
