"""
Convert infer_with_twinfer()'s output into a ranked edge list, for benchmarking or
downstream analysis.

Ranking is done by the directed significance p-value (result['directed_p_values']),
not raw correlation magnitude. Raw magnitude alone is not a fair ranking criterion here:
statistical significance depends on both magnitude and that specific gene pair's null
distribution width (which varies pair to pair), so a noisy, non-significant edge can have
a larger raw correlation than a genuinely significant one. Ranking by p-value first lets
non-significant edges sink to the bottom on their own merits; magnitude is used only as a
tie-breaker, since a fixed number of shuffles gives p-values a resolution floor (e.g. many
strong true edges can all report p=0.0 at n_shuffles=10000) where magnitude is the only
thing left to differentiate them.

This is agnostic to which infer_direction_for_which_edges mode produced `result`:
    - "all-edges": directed_p_values covers every gene pair -- intended for benchmarking
      against a full ground-truth network (unbiased, no pre-filtering by TwINFER's own
      earlier screening steps).
    - "single-state" (the default) / "all-potential-regulation": directed_p_values only
      covers pairs that survived the earlier undirected screen and single/multi-state
      classification -- for real analysis, the ranked list is expected to only include
      those pairs; that's TwINFER's actual staged inference, not a limitation to work around.
"""
import numpy as np
import pandas as pd


def twinfer_to_ranked_edges(result, magnitude_matrix='unfiltered'):
    """
    Parameters
    ----------
    result : dict
        Output of infer_with_twinfer(). Must contain 'directed_p_values' (added alongside
        'final_directed_edges' -- requires calling identify_actual_directed_edges /
        infer_with_twinfer from a version of this codebase that returns it).
    magnitude_matrix : {'unfiltered', 'filtered'}
        Which correlation matrix to pull the tie-breaking magnitude from.
        'unfiltered' -> result['unfiltered_direction_matrix'] (recommended: the raw value,
            not zeroed out by the significance threshold).
        'filtered' -> result['direction_matrix'].

    Returns
    -------
    pd.DataFrame with columns ['regulator', 'target', 'p_value', 'score'], sorted by
    p_value ascending with |score| descending as the tie-break.
    """
    p_values = result.get('directed_p_values')
    if not p_values:
        raise ValueError(
            "result['directed_p_values'] is empty or missing -- either no directed pairs "
            "were tested (check infer_direction_for_which_edges and the classification "
            "results in result['gene_lists']), or this result came from a version of "
            "infer_with_twinfer that doesn't return directed_p_values yet."
        )

    if magnitude_matrix not in ('unfiltered', 'filtered'):
        raise ValueError("magnitude_matrix must be 'unfiltered' or 'filtered'")
    matrix = result['unfiltered_direction_matrix'] if magnitude_matrix == 'unfiltered' else result['direction_matrix']

    rows = []
    for (gi, gj), p in p_values.items():
        score = matrix.loc[gi, gj] if (gi in matrix.index and gj in matrix.columns) else np.nan
        rows.append({'regulator': gi, 'target': gj, 'p_value': p, 'score': score})

    edge_df = pd.DataFrame(rows)
    edge_df['abs_score'] = edge_df['score'].abs()
    edge_df = (
        edge_df.sort_values(['p_value', 'abs_score'], ascending=[True, False])
        .drop(columns='abs_score')
        .reset_index(drop=True)
    )
    return edge_df
