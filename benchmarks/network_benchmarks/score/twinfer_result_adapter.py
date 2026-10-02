"""Adapter: current `infer_with_twinfer` result (nested dict / *_all_results.json) -> the OLD flat shape that the
benchmark scorers (`score_twinfer_dataset` in twinfer_scoring_extract, score_cyclic_g3456, real_networks_summary_table,
score_mixed_network_sweep, ...) were written for. Created 2026-09-30 from `adapt_result_schema` in score_mixed_network_sweep.py
(written by the project owner for the mixed sweep) so that every scorer can use it; the scoring math is untouched.

Mapping (new -> old key):
  direction.unfiltered_matrix            -> unfiltered_direction_matrix
  directed_edges (self pairs dropped)    -> final_directed_edges   (the new output always lists (g,g) for every gene; they are
                                                                    a structural placeholder, never a pairwise call, so they are
                                                                    removed when counting -- project owner's decision 2026-09-30)
  signed gamma of ranked_edges (fallback: unfiltered rho_cross) for each final edge -> direction_matrix
  classification.{single_state_regulation, multiple_states_and_reg, multiple_states_no_reg} -> gene_lists
  fan_out                                -> fan_out_log
  NaN entries of unfiltered_matrix (pair statistic not computable) are set to 0.0 = "no evidence", the convention score_mixed_network_sweep
                                           applies in build_master_ranked_table; the older scorers cannot handle NaN (sklearn raises).
  potential_regulation = pairs that PASSED STEP 1 (the correlation screen at whatever alpha_gene_gene_corr was set) = every pair
                         in classification except no_regulation. Checked on a stored mixed-sweep result: the four classification
                         lists partition the n(n-1)/2 unordered pairs.
`ensure_old_schema` is the identity for a result that is already in the old flat shape.
"""
import pandas as pd


def _reconstruct_matrix(raw):
    return pd.DataFrame(raw["data"], index=raw["index"], columns=raw["columns"])


def is_old_schema(record):
    return "unfiltered_direction_matrix" in record and "final_directed_edges" in record


def adapt_result_schema(record, gene_names):
    unfiltered_direction_matrix = record["direction"]["unfiltered_matrix"]
    _u = _reconstruct_matrix(unfiltered_direction_matrix)
    _u_filled = _u.fillna(0.0)   # see module docstring
    unfiltered_direction_matrix = {"data": _u_filled.values.tolist(), "index": list(_u.index), "columns": list(_u.columns)}

    ranked_cols = record["ranked_edges"]["columns"] if record.get("ranked_edges") else []
    gamma_by_pair = {}
    if record.get("ranked_edges") and record["ranked_edges"]["data"]:
        i_g1, i_g2, i_gamma = ranked_cols.index("gene_1"), ranked_cols.index("gene_2"), ranked_cols.index("gamma")
        for r in record["ranked_edges"]["data"]:
            gamma_by_pair[(r[i_g1], r[i_g2])] = r[i_gamma]

    final_directed_edges = [tuple(e) for e in record["directed_edges"] if e[0] != e[1]]

    direction_df = pd.DataFrame(0.0, index=gene_names, columns=gene_names)
    for gi, gj in final_directed_edges:
        # gamma = post-processing signed direction statistic; fall back to the raw unfiltered correlation if the edge has no
        # ranked_edges row. NaN -> 0 (sign unknown: counted as a wrong sign against a nonzero ground-truth sign).
        val = gamma_by_pair.get((gi, gj), _u.loc[gi, gj])
        direction_df.loc[gi, gj] = 0.0 if pd.isna(val) else val

    cl = record["classification"]
    gene_lists = {
        "single_state_regulation": cl.get("single_state_regulation", []),
        "multiple_states_and_reg": cl.get("multiple_states_and_reg", []),
        "multiple_states_no_reg": cl.get("multiple_states_no_reg", []),
    }
    potential_regulation = (
        gene_lists["single_state_regulation"] + gene_lists["multiple_states_and_reg"] + gene_lists["multiple_states_no_reg"]
    )
    n_genes = record.get("n_genes")
    if n_genes is None:
        n_genes = len(record["settings"]["genes"])
    return {
        "sim_type": record.get("dataset_id", record.get("sim_type")),
        "rep_id": record.get("label", record.get("rep_id")),
        "n_genes": n_genes,
        "unfiltered_direction_matrix": unfiltered_direction_matrix,
        "direction_matrix": {"data": direction_df.values.tolist(), "index": list(gene_names), "columns": list(gene_names)},
        "final_directed_edges": final_directed_edges,
        "fan_out_log": record.get("fan_out", []),
        "gene_lists": gene_lists,
        "potential_regulation": potential_regulation,
    }


def ensure_old_schema(record, gene_names=None):
    """Return `record` unchanged if it is already old-schema, else the adapted flat dict."""
    if is_old_schema(record):
        return record
    if gene_names is None:
        gene_names = record.get("gene_names") or list(record["settings"]["genes"])
    return adapt_result_schema(record, gene_names)
