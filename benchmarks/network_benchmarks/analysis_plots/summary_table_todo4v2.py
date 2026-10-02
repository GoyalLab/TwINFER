"""Reproduces the exact 'network_sweep_summary_metrics_table.csv' format the user provided
(columns: variant, dataset_id, method, f1_topk, precision_topk, auprc -- natural/GMM columns
dropped per instruction), with TODO4v2(nogate) -- established as the best-performing variant,
1.4-1.7x better than old/new on every benchmark -- added as a method alongside the 7 existing
BEELINE competitors, for each of 3 dataset TYPES: network_sweep (e13_pos100 + network_sweep_final
broader combined), mixed_sweep, real_networks. All at t1=1 (the best-performing measurement
point established earlier this session).

Reuses score_mixed_network_sweep.py's generic, already-validated primitives unchanged:
  dedupe_predictions/_undirected, add_predicted_sign, top_k_tie_aware_selection(+signed+
  undirected), compute_auprc(+signed+undirected), precision_recall_f1, load_beeline_ground_truth,
  build_edge_universe/build_signed_edge_universe/build_undirected_edge_universe.

TODO4v2's EdgeWeight (so it slots into this magnitude-ranked machinery the same way a
correlation/importance score does): sign(rho_cross_xy) * (score - min(score) over the dataset's
present pairs) -- the shift makes |EdgeWeight| monotonic with TODO4v2's own descending-score
rank (raw score, not abs, is what todo4v2_sim_scoring.py ranks by), and the sign is TODO4v2's
already-established best sign source (Part 3 of the 2026-09-18 handoff: sign(rho_cross_xy), not
sign of the composite score itself).
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/larry_hematopoiesis_validation")
from twinfer.scoring.analytic_zscores import DEFAULT_SD, z_signed
from benchmarks.network_benchmarks.score import todo4v2_sim_scoring as T
from benchmarks.network_benchmarks.score import score_mixed_network_sweep as M

ROOT = T.ROOT
HERE = T.HERE
VARIANT_SUFFIX = {"": "directed_unsigned", "_signed": "signed_directed", "_undirected": "undirected"}
METRICS = [("f1_topk", "f1"), ("precision_topk", "precision")]


def strip_rep(dataset_id):
    return re.sub(r"_rep\d+$", "", dataset_id)


def todo4v2_row_for_dataset(json_path, gt_df):
    d = json.load(open(json_path))
    tsi = d.get("twin_score_inputs")
    gr = d.get("gated_regulation")
    if tsi is None or gr is None or gr.get("z_reg_gated") is None:
        return None
    dd = pd.DataFrame(tsi["data"], columns=tsi["columns"])
    dd = dd[dd.gene_1 != dd.gene_2].reset_index(drop=True)
    dd = dd[dd.rho_cross_xy.notna()].reset_index(drop=True)
    if len(dd) < 3:
        return None

    U = list(zip(dd.gene_1, dd.gene_2))
    z_reg_map = gr["z_reg_gated"]
    sc, _ = T.todo4v2_score(dd, U, z_reg_map, "nogate")
    shifted = sc - np.nanmin(sc[np.isfinite(sc)])
    sign = np.sign(dd.rho_cross_xy.to_numpy())
    sign[sign == 0] = 1.0
    edge_weight = sign * shifted

    ranked_edges = pd.DataFrame({"Gene1": dd.gene_1, "Gene2": dd.gene_2, "EdgeWeight": edge_weight})

    possible_edges, true_edges = M.build_edge_universe(gt_df)
    true_signed_edges = M.build_signed_edge_universe(gt_df)
    possible_pairs_und, true_pairs_und = M.build_undirected_edge_universe(gt_df)
    num_true, num_true_signed, num_true_und = len(true_edges), len(true_signed_edges), len(true_pairs_und)

    predicted = M.dedupe_predictions(ranked_edges)
    predicted_signed = M.add_predicted_sign(predicted)
    predicted_und = M.dedupe_predictions_undirected(ranked_edges)

    auprc = M.compute_auprc(ranked_edges, gt_df)
    auprc_signed = M.compute_auprc_signed(ranked_edges, gt_df)
    auprc_und = M.compute_auprc_undirected(ranked_edges, gt_df)

    sel_topk, _ = M.top_k_tie_aware_selection(predicted, num_true)
    p_topk, _, f1_topk = M.precision_recall_f1(sel_topk, true_edges)
    sel_topk_s, _ = M.top_k_tie_aware_selection_signed(predicted_signed, num_true_signed)
    p_topk_s, _, f1_topk_s = M.precision_recall_f1(sel_topk_s, true_signed_edges)
    sel_topk_u, _ = M.top_k_tie_aware_selection_undirected(predicted_und, num_true_und)
    p_topk_u, _, f1_topk_u = M.precision_recall_f1(sel_topk_u, true_pairs_und)

    return dict(dataset_id=d.get("dataset_id") or d.get("sim_type"),
                algorithm="TODO4v2",
                auprc=auprc, auprc_signed=auprc_signed, auprc_undirected=auprc_und,
                precision_topk=p_topk, f1_topk=f1_topk,
                precision_topk_signed=p_topk_s, f1_topk_signed=f1_topk_s,
                precision_topk_undirected=p_topk_u, f1_topk_undirected=f1_topk_u)


def melt_for_summary(rows_df):
    """Same shape as real_networks_summary_table.py's melt_beeline_wide, restricted to
    metric_family='topk' per instruction to drop natural."""
    rows = []
    for _, r in rows_df.iterrows():
        base = {"method": r["algorithm"], "dataset_id": r["dataset_id"]}
        for suffix, variant in VARIANT_SUFFIX.items():
            rows.append({**base, "variant": variant, "score_name": "auprc", "value": r[f"auprc{suffix}"]})
            for score_name in ("precision", "f1"):
                rows.append({**base, "variant": variant, "score_name": score_name,
                             "value": r[f"{score_name}_topk{suffix}"]})
    return pd.DataFrame(rows)


def build_summary(scores_long):
    blocks = []
    for variant in scores_long["variant"].unique():
        v = scores_long[scores_long["variant"] == variant]
        for dataset_id in sorted(v["dataset_id"].unique()) + ["ALL"]:
            sub = v if dataset_id == "ALL" else v[v["dataset_id"] == dataset_id]
            row = {"variant": variant, "dataset_id": dataset_id}
            for method in sorted(sub["method"].unique()):
                m = sub[sub.method == method]
                out = {"variant": variant, "dataset_id": dataset_id, "method": method}
                out["f1_topk"] = m.loc[m.score_name == "f1", "value"].mean()
                out["precision_topk"] = m.loc[m.score_name == "precision", "value"].mean()
                out["auprc"] = m.loc[m.score_name == "auprc", "value"].mean()
                blocks.append(out)
    return pd.DataFrame(blocks).round(4)


def build_type(name, json_dirs_and_input_roots, beeline_csvs, group_dataset_id=None):
    """json_dirs_and_input_roots: list of (json_dir, gt_root_resolver) pairs (multiple for a
    combined type like network_sweep). gt_root_resolver(dataset_id) -> GroundTruthNetwork.csv
    Path, or None if not found (real_networks splits GSD/HSC/VSC/mCAD vs the other 4 across two
    different BEELINE input roots -- see real_networks_summary_table.py's BEELINE_GROUPS).
    beeline_csvs: list of existing competitor CSVs (already computed, t1=1) to reuse unchanged."""
    todo_rows = []
    for json_dir, gt_resolver in json_dirs_and_input_roots:
        for f in sorted(glob.glob(f"{json_dir}/*_all_results.json")):
            d = json.load(open(f))
            dataset_id = d.get("dataset_id") or d.get("sim_type")
            gt_path = gt_resolver(dataset_id)
            if gt_path is None or not gt_path.exists():
                continue
            gt_df = pd.read_csv(gt_path)
            row = todo4v2_row_for_dataset(f, gt_df)
            if row:
                todo_rows.append(row)
    todo_df = pd.DataFrame(todo_rows)
    if group_dataset_id:
        todo_df["dataset_id"] = todo_df["dataset_id"].apply(group_dataset_id)

    comp_dfs = []
    for csv in beeline_csvs:
        b = pd.read_csv(csv)
        if "scheme" in b.columns:
            b = b[b.scheme == "twin_paired"]
        if group_dataset_id:
            b["dataset_id"] = b["dataset_id"].apply(group_dataset_id)
        comp_dfs.append(b)
    comp_df = pd.concat(comp_dfs, ignore_index=True)

    combined = pd.concat([todo_df, comp_df], ignore_index=True)
    scores_long = melt_for_summary(combined)
    table = build_summary(scores_long)
    out_csv = f"{HERE}/summary_metrics_table_{name}_t1_1.csv"
    table.to_csv(out_csv, index=False)
    print(f"{name}: {len(table)} rows -> {out_csv}")
    return table


def main():
    from pathlib import Path
    beeline_root = Path(ROOT) / "code" / "Beeline" / "inputs"

    e13_root = beeline_root / "e13_pos100"
    nsf_root = beeline_root / "network_sweep_final"
    build_type("network_sweep",
        [(f"{ROOT}/analysis_data/network_sweep_final/e13_pos100/twinfer_inference_allpairs", lambda ds: e13_root / ds / "GroundTruthNetwork.csv"),
         (f"{ROOT}/analysis_data/network_sweep_final/twinfer_inference_allpairs", lambda ds: nsf_root / ds / "GroundTruthNetwork.csv")],
        [f"{ROOT}/analysis_data/network_sweep_final/e13_pos100/beeline_analysis_output.csv",
         f"{ROOT}/analysis_data/network_sweep_final/beeline_gmm_analysis_output.csv"],
        group_dataset_id=strip_rep)

    mns_root = beeline_root / "mixed_network_sweep"
    build_type("mixed_sweep",
        [(f"{ROOT}/analysis_data/mixed_network_sweep/twinfer_inference_allpairs", lambda ds: mns_root / ds / "GroundTruthNetwork.csv")],
        [f"{ROOT}/analysis_data/mixed_network_sweep/beeline_analysis_output.csv"],
        group_dataset_id=strip_rep)

    # real_networks splits across two BEELINE input roots (real_networks_summary_table.py's
    # BEELINE_GROUPS): GSD/HSC/VSC/mCAD live directly under inputs/<net>/, the other 4 under
    # inputs/real_networks/<net>/.
    def real_networks_gt(sim_type):
        p1 = beeline_root / sim_type / "GroundTruthNetwork.csv"
        if p1.exists():
            return p1
        return beeline_root / "real_networks" / sim_type / "GroundTruthNetwork.csv"

    build_type("real_networks",
        [(f"{ROOT}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs", real_networks_gt)],
        [f"{ROOT}/analysis_data/paper_analysis/real_networks/beeline_scores.csv"])


if __name__ == "__main__":
    main()
