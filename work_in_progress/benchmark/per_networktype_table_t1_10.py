"""Same format as the earlier 'unsigned auprc_x, t1=10' table (rows = method, values = mean
auprc_x), but broken out per network/sim type (topology group) within each benchmark family,
instead of pooling all replicates of a family into one number.

auprc_x per method per group = mean(raw auprc across that group's replicates) / mean(random
baseline across that group's replicates) -- same convention as todo4v2_sim_scoring.py's
_competitor_summary, just grouped by topology instead of by whole family.
"""
import glob
import json
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/larry_hematopoiesis_validation")
import todo4v2_sim_scoring as T

ROOT = T.ROOT
HERE = T.HERE
TOPO_MAP = {"GSD": "GSD.txt", "HSC": "HSC.txt", "VSC": "VSC.txt", "mCAD": "mCAD.txt",
            "B_cell_activation": "B_cell.txt", "Circadian_cycle": "circadian.txt",
            "EMT": "EMT.txt", "Pluripotent": "Pluripotent.txt"}


def strip_rep(dataset_id):
    return re.sub(r"_rep\d+$", "", dataset_id)


def todo4v2_and_crosscorr_rows(json_dir, resolver, group_fn):
    files = sorted(glob.glob(f"{json_dir}/*_all_results.json"))
    rows = []
    for f in files:
        d = json.load(open(f))
        tsi = d.get("twin_score_inputs")
        gr = d.get("gated_regulation")
        if tsi is None or gr is None or gr.get("z_reg_gated") is None:
            continue
        dd = pd.DataFrame(tsi["data"], columns=tsi["columns"])
        dd = dd[dd.gene_1 != dd.gene_2].reset_index(drop=True)
        dd = dd[dd.rho_cross_xy.notna()].reset_index(drop=True)
        if len(dd) < 3:
            continue
        gene_names = d["gene_names"]
        gt_path = resolver(d)
        if gt_path is None:
            continue
        import os
        if not os.path.exists(gt_path):
            continue
        true_edges = T.load_true_edges(gt_path, gene_names)
        U_present = list(zip(dd.gene_1, dd.gene_2))
        scored = set(U_present)
        U_full = [(a, b) for a in gene_names for b in gene_names if a != b and (a, b) in scored]
        y_full = np.array([1 if p in true_edges else 0 for p in U_full])
        if y_full.sum() < 1 or y_full.sum() == len(y_full):
            continue

        z_reg_map = gr["z_reg_gated"]
        sc_todo, _ = T.todo4v2_score(dd, U_present, z_reg_map, "nogate")
        score_map = dict(zip(U_present, sc_todo))
        floor = np.nanmin(sc_todo[np.isfinite(sc_todo)]) - 1.0 if np.isfinite(sc_todo).any() else -1.0
        sc_full = np.array([score_map.get(p, floor) for p in U_full])
        rep_todo = T.full_report(sc_full, y_full)

        cc_map = dict(zip(U_present, dd.rho_cross_xy.abs()))
        sc_cc = np.array([cc_map.get(p, 0.0) for p in U_full])
        rep_cc = T.full_report(sc_cc, y_full)

        rows.append(dict(group=group_fn(d), n_pairs=len(U_full), n_true=int(y_full.sum()),
                          todo4v2_auprc=rep_todo["auprc"], crosscorr_auprc=rep_cc["auprc"]))
    return pd.DataFrame(rows)


def build_table(name, own_rows, beeline_df):
    rand_by_group = own_rows.groupby("group").apply(lambda x: (x.n_true / x.n_pairs).mean())
    todo_by_group = own_rows.groupby("group")["todo4v2_auprc"].mean() / rand_by_group
    cc_by_group = own_rows.groupby("group")["crosscorr_auprc"].mean() / rand_by_group

    comp_by_group = beeline_df.groupby(["group", "algorithm"])["auprc"].mean().unstack("algorithm")
    comp_by_group = comp_by_group.div(rand_by_group.reindex(comp_by_group.index), axis=0)

    table = comp_by_group.copy()
    table.loc[:, "TODO4v2"] = todo_by_group.reindex(table.index)
    table.loc[:, "crosscorr"] = cc_by_group.reindex(table.index)
    table = table[["TODO4v2", "crosscorr"] + [c for c in table.columns if c not in ("TODO4v2", "crosscorr")]]
    print(f"\n=== {name}: mean auprc_x per network type (t1=10) ===")
    print(table.round(3).to_string())
    table.to_csv(f"{HERE}/per_networktype_table_t1_10_{name}.csv")
    return table


def main():
    # e13_pos100
    rows = todo4v2_and_crosscorr_rows(
        f"{ROOT}/analysis_data/network_sweep_final/e13_pos100/twinfer_inference_allpairs_t1_10",
        lambda d: d.get("ground_truth_matrix"),
        lambda d: strip_rep(d.get("dataset_id")))
    b = pd.read_csv(f"{ROOT}/analysis_data/network_sweep_final/e13_pos100/beeline_analysis_output_t1_10.csv")
    b["group"] = b["dataset_id"].apply(strip_rep)
    build_table("network_sweep_e13", rows, b)

    # mixed_network_sweep
    rows = todo4v2_and_crosscorr_rows(
        f"{ROOT}/analysis_data/mixed_network_sweep/twinfer_inference_allpairs_t1_10",
        lambda d: f"{ROOT}/input_data/mixed_network_sweep/{d.get('dataset_id')}.txt",
        lambda d: strip_rep(d.get("dataset_id")))
    b = pd.read_csv(f"{ROOT}/analysis_data/mixed_network_sweep/beeline_analysis_output_t1_10.csv")
    b["group"] = b["dataset_id"].apply(strip_rep)
    build_table("mixed_network_sweep", rows, b)

    # real_networks (group = sim_type directly, no rep suffix to strip)
    rows = todo4v2_and_crosscorr_rows(
        f"{ROOT}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs_t1_10",
        lambda d: f"{ROOT}/input_data/real_world_networks/{TOPO_MAP.get(d.get('sim_type'), '')}",
        lambda d: d.get("sim_type"))
    b = pd.read_csv(f"{ROOT}/analysis_data/paper_analysis/real_networks/beeline_analysis_output_t1_10.csv")
    b["group"] = b["dataset_id"]
    build_table("real_networks", rows, b)

    # network_sweep_final broader
    import os
    nsf_csv = f"{ROOT}/analysis_data/network_sweep_final/beeline_gmm_analysis_output_t1_10.csv"
    if os.path.exists(nsf_csv):
        rows = todo4v2_and_crosscorr_rows(
            f"{ROOT}/analysis_data/network_sweep_final/twinfer_inference_allpairs_t1_10",
            lambda d: d.get("ground_truth_matrix"),
            lambda d: strip_rep(d.get("dataset_id")))
        b = pd.read_csv(nsf_csv)
        b["group"] = b["dataset_id"].apply(strip_rep)
        build_table("network_sweep_final_broader", rows, b)


if __name__ == "__main__":
    main()
