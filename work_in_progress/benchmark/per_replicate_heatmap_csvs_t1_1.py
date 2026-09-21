"""One CSV per benchmark TYPE (network_sweep = e13_pos100 + network_sweep_final broader combined,
mixed_sweep = mixed_network_sweep, real_networks), t1=1 (the standard/best-performing
measurement point), every replicate scored separately, network name always explicit.

Columns: network (topology/sim-type name, rep-stripped), dataset_id (raw, rep-suffixed),
run_id (sim replicate), n_pairs, n_true, then auprc_x for: todo4v2_old, todo4v2_new,
todo4v2_nogate, crosscorr, and each of the 7 BEELINE competitors (PEARSON/PPCOR/PIDC/GENIE3/
GRNBOOST2/SCODE/SCSGL) -- all on the SAME per-replicate random baseline (n_true/n_pairs), so
every column in a row is directly comparable and heatmap-ready.

Competitor auprc pulled from the twin_paired scheme only (matching TwINFER's own convention;
network_sweep_final's beeline_gmm_analysis_output.csv also has a spread scheme, dropped here for
apples-to-apples comparison against TODO4v2/crosscorr, which only ever see twin_paired cells).
"""
import glob
import json
import os
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
HASH_RE = re.compile(r"([0-9a-f]{8})$")
ALGOS = ["PEARSON", "PPCOR", "PIDC", "GENIE3", "GRNBOOST2", "SCODE", "SCSGL"]


def strip_rep(dataset_id):
    return re.sub(r"_rep\d+$", "", dataset_id)


def score_replicate(json_path, resolver, family):
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
    gene_names = d["gene_names"]
    gt_path = resolver(d)
    if gt_path is None or not os.path.exists(gt_path):
        return None
    true_edges = T.load_true_edges(gt_path, gene_names)
    U_present = list(zip(dd.gene_1, dd.gene_2))
    scored = set(U_present)
    U_full = [(a, b) for a in gene_names for b in gene_names if a != b and (a, b) in scored]
    y_full = np.array([1 if p in true_edges else 0 for p in U_full])
    if y_full.sum() < 1 or y_full.sum() == len(y_full):
        return None

    z_reg_map = gr["z_reg_gated"]
    row = dict(n_pairs=len(U_full), n_true=int(y_full.sum()))
    for variant in ("old", "new", "nogate"):
        sc, _ = T.todo4v2_score(dd, U_present, z_reg_map, variant)
        score_map = dict(zip(U_present, sc))
        floor = np.nanmin(sc[np.isfinite(sc)]) - 1.0 if np.isfinite(sc).any() else -1.0
        sc_full = np.array([score_map.get(p, floor) for p in U_full])
        row[f"todo4v2_{variant}_auprc_x"] = T.full_report(sc_full, y_full)["auprc_x"]

    cc_map = dict(zip(U_present, dd.rho_cross_xy.abs()))
    sc_cc = np.array([cc_map.get(p, 0.0) for p in U_full])
    row["crosscorr_auprc_x"] = T.full_report(sc_cc, y_full)["auprc_x"]

    if family in ("network_sweep_e13", "network_sweep_final_broader", "mixed_network_sweep"):
        row["network"] = strip_rep(d.get("dataset_id"))
        row["dataset_id"] = d.get("dataset_id")
        row["run_id"] = f"simrep{d.get('label')}_twin_paired"
    else:  # real_networks
        row["network"] = d.get("sim_type")
        row["dataset_id"] = d.get("sim_type")
        m = HASH_RE.search(d.get("analysis_key") or "")
        row["run_id_hash"] = m.group(1) if m else None
    row["family"] = family
    return row


def attach_competitors(rows_df, beeline_csv, family):
    b = pd.read_csv(beeline_csv)
    if "scheme" in b.columns:
        b = b[b.scheme == "twin_paired"]
    if family == "real_networks":
        piv_rows = []
        for h in rows_df["run_id_hash"].dropna().unique():
            sub = b[b.run_id.str.contains(h)]
            for dataset_id, algo, auprc in zip(sub.dataset_id, sub.algorithm, sub.auprc):
                piv_rows.append((dataset_id, h, algo, auprc))
        piv = pd.DataFrame(piv_rows, columns=["dataset_id", "run_id_hash", "algorithm", "auprc"])
        wide = piv.pivot_table(index=["dataset_id", "run_id_hash"], columns="algorithm", values="auprc").reset_index()
        merged = rows_df.merge(wide, on=["dataset_id", "run_id_hash"], how="left")
    else:
        wide = b.pivot_table(index=["dataset_id", "run_id"], columns="algorithm", values="auprc").reset_index()
        merged = rows_df.merge(wide, on=["dataset_id", "run_id"], how="left")

    rand = merged.n_true / merged.n_pairs
    for algo in ALGOS:
        if algo in merged.columns:
            merged[f"{algo}_auprc_x"] = merged[algo] / rand
            merged = merged.drop(columns=[algo])
        else:
            merged[f"{algo}_auprc_x"] = np.nan
    return merged


def build_family(family, json_dir, resolver):
    files = sorted(glob.glob(f"{json_dir}/*_all_results.json"))
    rows = [r for r in (score_replicate(f, resolver, family) for f in files) if r]
    return pd.DataFrame(rows)


def main():
    # --- network_sweep: e13_pos100 + network_sweep_final broader combined ---
    df_e13 = build_family("network_sweep_e13",
                           f"{ROOT}/analysis_data/network_sweep_final/e13_pos100/twinfer_inference_allpairs",
                           lambda d: d.get("ground_truth_matrix"))
    df_e13 = attach_competitors(df_e13, f"{ROOT}/analysis_data/network_sweep_final/e13_pos100/beeline_analysis_output.csv", "network_sweep_e13")

    df_nsf = build_family("network_sweep_final_broader",
                           f"{ROOT}/analysis_data/network_sweep_final/twinfer_inference_allpairs",
                           lambda d: d.get("ground_truth_matrix"))
    df_nsf = attach_competitors(df_nsf, f"{ROOT}/analysis_data/network_sweep_final/beeline_gmm_analysis_output.csv", "network_sweep_final_broader")

    network_sweep = pd.concat([df_e13, df_nsf], ignore_index=True)
    cols = ["family", "network", "dataset_id", "run_id", "n_pairs", "n_true",
            "todo4v2_old_auprc_x", "todo4v2_new_auprc_x", "todo4v2_nogate_auprc_x", "crosscorr_auprc_x"] + \
           [f"{a}_auprc_x" for a in ALGOS]
    network_sweep = network_sweep[cols]
    network_sweep.to_csv(f"{HERE}/heatmap_network_sweep_t1_1.csv", index=False)
    print(f"network_sweep: {len(network_sweep)} rows -> {HERE}/heatmap_network_sweep_t1_1.csv")

    # --- mixed_sweep ---
    df_mns = build_family("mixed_network_sweep",
                           f"{ROOT}/analysis_data/mixed_network_sweep/twinfer_inference_allpairs",
                           lambda d: f"{ROOT}/input_data/mixed_network_sweep/{d.get('dataset_id')}.txt")
    df_mns = attach_competitors(df_mns, f"{ROOT}/analysis_data/mixed_network_sweep/beeline_analysis_output.csv", "mixed_network_sweep")
    df_mns = df_mns[cols]
    df_mns.to_csv(f"{HERE}/heatmap_mixed_sweep_t1_1.csv", index=False)
    print(f"mixed_sweep: {len(df_mns)} rows -> {HERE}/heatmap_mixed_sweep_t1_1.csv")

    # --- real_networks ---
    df_rn = build_family("real_networks",
                          f"{ROOT}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs",
                          lambda d: f"{ROOT}/input_data/real_world_networks/{TOPO_MAP.get(d.get('sim_type'), '')}")
    df_rn = attach_competitors(df_rn, f"{ROOT}/analysis_data/paper_analysis/real_networks/beeline_scores.csv", "real_networks")
    rn_cols = ["family", "network", "dataset_id", "n_pairs", "n_true",
               "todo4v2_old_auprc_x", "todo4v2_new_auprc_x", "todo4v2_nogate_auprc_x", "crosscorr_auprc_x"] + \
              [f"{a}_auprc_x" for a in ALGOS]
    df_rn = df_rn[rn_cols]
    df_rn.to_csv(f"{HERE}/heatmap_real_networks_t1_1.csv", index=False)
    print(f"real_networks: {len(df_rn)} rows -> {HERE}/heatmap_real_networks_t1_1.csv")


if __name__ == "__main__":
    main()
