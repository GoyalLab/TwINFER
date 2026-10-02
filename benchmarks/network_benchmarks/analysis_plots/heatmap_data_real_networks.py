"""Per-network data for real_networks. NOTE: unlike network_sweep/mixed_sweep, real_networks has
no '3 topology instances per family' structure -- each of the 8 named networks (GSD/HSC/VSC/mCAD/
B_cell_activation/Circadian_cycle/EMT/Pluripotent) IS one fixed real topology with multiple
simulation replicates, not 3 parametric instances. So this is one row per (network, method,
variant), mean over ALL that network's simulation replicates -- a single column per network, not
3 sub-columns. t1=1, twin_paired only.
"""
import glob
import json
import os
import sys

import pandas as pd

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/larry_hematopoiesis_validation")
from benchmarks.network_benchmarks.analysis_plots import summary_table_todo4v2 as S
from benchmarks.network_benchmarks.score import todo4v2_sim_scoring as T

ROOT = T.ROOT
HERE = T.HERE
VARIANTS = ["directed_unsigned", "signed_directed", "undirected"]
VARIANT_SUFFIX = {"directed_unsigned": "", "signed_directed": "_signed", "undirected": "_undirected"}
TOPO_MAP = {"GSD": "GSD.txt", "HSC": "HSC.txt", "VSC": "VSC.txt", "mCAD": "mCAD.txt",
            "B_cell_activation": "B_cell.txt", "Circadian_cycle": "circadian.txt",
            "EMT": "EMT.txt", "Pluripotent": "Pluripotent.txt"}


def main():
    json_dir = f"{ROOT}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs"
    beeline_csv = f"{ROOT}/analysis_data/paper_analysis/real_networks/beeline_scores.csv"
    gt_root = f"{ROOT}/code/Beeline/inputs"
    b = pd.read_csv(beeline_csv)
    b = b[b.scheme == "twin_paired"]

    rows = []
    files = sorted(glob.glob(f"{json_dir}/*_all_results.json"))
    for f in files:
        d = json.load(open(f))
        sim_type = d.get("sim_type")
        topo_file = TOPO_MAP.get(sim_type)
        if topo_file is None:
            continue
        gt_path = f"{ROOT}/input_data/real_world_networks/{topo_file}"
        if not os.path.exists(gt_path):
            continue
        gt_df = pd.DataFrame()  # todo4v2_row_for_dataset needs a BEELINE-format gt_df; build one
        import numpy as np
        M = np.loadtxt(gt_path, delimiter=",")
        gene_names = d["gene_names"]
        n = len(gene_names)
        recs = [(gene_names[i], gene_names[j], "+" if M[i, j] > 0 else "-")
                for i in range(n) for j in range(n) if i != j and M[i, j] != 0]
        gt_df = pd.DataFrame(recs, columns=["Gene1", "Gene2", "Type"])
        todo_row = S.todo4v2_row_for_dataset(f, gt_df)
        if todo_row is None:
            continue
        for variant in VARIANTS:
            suf = VARIANT_SUFFIX[variant]
            rows.append(dict(group=sim_type, method="TODO4v2", variant=variant,
                              auprc=todo_row[f"auprc{suf}"], f1_topk=todo_row[f"f1_topk{suf}"]))

    comp = b.copy()
    for variant in VARIANTS:
        suf = VARIANT_SUFFIX[variant]
        for (dataset_id, algo), grp in comp.groupby(["dataset_id", "algorithm"]):
            rows.append(dict(group=dataset_id, method=algo, variant=variant,
                              auprc=grp[f"auprc{suf}"].mean(), f1_topk=grp[f"f1_topk{suf}"].mean()))

    df = pd.DataFrame(rows)
    agg = df.groupby(["group", "method", "variant"], as_index=False)[["auprc", "f1_topk"]].mean()
    out_csv = f"{HERE}/heatmap_data_real_networks.csv"
    agg.to_csv(out_csv, index=False)
    print(f"{len(agg)} rows -> {out_csv}")


if __name__ == "__main__":
    main()
