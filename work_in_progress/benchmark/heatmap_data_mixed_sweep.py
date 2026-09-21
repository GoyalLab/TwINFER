"""Per-replicate data for mixed_sweep, same structure as heatmap_data_network_sweep.py: one
column per topology instance (rep0/rep1/rep2), each column = mean over that instance's ~9 sim
replicates. 16 topology families (n6/n10 x e6/e12/e10/e20 x c3/c4/c5/c8 x a0/a3/a5 combos already
established this session), 3 variants, 2 metrics, TODO4v2 + 7 competitors. t1=1, twin_paired only.
"""
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/larry_hematopoiesis_validation")
import summary_table_todo4v2 as S
import todo4v2_sim_scoring as T

ROOT = T.ROOT
HERE = T.HERE

VARIANTS = ["directed_unsigned", "signed_directed", "undirected"]
VARIANT_SUFFIX = {"directed_unsigned": "", "signed_directed": "_signed", "undirected": "_undirected"}

TOPOLOGIES = [
    "grn_n10_e10_c5_a0_pos50", "grn_n10_e10_c5_a5_pos50", "grn_n10_e10_c8_a0_pos50", "grn_n10_e10_c8_a5_pos50",
    "grn_n10_e20_c5_a0_pos50", "grn_n10_e20_c5_a5_pos50", "grn_n10_e20_c8_a0_pos50", "grn_n10_e20_c8_a5_pos50",
    "grn_n6_e12_c3_a0_pos50", "grn_n6_e12_c3_a3_pos50", "grn_n6_e12_c4_a0_pos50", "grn_n6_e12_c4_a3_pos50",
    "grn_n6_e6_c3_a0_pos50", "grn_n6_e6_c3_a3_pos50", "grn_n6_e6_c4_a0_pos50", "grn_n6_e6_c4_a3_pos50",
]


def main():
    json_dir = f"{ROOT}/analysis_data/mixed_network_sweep/twinfer_inference_allpairs"
    gt_root = f"{ROOT}/code/Beeline/inputs/mixed_network_sweep"
    beeline_csv = f"{ROOT}/analysis_data/mixed_network_sweep/beeline_analysis_output.csv"
    b = pd.read_csv(beeline_csv)
    b = b[b.scheme == "twin_paired"]

    rows = []
    for dataset_stem in TOPOLOGIES:
        for rep_idx in (0, 1, 2):
            dataset_id = f"{dataset_stem}_rep{rep_idx}"
            files = sorted(glob.glob(f"{json_dir}/{dataset_id}_rep*_all_results.json"))
            for f in files:
                d = json.load(open(f))
                sim_rep_label = d.get("label")
                gt_path = f"{gt_root}/{dataset_id}/GroundTruthNetwork.csv"
                if not os.path.exists(gt_path):
                    continue
                gt_df = pd.read_csv(gt_path)
                todo_row = S.todo4v2_row_for_dataset(f, gt_df)
                if todo_row is None:
                    continue
                run_id = f"simrep{sim_rep_label}_twin_paired"

                for variant in VARIANTS:
                    suf = VARIANT_SUFFIX[variant]
                    rows.append(dict(group=dataset_stem, dataset_id=dataset_id, rep=rep_idx,
                                      method="TODO4v2", variant=variant,
                                      auprc=todo_row[f"auprc{suf}"], f1_topk=todo_row[f"f1_topk{suf}"]))

                comp = b[(b.dataset_id == dataset_id) & (b.run_id == run_id)]
                for _, r in comp.iterrows():
                    for variant in VARIANTS:
                        suf = VARIANT_SUFFIX[variant]
                        rows.append(dict(group=dataset_stem, dataset_id=dataset_id, rep=rep_idx,
                                          method=r["algorithm"], variant=variant,
                                          auprc=r[f"auprc{suf}"], f1_topk=r[f"f1_topk{suf}"]))

    df = pd.DataFrame(rows)
    agg = df.groupby(["group", "dataset_id", "rep", "method", "variant"], as_index=False)[["auprc", "f1_topk"]].mean()
    out_csv = f"{HERE}/heatmap_data_mixed_sweep.csv"
    agg.to_csv(out_csv, index=False)
    print(f"{len(df)} raw rows -> {len(agg)} aggregated rows -> {out_csv}")


if __name__ == "__main__":
    main()
