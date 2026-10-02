"""Per-replicate data for the network_sweep heatmap (matching the reference PDF's layout, e17
dropped per user instruction): 4 topologies (grn_n6_e5_pos100_density = '5 edges', grn_n6_e9_
pos100_center = '9 edges' AND 'all positive' [same 3 replicates reused under both column groups,
confirmed by matching values in the reference PDF], grn_n6_e9_pos0_sign_ratio = 'all negative',
grn_n6_e9_pos50_sign_ratio = 'balanced'), 3 replicates each, t1=1.

2 metrics (AUPRC, F1 top-k -- F1(threshold)/natural dropped per instruction), 3 variants
(directed_unsigned, signed_directed, undirected), 8 methods (TODO4v2 + 7 BEELINE competitors).

Reuses todo4v2_row_for_dataset from summary_table_todo4v2.py (per-JSON = per-replicate already)
and beeline_gmm_analysis_output.csv (network_sweep_final's t1=1 competitor CSV, twin_paired
scheme only) joined on (dataset_id, run_id=f"simrep{label}_twin_paired").
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/larry_hematopoiesis_validation")
from benchmarks.network_benchmarks.analysis_plots import summary_table_todo4v2 as S
from benchmarks.network_benchmarks.score import todo4v2_sim_scoring as T

from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: project root for the clean_data/ repointing, see REPOINT_LOG.tsv]
ROOT = T.ROOT
HERE = T.HERE

TOPOLOGIES = {
    "5 edges": "grn_n6_e5_pos100_density",
    "9 edges": "grn_n6_e9_pos100_center",
    "13 edges": "n6_e13_pos100",
    "all negative": "grn_n6_e9_pos0_sign_ratio",
    "balanced": "grn_n6_e9_pos50_sign_ratio",
}
# e13_pos100 is a separate sub-benchmark: different JSON dir, different (non-_gmm) beeline CSV,
# 1-indexed reps (rep1/rep2/rep3) instead of the broader sweep's 0-indexed (rep0/rep1/rep2).
E13_STEM = "n6_e13_pos100"
ALGOS = ["PEARSON", "PPCOR", "PIDC", "GRNBOOST2", "SCSGL", "SCODE", "GENIE3"]
VARIANTS = ["directed_unsigned", "signed_directed", "undirected"]
VARIANT_SUFFIX = {"directed_unsigned": "", "signed_directed": "_signed", "undirected": "_undirected"}


def pull_group(rows, group_label, dataset_stem, rep_offset, json_dir, gt_root, beeline_csv):
    b = pd.read_csv(beeline_csv)
    b = b[b.scheme == "twin_paired"]
    for rep_idx in (0, 1, 2):
        dataset_id = f"{dataset_stem}_rep{rep_idx + rep_offset}"
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
                rows.append(dict(group=group_label, dataset_id=dataset_id, rep=rep_idx,
                                  method="TODO4v2", variant=variant,
                                  auprc=todo_row[f"auprc{suf}"],
                                  f1_topk=todo_row[f"f1_topk{suf}"]))

            comp = b[(b.dataset_id == dataset_id) & (b.run_id == run_id)]
            for _, r in comp.iterrows():
                for variant in VARIANTS:
                    suf = VARIANT_SUFFIX[variant]
                    rows.append(dict(group=group_label, dataset_id=dataset_id, rep=rep_idx,
                                      method=r["algorithm"], variant=variant,
                                      auprc=r[f"auprc{suf}"], f1_topk=r[f"f1_topk{suf}"]))


def main():
    nsf_json_dir = f"{ROOT}/analysis_data/network_sweep_final/twinfer_inference_allpairs"
    nsf_gt_root = f"{ROOT}/code/Beeline/inputs/network_sweep_final"
    nsf_beeline_csv = f"{ROOT}/analysis_data/network_sweep_final/beeline_gmm_analysis_output.csv"

    e13_json_dir = f"{ROOT}/analysis_data/network_sweep_final/e13_pos100/twinfer_inference_allpairs"
    e13_gt_root = f"{ROOT}/code/Beeline/inputs/e13_pos100"
    e13_beeline_csv = f"{ROOT}/analysis_data/network_sweep_final/e13_pos100/beeline_analysis_output.csv"

    rows = []
    for group_label, dataset_stem in TOPOLOGIES.items():
        if dataset_stem == E13_STEM:
            pull_group(rows, group_label, dataset_stem, rep_offset=1,
                       json_dir=e13_json_dir, gt_root=e13_gt_root, beeline_csv=e13_beeline_csv)
        else:
            pull_group(rows, group_label, dataset_stem, rep_offset=0,
                       json_dir=nsf_json_dir, gt_root=nsf_gt_root, beeline_csv=nsf_beeline_csv)

    df = pd.DataFrame(rows)
    # each dataset_id (one topology instance) has ~10 simulation replicates -- the reference PDF
    # shows one value per topology instance (columns 1/2/3), so average over sim replicates here.
    agg = df.groupby(["group", "dataset_id", "rep", "method", "variant"], as_index=False)[["auprc", "f1_topk"]].mean()
    # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] out_csv = f"{HERE}/heatmap_data_network_sweep.csv"
    out_csv = f"{TWINFER_PROJECT_ROOT}/analysis_data/benchmarks/network_benchmarks/score_results/heatmap_data_network_sweep.csv"
    agg.to_csv(out_csv, index=False)
    print(f"{len(df)} raw rows -> {len(agg)} aggregated rows -> {out_csv}")
    print(agg.groupby(["group", "method"]).size().unstack("group"))


if __name__ == "__main__":
    main()
