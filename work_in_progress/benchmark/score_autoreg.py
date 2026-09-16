"""
Scores autoregulation's 40 TwINFER JSONs and 40x7 BEELINE rankedEdges.csv
against ground truth, reusing the same extracted notebook scoring functions
as score_twinfer_20260824.py / score_beeline_20260824.py.
"""
import sys
sys.path.insert(0, "/tmp")

import json
from pathlib import Path
import pandas as pd

from twinfer_scoring_extract import (
    load_ground_truth_matrix, get_correct_gene_names, score_twinfer_dataset,
)
import beeline_scoring_extract as bse

TWINFER_JSON_DIR = Path("/home/gzu5140/TwINFER_KA/analysis_data/autoregulation_benchmark_20260824/twinfer_inference")
GROUND_TRUTH_DIR = Path("/home/gzu5140/TwINFER_KA/input_data")
BEELINE_OUTPUT_ROOT = Path("/home/gzu5140/TwINFER_KA/analysis_data/autoregulation_benchmark_20260824/beeline_inference")
bse.BEELINE_INPUT_ROOT = Path("/home/gzu5140/TwINFER_KA/code/Beeline/inputs/autoregulation_20260824")

TOPOLOGIES = ["autoreg_g2_pos", "autoreg_g2_neg", "autoreg_g3_pos", "autoreg_g3_neg"]


def score_twinfer():
    rows = []
    for ds in TOPOLOGIES:
        gt_path = GROUND_TRUTH_DIR / f"{ds}.txt"
        for json_path in sorted(TWINFER_JSON_DIR.glob(f"{ds}_label*_all_results.json")):
            label = json_path.stem.replace(f"{ds}_label", "").replace("_all_results", "")
            results = json.load(open(json_path))
            gene_names = get_correct_gene_names(results)
            gt = load_ground_truth_matrix(gt_path, gene_names=gene_names)
            row = score_twinfer_dataset(results, gene_names, gt)
            row["dataset_id"] = ds
            row["label"] = label
            rows.append(row)
    df = pd.DataFrame(rows)
    out = "/home/gzu5140/TwINFER_KA/analysis_data/autoregulation_benchmark_20260824/twinfer_analysis_output.csv"
    df.to_csv(out, index=False)
    print(f"Scored {len(df)} TwINFER autoregulation results -> {out}")
    print(df["dataset_id"].value_counts().sort_index())
    return df


def score_beeline():
    df = bse.score_beeline_sweep_folder(BEELINE_OUTPUT_ROOT, verbose=False)
    out = "/home/gzu5140/TwINFER_KA/analysis_data/autoregulation_benchmark_20260824/beeline_analysis_output.csv"
    df.to_csv(out, index=False)
    print(f"Scored {len(df)} BEELINE autoregulation rows -> {out}")
    print(df.groupby(["dataset_id", "algorithm"]).size().unstack())
    return df


if __name__ == "__main__":
    score_twinfer()
    print()
    score_beeline()
