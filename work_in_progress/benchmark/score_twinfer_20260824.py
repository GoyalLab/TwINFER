"""
Scores the 150 TwINFER JSON results (rerun_twinfer_150.py's output) against
ground truth, producing twinfer_analysis_output.csv for the 20260824 rerun.

Reuses score_twinfer_dataset and its dependencies verbatim from
benchmark_network_sweep.ipynb (extracted, not retyped, to avoid silently
diverging from the already-debugged scoring logic) -- only the file-discovery
layer is new, driven directly by the known 150-entry manifest instead of the
notebook's fuzzy topology-name matching (unnecessary here since every JSON's
dataset_id is already explicit).
"""
import sys
sys.path.insert(0, "/tmp")  # twinfer_scoring_extract.py
from twinfer_scoring_extract import (
    reconstruct_matrix, load_ground_truth_matrix, get_correct_gene_names,
    score_twinfer_dataset,
)

import json
import pandas as pd
from pathlib import Path

MANIFEST_PATH = "/home/gzu5140/TwINFER_KA/analysis_data/synthetic_network_benchmark_20260824/source_manifest_150.json"
JSON_DIR = Path("/home/gzu5140/TwINFER_KA/analysis_data/synthetic_network_benchmark_20260824/twinfer_inference")
GROUND_TRUTH_DIR = Path("/home/gzu5140/TwINFER_KA/input_data/network_sweep_final")
OUT_CSV = "/home/gzu5140/TwINFER_KA/analysis_data/synthetic_network_benchmark_20260824/twinfer_analysis_output.csv"


def main():
    manifest = json.load(open(MANIFEST_PATH))
    manifest.pop("logs", None)

    rows = []
    missing = []
    for ds, labels in manifest.items():
        gt_path = GROUND_TRUTH_DIR / f"{ds}.txt"
        for label in labels:
            json_path = JSON_DIR / f"{ds}_label{label}_all_results.json"
            if not json_path.exists():
                missing.append(str(json_path))
                continue
            results = json.load(open(json_path))
            gene_names = get_correct_gene_names(results)
            gt = load_ground_truth_matrix(gt_path, gene_names=gene_names)

            row = score_twinfer_dataset(results, gene_names, gt)
            row["dataset_id"] = ds
            row["label"] = label
            row["json_file"] = json_path.name
            rows.append(row)

    if missing:
        print(f"WARNING: {len(missing)} missing JSON file(s):")
        for m in missing:
            print(" ", m)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Scored {len(df)} TwINFER results -> {OUT_CSV}")
    print(df["dataset_id"].value_counts().sort_index())


if __name__ == "__main__":
    main()
