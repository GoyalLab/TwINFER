"""
Consolidate the per-task JSON files written by run_causal_direction_v2.py
(one {condition}_rep_{rep_id}.json per replicate, each holding matrix_record
and box_plot_record) into the CSVs analysis_v2.ipynb / plot_v2.ipynb expect:
box_plot_data.csv (feeds create_box_plot_and_save) plus per-condition
correlation-matrix CSVs.

Writes to stage_dir("causal_direction_inference", "analysis", None) -- a
fresh run_tag, with "latest" repointed at it.
"""
import argparse
import glob
import json
import os

import pandas as pd

from twinfer.utils.paths import stage_dir


def save_matrix_results_to_csv(matrix_records, output_dir):
    df = pd.DataFrame(matrix_records)
    os.makedirs(output_dir, exist_ok=True)
    matrix_types = ["directional", "gene_gene", "twin_t1", "twin_t2"]

    metadata_cols = ["condition", "rep_id", "analysis_key", "gene_gene_threshold"]
    claimed_columns = set()
    for matrix_type in sorted(matrix_types, key=len, reverse=True):
        prefix = f"{matrix_type}_"
        matrix_columns = [c for c in df.columns if c.startswith(prefix) and c not in claimed_columns]
        claimed_columns.update(matrix_columns)
        if matrix_columns:
            subset_cols = [c for c in metadata_cols if c in df.columns] + matrix_columns
            subset_df = df[subset_cols].copy()
            rename_dict = {col: col[len(prefix):] for col in matrix_columns}
            subset_df.rename(columns=rename_dict, inplace=True)
            file_path = os.path.join(output_dir, f"{matrix_type}_matrix_results.csv")
            subset_df.to_csv(file_path, index=False)
            print(f"Saved {matrix_type} matrix ({len(subset_df)} rows) to {file_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None,
                         help="defaults to a fresh stage_dir(\"causal_direction_inference\", \"analysis\") run")
    args = parser.parse_args()

    json_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    if not json_paths:
        raise SystemExit(f"No JSON files found in {args.input_dir}")

    matrix_records = []
    box_plot_records = []
    for p in json_paths:
        with open(p) as f:
            record = json.load(f)
        matrix_records.append(record["matrix_record"])
        box_plot_records.append(record["box_plot_record"])

    output_dir = args.output_dir or str(stage_dir("causal_direction_inference", "analysis", run_tag=None))
    print(f"Aggregating {len(json_paths)} task(s) -> {output_dir}")

    full_df = save_matrix_results_to_csv(matrix_records, output_dir)

    # Gated (pipeline-routed) z-scores are None/NaN whenever an earlier
    # step's threshold routes gene_1-gene_2 away from a later step; the
    # forced_* columns are infer.py's Steps 1-4 reproduced unconditionally
    # for gene_1-gene_2 (see run_forced_zscores docstring), so every row
    # has every z-score regardless of what the gated pipeline decided.
    gated_cols = [c for c in full_df.columns if c.startswith(("step1_", "step2_", "step4_"))]
    forced_cols = [c for c in full_df.columns if c.startswith("forced_")]
    zscore_cols = ["condition", "rep_id", "analysis_key", "gene_gene_threshold"] + sorted(gated_cols) + sorted(forced_cols)
    zscore_cols = [c for c in zscore_cols if c in full_df.columns]
    zscore_df = full_df[zscore_cols].sort_values(["condition", "rep_id"]).reset_index(drop=True)
    zscore_path = os.path.join(output_dir, "zscores.csv")
    zscore_df.to_csv(zscore_path, index=False)
    print(f"Saved z-scores ({len(zscore_df)} rows) to {zscore_path}")
    print(f"  gated columns:  {gated_cols}")
    print(f"  forced columns: {forced_cols}")
    n_missing_forced = zscore_df[forced_cols].isna().sum().sum() if forced_cols else 0
    print(f"  missing values in forced_* columns: {n_missing_forced} (should be 0)")

    box_plot_df = pd.DataFrame(box_plot_records)
    box_plot_df = box_plot_df.rename(columns={"condition": "Condition"})
    box_plot_path = os.path.join(output_dir, "box_plot_data.csv")
    box_plot_df.to_csv(box_plot_path, index=False)
    print(f"Saved box_plot_data ({len(box_plot_df)} rows) to {box_plot_path}")

    rescued_rows = []
    for r in matrix_records:
        for pair in r.get("rescued_from_no_regulation", []):
            rescued_rows.append({
                "condition": r["condition"], "rep_id": r["rep_id"],
                "gene_1": pair[0], "gene_2": pair[1],
            })
    rescued_df = pd.DataFrame(rescued_rows)
    rescued_path = os.path.join(output_dir, "rescued_from_no_regulation.csv")
    rescued_df.to_csv(rescued_path, index=False)
    print(f"Saved rescued-pair log ({len(rescued_df)} rows) to {rescued_path}")

    print("\ncondition counts:")
    print(pd.DataFrame(matrix_records)["condition"].value_counts())
