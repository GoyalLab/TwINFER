"""
Consolidate the per-task JSON files written by run_heterogeneity_v2.py
(one {sim_type}_rep_{rep_id}.json per replicate, each holding matrix_record
and d_vs_null_record) into the same CSVs analysis_v2.ipynb / plot_v2.ipynb
expect: directional/gene_gene/random_t1/random/twin_t1/twin_t2 matrix CSVs
(via the same save_results_to_csv logic) plus d_vs_null_summary.csv.

Writes to stage_dir("heterogeneity_vs_regulation_1k_v2", "analysis", None)
-- a fresh run_tag, with "latest" repointed at it -- so plot_v2.ipynb's
stage_dir(..., "latest") read picks this up.
"""
import argparse
import glob
import json
import os

import pandas as pd

from twinfer.utils.paths import stage_dir


def save_results_to_csv(results_list, output_dir):
    """Verbatim port of analysis_v2.ipynb's function of the same name."""
    df = pd.DataFrame(results_list)
    os.makedirs(output_dir, exist_ok=True)
    matrix_types = ["directional", "gene_gene", "random_t1", "random", "twin_t1", "twin_t2"]

    metadata_cols = ["sim_type", "rep_id", "analysis_key"]
    claimed_columns = set()
    for matrix_type in sorted(matrix_types, key=len, reverse=True):
        prefix = f"{matrix_type}_"
        matrix_columns = [c for c in df.columns if c.startswith(prefix) and c not in claimed_columns]
        claimed_columns.update(matrix_columns)
        if matrix_columns:
            subset_cols = metadata_cols + matrix_columns
            subset_df = df[subset_cols].copy()
            rename_dict = {col: col[len(prefix):] for col in matrix_columns}
            subset_df.rename(columns=rename_dict, inplace=True)
            file_path = os.path.join(output_dir, f"{matrix_type}_matrix_results.csv")
            subset_df.to_csv(file_path, index=False)
            print(f"Saved {matrix_type} matrix ({len(subset_df)} rows) to {file_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True,
                         help="directory of per-task JSON files from run_heterogeneity_v2.py")
    parser.add_argument("--output-dir", type=str, default=None,
                         help="defaults to a fresh stage_dir(\"heterogeneity_vs_regulation_1k_v2\", \"analysis\") run")
    args = parser.parse_args()

    json_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    if not json_paths:
        raise SystemExit(f"No JSON files found in {args.input_dir}")

    matrix_records = []
    d_vs_null_records = []
    for p in json_paths:
        with open(p) as f:
            record = json.load(f)
        matrix_records.append(record["matrix_record"])
        d_vs_null_records.append(record["d_vs_null_record"])

    output_dir = args.output_dir or str(stage_dir("heterogeneity_vs_regulation_1k_v2", "analysis", run_tag=None))
    print(f"Aggregating {len(json_paths)} task(s) -> {output_dir}")

    df = save_results_to_csv(matrix_records, output_dir)

    d_vs_null_df = pd.DataFrame(d_vs_null_records)
    d_vs_null_csv_path = os.path.join(output_dir, "d_vs_null_summary.csv")
    d_vs_null_df.to_csv(d_vs_null_csv_path, index=False)
    print(f"Saved d_vs_null summary ({len(d_vs_null_df)} rows) to {d_vs_null_csv_path}")

    # Step 2 (twin-vs-random gate, computed once per replicate by
    # run_heterogeneity_v2.py via infer_with_twinfer(..., return_diagnostics=True))
    # already carries the null distribution's median (the "medians" metric),
    # the twin z-score ("z_scores"), and a +/-Z_STAR-sigma decision boundary
    # ("z_threshold_list", formatted as a "(lower, upper)" string to match
    # plot_v2.ipynb's parse_threshold_pair) in the step2_* record fields --
    # no need to re-derive any of this by re-reading raw simulation CSVs.
    Z_STAR = 4.51  # matches z_score_threshold_two_states passed to infer_with_twinfer
    zscore_summary_rows = []
    for network_type in ["A_to_B", "A_to_B_2_states", "A_B_2_states"]:
        sub = df[df["sim_type"] == network_type]
        for _, row in sub.iterrows():
            if pd.isna(row.get("step2_null_mean")):
                continue
            mean, std, median, z = (
                row["step2_null_mean"], row["step2_null_std"],
                row["step2_null_median"], row["step2_z_score"],
            )
            lower, upper = mean - Z_STAR * std, mean + Z_STAR * std
            zscore_summary_rows.extend([
                {"network_type": network_type, "metric": "medians", "values": median},
                {"network_type": network_type, "metric": "z_scores", "values": z},
                {"network_type": network_type, "metric": "z_threshold_list", "values": f"({lower:.6f}, {upper:.6f})"},
            ])

    zscore_summary_df = pd.DataFrame(zscore_summary_rows)
    zscore_summary_csv_path = os.path.join(output_dir, "twins_random_zscore_summary.csv")
    zscore_summary_df.to_csv(zscore_summary_csv_path, index=False)
    print(f"Saved twins_random_zscore_summary ({len(zscore_summary_df)} rows) to {zscore_summary_csv_path}")

    print("\nsim_type counts:")
    print(pd.DataFrame(matrix_records)["sim_type"].value_counts())
