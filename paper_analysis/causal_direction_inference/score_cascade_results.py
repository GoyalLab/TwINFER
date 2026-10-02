"""
Precision/recall/F1 scoring for the 5-gene cascade (Part C), updated for
run_cascade_v2.py's nested JSON schema instead of the old notebook's flat
infer_with_twinfer dict. Field mapping:

    old flat dict                              -> new nested schema
    cm["final_directed_edges"]                 -> cm["real"]["directed_edges"]
    cm["unfiltered_direction_matrix"]           -> cm["real"]["unfiltered_direction_matrix"]
    cm["gene_lists"]["multiple_states_and_reg"] -> cm["real"]["classification"]["multiple_states_and_reg"]
    cm["all_gene_pairs"]                        -> cm["real"]["all_gene_pairs"]

Scoring logic itself is otherwise unchanged from the old cell 24. The old
EXCEPTION_PAIR = frozenset() special case (a no-op -- empty set matches
nothing) has been dropped rather than ported forward.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd


def score_one(cm, gt_df, genes):
    real = cm["real"]
    final_directed_edges = real["directed_edges"]
    unfiltered_direction_matrix = pd.DataFrame(real["unfiltered_direction_matrix"])
    unfiltered_direction_matrix.index = genes
    unfiltered_direction_matrix.columns = genes
    multiple_states_and_reg = real["classification"]["multiple_states_and_reg"]

    predicted_edges = set(tuple(e) for e in final_directed_edges)

    TP = FP = FN = TN = 0
    FP_edge, FN_edge = [], []

    for gi in genes:
        for gj in genes:
            if gi == gj:
                continue
            gt_edge = gt_df.loc[gi, gj] == 1
            pred_edge = (gi, gj) in predicted_edges

            if gt_edge and pred_edge:
                if [gi, gj] in multiple_states_and_reg:
                    FN += 1
                    FN_edge.append((gi, gj))
                else:
                    TP += 1
            elif gt_edge and not pred_edge:
                FN += 1
                FN_edge.append((gi, gj))
            elif not gt_edge and pred_edge:
                FP += 1
                FP_edge.append((gi, gj))
            else:
                TN += 1

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "precision": precision, "recall": recall, "f1": f1,
        "FP_edges": FP_edge, "FN_edges": FN_edge,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True,
                         help="directory of cascade_rep_*.json files from run_cascade_v2.py")
    parser.add_argument(
        "--gt-matrix",
        type=str,
        default=f'{TWINFER_PROJECT_ROOT}/code/TwINFER/simulation_example_input_data/connectivity_matrix_5_gene_linear_cascade.txt',
    )
    parser.add_argument("--output-csv", type=str, default=None)
    args = parser.parse_args()

    gt_matrix = np.loadtxt(args.gt_matrix, delimiter=",")
    n = gt_matrix.shape[0]
    genes = [f"gene_{i+1}" for i in range(n)]
    gt_df = pd.DataFrame(gt_matrix, index=genes, columns=genes)

    json_paths = sorted(glob.glob(os.path.join(args.input_dir, "cascade_rep_*.json")))
    if not json_paths:
        raise SystemExit(f"No cascade_rep_*.json files found in {args.input_dir}")

    rows = []
    for p in json_paths:
        with open(p) as f:
            cm = json.load(f)
        result = score_one(cm, gt_df, genes)
        print(
            f"{os.path.basename(p)}: TP={result['TP']}, FP={result['FP']}, "
            f"FN={result['FN']}, precision={result['precision']:.4f}, "
            f"recall={result['recall']:.4f}, f1={result['f1']:.4f}"
        )
        rows.append({
            "json": os.path.basename(p),
            "TP": result["TP"], "FP": result["FP"], "FN": result["FN"],
            "precision": result["precision"], "recall": result["recall"], "f1": result["f1"],
        })

    metrics_df = pd.DataFrame(rows)
    if args.output_csv:
        metrics_df.to_csv(args.output_csv, index=False)
        print(f"\nSaved metrics to {args.output_csv}")
    print("\nSummary:")
    print(metrics_df[["precision", "recall", "f1"]].describe())
