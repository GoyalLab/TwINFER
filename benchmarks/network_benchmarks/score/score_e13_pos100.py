"""
Scores TwINFER and BEELINE results for the e13_pos100 benchmark (3 of the
network_sweep_final OFAT "center" topologies, n6_e13_pos100_rep{1,2,3} ->
grn_n6_e13_pos100_center_rep{1,2,0}.txt respectively -- see
infer_e13_pos100.py's docstring for how that mapping was verified).

Reuses every scoring function from score_mixed_network_sweep.py unchanged
(same math, same current-schema adapter) -- only the dataset discovery/paths
differ, and ground truth is read from each JSON's own "ground_truth_matrix"
field (written by infer_e13_pos100.py) rather than a <dataset_id>.txt lookup,
since e13_pos100's dataset_id doesn't match its ground-truth filename 1:1.

Usage:
    score_twinfer()
    score_beeline()
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]

import glob
import json
import os
from pathlib import Path

import pandas as pd

from benchmarks.network_benchmarks.score import score_mixed_network_sweep as M

PROJECT_ROOT = Path(f'{TWINFER_PROJECT_ROOT}')
TWINFER_JSON_DIR = PROJECT_ROOT / "analysis_data" / "network_sweep_final" / "e13_pos100" / "twinfer_inference"
BEELINE_INPUT_ROOT = PROJECT_ROOT / "code" / "Beeline" / "inputs" / "e13_pos100"
BEELINE_OUTPUT_ROOT = PROJECT_ROOT / "analysis_data" / "network_sweep_final" / "e13_pos100" / "beeline_inference"

TWINFER_SCORES_CSV = PROJECT_ROOT / "analysis_data" / "network_sweep_final" / "e13_pos100" / "twinfer_analysis_output.csv"
BEELINE_SCORES_CSV = PROJECT_ROOT / "analysis_data" / "network_sweep_final" / "e13_pos100" / "beeline_analysis_output.csv"


def score_twinfer():
    records = []
    for jf in sorted(glob.glob(str(TWINFER_JSON_DIR / "*_all_results.json"))):
        with open(jf) as f:
            record = json.load(f)
        dataset_id = record["dataset_id"]
        n_genes = record["n_genes"]
        gene_names = record.get("gene_names") or [f"gene_{i+1}" for i in range(n_genes)]
        gt = M.load_ground_truth_matrix(record["ground_truth_matrix"], gene_names)

        adapted = M.adapt_result_schema(record, gene_names)
        row = M.score_twinfer_dataset(adapted, gene_names, gt)
        row["dataset_id"] = dataset_id
        row["ground_truth_matrix"] = record["ground_truth_matrix"]
        row["json_file"] = os.path.basename(jf)
        import numpy as np
        row["n_true_self_loops"] = int(np.count_nonzero(np.diag(gt.values)))
        records.append(row)

    df = pd.DataFrame(records)
    TWINFER_SCORES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TWINFER_SCORES_CSV, index=False)
    print(f"Scored {len(df)} TwINFER (dataset, rep) result(s) -> {TWINFER_SCORES_CSV}")
    return df


def score_beeline():
    # BEELINE ground truth is already correctly written per-dataset-dir by
    # e13_pos100_to_beeline_format.py, so M.score_beeline()'s generic
    # dataset-dir-name -> GroundTruthNetwork.csv lookup works unchanged; just
    # point it at this benchmark's paths.
    orig = (M.BEELINE_INPUT_ROOT, M.BEELINE_OUTPUT_ROOT, M.BEELINE_SCORES_CSV)
    M.BEELINE_INPUT_ROOT, M.BEELINE_OUTPUT_ROOT, M.BEELINE_SCORES_CSV = (
        BEELINE_INPUT_ROOT, BEELINE_OUTPUT_ROOT, BEELINE_SCORES_CSV)
    try:
        df = M.score_beeline()
    finally:
        M.BEELINE_INPUT_ROOT, M.BEELINE_OUTPUT_ROOT, M.BEELINE_SCORES_CSV = orig
    return df


if __name__ == "__main__":
    score_twinfer()
    score_beeline()
