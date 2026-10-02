"""Competitor (BEELINE) scoring for real_networks at t1=10 -- previously unscored because
score_mixed_network_sweep.py's generic score_beeline() filters run_id dirs through
RUN_ID_RE = ^simrep([0-9a-f]+)_(spread|twin_paired)$, which doesn't match real_networks' actual
run_id format (simrep{i}_{j}_{hash}_twin_paired, e.g. simrep1_0_3f3127d1_twin_paired -- an extra
integer index before the hash). Reuses every per-pair scoring primitive from
score_mixed_network_sweep.py verbatim (load_beeline_ground_truth, compute_auprc/_signed/
_undirected, dedupe_predictions, top_k_tie_aware_selection*) -- only the directory-walk is new,
built to accept whatever run_id string is actually there instead of pattern-matching it.

GENIE3/GRNBOOST2 for this benchmark were only added by run_real_network_genie3_grnboost2_t1_10.sh
(2026-09-18, filling a gap where the split config existed but no launcher had ever been written).
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
from pathlib import Path

import pandas as pd

from benchmarks.network_benchmarks.score import score_mixed_network_sweep as M

PROJECT_ROOT = Path(f'{TWINFER_PROJECT_ROOT}')
BEELINE_INPUT_ROOT = PROJECT_ROOT / "code" / "Beeline" / "inputs" / "real_networks_t1_10"
BEELINE_OUTPUT_ROOT = PROJECT_ROOT / "analysis_data" / "paper_analysis" / "real_networks" / "beeline_inference_t1_10"
BEELINE_SCORES_CSV = PROJECT_ROOT / "analysis_data" / "paper_analysis" / "real_networks" / "beeline_analysis_output_t1_10.csv"


def score_beeline():
    M.BEELINE_INPUT_ROOT = BEELINE_INPUT_ROOT
    if not BEELINE_OUTPUT_ROOT.exists():
        raise FileNotFoundError(f"{BEELINE_OUTPUT_ROOT} doesn't exist yet")

    rows = []
    for dataset_dir in sorted(BEELINE_OUTPUT_ROOT.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name == "logs":
            continue
        dataset_id = dataset_dir.name
        if not (BEELINE_INPUT_ROOT / dataset_id / "GroundTruthNetwork.csv").exists():
            continue
        gt_df = M.load_beeline_ground_truth(dataset_id)
        possible_edges, true_edges = M.build_edge_universe(gt_df)
        num_true_edges = len(true_edges)

        for run_dir in sorted(dataset_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            run_id = run_dir.name  # e.g. simrep1_0_3f3127d1_twin_paired -- not pattern-matched

            for algo_dir in sorted(run_dir.iterdir()):
                ranked_path = algo_dir / "rankedEdges.csv"
                if not ranked_path.exists():
                    continue
                algorithm = algo_dir.name
                ranked_edges = pd.read_csv(ranked_path, sep="\t", header=0)
                predicted = M.dedupe_predictions(ranked_edges)

                auprc = M.compute_auprc(ranked_edges, gt_df)
                auprc_signed = M.compute_auprc_signed(ranked_edges, gt_df)
                auprc_undirected = M.compute_auprc_undirected(ranked_edges, gt_df)

                selected_topk, _ = M.top_k_tie_aware_selection(predicted, num_true_edges)
                p_topk, r_topk, f1_topk = M.precision_recall_f1(selected_topk, true_edges)

                rows.append({
                    "dataset_id": dataset_id, "run_id": run_id, "algorithm": algorithm,
                    "n_true_edges": num_true_edges,
                    "auprc": auprc, "auprc_signed": auprc_signed, "auprc_undirected": auprc_undirected,
                    "precision_topk": p_topk, "recall_topk": r_topk, "f1_topk": f1_topk,
                })

    df = pd.DataFrame(rows)
    BEELINE_SCORES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(BEELINE_SCORES_CSV, index=False)
    print(f"Scored {len(df)} (dataset, run, algorithm) combination(s) -> {BEELINE_SCORES_CSV}")
    return df


if __name__ == "__main__":
    df = score_beeline()
    print(df.groupby("algorithm")[["auprc", "auprc_signed"]].agg(["mean", "count"]))
