"""t1=10,t2=20 variant of todo4v2_sim_scoring.py (Part 6 of
handoff/2026-09-18_todo4v2_signed_metrics_and_t1_10_pipeline.md) -- glob-path change only, no
formula code changes. Reuses every function from todo4v2_sim_scoring.py unchanged; only the
JSON_DIR / BEELINE_CSV paths are repointed at the *_t1_10 outputs (Part 6's SLURM rerun,
jobs 6582042-6582045 TwINFER inference, 6582068-6582073 BEELINE, all COMPLETED).

network_sweep_final's broader OFAT sweep and real_networks don't have a competitor CSV compare
in the t1=1 script either (todo4v2 vs random only); e13_pos100 and mixed_network_sweep's
competitor CSVs were regenerated at t1_10 by score_beeline_t1_10.py.
"""
import os

import pandas as pd

from benchmarks.network_benchmarks.score import todo4v2_sim_scoring as T

from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: project root for the clean_data/ repointing, see REPOINT_LOG.tsv]
ROOT = T.ROOT
HERE = T.HERE


def run_network_sweep_e13_t1_10():
    JSON_DIR = f"{ROOT}/analysis_data/network_sweep_final/e13_pos100/twinfer_inference_allpairs_t1_10"
    BEELINE_CSV = f"{ROOT}/analysis_data/network_sweep_final/e13_pos100/beeline_analysis_output_t1_10.csv"
    files = sorted(__import__("glob").glob(f"{JSON_DIR}/*_all_results.json"))
    print(f"network_sweep (e13_pos100) t1_10: {len(files)} files")
    resolver = lambda d: d.get("ground_truth_matrix")
    rows = [r for r in (T.score_one_json(f, resolver) for f in files) if r]
    df = pd.DataFrame(rows)
    df.to_csv(f"{HERE}/todo4v2_network_sweep_e13_results_t1_10.csv", index=False)
    print(f"mean todo4v2_old={df.todo4v2_old_auprc_x.mean():.3f}x  "
          f"todo4v2_new={df.todo4v2_new_auprc_x.mean():.3f}x  "
          f"todo4v2_nogate={df.todo4v2_nogate_auprc_x.mean():.3f}x  (n={len(df)}/{len(files)} usable)")
    best_algo, best_auprc = T._competitor_summary(BEELINE_CSV)
    if best_algo:
        print(f"best competitor (raw AUPRC, not x-random): {best_algo}={best_auprc:.3f}")
    return df


def run_network_sweep_final_broader_t1_10():
    JSON_DIR = f"{ROOT}/analysis_data/network_sweep_final/twinfer_inference_allpairs_t1_10"
    BEELINE_CSV = f"{ROOT}/analysis_data/network_sweep_final/beeline_gmm_analysis_output_t1_10.csv"
    files = sorted(__import__("glob").glob(f"{JSON_DIR}/*_all_results.json"))
    print(f"network_sweep_final (broader OFAT sweep) t1_10: {len(files)} files")
    resolver = lambda d: d.get("ground_truth_matrix")
    rows = [r for r in (T.score_one_json(f, resolver) for f in files) if r]
    df = pd.DataFrame(rows)
    df.to_csv(f"{HERE}/todo4v2_network_sweep_final_broader_results_t1_10.csv", index=False)
    print(f"mean todo4v2_old={df.todo4v2_old_auprc_x.mean():.3f}x  "
          f"todo4v2_new={df.todo4v2_new_auprc_x.mean():.3f}x  "
          f"todo4v2_nogate={df.todo4v2_nogate_auprc_x.mean():.3f}x  (n={len(df)}/{len(files)} usable)")
    best_algo, best_auprc = T._competitor_summary(BEELINE_CSV)
    if best_algo:
        print(f"best competitor (raw AUPRC, not x-random): {best_algo}={best_auprc:.3f}")
    return df


def run_mixed_network_sweep_t1_10():
    JSON_DIR = f"{ROOT}/analysis_data/mixed_network_sweep/twinfer_inference_allpairs_t1_10"
    GT_DIR = f"{ROOT}/input_data/mixed_network_sweep"
    BEELINE_CSV = f"{ROOT}/analysis_data/mixed_network_sweep/beeline_analysis_output_t1_10.csv"
    files = sorted(__import__("glob").glob(f"{JSON_DIR}/*_all_results.json"))
    print(f"mixed_network_sweep t1_10: {len(files)} files")
    resolver = lambda d: f"{GT_DIR}/{d.get('dataset_id')}.txt"
    rows = [r for r in (T.score_one_json(f, resolver) for f in files) if r]
    df = pd.DataFrame(rows)
    df.to_csv(f"{HERE}/todo4v2_mixed_network_sweep_results_t1_10.csv", index=False)
    print(f"mean todo4v2_old={df.todo4v2_old_auprc_x.mean():.3f}x  "
          f"todo4v2_new={df.todo4v2_new_auprc_x.mean():.3f}x  "
          f"todo4v2_nogate={df.todo4v2_nogate_auprc_x.mean():.3f}x  (n={len(df)}/{len(files)} usable)")
    best_algo, best_auprc = T._competitor_summary(BEELINE_CSV)
    if best_algo:
        print(f"best competitor (raw AUPRC, not x-random): {best_algo}={best_auprc:.3f}")
    return df


def run_real_networks_t1_10():
    JSON_DIR = f"{ROOT}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs_t1_10"
    TOPO_DIR = f"{ROOT}/input_data/real_world_networks"
    TOPO_MAP = {
        "GSD": "GSD.txt", "HSC": "HSC.txt", "VSC": "VSC.txt", "mCAD": "mCAD.txt",
        "B_cell_activation": "B_cell.txt", "Circadian_cycle": "circadian.txt",
        "EMT": "EMT.txt", "Pluripotent": "Pluripotent.txt",
    }
    files = sorted(__import__("glob").glob(f"{JSON_DIR}/*_all_results.json"))
    print(f"real_networks t1_10: {len(files)} files")
    resolver = lambda d: f"{TOPO_DIR}/{TOPO_MAP.get(d.get('sim_type'), '')}"
    rows = [r for r in (T.score_one_json(f, resolver) for f in files) if r]
    df = pd.DataFrame(rows)
    df.to_csv(f"{HERE}/todo4v2_real_networks_results_t1_10.csv", index=False)
    print(f"mean todo4v2_old={df.todo4v2_old_auprc_x.mean():.3f}x  "
          f"todo4v2_new={df.todo4v2_new_auprc_x.mean():.3f}x  "
          f"todo4v2_nogate={df.todo4v2_nogate_auprc_x.mean():.3f}x  (n={len(df)}/{len(files)} usable)")
    return df


def compare_to_t1_1():
    """Side-by-side t1=1 vs t1=10 mean auprc_x table, all variants, all 4 benchmarks."""
    pairs = [
        # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] ("network_sweep_e13", f"{HERE}/todo4v2_network_sweep_e13_results.csv", f"{HERE}/todo4v2_network_sweep_e13_results_t1_10.csv"),
        ("network_sweep_e13", f"{TWINFER_PROJECT_ROOT}/analysis_data/benchmarks/network_benchmarks/score_results/todo4v2_network_sweep_e13_results.csv", f"{HERE}/todo4v2_network_sweep_e13_results_t1_10.csv"),
        # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] ("network_sweep_final_broader", f"{HERE}/todo4v2_network_sweep_final_broader_results.csv", f"{HERE}/todo4v2_network_sweep_final_broader_results_t1_10.csv"),
        ("network_sweep_final_broader", f"{TWINFER_PROJECT_ROOT}/analysis_data/benchmarks/network_benchmarks/score_results/todo4v2_network_sweep_final_broader_results.csv", f"{HERE}/todo4v2_network_sweep_final_broader_results_t1_10.csv"),
        # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] ("mixed_network_sweep", f"{HERE}/todo4v2_mixed_network_sweep_results.csv", f"{HERE}/todo4v2_mixed_network_sweep_results_t1_10.csv"),
        ("mixed_network_sweep", f"{TWINFER_PROJECT_ROOT}/analysis_data/benchmarks/network_benchmarks/score_results/todo4v2_mixed_network_sweep_results.csv", f"{HERE}/todo4v2_mixed_network_sweep_results_t1_10.csv"),
        # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] ("real_networks", f"{HERE}/todo4v2_real_networks_results.csv", f"{HERE}/todo4v2_real_networks_results_t1_10.csv"),
        ("real_networks", f"{TWINFER_PROJECT_ROOT}/analysis_data/benchmarks/network_benchmarks/score_results/todo4v2_real_networks_results.csv", f"{HERE}/todo4v2_real_networks_results_t1_10.csv"),
    ]
    rows = []
    for name, p1, p10 in pairs:
        if not (os.path.exists(p1) and os.path.exists(p10)):
            continue
        d1, d10 = pd.read_csv(p1), pd.read_csv(p10)
        for variant in ("old", "new", "nogate"):
            rows.append(dict(
                benchmark=name, variant=variant,
                t1_1_mean_auprc_x=d1[f"todo4v2_{variant}_auprc_x"].mean(),
                t1_10_mean_auprc_x=d10[f"todo4v2_{variant}_auprc_x"].mean(),
                t1_1_n=len(d1), t1_10_n=len(d10),
            ))
    cdf = pd.DataFrame(rows)
    cdf["delta"] = cdf.t1_10_mean_auprc_x - cdf.t1_1_mean_auprc_x
    cdf.to_csv(f"{HERE}/todo4v2_t1_1_vs_t1_10_comparison.csv", index=False)
    print(cdf.to_string(index=False))
    return cdf


if __name__ == "__main__":
    which = os.environ.get("WHICH", "all")
    if which in ("network_sweep", "all"):
        run_network_sweep_e13_t1_10()
    if which in ("network_sweep_final_broader", "all"):
        run_network_sweep_final_broader_t1_10()
    if which in ("mixed_network_sweep", "all"):
        run_mixed_network_sweep_t1_10()
    if which in ("real_networks", "all"):
        run_real_networks_t1_10()
    if which == "all":
        print("\n=== t1=1 vs t1=10 comparison ===")
        compare_to_t1_1()
