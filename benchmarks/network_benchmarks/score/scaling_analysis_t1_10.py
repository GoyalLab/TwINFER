"""t1=10 variant of scaling_analysis.py -- glob-path change only, reusing every function
(net_properties, cohens_d, process_one) unchanged. Part 6 follow-up."""
import glob
import os

import pandas as pd
from scipy.stats import spearmanr

from benchmarks.network_benchmarks.score import scaling_analysis as S

ROOT = S.ROOT
# [2026-09-30 commented out: result files now in clean_data/, see REPOINT_LOG.tsv] HERE = S.HERE
HERE = f'{S.TWINFER_PROJECT_ROOT}/clean_data/benchmarks/network_benchmarks/score_results'


def collect_network_sweep_final_t1_10():
    JSON_DIR = f"{ROOT}/analysis_data/network_sweep_final/twinfer_inference_allpairs_t1_10"
    files = sorted(glob.glob(f"{JSON_DIR}/*_all_results.json"))
    resolver = lambda d: d.get("ground_truth_matrix")
    return [r for r in (S.process_one(f, resolver) for f in files) if r], "network_sweep_final"


def collect_mixed_network_sweep_t1_10():
    JSON_DIR = f"{ROOT}/analysis_data/mixed_network_sweep/twinfer_inference_allpairs_t1_10"
    GT_DIR = f"{ROOT}/input_data/mixed_network_sweep"
    files = sorted(glob.glob(f"{JSON_DIR}/*_all_results.json"))
    resolver = lambda d: f"{GT_DIR}/{d.get('dataset_id')}.txt"
    return [r for r in (S.process_one(f, resolver) for f in files) if r], "mixed_network_sweep"


def collect_real_networks_t1_10():
    JSON_DIR = f"{ROOT}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs_t1_10"
    TOPO_DIR = f"{ROOT}/input_data/real_world_networks"
    TOPO_MAP = {"GSD": "GSD.txt", "HSC": "HSC.txt", "VSC": "VSC.txt", "mCAD": "mCAD.txt",
                "B_cell_activation": "B_cell.txt", "Circadian_cycle": "circadian.txt",
                "EMT": "EMT.txt", "Pluripotent": "Pluripotent.txt"}
    files = sorted(glob.glob(f"{JSON_DIR}/*_all_results.json"))
    resolver = lambda d: f"{TOPO_DIR}/{TOPO_MAP.get(d.get('sim_type'), '')}"
    return [r for r in (S.process_one(f, resolver) for f in files) if r], "real_networks"


def main():
    all_rows = []
    for collector in (collect_network_sweep_final_t1_10, collect_mixed_network_sweep_t1_10, collect_real_networks_t1_10):
        rows, name = collector()
        for r in rows:
            r["benchmark"] = name
        print(f"{name} t1_10: {len(rows)} usable replicates")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_csv = f"{HERE}/scaling_analysis_pooled_t1_10.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv} ({len(df)} rows total)\n")

    metrics = [c for c in df.columns if c.startswith("d_") or c.startswith("auprc_x_")]
    props = ["n_genes", "density", "neg_frac"]

    corr_rows = []
    for prop in props:
        for metric in metrics:
            sub = df[[prop, metric]].dropna()
            if len(sub) < 8:
                continue
            rho, p = spearmanr(sub[prop], sub[metric])
            corr_rows.append(dict(property=prop, metric=metric, rho=rho, p=p, n=len(sub)))
    cdf_sorted = pd.DataFrame(corr_rows)
    for prop in props:
        sub = cdf_sorted[cdf_sorted.property == prop].reindex(
            cdf_sorted[cdf_sorted.property == prop].rho.abs().sort_values(ascending=False).index)
        print(f"\n--- vs {prop} (t1_10) ---")
        print(sub[["metric", "rho", "p", "n"]].to_string(index=False))
    cdf_sorted.to_csv(f"{HERE}/scaling_analysis_correlations_t1_10.csv", index=False)

    # side-by-side t1=1 vs t1=10 for the two headline metrics
    t1_1_csv = f"{HERE}/scaling_analysis_correlations.csv"
    if os.path.exists(t1_1_csv):
        c1 = pd.read_csv(t1_1_csv)
        merged = c1.merge(cdf_sorted, on=["property", "metric"], suffixes=("_t1_1", "_t1_10"))
        merged.to_csv(f"{HERE}/scaling_analysis_correlations_t1_1_vs_t1_10.csv", index=False)
        print("\n=== t1=1 vs t1=10 correlation comparison (neg_frac vs auprc_x_todo4v2 is the headline check) ===")
        headline = merged[(merged.property == "neg_frac") & (merged.metric == "auprc_x_todo4v2")]
        print(headline[["property", "metric", "rho_t1_1", "rho_t1_10"]].to_string(index=False))


if __name__ == "__main__":
    main()
