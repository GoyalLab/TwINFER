"""2026-09-17: What scales with network size / density / negative-edge-fraction, pooled across
ALL replicates of ALL three simulated benchmarks (network_sweep_final, mixed_network_sweep,
real_networks) -- not just the 6 hand-picked topologies from the earlier Cohen's-d ablation
(HANDOFF_2026-09-17_todo4v2_benchmarks.md Part 5), which never got checked into the repo.

For every replicate: extract (n_genes, density, neg_frac) from its ground-truth matrix, compute
Cohen's d (true-edge vs false-edge) for each raw z-score feature AND auprc_x for cross-corr-only
+ todo4v2(nogate), then Spearman-correlate every quantity against the three network properties
across the pooled replicate set.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/larry_hematopoiesis_validation")
from twinfer.scoring.analytic_zscores import DEFAULT_SD, z_signed
from benchmarks.network_benchmarks.score.todo4v2_sim_scoring import todo4v2_score, full_report, load_true_edges

ROOT = f'{TWINFER_PROJECT_ROOT}'
# [2026-09-30 commented out: HERE was the original code directory and is used only for outputs; results now go to clean_data/, see REPOINT_LOG.tsv] HERE = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/work_in_progress/benchmark'
HERE = f'{TWINFER_PROJECT_ROOT}/analysis_data/benchmarks/network_benchmarks/score_results'

FEATURES = ["z_abs_rho_t1", "z_abs_rho_t2", "z_abs_rho_change", "z_rho_change", "z_div",
            "z_het", "z_d_het", "z_gamma", "abs_rho_cross_xy", "z_dagger", "z_flux", "z_reg_gated"]


def net_properties(gt_path, n_genes):
    M = np.loadtxt(gt_path, delimiter=",")
    off_diag = ~np.eye(n_genes, dtype=bool)
    nz = M[off_diag] != 0
    n_edges = int(nz.sum())
    density = n_edges / (n_genes * (n_genes - 1))
    neg_frac = float((M[off_diag][nz] < 0).mean()) if n_edges > 0 else np.nan
    return n_edges, density, neg_frac


def cohens_d(vals, y):
    vals = np.asarray(vals, float)
    f = np.isfinite(vals)
    vals, y = vals[f], y[f]
    a, b = vals[y == 1], vals[y == 0]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    n1, n2 = len(a), len(b)
    pooled_sd = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    return np.nan if pooled_sd == 0 else (a.mean() - b.mean()) / pooled_sd


def process_one(json_path, gt_resolver):
    d = json.load(open(json_path))
    tsi = d.get("twin_score_inputs")
    gr = d.get("gated_regulation")
    if tsi is None or gr is None or gr.get("z_reg_gated") is None:
        return None
    dd = pd.DataFrame(tsi["data"], columns=tsi["columns"])
    dd = dd[dd.gene_1 != dd.gene_2].reset_index(drop=True)
    dd = dd[dd.rho_cross_xy.notna()].reset_index(drop=True)
    if len(dd) < 5:
        return None
    z_reg_map = gr["z_reg_gated"]
    gene_names = d["gene_names"]
    n_genes = len(gene_names)
    gt_path = gt_resolver(d)
    if gt_path is None or not os.path.exists(gt_path):
        return None
    true_edges = load_true_edges(gt_path, gene_names)
    n_edges, density, neg_frac = net_properties(gt_path, n_genes)

    U = list(zip(dd.gene_1, dd.gene_2))
    y = np.array([1 if p in true_edges else 0 for p in U])
    if y.sum() < 2 or y.sum() == len(y):
        return None

    def lookup(a, b):
        v = z_reg_map.get(f"{a}__{b}", z_reg_map.get(f"{b}__{a}"))
        return np.nan if v is None else v

    dd = dd.assign(
        abs_rho_cross_xy=dd.rho_cross_xy.abs(),
        z_dagger=z_signed(dd.rho_cross_xy.to_numpy(), DEFAULT_SD["cross"]),
        z_reg_gated=[lookup(a, b) for a, b in U],
    )
    # z_flux needs the per-gene REG construction (same as todo4v2_score's signed version)
    z_out = z_signed(dd.rho_cross_xy.to_numpy(), DEFAULT_SD["cross"])
    z_in = z_signed(dd.rho_cross_yx.to_numpy(), DEFAULT_SD["cross"])
    asym = dd.assign(z_out=z_out, z_in=z_in)
    regd = asym.groupby("gene_1")["z_out"].mean() - asym.groupby("gene_1")["z_in"].mean()
    REG = regd.to_dict()
    dd = dd.assign(z_flux=[REG.get(a, 0.0) - REG.get(b, 0.0) for a, b in zip(dd.gene_1, dd.gene_2)])

    row = dict(dataset_id=d.get("dataset_id", os.path.basename(json_path)),
               n_genes=n_genes, n_edges=n_edges, density=density, neg_frac=neg_frac,
               n_pairs=len(U), n_true=int(y.sum()))
    for feat in FEATURES:
        row[f"d_{feat}"] = cohens_d(dd[feat].to_numpy(), y)

    sc_cc = dd.abs_rho_cross_xy.to_numpy()
    row["auprc_x_crosscorr"] = full_report(sc_cc, y)["auprc_x"]
    sc_todo4v2, _ = todo4v2_score(dd, U, z_reg_map, "nogate")
    row["auprc_x_todo4v2"] = full_report(sc_todo4v2, y)["auprc_x"]
    return row


def collect_network_sweep_final():
    JSON_DIR = f"{ROOT}/analysis_data/network_sweep_final/twinfer_inference_allpairs"
    files = sorted(glob.glob(f"{JSON_DIR}/*_all_results.json"))
    resolver = lambda d: d.get("ground_truth_matrix")
    return [r for r in (process_one(f, resolver) for f in files) if r], "network_sweep_final"


def collect_mixed_network_sweep():
    JSON_DIR = f"{ROOT}/analysis_data/mixed_network_sweep/twinfer_inference_allpairs"
    GT_DIR = f"{ROOT}/input_data/mixed_network_sweep"
    files = sorted(glob.glob(f"{JSON_DIR}/*_all_results.json"))
    resolver = lambda d: f"{GT_DIR}/{d.get('dataset_id')}.txt"
    return [r for r in (process_one(f, resolver) for f in files) if r], "mixed_network_sweep"


def collect_real_networks():
    JSON_DIR = f"{ROOT}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs"
    TOPO_DIR = f"{ROOT}/input_data/real_world_networks"
    TOPO_MAP = {"GSD": "GSD.txt", "HSC": "HSC.txt", "VSC": "VSC.txt", "mCAD": "mCAD.txt",
                "B_cell_activation": "B_cell.txt", "Circadian_cycle": "circadian.txt",
                "EMT": "EMT.txt", "Pluripotent": "Pluripotent.txt"}
    files = sorted(glob.glob(f"{JSON_DIR}/*_all_results.json"))
    resolver = lambda d: f"{TOPO_DIR}/{TOPO_MAP.get(d.get('sim_type'), '')}"
    return [r for r in (process_one(f, resolver) for f in files) if r], "real_networks"


def main():
    all_rows = []
    for collector in (collect_network_sweep_final, collect_mixed_network_sweep, collect_real_networks):
        rows, name = collector()
        for r in rows:
            r["benchmark"] = name
        print(f"{name}: {len(rows)} usable replicates")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] out_csv = f"{HERE}/scaling_analysis_pooled.csv"
    out_csv = f"{TWINFER_PROJECT_ROOT}/analysis_data/benchmarks/network_benchmarks/score_results/scaling_analysis_pooled.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv} ({len(df)} rows total)\n")

    metrics = [c for c in df.columns if c.startswith("d_") or c.startswith("auprc_x_")]
    props = ["n_genes", "density", "neg_frac"]

    print("=== Spearman rho(metric, network property), pooled across all 3 benchmarks ===")
    corr_rows = []
    for prop in props:
        for metric in metrics:
            sub = df[[prop, metric]].dropna()
            if len(sub) < 8:
                continue
            rho, p = spearmanr(sub[prop], sub[metric])
            corr_rows.append(dict(property=prop, metric=metric, rho=rho, p=p, n=len(sub)))
    cdf = pd.DataFrame(corr_rows).sort_values(["property", "rho"], key=lambda s: s.abs() if s.name == "rho" else s)
    cdf_sorted = pd.DataFrame(corr_rows)
    for prop in props:
        sub = cdf_sorted[cdf_sorted.property == prop].reindex(
            cdf_sorted[cdf_sorted.property == prop].rho.abs().sort_values(ascending=False).index)
        print(f"\n--- vs {prop} ---")
        print(sub[["metric", "rho", "p", "n"]].to_string(index=False))
    # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] cdf_sorted.to_csv(f"{HERE}/scaling_analysis_correlations.csv", index=False)
    cdf_sorted.to_csv(f"{TWINFER_PROJECT_ROOT}/analysis_data/benchmarks/network_benchmarks/score_results/scaling_analysis_correlations.csv", index=False)

    print("\n=== per-benchmark n_genes / density / neg_frac ranges ===")
    print(df.groupby("benchmark")[props].agg(["min", "max", "mean"]).to_string())


if __name__ == "__main__":
    main()
