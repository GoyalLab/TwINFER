"""2026-09-17: Full unsigned+signed auprc_x comparison table -- TODO4v2(nogate), cross-corr-only,
and all 7 BEELINE competitors -- across the 3 simulated benchmarks. TODO4v2 and cross-corr-only
per-replicate numbers come from signed_scoring_analysis.py's already-computed CSVs (crosscorr/
todo4v2 unsigned from todo4v2_sim_scoring.py's output, signed from signed_scoring_analysis.py's
output, both now using sign(rho_cross_xy) for TODO4v2 per user instruction). BEELINE competitors'
raw auprc/auprc_signed come from the beeline CSVs; x-random ratios are computed here by resolving
each dataset_id/sim_type to its ground-truth matrix to get n_genes (beeline CSVs don't carry
n_genes directly).
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob
import os

import numpy as np
import pandas as pd

ROOT = f'{TWINFER_PROJECT_ROOT}'
# [2026-09-30 commented out: HERE was the original code directory and is used only for outputs; results now go to clean_data/, see REPOINT_LOG.tsv] HERE = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/work_in_progress/benchmark'
HERE = f'{TWINFER_PROJECT_ROOT}/analysis_data/benchmarks/network_benchmarks/score_results'


def n_genes_from_gt(path):
    return np.loadtxt(path, delimiter=",").shape[0]


def beeline_auprc_x_table(beeline_csv, gt_resolver):
    """Per-algorithm mean unsigned+signed auprc_x, resolving n_genes per dataset_id via gt_resolver."""
    b = pd.read_csv(beeline_csv)
    gt_cache = {}
    def genes_for(ds):
        if ds not in gt_cache:
            p = gt_resolver(ds)
            gt_cache[ds] = n_genes_from_gt(p) if p and os.path.exists(p) else np.nan
        return gt_cache[ds]
    b = b.assign(n_genes=b.dataset_id.map(genes_for))
    b = b[b.n_genes.notna()]
    total_pairs = b.n_genes * (b.n_genes - 1)
    b = b.assign(
        rand_unsigned=b.n_true_edges / total_pairs,
        rand_signed=b.n_true_edges / (2 * total_pairs),
    )
    b = b.assign(
        auprc_x=b.auprc / b.rand_unsigned,
        auprc_x_signed=b.auprc_signed / b.rand_signed,
    )
    return b.groupby("algorithm")[["auprc_x", "auprc_x_signed"]].mean()


def run(name, todo4v2_csv, crosscorr_csv, signed_csv, beeline_csv, gt_resolver):
    t4 = pd.read_csv(todo4v2_csv)
    cc = pd.read_csv(crosscorr_csv)
    cc_col = "crosscorr_auprc_x" if "crosscorr_auprc_x" in cc.columns else "auprc_x"
    sg = pd.read_csv(signed_csv)

    rows = [
        dict(method="TODO4v2(nogate)", auprc_x=t4.todo4v2_nogate_auprc_x.mean(),
             auprc_x_signed=sg.todo4v2_signed_auprc_x.mean()),
        dict(method="cross-corr-only", auprc_x=cc[cc_col].mean(),
             auprc_x_signed=sg.crosscorr_signed_auprc_x.mean()),
    ]
    if beeline_csv and os.path.exists(beeline_csv):
        bt = beeline_auprc_x_table(beeline_csv, gt_resolver)
        for algo, r in bt.iterrows():
            rows.append(dict(method=algo, auprc_x=r.auprc_x, auprc_x_signed=r.auprc_x_signed))

    df = pd.DataFrame(rows).sort_values("auprc_x", ascending=False).reset_index(drop=True)
    df["retention"] = df.auprc_x_signed / df.auprc_x
    print(f"\n=== {name} ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    df.to_csv(f"{HERE}/full_comparison_{name}.csv", index=False)
    return df


def main():
    run("network_sweep_final",
        # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] f"{HERE}/todo4v2_network_sweep_final_broader_results.csv",
        f"{TWINFER_PROJECT_ROOT}/analysis_data/benchmarks/network_benchmarks/score_results/todo4v2_network_sweep_final_broader_results.csv",
        # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] f"{HERE}/crosscorr_network_sweep_final_broader_results.csv",
        f"{TWINFER_PROJECT_ROOT}/analysis_data/benchmarks/network_benchmarks/score_results/crosscorr_network_sweep_final_broader_results.csv",
        # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] f"{HERE}/signed_network_sweep_final_results.csv",
        f"{TWINFER_PROJECT_ROOT}/analysis_data/benchmarks/network_benchmarks/score_results/signed_network_sweep_final_results.csv",
        f"{ROOT}/analysis_data/network_sweep_final/beeline_gmm_analysis_output.csv",
        lambda ds: f"{ROOT}/input_data/network_sweep_final/{ds}.txt")

    run("mixed_network_sweep",
        # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] f"{HERE}/todo4v2_mixed_network_sweep_results.csv",
        f"{TWINFER_PROJECT_ROOT}/analysis_data/benchmarks/network_benchmarks/score_results/todo4v2_mixed_network_sweep_results.csv",
        # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] f"{HERE}/crosscorr_mixed_network_sweep_results.csv",
        f"{TWINFER_PROJECT_ROOT}/analysis_data/benchmarks/network_benchmarks/score_results/crosscorr_mixed_network_sweep_results.csv",
        # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] f"{HERE}/signed_mixed_network_sweep_results.csv",
        f"{TWINFER_PROJECT_ROOT}/analysis_data/benchmarks/network_benchmarks/score_results/signed_mixed_network_sweep_results.csv",
        f"{ROOT}/analysis_data/mixed_network_sweep/beeline_analysis_output.csv",
        lambda ds: f"{ROOT}/input_data/mixed_network_sweep/{ds}.txt")

    TOPO_MAP = {"GSD": "GSD.txt", "HSC": "HSC.txt", "VSC": "VSC.txt", "mCAD": "mCAD.txt",
                "B_cell_activation": "B_cell.txt", "Circadian_cycle": "circadian.txt",
                "EMT": "EMT.txt", "Pluripotent": "Pluripotent.txt"}
    run("real_networks",
        # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] f"{HERE}/todo4v2_real_networks_results.csv",
        f"{TWINFER_PROJECT_ROOT}/analysis_data/benchmarks/network_benchmarks/score_results/todo4v2_real_networks_results.csv",
        # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] f"{HERE}/crosscorr_real_networks_results.csv",
        f"{TWINFER_PROJECT_ROOT}/analysis_data/benchmarks/network_benchmarks/score_results/crosscorr_real_networks_results.csv",
        # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] f"{HERE}/signed_real_networks_results.csv",
        f"{TWINFER_PROJECT_ROOT}/analysis_data/benchmarks/network_benchmarks/score_results/signed_real_networks_results.csv",
        f"{ROOT}/analysis_data/paper_analysis/real_networks/beeline_scores.csv",
        lambda ds: f"{ROOT}/input_data/real_world_networks/{TOPO_MAP.get(ds, '')}")


if __name__ == "__main__":
    main()
