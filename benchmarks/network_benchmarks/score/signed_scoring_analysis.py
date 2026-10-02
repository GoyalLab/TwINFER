"""2026-09-17: Signed edge-detection metrics (existence AND correct activation/repression sign)
for the 3 simulated-GRN benchmarks, per user instruction: "if the method gives sign, great, if it
does not, take it as positive (or sign = sign of pearson correlation)".

Sign source per method:
  - cross-corr-only: sign(rho_cross_xy) -- it's a signed statistic already, use its own sign.
  - TODO4v2(nogate): ALSO sign(rho_cross_xy), per user instruction (2026-09-17) -- gamma's
    directional call retained noticeably less signal on the signed task (retention 1.14-1.50x vs
    cross-corr's 1.90-1.96x across the 3 benchmarks), so TODO4v2's own sign call was switched to
    match cross-corr-only's better-calibrated source.
  - BEELINE competitors: reuses the ALREADY-COMPUTED beeline_analysis_output.csv auprc_signed /
    f1_topk_signed columns -- that pipeline's add_predicted_sign() already implements exactly the
    user's fallback rule (EdgeWeight>=0 -> "+"), so GENIE3/GRNBOOST2/PIDC (which never emit
    negative importance) are already defaulted to all-positive-sign there. No need to recompute.

A signed "hit" requires: the pair is a true edge in ground truth AND predicted sign == true sign.
Random baseline = n_true_signed / n_scored_signed_positions (each scored pair contributes 2
signed positions, +/-).
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/larry_hematopoiesis_validation")
from twinfer.scoring.analytic_zscores import DEFAULT_SD, z_signed
from benchmarks.network_benchmarks.score.todo4v2_sim_scoring import todo4v2_score, full_report

ROOT = f'{TWINFER_PROJECT_ROOT}'
# [2026-09-30 commented out: HERE was the original code directory and is used only for outputs; results now go to clean_data/, see REPOINT_LOG.tsv] HERE = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/work_in_progress/benchmark'
HERE = f'{TWINFER_PROJECT_ROOT}/clean_data/benchmarks/network_benchmarks/score_results'


def load_true_signed_edges(matrix_path, gene_names):
    M = np.loadtxt(matrix_path, delimiter=",")
    n = len(gene_names)
    assert M.shape == (n, n), f"{matrix_path}: shape {M.shape} != ({n},{n})"
    return {(gene_names[i], gene_names[j], int(np.sign(M[i, j])))
            for i in range(n) for j in range(n) if i != j and M[i, j] != 0}


def score_one_signed(json_path, gt_resolver):
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
    gt_path = gt_resolver(d)
    if gt_path is None or not os.path.exists(gt_path):
        return None
    true_signed = load_true_signed_edges(gt_path, gene_names)

    U = list(zip(dd.gene_1, dd.gene_2))
    scored_pairs = set(U)
    U_full = [(a, b) for a in gene_names for b in gene_names if a != b and (a, b) in scored_pairs]
    signed_positions = [(a, b, s) for a, b in U_full for s in (1, -1)]
    y_signed = np.array([1 if p in true_signed else 0 for p in signed_positions])
    if y_signed.sum() < 1:
        return None

    # cross-corr-only: natural sign = sign(rho_cross_xy)
    cc_mag = dict(zip(U, dd.rho_cross_xy.abs()))
    cc_sign = dict(zip(U, np.sign(dd.rho_cross_xy).astype(int)))
    floor_cc = -1.0
    sc_cc_signed = np.array([cc_mag.get((a, b), 0.0) if cc_sign.get((a, b)) == s else floor_cc
                              for a, b, s in signed_positions])

    # TODO4v2(nogate): magnitude = composite score, sign = sign(rho_cross_xy) (per user instruction)
    sc_todo4v2, _ = todo4v2_score(dd, U, z_reg_map, "nogate")
    t4v2_mag = dict(zip(U, sc_todo4v2))
    floor_t4v2 = float(np.min(sc_todo4v2)) - 1.0
    sc_t4v2_signed = np.array([t4v2_mag.get((a, b), floor_t4v2) if cc_sign.get((a, b)) == s else floor_t4v2
                                for a, b, s in signed_positions])

    rep_cc = full_report(sc_cc_signed, y_signed)
    rep_t4v2 = full_report(sc_t4v2_signed, y_signed)
    return dict(dataset_id=d.get("dataset_id", os.path.basename(json_path)),
                sim_type=d.get("sim_type"),
                n_pairs=len(U_full), n_true_signed=int(y_signed.sum()),
                crosscorr_signed_auprc_x=rep_cc["auprc_x"], crosscorr_signed_f1=rep_cc["topk_f1"],
                todo4v2_signed_auprc_x=rep_t4v2["auprc_x"], todo4v2_signed_f1=rep_t4v2["topk_f1"])


def competitor_signed_summary(beeline_csv, dataset_ids=None):
    if not os.path.exists(beeline_csv):
        return None, None, None
    b = pd.read_csv(beeline_csv)
    if dataset_ids is not None:
        b = b[b.dataset_id.isin(dataset_ids)]
    if b.empty or "auprc_signed" not in b.columns:
        return None, None, None
    means = b.groupby("algorithm")["auprc_signed"].mean().sort_values(ascending=False)
    best = means.index[0]
    f1_mean = b.loc[b.algorithm == best, "f1_topk_signed"].mean() if "f1_topk_signed" in b.columns else np.nan
    return best, float(means.iloc[0]), float(f1_mean)


def run(name, json_dir, gt_resolver, beeline_csv=None):
    files = sorted(glob.glob(f"{json_dir}/*_all_results.json"))
    rows = [r for r in (score_one_signed(f, gt_resolver) for f in files) if r]
    df = pd.DataFrame(rows)
    out_csv = f"{HERE}/signed_{name}_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n=== {name} ({len(df)}/{len(files)} usable replicates) ===")
    print(f"mean crosscorr_signed_auprc_x = {df.crosscorr_signed_auprc_x.mean():.3f}")
    print(f"mean todo4v2_signed_auprc_x   = {df.todo4v2_signed_auprc_x.mean():.3f}")
    if beeline_csv:
        ids = set(df.sim_type.dropna()) if "sim_type" in df.columns and df.sim_type.notna().any() else set(df.dataset_id)
        best, best_auprc_signed, best_f1 = competitor_signed_summary(beeline_csv, ids)
        if best:
            print(f"best signed competitor (raw auprc_signed, not x-random): {best}={best_auprc_signed:.3f} (f1_topk_signed={best_f1:.3f})")
    return df


def main():
    run("network_sweep_final",
        f"{ROOT}/analysis_data/network_sweep_final/twinfer_inference_allpairs",
        lambda d: d.get("ground_truth_matrix"))

    run("mixed_network_sweep",
        f"{ROOT}/analysis_data/mixed_network_sweep/twinfer_inference_allpairs",
        lambda d: f"{ROOT}/input_data/mixed_network_sweep/{d.get('dataset_id')}.txt",
        beeline_csv=f"{ROOT}/analysis_data/mixed_network_sweep/beeline_analysis_output.csv")

    TOPO_MAP = {"GSD": "GSD.txt", "HSC": "HSC.txt", "VSC": "VSC.txt", "mCAD": "mCAD.txt",
                "B_cell_activation": "B_cell.txt", "Circadian_cycle": "circadian.txt",
                "EMT": "EMT.txt", "Pluripotent": "Pluripotent.txt"}
    run("real_networks",
        f"{ROOT}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs",
        lambda d: f"{ROOT}/input_data/real_world_networks/{TOPO_MAP.get(d.get('sim_type'), '')}",
        beeline_csv=f"{ROOT}/analysis_data/paper_analysis/real_networks/beeline_scores.csv")


if __name__ == "__main__":
    main()
