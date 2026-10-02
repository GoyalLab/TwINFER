#!/usr/bin/env python
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_807d072d_2026-09-20_real_network_sim; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""
Generalization check: does swapping PIDC's raw per-pair score in for
z_abs_rho_t1 (as tested on VSC, +26% mean auprc_x, 10/10 reps) help or hurt
across the synthetic-network benchmark families (network_sweep_final broader
OFAT sweep, network_sweep_e13, mixed_network_sweep)? These all use native
gene_N naming in both TwINFER and PIDC's rankedEdges.csv -- no biological
gene-name mapping risk like VSC/mCAD needed.

For each JSON's dataset_id, average PIDC EdgeWeight per (gene_a, gene_b) pair
across whatever simrepN_twin_paired/PIDC/rankedEdges.csv replicates exist for
that exact dataset_id, then score nogate(rho) vs nogate(pidc-swap).
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
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/work_in_progress/benchmark")
from benchmarks.network_benchmarks.score.todo4v2_sim_scoring import (
    s, full_report, load_true_edges, todo4v2_score, _reg_and_flux, Z_HET_THR_NEW,
)
from twinfer.scoring.analytic_zscores import DEFAULT_SD, z_signed

ROOT = f'{TWINFER_PROJECT_ROOT}'

FAMILIES = {
    "mixed_network_sweep": dict(
        json_glob=f"{ROOT}/analysis_data/mixed_network_sweep/twinfer_inference_allpairs/*_all_results.json",
        beeline_root=f"{ROOT}/analysis_data/mixed_network_sweep/beeline_inference",
        gt_resolver=lambda d: f"{ROOT}/input_data/mixed_network_sweep/{d.get('dataset_id')}.txt",
    ),
    "network_sweep_final_broader": dict(
        json_glob=f"{ROOT}/analysis_data/network_sweep_final/twinfer_inference_allpairs/*_all_results.json",
        beeline_root=f"{ROOT}/analysis_data/network_sweep_final/beeline_inference",
        gt_resolver=lambda d: d.get("ground_truth_matrix"),
    ),
    "network_sweep_e13": dict(
        json_glob=f"{ROOT}/analysis_data/network_sweep_final/e13_pos100/twinfer_inference_allpairs/*_all_results.json",
        beeline_root=f"{ROOT}/analysis_data/network_sweep_final/e13_pos100/beeline_inference",
        gt_resolver=lambda d: d.get("ground_truth_matrix"),
    ),
}

_pidc_cache = {}


def load_pidc_for_dataset(beeline_root, dataset_id):
    key = (beeline_root, dataset_id)
    if key in _pidc_cache:
        return _pidc_cache[key]
    pattern = f"{beeline_root}/{dataset_id}/simrep*_twin_paired/PIDC/rankedEdges.csv"
    paths = sorted(glob.glob(pattern))
    if not paths:
        _pidc_cache[key] = None
        return None
    frames = [pd.read_csv(p, sep="\t") for p in paths]
    all_df = pd.concat(frames, ignore_index=True)
    mean_w = all_df.groupby(["Gene1", "Gene2"])["EdgeWeight"].mean()
    d = mean_w.to_dict()
    _pidc_cache[key] = d
    return d


def score_with_pidc_swap(dd, U, pidc_scores):
    z_abs_change = dd.z_abs_rho_change.to_numpy()
    z_het = dd.z_het.to_numpy()
    z_div = dd.z_div.to_numpy()
    z_gamma = dd.z_gamma.to_numpy()
    z_stable = -dd.z_rho_change.to_numpy()

    pidc_w = np.array([pidc_scores.get((a, b), np.nan) for a, b in U])
    if np.isnan(pidc_w).all():
        return None
    z_abs_pidc = s(pidc_w)

    Cc = -s(z_abs_change)
    divp = -np.abs(z_div) * (z_het < Z_HET_THR_NEW).astype(float)
    s_zg = s(z_gamma)
    new_gamma = (np.abs(s_zg) >= 1.0).astype(float) * s_zg
    hinge_stable = -np.where(z_stable > 2.326, z_stable, 0.0)
    hinge_het = np.where(np.abs(z_het) > 2.576, np.abs(z_het), 0.0)
    z_flux = _reg_and_flux(dd, abs_valued=False)
    z_dagger = z_signed(dd.rho_cross_xy.to_numpy(), DEFAULT_SD["cross"])
    s_zdagger = s(np.abs(z_dagger))

    return z_abs_pidc + Cc + divp + new_gamma + s(z_flux) + hinge_stable + hinge_het + s_zdagger


def main():
    overall = []
    for fam_name, cfg in FAMILIES.items():
        paths = sorted(glob.glob(cfg["json_glob"]))
        rows = []
        n_no_pidc = 0
        for p in paths:
            d = json.load(open(p))
            tsi = d["twin_score_inputs"]
            dd = pd.DataFrame(tsi["data"], columns=tsi["columns"])
            dd = dd[dd.gene_1 != dd.gene_2].reset_index(drop=True)
            dd = dd[dd.rho_cross_xy.notna()].reset_index(drop=True)
            if len(dd) == 0:
                continue
            z_reg_map = d.get("gated_regulation", {}).get("z_reg_gated", {})
            gene_names = d["gene_names"]
            matrix_path = cfg["gt_resolver"](d)
            if not matrix_path or not os.path.exists(matrix_path):
                continue
            true_edges = load_true_edges(matrix_path, gene_names)
            U = list(zip(dd.gene_1, dd.gene_2))
            y = np.array([1 if pr in true_edges else 0 for pr in U])
            if y.sum() < 1:
                continue

            pidc_scores = load_pidc_for_dataset(cfg["beeline_root"], d.get("dataset_id", ""))
            if pidc_scores is None:
                n_no_pidc += 1
                continue

            score_orig, _ = todo4v2_score(dd, U, z_reg_map, "nogate")
            m_orig = full_report(score_orig, y)

            score_pidc = score_with_pidc_swap(dd, U, pidc_scores)
            if score_pidc is None:
                n_no_pidc += 1
                continue
            m_pidc = full_report(score_pidc, y)

            rows.append(dict(dataset_id=d.get("dataset_id", ""),
                              orig_auprc_x=m_orig["auprc_x"], pidc_auprc_x=m_pidc["auprc_x"]))

        df = pd.DataFrame(rows)
        print(f"\n=== {fam_name} ===  ({len(paths)} json files, {n_no_pidc} skipped -- no PIDC match, "
              f"{len(df)} scored)")
        if len(df):
            wins = (df.pidc_auprc_x > df.orig_auprc_x).sum()
            print(f"  mean nogate(rho)      = {df.orig_auprc_x.mean():.3f}x")
            print(f"  mean nogate(pidc-swap)= {df.pidc_auprc_x.mean():.3f}x")
            print(f"  pidc-swap wins on {wins}/{len(df)} ({100*wins/len(df):.0f}%)")
            df["family"] = fam_name
            overall.append(df)

    if overall:
        all_df = pd.concat(overall, ignore_index=True)
        # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] all_df.to_csv(f"{ROOT}/code/TwINFER/work_in_progress/benchmark/pidc_vs_rho_generalization.csv", index=False)
        all_df.to_csv(f"{TWINFER_PROJECT_ROOT}/clean_data/benchmarks/network_benchmarks/score_results/pidc_vs_rho_generalization.csv", index=False)
        print(f"\n=== GRAND TOTAL (all synthetic families, n={len(all_df)}) ===")
        wins = (all_df.pidc_auprc_x > all_df.orig_auprc_x).sum()
        print(f"mean nogate(rho)      = {all_df.orig_auprc_x.mean():.3f}x")
        print(f"mean nogate(pidc-swap)= {all_df.pidc_auprc_x.mean():.3f}x")
        print(f"pidc-swap wins on {wins}/{len(all_df)} ({100*wins/len(all_df):.0f}%)")


if __name__ == "__main__":
    main()
