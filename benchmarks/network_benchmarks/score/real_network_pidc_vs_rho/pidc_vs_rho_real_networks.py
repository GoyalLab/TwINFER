#!/usr/bin/env python
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_807d072d_2026-09-20_real_network_sim; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""
Extend the PIDC-swap-for-z_abs_rho_t1 generalization test to 7 of the 8 real
networks (all except GSD, whose gene_order.txt doesn't match its matrix and
whose 19! search space makes brute-force verification infeasible -- left out
rather than guessed).

Gene-name mappings verified this session:
  VSC, mCAD    -- brute-force permutation search against GroundTruthNetwork.csv
  HSC          -- direct check against GroundTruthNetwork.csv (network_labels.txt's
                  listed order): predicted edges are a 26/26 exact subset of the
                  30-edge literature network (4 edges legitimately pruned in the
                  sim, zero false positives) -- correct mapping, not a full model
  EMT, Pluripotent, Circadian_cycle, B_cell_activation
               -- PIDC output already uses native gene_N naming, no mapping needed
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
    REAL_NETWORK_MATRIX,
)
from twinfer.scoring.analytic_zscores import DEFAULT_SD, z_signed

ROOT = f'{TWINFER_PROJECT_ROOT}'
JSON_GLOB = f"{ROOT}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs/*_all_results.json"

NAME_TO_GENE = {
    "VSC": {  # verified via brute-force permutation search
        "Pax6": "gene_1", "Nkx22": "gene_2", "Dbx2": "gene_3", "Olig2": "gene_4",
        "Nkx61": "gene_5", "Irx3": "gene_6", "Dbx1": "gene_7", "Nkx62": "gene_8",
    },
    "mCAD": {  # verified via brute-force permutation search
        "Pax6": "gene_1", "Coup": "gene_2", "Emx2": "gene_3", "Fgf8": "gene_4", "Sp8": "gene_5",
    },
    "HSC": {  # verified: predicted edges are an exact 26/26 subset of the 30-edge literature GT
        "cJun": "gene_1", "SCL": "gene_2", "Eklf": "gene_3", "Gata1": "gene_4",
        "PU.1": "gene_5", "CEBPa": "gene_6", "Gata2": "gene_7", "Gfi1": "gene_8",
        "EgrNab": "gene_9", "Fli1": "gene_10", "Fog": "gene_11",
    },
    # EMT/Pluripotent/Circadian_cycle/B_cell_activation: PIDC already uses gene_N -> identity map
}

PIDC_ROOT = {
    "VSC": f"{ROOT}/analysis_data/real_networks/beeline_inference/VSC",
    "mCAD": f"{ROOT}/analysis_data/real_networks/beeline_inference/mCAD",
    "HSC": f"{ROOT}/analysis_data/real_networks/beeline_inference/HSC",
    "EMT": f"{ROOT}/analysis_data/paper_analysis/real_networks/beeline_inference/EMT",
    "Pluripotent": f"{ROOT}/analysis_data/paper_analysis/real_networks/beeline_inference/Pluripotent",
    "Circadian_cycle": f"{ROOT}/analysis_data/paper_analysis/real_networks/beeline_inference/Circadian_cycle",
    "B_cell_activation": f"{ROOT}/analysis_data/paper_analysis/real_networks/beeline_inference/B_cell_activation",
    # GSD_multistate: same GSD.txt topology as vanilla GSD (only Hill n/k_add tuning
    # differs, not the connectivity matrix, so gene_N numbering is identical) --
    # uses native gene_N naming, sidesteps the unresolved biological-name mapping.
    "GSD": f"{ROOT}/analysis_data/paper_analysis/real_networks/beeline_inference/GSD_multistate",
}

_pidc_cache = {}


def load_pidc_for_network(sim_type):
    if sim_type in _pidc_cache:
        return _pidc_cache[sim_type]
    root = PIDC_ROOT.get(sim_type)
    if root is None:
        _pidc_cache[sim_type] = None
        return None
    paths = sorted(glob.glob(f"{root}/simrep*_twin_paired/PIDC/rankedEdges.csv"))
    if not paths:
        _pidc_cache[sim_type] = None
        return None
    frames = [pd.read_csv(p, sep="\t") for p in paths]
    all_df = pd.concat(frames, ignore_index=True)
    name_map = NAME_TO_GENE.get(sim_type)
    if name_map:
        all_df["gene_1"] = all_df["Gene1"].map(name_map)
        all_df["gene_2"] = all_df["Gene2"].map(name_map)
    else:
        all_df["gene_1"] = all_df["Gene1"]
        all_df["gene_2"] = all_df["Gene2"]
    mean_w = all_df.groupby(["gene_1", "gene_2"])["EdgeWeight"].mean()
    d = mean_w.to_dict()
    _pidc_cache[sim_type] = d
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
    paths = sorted(glob.glob(JSON_GLOB))
    rows = []
    skipped = {"no_pidc_root": 0, "no_matrix": 0, "no_true_edges": 0}
    for p in paths:
        d = json.load(open(p))
        sim_type = d.get("sim_type", "")
        if sim_type not in PIDC_ROOT:
            skipped["no_pidc_root"] += 1
            continue
        tsi = d["twin_score_inputs"]
        dd = pd.DataFrame(tsi["data"], columns=tsi["columns"])
        dd = dd[dd.gene_1 != dd.gene_2].reset_index(drop=True)
        dd = dd[dd.rho_cross_xy.notna()].reset_index(drop=True)
        if len(dd) == 0:
            continue
        z_reg_map = d.get("gated_regulation", {}).get("z_reg_gated", {})
        gene_names = d["gene_names"]
        matrix_path = f"{ROOT}/input_data/real_world_networks/{REAL_NETWORK_MATRIX.get(sim_type, '')}"
        if not os.path.exists(matrix_path):
            skipped["no_matrix"] += 1
            continue
        true_edges = load_true_edges(matrix_path, gene_names)
        U = list(zip(dd.gene_1, dd.gene_2))
        y = np.array([1 if pr in true_edges else 0 for pr in U])
        if y.sum() < 1:
            skipped["no_true_edges"] += 1
            continue

        pidc_scores = load_pidc_for_network(sim_type)
        if pidc_scores is None:
            skipped["no_pidc_root"] += 1
            continue

        score_orig, _ = todo4v2_score(dd, U, z_reg_map, "nogate")
        m_orig = full_report(score_orig, y)
        score_pidc = score_with_pidc_swap(dd, U, pidc_scores)
        if score_pidc is None:
            continue
        m_pidc = full_report(score_pidc, y)

        rows.append(dict(sim_type=sim_type, rep_id=d.get("rep_id", ""),
                          orig_auprc_x=m_orig["auprc_x"], pidc_auprc_x=m_pidc["auprc_x"]))

    df = pd.DataFrame(rows)
    print(f"scored {len(df)} / {len(paths)} json files  (skipped: {skipped})\n")
    print("=== per sim_type ===")
    for st, g in df.groupby("sim_type"):
        wins = (g.pidc_auprc_x > g.orig_auprc_x).sum()
        print(f"{st:<20} n={len(g):3d}  mean_rho={g.orig_auprc_x.mean():.3f}x  "
              f"mean_pidc={g.pidc_auprc_x.mean():.3f}x  delta={g.pidc_auprc_x.mean()-g.orig_auprc_x.mean():+.3f}  "
              f"wins={wins}/{len(g)}")

    # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] df.to_csv(f"{ROOT}/code/TwINFER/work_in_progress/benchmark/pidc_vs_rho_real_networks.csv", index=False)
    df.to_csv(f"{TWINFER_PROJECT_ROOT}/clean_data/benchmarks/network_benchmarks/score_results/pidc_vs_rho_real_networks.csv", index=False)
    wins = (df.pidc_auprc_x > df.orig_auprc_x).sum()
    print(f"\n=== TOTAL (real networks, n={len(df)}) ===")
    print(f"mean nogate(rho)      = {df.orig_auprc_x.mean():.3f}x")
    print(f"mean nogate(pidc-swap)= {df.pidc_auprc_x.mean():.3f}x")
    print(f"pidc-swap wins on {wins}/{len(df)} ({100*wins/len(df):.0f}%)")


if __name__ == "__main__":
    main()
