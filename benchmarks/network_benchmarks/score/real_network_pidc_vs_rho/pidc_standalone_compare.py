#!/usr/bin/env python
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_807d072d_2026-09-20_real_network_sim; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""
Does the PIDC-swapped TwINFER formula actually beat raw/standalone PIDC (not
just TwINFER's own prior rho-based formula)? Since we're literally feeding
PIDC's own score into the formula, the real test is whether TwINFER's OTHER
terms (twin heterogeneity/divergence, cross-time z_dagger, stability) ADD
signal on top of PIDC, evaluated on the identical candidate-pair universe.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob
import json
import sys

import numpy as np
import pandas as pd

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/work_in_progress/benchmark")
from benchmarks.network_benchmarks.score.todo4v2_sim_scoring import s, full_report, load_true_edges, todo4v2_score, REAL_NETWORK_MATRIX
from benchmarks.network_benchmarks.score.real_network_pidc_vs_rho.pidc_vs_rho_real_networks import PIDC_ROOT, NAME_TO_GENE, load_pidc_for_network, score_with_pidc_swap

ROOT = f'{TWINFER_PROJECT_ROOT}'
JSON_GLOB = f"{ROOT}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs/*_all_results.json"


def main():
    rows = []
    for p in sorted(glob.glob(JSON_GLOB)):
        d = json.load(open(p))
        sim_type = d.get("sim_type", "")
        if sim_type not in PIDC_ROOT:
            continue
        tsi = d["twin_score_inputs"]
        dd = pd.DataFrame(tsi["data"], columns=tsi["columns"])
        dd = dd[dd.gene_1 != dd.gene_2].reset_index(drop=True)
        dd = dd[dd.rho_cross_xy.notna()].reset_index(drop=True)
        if len(dd) == 0:
            continue
        z_reg_map = d.get("gated_regulation", {}).get("z_reg_gated", {})
        matrix_path = f"{ROOT}/input_data/real_world_networks/{REAL_NETWORK_MATRIX.get(sim_type, '')}"
        true_edges = load_true_edges(matrix_path, d["gene_names"])
        U = list(zip(dd.gene_1, dd.gene_2))
        y = np.array([1 if pr in true_edges else 0 for pr in U])
        if y.sum() < 1:
            continue

        pidc_scores = load_pidc_for_network(sim_type)
        if pidc_scores is None:
            continue
        pidc_w = np.array([pidc_scores.get((a, b), np.nan) for a, b in U])
        if np.isnan(pidc_w).all():
            continue

        m_rho, _ = todo4v2_score(dd, U, z_reg_map, "nogate")
        m_rho = full_report(m_rho, y)
        m_swap = full_report(score_with_pidc_swap(dd, U, pidc_scores), y)
        m_pidc_alone = full_report(pidc_w, y)  # raw PIDC weight, same U, ranked directly

        rows.append(dict(sim_type=sim_type, n_pairs=len(U), n_true=int(y.sum()),
                          nogate_rho=m_rho["auprc_x"], nogate_pidc_swap=m_swap["auprc_x"],
                          pidc_alone=m_pidc_alone["auprc_x"]))

    df = pd.DataFrame(rows)
    print("=== per sim_type: nogate(rho) vs nogate(pidc-swap) vs PIDC ALONE (same pairs) ===")
    for st, g in df.groupby("sim_type"):
        print(f"{st:<20} n={len(g):3d}  nogate(rho)={g.nogate_rho.mean():.3f}x  "
              f"nogate(pidc-swap)={g.nogate_pidc_swap.mean():.3f}x  PIDC-alone={g.pidc_alone.mean():.3f}x  "
              f"swap-beats-PIDC-alone={100*(g.nogate_pidc_swap>g.pidc_alone).mean():.0f}%")

    print(f"\n=== TOTAL (n={len(df)}) ===")
    print(f"mean nogate(rho)       = {df.nogate_rho.mean():.3f}x")
    print(f"mean nogate(pidc-swap) = {df.nogate_pidc_swap.mean():.3f}x")
    print(f"mean PIDC-alone        = {df.pidc_alone.mean():.3f}x")
    print(f"pidc-swap beats PIDC-alone on {(df.nogate_pidc_swap > df.pidc_alone).sum()}/{len(df)} "
          f"({100*(df.nogate_pidc_swap > df.pidc_alone).mean():.0f}%)")
    # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] df.to_csv(f"{ROOT}/code/TwINFER/work_in_progress/benchmark/pidc_standalone_compare.csv", index=False)
    df.to_csv(f"{TWINFER_PROJECT_ROOT}/clean_data/benchmarks/network_benchmarks/score_results/pidc_standalone_compare.csv", index=False)


if __name__ == "__main__":
    main()
