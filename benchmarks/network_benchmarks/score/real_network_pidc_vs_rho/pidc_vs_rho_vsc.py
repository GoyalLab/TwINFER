#!/usr/bin/env python
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_807d072d_2026-09-20_real_network_sim; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""
Test: does substituting PIDC's raw per-pair score for TwINFER's own Step-1
same-timepoint rho (z_abs_rho_t1) fix VSC's diagnosed weak point (indirect/
2-hop correlation confound -- a bivariate rho can't tell a true edge from two
genes that are both downstream of a shared regulator; PIDC's context-
likelihood discounting is built to handle exactly this)?

NOT substituting into z_dagger (the cross-time term) -- that has no PIDC
analog (PIDC sees one static snapshot, no time/twin structure). z_abs_rho_t1
is the natural swap point: both are single-snapshot bivariate "does this pair
look related" statistics.

Method: keep every other todo4v2(nogate) term (Cc, divp, new_gamma, z_flux/
z_dagger from cross-time rho, hinge_stable/hinge_het from twin heterogeneity)
exactly as-is; replace z_abs_rho_t1 with s(pidc_weight) for that pair.
PIDC weight comes from BEELINE's rankedEdges.csv, gene-name-mapped to gene_N
and averaged across all 10 simrepN_twin_paired replicates (PIDC is a
network-topology detector, not tied to a specific noise realization, so
averaging across replicates gives a more stable per-pair estimate).

Ground truth is the same VSC.txt connectivity matrix used everywhere else.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob
import sys

import numpy as np
import pandas as pd

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/work_in_progress/benchmark")
from benchmarks.network_benchmarks.score.todo4v2_sim_scoring import (
    s, full_report, lookup, _reg_and_flux, load_true_edges,
    Z_HET_THR_NEW,
)

ROOT = f'{TWINFER_PROJECT_ROOT}'
VSC_JSON_GLOB = f"{ROOT}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs/VSC_rep_2_*_all_results.json"
PIDC_GLOB = f"{ROOT}/analysis_data/real_networks/beeline_inference/VSC/simrep*_twin_paired/PIDC/rankedEdges.csv"
MATRIX_PATH = f"{ROOT}/input_data/real_world_networks/VSC.txt"

# verified against code/Beeline/inputs/VSC/GroundTruthNetwork.csv by brute-force
# permutation search over the connectivity matrix (Nkx62/Dbx1 are topologically
# symmetric pure mutual-repressors -- either assignment for the last pair gives
# identical results)
NAME_TO_GENE = {
    "Pax6": "gene_1", "Nkx22": "gene_2", "Dbx2": "gene_3", "Olig2": "gene_4",
    "Nkx61": "gene_5", "Irx3": "gene_6", "Nkx62": "gene_7", "Dbx1": "gene_8",
}


def load_pidc_scores():
    """Mean PIDC EdgeWeight per (gene_a, gene_b) pair (both orderings, PIDC is
    symmetric) across all 10 twin_paired replicates."""
    frames = []
    for p in sorted(glob.glob(PIDC_GLOB)):
        df = pd.read_csv(p, sep="\t")
        df["gene_1"] = df["Gene1"].map(NAME_TO_GENE)
        df["gene_2"] = df["Gene2"].map(NAME_TO_GENE)
        frames.append(df[["gene_1", "gene_2", "EdgeWeight"]])
    all_df = pd.concat(frames, ignore_index=True)
    mean_w = all_df.groupby(["gene_1", "gene_2"])["EdgeWeight"].mean()
    return mean_w.to_dict()


def todo4v2_score_pidc_swap(dd, U, z_reg_map, pidc_scores):
    """nogate formula, with z_abs_rho_t1 replaced by s(pidc_weight)."""
    z_abs_change = dd.z_abs_rho_change.to_numpy()
    z_het = dd.z_het.to_numpy()
    z_div = dd.z_div.to_numpy()
    z_gamma = dd.z_gamma.to_numpy()
    z_stable = -dd.z_rho_change.to_numpy()

    pidc_w = np.array([pidc_scores.get((a, b), np.nan) for a, b in U])
    z_abs_pidc = s(pidc_w)  # PIDC weights aren't already z-scored -- standardize per-panel

    Cc = -s(z_abs_change)
    divp = -np.abs(z_div) * (z_het < Z_HET_THR_NEW).astype(float)
    s_zg = s(z_gamma)
    new_gamma = (np.abs(s_zg) >= 1.0).astype(float) * s_zg
    hinge_stable = -np.where(z_stable > 2.326, z_stable, 0.0)
    hinge_het = np.where(np.abs(z_het) > 2.576, np.abs(z_het), 0.0)
    z_flux = _reg_and_flux(dd, abs_valued=False)
    z_dagger = None
    from twinfer.scoring.analytic_zscores import DEFAULT_SD, z_signed
    z_dagger = z_signed(dd.rho_cross_xy.to_numpy(), DEFAULT_SD["cross"])
    s_zdagger = s(np.abs(z_dagger))

    score = z_abs_pidc + Cc + divp + new_gamma + s(z_flux) + hinge_stable + hinge_het + s_zdagger
    return score


def main():
    from benchmarks.network_benchmarks.score.todo4v2_sim_scoring import todo4v2_score
    import json, os

    pidc_scores = load_pidc_scores()
    print(f"loaded PIDC scores for {len(pidc_scores)} directed pairs (from 10 replicates, VSC)")

    paths = sorted(glob.glob(VSC_JSON_GLOB))
    print(f"scoring {len(paths)} VSC TwINFER reps\n")

    rows = []
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
        true_edges = load_true_edges(MATRIX_PATH, gene_names)
        U = list(zip(dd.gene_1, dd.gene_2))
        y = np.array([1 if pr in true_edges else 0 for pr in U])
        if y.sum() < 1:
            continue

        score_orig, _ = todo4v2_score(dd, U, z_reg_map, "nogate")
        m_orig = full_report(score_orig, y)

        score_pidc = todo4v2_score_pidc_swap(dd, U, z_reg_map, pidc_scores)
        m_pidc = full_report(score_pidc, y)

        rows.append(dict(rep=os.path.basename(p), n_pairs=len(U), n_true=int(y.sum()),
                          orig_auprc_x=m_orig["auprc_x"], pidc_auprc_x=m_pidc["auprc_x"]))
        print(f"[{os.path.basename(p)}] n_true={int(y.sum())}/{len(U)}  "
              f"nogate(rho)={m_orig['auprc_x']:.3f}x  nogate(pidc-swap)={m_pidc['auprc_x']:.3f}x  "
              f"delta={m_pidc['auprc_x']-m_orig['auprc_x']:+.3f}")

    df = pd.DataFrame(rows)
    print(f"\n=== VSC summary (n={len(df)} reps) ===")
    print(f"mean nogate(rho)      = {df.orig_auprc_x.mean():.3f}x")
    print(f"mean nogate(pidc-swap)= {df.pidc_auprc_x.mean():.3f}x")
    print(f"pidc-swap wins on {(df.pidc_auprc_x > df.orig_auprc_x).sum()}/{len(df)} reps")


if __name__ == "__main__":
    main()
