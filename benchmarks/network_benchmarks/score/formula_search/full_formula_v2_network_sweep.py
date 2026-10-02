"""Recompute TwINFER's numbers for the network_sweep_final heatmap using THIS
SESSION's from-scratch 'full' formula (analytic_core_general.full_table_disjoint +
compute_scores), instead of the production package's shipped twinScore that used to
populate twinfer_analysis_output.csv / e13_pos100/twinfer_analysis_output.csv.

Column groups (density family + sign-fraction family, sharing the 9-edge/all-positive
topology replicates):
    5 edges   <- grn_n6_e5_pos100_density_rep{0,1,2}
    9 edges   <- grn_n6_e9_pos100_center_rep{0,1,2}      (== "all-pos" below)
    13 edges  <- grn_n6_e13_pos100_center_rep{0,1,2}     (raw sims under e13_pos100/,
                 tag rep1->center_rep1, rep2->center_rep2, rep3->center_rep0)
    17 edges  <- grn_n6_e17_pos100_density_rep{0,1,2}
    all-neg   <- grn_n6_e9_pos0_sign_ratio_rep{0,1,2}
    balanced  <- grn_n6_e9_pos50_sign_ratio_rep{0,1,2}
    all-pos   <- grn_n6_e9_pos100_center_rep{0,1,2}      (duplicate of "9 edges")

T1=1, T2=20 throughout. Competitor (BEELINE) numbers are untouched: read straight from
beeline_gmm_analysis_output.csv (main families) / e13_pos100/beeline_analysis_output.csv
(e13), scheme=='spread', averaged per dataset_id across sim_rep -- 'auprc' and 'f1_topk'
columns (unsigned, directed).
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import os
import sys
import time

import numpy as np
import pandas as pd

# HERE = os.path.dirname(os.path.abspath(__file__))   [2026-09-30 replaced: these scripts used to sit in analysis_data/network_sweep_final and read/write their CSVs next to themselves; that data dir is now referenced explicitly]
HERE = f'{TWINFER_PROJECT_ROOT}/analysis_data/network_sweep_final'
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, HERE)
from benchmarks.network_benchmarks.score.formula_search.full_formula_v2_common import sim_files_for_topology, score_topology_replicate

ROOT = f'{TWINFER_PROJECT_ROOT}'
SIM_DIR = f"{ROOT}/simulation_data/network_sweep_final"
E13_SIM_DIR = f"{ROOT}/simulation_data/network_sweep_final/e13_pos100"
TOPO_DIR = f"{ROOT}/input_data/network_sweep_final"

BEELINE_MAIN_CSV = f"{HERE}/beeline_gmm_analysis_output.csv"
BEELINE_E13_CSV = f"{HERE}/e13_pos100/beeline_analysis_output.csv"

ALGOS = ["GENIE3", "GRNBOOST2", "PPCOR", "SCODE", "SCSGL", "PEARSON", "PIDC"]

# (group_label, topology_family_prefix) -- 3 reps (0,1,2) each, standard families
STANDARD_FAMILIES = [
    ("5 edges", "grn_n6_e5_pos100_density"),
    ("9 edges", "grn_n6_e9_pos100_center"),
    ("17 edges", "grn_n6_e17_pos100_density"),
    ("all-neg", "grn_n6_e9_pos0_sign_ratio"),
    ("balanced", "grn_n6_e9_pos50_sign_ratio"),
    ("all-pos", "grn_n6_e9_pos100_center"),  # duplicate of "9 edges" -- same data, on purpose
]

# e13: tag -> center_rep index (rep3 -> 0 is the confirmed wraparound)
E13_TAG_TO_REP = {"rep3": 0, "rep1": 1, "rep2": 2}


def compute_twinfer_full_scores():
    rows = []
    t0 = time.time()

    for group_label, prefix in STANDARD_FAMILIES:
        for rep_idx in (0, 1, 2):
            topo_name = f"{prefix}_rep{rep_idx}"
            topo_txt = f"{TOPO_DIR}/{topo_name}.txt"
            sims = sim_files_for_topology(SIM_DIR, topo_name)
            r = score_topology_replicate(sims, topo_txt)
            rows.append(dict(group=group_label, col=rep_idx + 1, topology=topo_name,
                              n_sim_reps=r["n_sim_reps"], auprc=r["auprc"], f1_topk=r["f1"]))
            print(f"[{time.time()-t0:.0f}s] {group_label} col{rep_idx+1} ({topo_name}): "
                  f"n_sims={r['n_sim_reps']} auprc={r['auprc']:.3f} f1={r['f1']:.3f}", flush=True)

    # e13 family
    for tag, rep_idx in E13_TAG_TO_REP.items():
        topo_name = f"grn_n6_e13_pos100_center_rep{rep_idx}"
        topo_txt = f"{TOPO_DIR}/{topo_name}.txt"
        import glob
        pattern = os.path.join(E13_SIM_DIR, f"*n6_e13_pos100_{tag}_rep_*.csv")
        sims = sorted(f for f in glob.glob(pattern) if "simulation_before_division" not in os.path.basename(f))
        r = score_topology_replicate(sims, topo_txt)
        rows.append(dict(group="13 edges", col=rep_idx + 1, topology=topo_name,
                          n_sim_reps=r["n_sim_reps"], auprc=r["auprc"], f1_topk=r["f1"]))
        print(f"[{time.time()-t0:.0f}s] 13 edges col{rep_idx+1} ({topo_name}, tag={tag}): "
              f"n_sims={r['n_sim_reps']} auprc={r['auprc']:.3f} f1={r['f1']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{HERE}/full_formula_v2_network_sweep_twinfer.csv", index=False)
    print(f"\nWrote {HERE}/full_formula_v2_network_sweep_twinfer.csv")
    return df


def load_competitor_scores():
    """Per (dataset_id, algorithm): mean auprc / f1_topk across scheme=='spread' sim_rep rows."""
    main = pd.read_csv(BEELINE_MAIN_CSV)
    e13 = pd.read_csv(BEELINE_E13_CSV)
    main = main[main.scheme == "spread"]
    e13 = e13[e13.scheme == "spread"]

    main_g = main.groupby(["dataset_id", "algorithm"], as_index=False)[["auprc", "f1_topk"]].mean()
    e13_g = e13.groupby(["dataset_id", "algorithm"], as_index=False)[["auprc", "f1_topk"]].mean()

    # map e13 dataset_id tags -> topology name used above, so it merges the same way
    tag_to_topo = {f"n6_e13_pos100_{tag}": f"grn_n6_e13_pos100_center_rep{rep_idx}"
                   for tag, rep_idx in E13_TAG_TO_REP.items()}
    e13_g["dataset_id"] = e13_g["dataset_id"].map(tag_to_topo)

    return pd.concat([main_g, e13_g], ignore_index=True)


def main():
    twinfer_df = compute_twinfer_full_scores()
    comp_df = load_competitor_scores()
    comp_df.to_csv(f"{HERE}/full_formula_v2_network_sweep_competitors.csv", index=False)
    print(f"Wrote {HERE}/full_formula_v2_network_sweep_competitors.csv")


if __name__ == "__main__":
    main()
