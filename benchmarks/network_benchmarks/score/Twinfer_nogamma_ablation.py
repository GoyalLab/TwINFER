"""Ablation: drop new_gamma from todo4v2_score's nogate variant, per the finding that
new_gamma (z_gamma) and s_zdagger (z_dagger) are correlated (Spearman 0.38-0.59 across all 3
sim benchmarks, pooled) -- both built from the same raw rho_cross_xy/yx ingredient, so summing
both under todo4v2's equal-unit-variance convention double-weights one fragile signal source.
z_flux is NOT dropped: despite also deriving from z_dagger, it's a gene-level aggregate
(orthogonal, Spearman ~0.0-0.14 to the other two) carrying independent information.

no_gamma score = z_abs_t1 + Cc + divp + s(z_flux) + hinge_stable + hinge_het + s_zdagger
(current nogate score, minus new_gamma)

Compares mean auprc_x, no_gamma vs current nogate, at both t1=1 and t1=10, all 4 benchmark
families the rest of this session already scored.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/larry_hematopoiesis_validation")
from twinfer.scoring.analytic_zscores import DEFAULT_SD, z_signed
from benchmarks.network_benchmarks.score import todo4v2_sim_scoring as T

HERE = T.HERE
ROOT = T.ROOT


def todo4v2_score_no_gamma(dd, U, z_reg_map):
    """Identical to todo4v2_score(dd, U, z_reg_map, 'nogate') minus the new_gamma term."""
    z_flux = T._reg_and_flux(dd, abs_valued=False)
    z_abs_t1 = dd.z_abs_rho_t1.to_numpy()
    z_abs_change = dd.z_abs_rho_change.to_numpy()
    z_het = dd.z_het.to_numpy(); z_div = dd.z_div.to_numpy()
    z_stable = -dd.z_rho_change.to_numpy()
    z_dagger = z_signed(dd.rho_cross_xy.to_numpy(), DEFAULT_SD["cross"])

    Cc = -T.s(z_abs_change)
    divp = -np.abs(z_div) * (z_het < T.Z_HET_THR_NEW).astype(float)
    hinge_stable = -np.where(z_stable > T.Z_ONE_SIDED, z_stable, 0.0)
    hinge_het = np.where(np.abs(z_het) > T.Z_TWO_SIDED, np.abs(z_het), 0.0)
    s_zdagger = T.s(np.abs(z_dagger))

    score = z_abs_t1 + Cc + divp + T.s(z_flux) + hinge_stable + hinge_het + s_zdagger
    return score  # nogate: no -inf floor applied


def score_one_json_ablated(json_path, gt_resolver):
    import json
    d = json.load(open(json_path))
    tsi = d.get("twin_score_inputs")
    gr = d.get("gated_regulation")
    if tsi is None or gr is None or gr.get("z_reg_gated") is None:
        return None
    dd = pd.DataFrame(tsi["data"], columns=tsi["columns"])
    dd = dd[dd.gene_1 != dd.gene_2].reset_index(drop=True)
    dd = dd[dd.rho_cross_xy.notna()].reset_index(drop=True)
    if len(dd) < 3:
        return None
    gene_names = d["gene_names"]
    gt_path = gt_resolver(d)
    if gt_path is None or not os.path.exists(gt_path):
        return None
    true_edges = T.load_true_edges(gt_path, gene_names)

    U_present = list(zip(dd.gene_1, dd.gene_2))
    scored_pairs = set(U_present)
    U_full = [(a, b) for a in gene_names for b in gene_names if a != b and (a, b) in scored_pairs]
    y_full = np.array([1 if p in true_edges else 0 for p in U_full])
    if y_full.sum() < 1 or y_full.sum() == len(y_full):
        return None

    z_reg_map = gr["z_reg_gated"]
    sc_present = todo4v2_score_no_gamma(dd, U_present, z_reg_map)
    score_map = dict(zip(U_present, sc_present))
    floor = np.nanmin(sc_present[np.isfinite(sc_present)]) - 1.0 if np.isfinite(sc_present).any() else -1.0
    sc_full = np.array([score_map.get(p, floor) for p in U_full])
    rep = T.full_report(sc_full, y_full)
    return dict(dataset_id=d.get("dataset_id"), n_pairs=len(U_full), n_true=int(y_full.sum()),
                no_gamma_auprc_x=rep["auprc_x"])


def run_family(name, json_dir, resolver):
    files = sorted(glob.glob(f"{json_dir}/*_all_results.json"))
    rows = [r for r in (score_one_json_ablated(f, resolver) for f in files) if r]
    df = pd.DataFrame(rows)
    return df.no_gamma_auprc_x.mean(), len(df)


TOPO_MAP = {"GSD": "GSD.txt", "HSC": "HSC.txt", "VSC": "VSC.txt", "mCAD": "mCAD.txt",
            "B_cell_activation": "B_cell.txt", "Circadian_cycle": "circadian.txt",
            "EMT": "EMT.txt", "Pluripotent": "Pluripotent.txt"}

FAMILIES = {
    "network_sweep_e13": dict(
        t1_1=f"{ROOT}/analysis_data/network_sweep_final/e13_pos100/twinfer_inference_allpairs",
        t1_10=f"{ROOT}/analysis_data/network_sweep_final/e13_pos100/twinfer_inference_allpairs_t1_10",
        resolver=lambda d: d.get("ground_truth_matrix")),
    "network_sweep_final_broader": dict(
        t1_1=f"{ROOT}/analysis_data/network_sweep_final/twinfer_inference_allpairs",
        t1_10=f"{ROOT}/analysis_data/network_sweep_final/twinfer_inference_allpairs_t1_10",
        resolver=lambda d: d.get("ground_truth_matrix")),
    "mixed_network_sweep": dict(
        t1_1=f"{ROOT}/analysis_data/mixed_network_sweep/twinfer_inference_allpairs",
        t1_10=f"{ROOT}/analysis_data/mixed_network_sweep/twinfer_inference_allpairs_t1_10",
        resolver=lambda d: f"{ROOT}/input_data/mixed_network_sweep/{d.get('dataset_id')}.txt"),
    "real_networks": dict(
        t1_1=f"{ROOT}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs",
        t1_10=f"{ROOT}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs_t1_10",
        resolver=lambda d: f"{ROOT}/input_data/real_world_networks/{TOPO_MAP.get(d.get('sim_type'), '')}"),
}

CURRENT_NOGATE = {
    ("network_sweep_e13", "t1_1"): 1.378092, ("network_sweep_e13", "t1_10"): 1.178980,
    ("network_sweep_final_broader", "t1_1"): 2.305859, ("network_sweep_final_broader", "t1_10"): 1.710538,
    ("mixed_network_sweep", "t1_1"): 2.453570, ("mixed_network_sweep", "t1_10"): 1.932397,
    ("real_networks", "t1_1"): 2.099582, ("real_networks", "t1_10"): 1.615004,
}


def main():
    rows = []
    for name, cfg in FAMILIES.items():
        for t_label in ("t1_1", "t1_10"):
            mean_x, n = run_family(name, cfg[t_label], cfg["resolver"])
            rows.append(dict(benchmark=name, t=t_label, no_gamma_auprc_x=mean_x, n=n,
                              current_nogate_auprc_x=CURRENT_NOGATE[(name, t_label)]))
            print(f"{name} {t_label}: no_gamma={mean_x:.3f}x (n={n})  current_nogate={CURRENT_NOGATE[(name, t_label)]:.3f}x")
    df = pd.DataFrame(rows)
    df["delta"] = df.no_gamma_auprc_x - df.current_nogate_auprc_x
    df.to_csv(f"{HERE}/todo4v2_nogamma_ablation_results.csv", index=False)
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
