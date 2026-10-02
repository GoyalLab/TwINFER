#!/usr/bin/env python3
"""TwinScore_supplement, gated + bootstrapped version (apply_twinscore_supplement_fatemap_gated_bootstrap.py),
applied to SIMULATED twin data (BoolODE real-network twin sims; also usable on any file in the same
clone_id / cell_id / time_step / gene_i_mRNA layout, e.g. the Gillespie df_rows files).

Same formula and same bootstrap / phi-gate machinery as the FateMap runner (imported from it, not copied):
    TwinScore = w_rel * s(phi_x) + s(PAIR) + direction_term,  PAIR = s(D) + gate_g*s(R)   [+ gate_g*v*s(Wz)]
Differences from the FateMap runner, all forced by the data:
  * t1_raw / t2_raw are the twin cells (both twins of every clone) at two post-division time points t1 < t2
    (the FateMap runner uses replicate A / B). S / z_het / R / gate_g / D use the t2 frame only, so no cell is
    counted twice in one statistic.
  * Wz is NOT computed (v_gate = 0): it needs integer counts for split-half binomial thinning; BoolODE values are
    continuous. PAIR = s(D) + gate_g*s(R) here.
  * Truth = directed edges of the topology matrix (rows=regulator, cols=target) instead of CollecTRI.
  * z_dagger (direction term) comes from the all-pairs infer_with_twinfer JSON (boolode_real_infer_allpairs.py,
    direction.rho_cross_null[a__b].z_rho_cross), same source the FateMap runner uses.

    python apply_twinscore_supplement_sim_gated_bootstrap.py NET REP [--source boolode|file:<csv>] [--n-cores 8]
Output: analysis_data/boolode_sims_real_networks/twinscore_supp_gated_bootstrap/<NET>_rep_<REP>_{pair_terms,gene_terms}.csv
        and <NET>_rep_<REP>_metrics.json
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, average_precision_score

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_analysis.fatemap_pipeline import twinscore_supp_helpers as supp
from paper_analysis.fatemap_pipeline import apply_twinscore_supplement_fatemap_gated_bootstrap as gb

log = supp.log
s = supp.s

ROOT = f'{TWINFER_PROJECT_ROOT}'
FMT = f"{ROOT}/analysis_data/boolode_sims_real_networks/twinfer_format"
INF = f"{ROOT}/analysis_data/boolode_sims_real_networks/twinfer_inference_allpairs"
OUT = f"{ROOT}/analysis_data/boolode_sims_real_networks/twinscore_supp_gated_bootstrap"
RW = f"{ROOT}/input_data/real_world_networks"
OLD = f"{ROOT}/simulation_data/twinfer_format"
TOPO = {"B_cell_activation": f"{RW}/B_cell.txt", "EMT_real": f"{RW}/EMT.txt", "Pluripotent_real": f"{RW}/Pluripotent.txt",
        **{n: f"{OLD}/{n}/interaction_matrix.txt" for n in ("GSD", "HSC", "mCAD", "VSC")}}
T1T2 = {"B_cell_activation": (500, 799), "EMT_real": (900, 1599), "Pluripotent_real": (900, 1599),
        "GSD": (500, 799), "HSC": (500, 799), "mCAD": (300, 499), "VSC": (300, 499)}


def load_z_dagger(net, rep):
    path = f"{INF}/{net}_rep_{rep}_all_results.json"
    if not os.path.exists(path):
        log(f"    WARNING: {path} missing -> direction term = 0")
        return {}
    d = json.load(open(path))
    rn = d["direction"]["rho_cross_null"]
    return {k: float(v["z_rho_cross"]) for k, v in rn.items()
            if isinstance(v, dict) and v.get("z_rho_cross") is not None and np.isfinite(v["z_rho_cross"])}


def metrics_for(y, cols):
    out = {}
    base = y.mean()
    for name, v in cols.items():
        v = np.nan_to_num(np.asarray(v, float), nan=np.nanmin(v) - 1 if np.isfinite(v).any() else 0)
        out[name] = dict(auroc=float(roc_auc_score(y, v)), auprc_x=float(average_precision_score(y, v) / base))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("net", choices=list(TOPO))
    ap.add_argument("rep", type=int)
    ap.add_argument("--n-shuffles", type=int, default=500)
    ap.add_argument("--n-cores", type=int, default=8)
    ap.add_argument("--out-dir", default=OUT)
    args = ap.parse_args()
    net, rep = args.net, args.rep
    os.makedirs(args.out_dir, exist_ok=True)

    M = np.loadtxt(TOPO[net], delimiter=",")
    genes = [f"gene_{i + 1}" for i in range(M.shape[0])]
    df = pd.read_csv(f"{FMT}/{net}/replicate_{rep}_simulation.csv")
    t1, t2 = T1T2[net]
    t1_raw = df[df.time_step == t1].reset_index(drop=True)
    t2_raw = df[df.time_step == t2].reset_index(drop=True)
    raw = t2_raw
    log(f"[{net} rep {rep}] {len(genes)} genes, t1={t1} ({len(t1_raw)} cells) t2={t2} ({len(t2_raw)} cells), "
        f"{t2_raw.clone_id.nunique()} clones")

    log("S, z_het (fresh permutation)")
    zhet_matrix, S, _ = supp.compute_zhet_zrho(raw, genes, n_shuffles=args.n_shuffles, n_cores=args.n_cores)
    log("C, heritability h_t1 / h_t2")
    C_t1, meff_t1 = supp.sister_matrix_from_frame(t1_raw, genes)
    C_t2, meff_t2 = supp.sister_matrix_from_frame(t2_raw, genes)
    h1 = {g: C_t1.loc[g, g] for g in genes}
    h2 = {g: C_t2.loc[g, g] for g in genes}
    log("cross-time within-clone correlation (persistence numerator)")
    rho_dagger, meff_cross = supp.cross_matrix(t1_raw, t2_raw, genes)
    rdd = {g: rho_dagger.loc[g, g] for g in genes}

    log("bootstrap noise floors")
    b1, n1 = gb.bootstrap_null_sd_diag(t1_raw, genes, seed=supp.SEED)
    b2, n2 = gb.bootstrap_null_sd_diag(t2_raw, genes, seed=supp.SEED + 1)
    nf1 = {g: (b1[g] if np.isfinite(b1[g]) else supp.null_sd(meff_t1)) for g in genes}
    nf2 = {g: (b2[g] if np.isfinite(b2[g]) else supp.null_sd(meff_t2)) for g in genes}
    phi, w_rel, phi_info = gb.persistence_gated_bootstrap(h1, h2, rdd, genes, nf1, nf2, t1_raw, t2_raw)

    lam = np.minimum(1.0, zhet_matrix.abs() / 2.33)
    sd_reg, _ = gb.bootstrap_null_sd_offdiag(t2_raw, genes, seed=supp.SEED + 2, kind="reg", extra=lam)
    sd_tw, _ = gb.bootstrap_null_sd_offdiag(t2_raw, genes, seed=supp.SEED + 3, kind="C")
    if not np.isfinite(sd_reg) or sd_reg <= 0:
        sd_reg = supp.null_sd(supp.m_eff_step1(raw.groupby("clone_id").size().to_numpy()))
    if not np.isfinite(sd_tw) or sd_tw <= 0:
        sd_tw = supp.null_sd(meff_t2)
    zreg = (S - lam * C_t2) / sd_reg
    R_fn = supp.clr_calibrate(zreg, genes)
    n = len(genes)
    gate_g = supp.signal_share((C_t2 / sd_tw).to_numpy()[~np.eye(n, dtype=bool)])

    log("D (PIDC)")
    D_mat = supp.compute_D(t2_raw, genes)

    log("direction: z_dagger, gamma, kappa_gamma, q")
    zmap = load_z_dagger(net, rep)
    U = [(a, b) for a in genes for b in genes if a != b]
    zxy = np.array([zmap.get(f"{a}__{b}", np.nan) for a, b in U])
    zyx = np.array([zmap.get(f"{b}__{a}", np.nan) for a, b in U])
    gamma = (np.abs(zxy) - np.abs(zyx)) / np.sqrt(2 * (1 - 2 / np.pi))
    kappa = supp.signal_share(gamma)
    q = 0.5 + (supp.R0 - 0.5) * (2 * norm.cdf(np.abs(gamma)) - 1) * np.sign(gamma)
    q = np.clip(np.nan_to_num(q, nan=0.5), 1e-6, 1 - 1e-6)
    gate_dir = (np.nan_to_num(np.maximum(np.abs(zxy), np.abs(zyx)), nan=0.0) > supp.Z_DIRECTION_GATE).astype(float)
    direction_term = kappa * gate_dir * np.log(q)

    D_vec = np.array([D_mat.loc[a, b] for a, b in U])
    R_vec = np.array([R_fn(a, b) for a, b in U])
    PAIR = s(D_vec) + gate_g * s(R_vec)
    phi_x = np.array([phi[a] for a, b in U])
    score = w_rel * s(phi_x) + s(PAIR) + direction_term

    gi = {g: i for i, g in enumerate(genes)}
    truth = np.array([1 if M[gi[a], gi[b]] != 0 else 0 for a, b in U])
    pair_df = pd.DataFrame({
        "gene_1": [a for a, b in U], "gene_2": [b for a, b in U], "S": [S.loc[a, b] for a, b in U],
        "C": [C_t2.loc[a, b] for a, b in U], "zhet": [zhet_matrix.loc[a, b] for a, b in U],
        "zreg": [zreg.loc[a, b] for a, b in U], "R": R_vec, "D": D_vec, "PAIR": PAIR,
        "z_dagger_xy": zxy, "z_dagger_yx": zyx, "gamma": gamma, "direction_term": direction_term,
        "phi_x": phi_x, "TwinScore": score, "true_edge": truth,
    })
    gene_df = pd.DataFrame({
        "gene": genes, "h_t1": [h1[g] for g in genes], "h_t2": [h2[g] for g in genes], "rho_dagger_gg": [rdd[g] for g in genes],
        "phi": [phi[g] for g in genes], "gated": [g in set(phi_info["gated"]) for g in genes],
        "nf_h1": [nf1[g] for g in genes], "nf_h2": [nf2[g] for g in genes],
    })
    stem = f"{args.out_dir}/{net}_rep_{rep}"
    pair_df.to_csv(f"{stem}_pair_terms.csv", index=False)
    gene_df.to_csv(f"{stem}_gene_terms.csv", index=False)
    cols = {"TwinScore": score, "PAIR": PAIR, "D(PIDC)": D_vec, "R": R_vec, "phi_x": phi_x,
            "abs_S": np.abs(pair_df.S.to_numpy()), "abs_C": np.abs(pair_df.C.to_numpy())}
    met = metrics_for(truth, cols)
    diag = dict(net=net, rep=rep, t1=t1, t2=t2, n_genes=n, n_true=int(truth.sum()), n_pairs=len(U), w_rel=w_rel,
                gate_g=float(gate_g), kappa_gamma=float(kappa), n_phi_gated=len(phi_info["gated"]),
                z_dagger_coverage=len(zmap), wz="not computed (continuous data)", metrics=met)
    json.dump(diag, open(f"{stem}_metrics.json", "w"), indent=2, default=str)
    log(f"[{net} rep {rep}] " + "  ".join(f"{k}: AUROC {v['auroc']:.3f} AUPRCx {v['auprc_x']:.2f}" for k, v in met.items()))


if __name__ == "__main__":
    main()
