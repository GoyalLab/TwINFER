# [UNREVIEWED: rescued 2026-09-30 from scratchpad_175bdfed_2026-09-17_todo4v2; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Test whether a conditional (partial-correlation, controlling for all other genes in the panel)
term is a consistently safe addition -- i.e. does it separate true from false edges with the SAME
sign everywhere (not just VSC), and does it noticeably beat |rho_t1| (the plain marginal
correlation TwINFER already uses)?

Uses the raw ExpressionData.csv (cells x genes, twin_paired scheme = both timepoints pooled)
already generated for each topology's beeline competitor runs -- no need to touch simulation_data
directly.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

R = f'{TWINFER_PROJECT_ROOT}/analysis_data'


def auprc_x(sc, y):
    sc = np.asarray(sc, float)
    sc = np.nan_to_num(sc, nan=np.nanmin(sc) - 1.0 if np.isfinite(sc).any() else 0.0)
    rand = y.mean()
    if rand <= 0 or rand >= 1:
        return np.nan
    prec, rec, _ = precision_recall_curve(y, sc)
    return auc(rec, prec) / rand


def partial_corr_matrix(X, ridge=1e-2):
    """X: cells x genes (log1p). Returns |partial correlation| matrix via ridge-regularized
    precision matrix inversion (small ridge needed since n_genes << n_cells but some genes can be
    near-collinear in these small networks). Zero-variance genes get an all-zero row/col (no
    signal, not a crash)."""
    std = X.std(axis=0)
    dead = std < 1e-10
    Xc = X - X.mean(axis=0, keepdims=True)
    Xc = np.divide(Xc, std[None, :] + 1e-12, out=np.zeros_like(Xc), where=~dead[None, :])
    Rmat = np.corrcoef(Xc, rowvar=False)
    Rmat = np.nan_to_num(Rmat, nan=0.0)
    np.fill_diagonal(Rmat, 1.0)
    p = Rmat.shape[0]
    P = np.linalg.pinv(Rmat + ridge * np.eye(p))
    d = np.sqrt(np.abs(np.diag(P))) + 1e-12
    pcorr = -P / np.outer(d, d)
    np.fill_diagonal(pcorr, 0.0)
    pcorr[dead, :] = 0.0
    pcorr[:, dead] = 0.0
    return pcorr


def load_true_edges_matrix(matrix_path, gene_names):
    M = np.loadtxt(matrix_path, delimiter=",")
    n = len(gene_names)
    return {(gene_names[i], gene_names[j]) for i in range(n) for j in range(n) if i != j and M[i, j] != 0}


def score_topology(expr_dirs, gt_path, label):
    rows = []
    for expr_path in expr_dirs:
        if not os.path.exists(expr_path):
            continue
        df = pd.read_csv(expr_path, sep=None, engine="python", index_col=0)
        gene_names = list(df.columns)
        X = np.log1p(df.to_numpy(dtype=float))
        pcorr = partial_corr_matrix(X)
        Rraw = np.corrcoef(X, rowvar=False)

        true_edges = load_true_edges_matrix(gt_path, gene_names)
        n = len(gene_names)
        U = [(gene_names[i], gene_names[j]) for i in range(n) for j in range(n) if i != j]
        y = np.array([1 if p in true_edges else 0 for p in U])
        if y.sum() < 2 or y.sum() == len(y):
            continue
        idx = [(gene_names.index(a), gene_names.index(b)) for a, b in U]
        sc_pcorr = np.array([abs(pcorr[i, j]) for i, j in idx])
        sc_rho = np.array([abs(Rraw[i, j]) for i, j in idx])

        rows.append(dict(
            topology=label, n_genes=n, n_true=int(y.sum()),
            auprc_x_pcorr=auprc_x(sc_pcorr, y),
            auprc_x_rho=auprc_x(sc_rho, y),
        ))
    return rows


all_rows = []

REAL_TOPOS = {
    "VSC": ("beeline_inference/VSC_seeded", f'{TWINFER_PROJECT_ROOT}/input_data/real_world_networks/VSC.txt'),
    "mCAD": ("beeline_inference/mCAD_seeded", f'{TWINFER_PROJECT_ROOT}/input_data/real_world_networks/mCAD.txt'),
    "Circadian_cycle": ("beeline_inference/Circadian_cycle", f'{TWINFER_PROJECT_ROOT}/input_data/real_world_networks/circadian.txt'),
    "Pluripotent": ("beeline_inference/Pluripotent", f'{TWINFER_PROJECT_ROOT}/input_data/real_world_networks/Pluripotent.txt'),
}
for label, (subdir, gt) in REAL_TOPOS.items():
    base = f"{R}/paper_analysis/real_networks/{subdir}"
    expr_dirs = sorted(glob.glob(f"{base}/simrep*_twin_paired/GENIE3/working_dir/ExpressionData.csv"))
    all_rows += score_topology(expr_dirs, gt, label)

# network_sweep_final: e9_pos0_sign_ratio, e5_pos100_density
NS_BASE = f"{R}/network_sweep_final/beeline_inference"
NS_INPUT = f'{TWINFER_PROJECT_ROOT}/input_data/network_sweep_final'
for target in ["e9_pos0_sign_ratio", "e5_pos100_density"]:
    dirs = sorted(glob.glob(f"{NS_BASE}/grn_n6_{target}_rep*"))
    for d in dirs:
        rep_name = os.path.basename(d)  # e.g. grn_n6_e9_pos0_sign_ratio_rep0
        gt = f"{NS_INPUT}/{rep_name}.txt"
        expr_dirs = sorted(glob.glob(f"{d}/simrep*_twin_paired/GENIE3/working_dir/ExpressionData.csv"))
        if not os.path.exists(gt) or not expr_dirs:
            continue
        all_rows += score_topology(expr_dirs, gt, target)

df = pd.DataFrame(all_rows)
df.to_csv(f'{TWINFER_PROJECT_ROOT}/clean_data/benchmarks/network_benchmarks/score/formula_search/todo4v2_exploration/partial_corr_results.csv', index=False)
print(df.to_string(index=False))
print()
print("=== mean by topology ===")
print(df.groupby("topology")[["auprc_x_pcorr", "auprc_x_rho"]].mean().round(3))
