from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_175bdfed_2026-09-17_todo4v2; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import glob, os
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

def partial_corr_matrix(X, ridge):
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

def score_topology(expr_dirs, gt_path, ridge_fn):
    xs = []
    for expr_path in expr_dirs:
        if not os.path.exists(expr_path):
            continue
        df = pd.read_csv(expr_path, sep=None, engine="python", index_col=0)
        gene_names = list(df.columns)
        X = np.log1p(df.to_numpy(dtype=float))
        ridge = ridge_fn(len(gene_names))
        pcorr = partial_corr_matrix(X, ridge)
        true_edges = load_true_edges_matrix(gt_path, gene_names)
        n = len(gene_names)
        U = [(gene_names[i], gene_names[j]) for i in range(n) for j in range(n) if i != j]
        y = np.array([1 if p in true_edges else 0 for p in U])
        if y.sum() < 2 or y.sum() == len(y):
            continue
        idx = [(gene_names.index(a), gene_names.index(b)) for a, b in U]
        sc_pcorr = np.array([abs(pcorr[i, j]) for i, j in idx])
        xs.append(auprc_x(sc_pcorr, y))
    return np.nanmean(xs) if xs else np.nan

REAL_TOPOS = {
    "VSC": ("beeline_inference/VSC_seeded", f'{TWINFER_PROJECT_ROOT}/input_data/real_world_networks/VSC.txt'),
    "Pluripotent": ("beeline_inference/Pluripotent", f'{TWINFER_PROJECT_ROOT}/input_data/real_world_networks/Pluripotent.txt'),
}
dirs_cache = {}
for label, (subdir, gt) in REAL_TOPOS.items():
    base = f"{R}/paper_analysis/real_networks/{subdir}"
    expr_dirs = sorted(glob.glob(f"{base}/simrep*_twin_paired/GENIE3/working_dir/ExpressionData.csv"))
    dirs_cache[label] = (expr_dirs, gt)

for ridge in [0.01, 0.03, 0.06, 0.1, 0.2, 0.3, 0.5, 1.0]:
    row = {}
    for label, (expr_dirs, gt) in dirs_cache.items():
        row[label] = score_topology(expr_dirs, gt, lambda n: ridge)
    print(f"ridge={ridge:5.2f}  VSC={row['VSC']:.3f}  Pluripotent={row['Pluripotent']:.3f}")
