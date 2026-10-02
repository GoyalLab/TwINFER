#!/usr/bin/env python3
"""n_genes / mito% distribution for HS054_Sot48h (raw Cell Ranger filtered matrix, no QC
filters applied yet) -- to pick sensible mito/gene-count cutoffs before writing
build_hs054_qc_matrix.py's thresholds."""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import numpy as np
import pandas as pd
import scanpy as sc

RAW_DIR = f'{TWINFER_PROJECT_ROOT}/real_data/pancreatic_cancer/HS054_Sot48h/outs/filtered_feature_bc_matrix'


def load():
    adata = sc.read_mtx(f"{RAW_DIR}/matrix.mtx.gz").T
    features = pd.read_csv(f"{RAW_DIR}/features.tsv.gz", sep="\t", header=None)
    barcodes = pd.read_csv(f"{RAW_DIR}/barcodes.tsv.gz", sep="\t", header=None)
    assert adata.shape == (len(barcodes), len(features))
    adata.var_names = features[1].to_numpy()
    adata.var_names_make_unique()
    adata.obs_names = barcodes[0].str.replace(r"-1$", "", regex=True).to_numpy()
    return adata


adata = load()
genes = pd.Series(adata.var_names)
mito_mask = genes.str.upper().str.startswith("MT-").to_numpy()
print(f"n cells (Cell Ranger filtered) = {adata.shape[0]}, n mito genes = {mito_mask.sum()}")

X = adata.X.tocsr()
total = np.asarray(X.sum(axis=1)).ravel()
mito = np.asarray(X[:, mito_mask].sum(axis=1)).ravel()
pct_mt = np.divide(100 * mito, total, out=np.zeros_like(total), where=total > 0)
n_genes = np.asarray((X > 0).sum(axis=1)).ravel()

df = pd.DataFrame({"total_counts": total, "pct_mt": pct_mt, "n_genes": n_genes}, index=adata.obs_names)
pd.set_option("display.width", 200)
print(df.describe(percentiles=[.01, .05, .1, .25, .5, .75, .9, .95, .99]).round(2))

for cut in [10, 15, 20, 21, 25, 26, 30]:
    print(f"mito>{cut}%: {(df['pct_mt'] > cut).mean():.3f}")
for g in [200, 500, 1000, 1500, 2000]:
    print(f"n_genes<{g}: {(df['n_genes'] < g).mean():.3f}")
for g in [5000, 6000, 7000, 7500, 8000]:
    print(f"n_genes>{g}: {(df['n_genes'] > g).mean():.3f}")

singlets = pd.read_csv(
    f'{TWINFER_PROJECT_ROOT}/real_data/pancreatic_cancer/HS054_Sot48h__singlets_all.txt', header=None
)[0]
is_singlet = df.index.isin(set(singlets))
print(f"\nmatrix cells matched to singlets_all.txt: {is_singlet.sum()} / {len(singlets)} listed")
print(df[is_singlet].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).round(2))
