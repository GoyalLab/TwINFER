"""Assemble the Novo Nordisk hPSC endo h5ad from the files written by export_seurat.R (step 2 of 2).

X            = RNA raw counts (int32 CSR, cells x genes)
layers       = SCT_counts (int32), SCT_data (float32), both aligned to the RNA gene space (genes not in SCT = 0; var['in_SCT'])
obs          = all Seurat meta.data columns (+ derived clone_size)
obsm         = X_pca, X_umap_rna, X_umap_rpca, SCT_scale_data (3000 variable genes, names in uns)
obsp         = SCT_nn, SCT_snn
varm['PCs']  = PCA loadings (zeros outside the variable genes; var['in_pca'])

usage: build_h5ad.py OUT.h5ad [--n-cells N]   (--n-cells = test mode on the first N cells)
"""
import argparse, json, os, sys, time
from collections import defaultdict
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad

EXP = "/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/export"
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--n-cells", type=int, default=None)
a = ap.parse_args()
if os.path.exists(a.out): sys.exit(f"refusing to overwrite {a.out}")
T0 = time.time()
def log(*m): print(f"[{time.time()-T0:7.1f}s]", *m, flush=True)

cells_all = pd.read_csv(f"{EXP}/cells.csv")["cell"].astype(str).values
N = a.n_cells or len(cells_all)
cells = cells_all[:N]

def read_csc(name):
    """genes x cells CSC from raw binary -> cells x genes CSR (zero-copy transpose); first N cells only."""
    nr, nc = [int(v) for v in open(f"{EXP}/{name}_dim.txt").read().split()]
    p = np.fromfile(f"{EXP}/{name}_p.bin", dtype=np.float64)
    assert len(p) == nc + 1 and p[0] == 0
    p = p[: N + 1]; nnz = int(p[-1]); assert nnz < 2**31 - 1
    i = np.fromfile(f"{EXP}/{name}_i.bin", dtype=np.int32, count=nnz)
    x = np.fromfile(f"{EXP}/{name}_x.bin", dtype=np.float64, count=nnz)
    return nr, i, p.astype(np.int32), x

def csr(nr, i, p, x):
    return sp.csc_matrix((x, i, p), shape=(nr, len(p) - 1)).T      # CSC transpose == CSR, no copy

def to_int32(x):
    out = np.empty(len(x), dtype=np.int32)
    for s in range(0, len(x), 100_000_000):
        c = x[s:s + 100_000_000]; r = np.rint(c)
        assert np.array_equal(c, r) and np.abs(r).max() < 2**31 - 1, "counts are not int32-safe integers"
        out[s:s + 100_000_000] = r
    return out

def features(name): return pd.read_csv(f"{EXP}/{name}_features.csv")["feature"].astype(str).values

## ---- var / X --------------------------------------------------------------------------------------
rna_feat = features("RNA_counts"); assert len(set(rna_feat)) == len(rna_feat)
nr, i, p, x = read_csc("RNA_counts"); assert nr == len(rna_feat)
X = csr(nr, i, p, to_int32(x)); del x; X.sort_indices()
log("X", X.shape, X.nnz, X.dtype, "sum", int(X.sum()))
var = pd.DataFrame(index=pd.Index(rna_feat, name="gene"))

## ---- SCT layers aligned to the RNA gene space -------------------------------------------------
sct_feat = features("SCT_counts"); assert (features("SCT_data") == sct_feat).all()
cmap = pd.Index(rna_feat).get_indexer(sct_feat); assert (cmap >= 0).all(), "SCT feature missing from RNA"
var["in_SCT"] = False; var.iloc[cmap, var.columns.get_loc("in_SCT")] = True
mono = bool(np.all(np.diff(cmap) > 0))
def sct_layer(name, conv):
    nr, i, p, x = read_csc(name); assert nr == len(sct_feat)
    m = sp.csr_matrix((conv(x), cmap.astype(np.int32)[i], p), shape=(len(p) - 1, len(rna_feat)))
    if not mono: m.sort_indices()
    return m
layers = {}
layers["SCT_counts"] = sct_layer("SCT_counts", to_int32); log("SCT_counts", layers["SCT_counts"].nnz, "sum", int(layers["SCT_counts"].sum()))
layers["SCT_data"] = sct_layer("SCT_data", lambda x: x.astype(np.float32)); log("SCT_data", layers["SCT_data"].nnz)

## ---- obs ---------------------------------------------------------------------------------------------
obs = pd.read_csv(f"{EXP}/obs.csv", index_col=0, low_memory=False, keep_default_na=True, dtype={"cell.barcode.ct": str})
assert (obs.index.astype(str).values[:N] == cells).all() and len(obs) == len(cells_all)
obs = obs.iloc[:N].copy(); obs.index = obs.index.astype(str); obs.index.name = "cell"
for col, lv in json.load(open(f"{EXP}/obs_factor_levels.json")).items():
    obs[col] = pd.Categorical(obs[col].astype(object), categories=lv)
obs["clone.id"] = obs["clone.id"].astype("Int64")
for col in ("Phase", "clust_annot_02"): obs[col] = obs[col].astype("category")
# derived: clone size over ALL cells of the full dataset (not only the first N in test mode)
full_ids = pd.read_csv(f"{EXP}/obs.csv", usecols=["clone.id"])["clone.id"]
obs["clone_size"] = obs["clone.id"].map(full_ids.value_counts()).astype("Int64")

## ---- obsm / varm / obsp / uns ----------------------------------------------------------------------
obsm = {}
for r, key in (("pca", "X_pca"), ("umap_rna", "X_umap_rna"), ("umap_rpca", "X_umap_rpca")):
    d = pd.read_csv(f"{EXP}/red_{r}.csv", index_col=0); assert (d.index.astype(str).values[:N] == cells).all()
    obsm[key] = d.iloc[:N].to_numpy(np.float32); log(key, obsm[key].shape)
hdr = pd.read_csv(f"{EXP}/SCT_scale_data.csv", nrows=0).columns.tolist(); assert hdr[1:] == list(cells_all), "scale.data column order != cells.csv"
use = [0] + list(range(1, N + 1))
sd = pd.read_csv(f"{EXP}/SCT_scale_data.csv", usecols=use, index_col=0,
                 dtype=defaultdict(lambda: np.float32, {hdr[0]: str}))
scale_feat = sd.index.astype(str).tolist(); obsm["SCT_scale_data"] = np.ascontiguousarray(sd.to_numpy(np.float32).T); del sd
log("SCT_scale_data", obsm["SCT_scale_data"].shape)

ld = pd.read_csv(f"{EXP}/red_pca_loadings.csv", index_col=0)
pcs = np.zeros((len(rna_feat), ld.shape[1]), dtype=np.float32)
idx = pd.Index(rna_feat).get_indexer(ld.index.astype(str)); assert (idx >= 0).all()
pcs[idx] = ld.to_numpy(np.float32); var["in_pca"] = False; var.iloc[idx, var.columns.get_loc("in_pca")] = True

obsp = {}
for g in ("SCT_nn", "SCT_snn"):
    assert (pd.read_csv(f"{EXP}/graph_{g}_cells.csv")["cell"].astype(str).values == cells_all).all()
    nr, i, p, x = read_csc(f"graph_{g}")                     # N columns of the full cells x cells graph
    G = sp.csc_matrix((x, i, p), shape=(nr, len(p) - 1)).tocsr()[:N, :N]    # square top-left block (test mode); full = whole graph
    obsp[g] = G.astype(np.float32); log("graph", g, G.shape, G.nnz)

uns = {"SCT_variable_features": pd.read_csv(f"{EXP}/SCT_variable_features.csv")["feature"].astype(str).tolist(),
       "SCT_scale_data_features": scale_feat,
       "source": {"file": "seurat_lineage_tracing_linneberg_agerholm_2026.qs", "seurat_version": "5.0.2",
                  "X": "RNA raw counts", "layers": "SCT_counts, SCT_data (RNA gene space, absent genes = 0)",
                  "note_nCount_RNA": "obs.nCount_RNA / nFeature_RNA are the original Seurat values and do NOT equal X row sums / non-zero gene counts: nCount_RNA >= X row sum in all cells (equal in 9%), median gap 50 counts (0.2%), max 2.4%, larger in hESC/PS samples than DE; likely computed on a slightly larger gene set before genes were removed (unverified). Recompute from X if needed.", "note_cell_barcode_ct": "obs.cell.barcode.ct is the 10x cell barcode (Sample-N_<16mer>-1) linking cells to clone.id, not a lineage-tag sequence; lineage tag sequences are not in the source object"}}

adata = ad.AnnData(X=X, obs=obs, var=var, obsm=obsm, varm={"PCs": pcs}, obsp=obsp, layers=layers, uns=uns)
log(adata)
adata.write_h5ad(a.out, compression="lzf")
log("written", a.out, f"{os.path.getsize(a.out)/1e9:.2f} GB")
