"""Validate the h5ad built by build_h5ad.py against R-side checksums and the raw binary export.
usage: validate_h5ad.py FILE.h5ad [--n-cells N]   (N only if the file was built in test mode)
Exit code 1 if any check fails.
"""
import argparse, json, sys
import h5py, numpy as np, pandas as pd, anndata as ad

EXP = "/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/export"
ap = argparse.ArgumentParser(); ap.add_argument("h5ad"); ap.add_argument("--n-cells", type=int, default=None)
a = ap.parse_args()
R = json.load(open(f"{EXP}/R_checksums.json"))
cells_all = pd.read_csv(f"{EXP}/cells.csv")["cell"].astype(str).values
N = a.n_cells or len(cells_all); full = N == len(cells_all)
fails = []
def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""));
    if not ok: fails.append(name)

adata = ad.read_h5ad(a.h5ad, backed="r")
check("dims", adata.shape == (N, R["RNA_counts_dim"][0]), str(adata.shape))
check("obs_names == Seurat cell order", (adata.obs_names.values == cells_all[:N]).all())
check("obs_names unique / var_names unique", adata.obs_names.is_unique and adata.var_names.is_unique)

def csr_stats(path):   # chunked sum / nnz / max over a stored CSR group
    with h5py.File(a.h5ad) as f:
        d = f[f"{path}/data"]; tot, mx = 0.0, -np.inf
        for s in range(0, d.shape[0], 50_000_000):
            c = d[s:s + 50_000_000].astype(np.float64); tot += c.sum(); mx = max(mx, c.max())
        return tot, d.shape[0], mx, f[f"{path}/data"].dtype
for key, grp, rk in (("X", "X", "RNA_counts"), ("SCT_counts", "layers/SCT_counts", "SCT_counts"), ("SCT_data", "layers/SCT_data", "SCT_data")):
    tot, nnz, mx, dt = csr_stats(grp)
    if full:
        check(f"{key} nnz == R", nnz == R[f"{rk}_nnz"], f"{nnz} vs {R[rk + '_nnz']}")
        check(f"{key} sum == R", abs(tot - R[f"{rk}_sum"]) <= 1e-6 * abs(R[f"{rk}_sum"]) + 1e-9, f"{tot:.6f} vs {R[rk + '_sum']:.6f} ({dt})")
    else:
        print(f"info {key}: nnz {nnz} sum {tot:.1f} dtype {dt}")

# independent re-read of random cells straight from the raw binary export (offset reads, no build code)
rng = np.random.default_rng(0); pick = np.sort(rng.choice(N, size=min(60, N), replace=False))
def raw_col(name, j, nr):
    p = np.memmap(f"{EXP}/{name}_p.bin", dtype=np.float64, mode="r")
    s, e = int(p[j]), int(p[j + 1])
    i = np.fromfile(f"{EXP}/{name}_i.bin", dtype=np.int32, count=e - s, offset=4 * s)
    x = np.fromfile(f"{EXP}/{name}_x.bin", dtype=np.float64, count=e - s, offset=8 * s)
    v = np.zeros(nr); v[i] = x; return v
nrna = R["RNA_counts_dim"][0]
sct_feat = pd.read_csv(f"{EXP}/SCT_counts_features.csv")["feature"].astype(str).values
cmap = adata.var_names.get_indexer(sct_feat)
ok_x = ok_c = ok_d = True
for j in pick:
    ok_x &= np.array_equal(np.asarray(adata.X[j].todense()).ravel() if hasattr(adata.X[j], "todense") else np.asarray(adata.X[j]).ravel(), raw_col("RNA_counts", j, nrna))
    for lay, name in (("SCT_counts", "SCT_counts"), ("SCT_data", "SCT_data")):
        row = adata.layers[lay][j]; row = np.asarray(row.todense()).ravel() if hasattr(row, "todense") else np.asarray(row).ravel()
        ref = np.zeros(nrna); ref[cmap] = raw_col(name, j, len(sct_feat))
        good = np.allclose(row, ref, rtol=1e-6, atol=0) if lay == "SCT_data" else np.array_equal(row, ref)
        if lay == "SCT_counts": ok_c &= good
        else: ok_d &= good
check(f"X identical to raw export in {len(pick)} random cells", ok_x)
check("SCT_counts identical (aligned) in random cells", ok_c)
check("SCT_data identical (aligned, float32 tol) in random cells", ok_d)
check("row sums of first 5 cells == R colSums", np.allclose(np.asarray(adata.X[:5].sum(axis=1)).ravel(), R["RNA_counts_colsums_first5"]), str(np.asarray(adata.X[:5].sum(axis=1)).ravel()))
ins = adata.var["in_SCT"].values
check("SCT layers are zero outside in_SCT genes", float(abs(adata.layers["SCT_counts"][:min(N, 2000)][:, ~ins]).sum()) == 0.0)

# obs / clones
obs = adata.obs; src = pd.read_csv(f"{EXP}/obs.csv", index_col=0, low_memory=False).iloc[:N]
check("obs columns (minus derived clone_size) == Seurat meta.data columns", list(obs.columns.drop("clone_size")) == list(src.columns), f"{obs.shape[1]-1} cols")
check("NA counts per column equal", all(int(obs[c].isna().sum()) == int(src[c].isna().sum()) for c in src.columns))
check("clone.id NA <=> cell.barcode.ct NA", (obs["clone.id"].isna() == obs["cell.barcode.ct"].isna()).all())
if full:
    cs = obs["clone.id"].dropna().astype(int).value_counts(); Rcs = pd.read_csv(f"{EXP}/clone_sizes_R.csv").set_index("clone.id")["size"]
    mine = {int(k): int(v) for k, v in cs.items()}; theirs = {int(k): int(v) for k, v in Rcs.items()}
    check("clone sizes recomputed == R table", mine == theirs, f"{len(mine)} clones, max {max(mine.values())}")
    check("n_clones / cells with clone / max size == R", (len(cs), int(cs.sum()), int(cs.max())) == (R["n_clones"], R["n_cells_with_clone"], R["max_clone_size"]))
    chk = obs.dropna(subset=["clone.id"]); check("obs.clone_size == recomputed per-clone count", (chk["clone_size"].astype(int).values == chk["clone.id"].astype(int).map(cs).values).all())
print("obsm:", {k: adata.obsm[k].shape for k in adata.obsm}, "| obsp:", list(adata.obsp), "| layers:", list(adata.layers), "| varm:", list(adata.varm))
print("\nSUMMARY:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"); sys.exit(1 if fails else 0)
