"""Compare per-cell values in the h5ad against the original .qs (cell_checks_R.csv from cell_checks.R), 10% of cells.
usage: validate_cells_vs_R.py FILE.h5ad     exit 1 on any mismatch"""
import sys, numpy as np, pandas as pd, anndata as ad
R = pd.read_csv("/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/cell_checks_R.csv")
a = ad.read_h5ad(sys.argv[1], backed="r"); idx = R["row"].values - 1          # R rows are 1-based
assert (a.obs_names.values[idx] == R["cell"].values).all(), "cell order differs"
def gw(g):
    out = []
    for s in g:
        h = 0
        for c in map(ord, s): h = (h * 131 + c) % 1000000007
        out.append(h)
    return np.array(out, dtype=np.float64)
w = gw(a.var_names)
def get(mat):
    m = mat[idx]; m = m.tocsr() if hasattr(m, "tocsr") else m
    m = m.astype(np.float64)
    return np.asarray(m.sum(1)).ravel(), np.diff(m.indptr), np.asarray(m @ w).ravel()
fails = []
for pre, mat, rtol in (("rna", a.X, 0), ("sctc", a.layers["SCT_counts"], 0), ("sctd", a.layers["SCT_data"], 2e-5)):
    s, n, ww = get(mat)
    for name, mine, ref in (("sum", s, R[f"{pre}_sum"].values), ("nnz", n, R[f"{pre}_nnz"].values), ("w", ww, R[f"{pre}_w"].values)):
        ok = np.allclose(mine, ref, rtol=rtol, atol=0) if rtol else np.array_equal(mine, ref)
        mx = np.max(np.abs(mine - ref) / np.maximum(np.abs(ref), 1))
        print(("PASS " if ok else "FAIL ") + f"{pre}_{name}: {len(idx)} cells, max rel diff {mx:.2e}"); 
        if not ok: fails.append(f"{pre}_{name}")
S = np.asarray(a.obsm["SCT_scale_data"][idx], dtype=np.float64)
for name, mine, ref in (("sum", S.sum(1), R["scale_sum"].values), ("sumsq", (S**2).sum(1), R["scale_sumsq"].values)):
    ok = np.allclose(mine, ref, rtol=1e-4, atol=1e-3); print(("PASS " if ok else "FAIL ") + f"scale_{name}: max abs diff {np.max(np.abs(mine-ref)):.2e}")
    if not ok: fails.append(f"scale_{name}")
print("\nSUMMARY:", "ALL PASS" if not fails else f"FAILED {fails}"); sys.exit(1 if fails else 0)
