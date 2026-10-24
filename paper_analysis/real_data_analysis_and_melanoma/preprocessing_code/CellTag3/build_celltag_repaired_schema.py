"""CellTag schema whose clones are the CellTag authors' full caller: Jaccard >= 0.6 on cells carrying 2-25 CellTags, connected components,
and the 10 largest components with edge density below 0.2 decomposed into maximum cliques (their ident_sparse_clones / fix_sparse_clones;
exports/legacy_reproductions/pairing/celltag_clones/clone_table_repaired.csv, written by helpers/celltag_clones_repaired.py).
Built on the quality-controlled cell set of celltag_clones_published_nopad(_counts).h5ad: every obs column kept, one column added,
repaired_clone (the authors' clone id; NaN for cells in no clone). The join is on the cell barcode exactly as stored in both files; the
script asserts that every repaired-table cell is found. Writes datasets/celltag/celltag_clones_repaired_nopad(_counts).h5ad."""
import os, numpy as np, pandas as pd, anndata as ad
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # work root: datasets/ and exports/ sit next to the folder holding this script (see run_rebuild_celltag.sbatch)
R = pd.read_csv("exports/legacy_reproductions/pairing/celltag_clones/clone_table_repaired.csv"); m = dict(zip(R["cell.bc"].astype(str), R["clone.id"].astype(float)))
for src, dst in (("datasets/celltag/celltag_clones_published_nopad.h5ad", "datasets/celltag/celltag_clones_repaired_nopad.h5ad"), ("datasets/celltag/celltag_clones_published_nopad_counts.h5ad", "datasets/celltag/celltag_clones_repaired_nopad_counts.h5ad")):
    A = ad.read_h5ad(src); names = A.obs_names.astype(str)
    found = pd.Index(names).isin(list(m)).sum(); assert found == len(m), f"{src}: {found} of {len(m)} repaired cells found"
    A.obs["repaired_clone"] = [m.get(n, np.nan) for n in names]
    cl = A.obs["repaired_clone"].dropna(); sizes = cl.value_counts()
    print(f"{dst}: cells {A.n_obs}, in a clone {len(cl)}, clones {cl.nunique()}, clone size median {int(sizes.median())} max {int(sizes.max())}, clones spanning both timepoints {int(A.obs.dropna(subset=['repaired_clone']).groupby('repaired_clone')['Time point'].nunique().eq(2).sum())}", flush=True)
    A.write_h5ad(dst)
print("SCHEMA_DONE")
