"""Finalize the no-singletCode dataset: drop ambiguous duplicate cell names, then count Jaccard>=0.6 pairs
using sparse matrix algebra (same formulation as the Morris lab's jaccard_similarities) instead of a Python
loop over 19.6M tuples, which was the bottleneck in the original build.

intersection = M @ M.T on the binary cell x tag matrix; union = |A| + |B| - intersection. Both operate only on
the stored non-zeros, i.e. only on cell pairs sharing at least one tag.
"""
import os, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
HERE=os.path.dirname(os.path.abspath(__file__))
F=os.path.join(HERE,"..","datasets", "celltag", "celltag_nosinglet_counts.h5ad")
def log(m): print(m,flush=True)

A=ad.read_h5ad(F)
n=pd.Index(A.obs_names); dup=n.duplicated(keep=False)
log(f"dropping {int(dup.sum())} rows with ambiguous duplicated names ({100*dup.mean():.3f}%)")
A=A[~dup].copy()
A.var_names_make_unique()
A.write_h5ad(F)
log(f"clean dataset: {A.shape} | timepoints={dict(A.obs['timepoint'].value_counts())}")

# ---- sparse pair counting ----
tagsets=[s.split(',') for s in A.obs['lineage_barcodes']]
nt=np.array([len(set(t)) for t in tagsets])
elig=np.where((nt>=2)&(nt<=25))[0]
log(f"eligible cells (2-25 tags): {len(elig)}")
sub=[sorted(set(tagsets[i])) for i in elig]
vocab={t:j for j,t in enumerate(sorted({t for s in sub for t in s}))}
rows=[]; cols=[]
for i,s in enumerate(sub):
    for t in s: rows.append(i); cols.append(vocab[t])
M=sp.csr_matrix((np.ones(len(rows),dtype=np.int32),(rows,cols)),shape=(len(sub),len(vocab)))
log(f"cell x tag matrix: {M.shape}, nnz={M.nnz}")

inter=(M @ M.T).tocoo()
sizes=np.asarray(M.sum(1)).ravel()
mask=inter.row < inter.col                      # upper triangle, excludes diagonal
r,c,v=inter.row[mask],inter.col[mask],inter.data[mask]
union=sizes[r]+sizes[c]-v
jac=v/union
log(f"co-occurring pairs (Jaccard computed): {len(jac):,}")
for th in [0.6]:
    keep=jac>=th
    log(f"pairs with Jaccard >= {th}: {int(keep.sum()):,}")
    tp=A.obs['timepoint'].values[elig]
    a,b=tp[r[keep]],tp[c[keep]]
    log(f"   within day2: {int(((a=='day2')&(b=='day2')).sum()):,}")
    log(f"   within day5: {int(((a=='day5')&(b=='day5')).sum()):,}")
    log(f"   CROSS-TIMEPOINT: {int((a!=b).sum()):,}   [filtered dataset gave 142]")
log("FINALIZE_DONE")
