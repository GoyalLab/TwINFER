"""Compute every clone-calling method on the CORRECTED (no-singletCode, >1 UMI) dataset.

Jaccard is computed exactly by sparse algebra: intersection = M @ M.T on the binary cell x tag matrix,
union = |A| + |B| - intersection. Only co-occurring pairs are materialized, which is exact (a pair sharing
no tag has Jaccard 0 < 0.6 and cannot qualify).

Methods, matching the definitions used throughout:
  jaccard_overlap   2-25 tags, Jaccard>=0.6, CONNECTED COMPONENTS      (the published rule)
  no_tag_filter  >=1 tag,  Jaccard>=0.6, connected components
  exact_tag_match     identical tag sets
  barcode_overlap_2to25tags     2-25 tags, Jaccard>=0.6 pairs used DIRECTLY (no transitivity)
  barcode_overlap_min1tag   >=1 tag,  Jaccard>=0.6 pairs used directly
  master_clone_table    lookup in the authors' published clone table (lane-aware)
Cell-type labels are taken from the authors' master table (cell_type2), which covers far more cells than the
Seurat annotation available for the old filtered subset.
"""
import os, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
HERE=os.path.dirname(os.path.abspath(__file__))
# CT_SOURCE / TAG_MIN / TAG_MAX / PAIR_SUFFIX exist so the same pairing rule can be re-derived on a
# quality-controlled cell set without touching the artifacts already on disk. Unset, every value below
# is what it was before they existed, so a default run is unchanged. PAIR_SUFFIX is what keeps the
# outputs distinct: without it a rebuild on a different cell set would silently overwrite the old
# pair lists under the same names.
DS=os.environ.get("CT_SOURCE", os.path.join(HERE,"..","datasets", "celltag", "celltag_nosinglet_counts.h5ad"))
sys.path.insert(0, os.path.join(HERE, ".."))      # fig5_paths.py lives in preprocessing_code/
from fig5_paths import CELLTAG_GEO      # noqa: E402
MASTER=os.path.join(CELLTAG_GEO, "hsc.rna&atac.r1&2_master_v2.csv")      # the authors' master clone table (raw_data link)
SUFFIX=os.environ.get("PAIR_SUFFIX","")
OUT=os.path.join(HERE,"..","exports","legacy_reproductions", "pairing", "celltag_clones",f"clone_assignments_corrected{SUFFIX}.csv")
TAG_MIN=int(os.environ.get("TAG_MIN","2")); TAG_MAX=int(os.environ.get("TAG_MAX","25"))
assert 1<=TAG_MIN<=TAG_MAX, f"tag bounds out of order: {TAG_MIN}..{TAG_MAX}"
assert os.path.exists(DS), f"missing input: {DS}"
THR=0.6
def log(m): print(m,flush=True)

A=ad.read_h5ad(DS, backed='r')
obs=A.obs.copy()
tags=[sorted(set(s.split(','))) for s in obs['lineage_barcodes']]
nt=np.array([len(t) for t in tags])
log(f"cells={len(obs)} | ntags median={int(np.median(nt))} mean={nt.mean():.2f}")

def jac_pairs(idx):
    """exact Jaccard>=THR pairs among cells idx, via sparse matmul"""
    sub=[tags[i] for i in idx]
    vocab={t:j for j,t in enumerate(sorted({t for s in sub for t in s}))}
    r=[];c=[]
    for i,s in enumerate(sub):
        for t in s: r.append(i); c.append(vocab[t])
    M=sp.csr_matrix((np.ones(len(r),dtype=np.int32),(r,c)),shape=(len(sub),len(vocab)))
    inter=(M@M.T).tocoo(); sizes=np.asarray(M.sum(1)).ravel()
    m=inter.row<inter.col
    rr,cc,vv=inter.row[m],inter.col[m],inter.data[m]
    j=vv/(sizes[rr]+sizes[cc]-vv)
    k=j>=THR
    return rr[k],cc[k],len(sub)

def components_labels(idx,prefix):
    rr,cc,n=jac_pairs(idx)
    g=sp.coo_matrix((np.ones(len(rr)),(rr,cc)),shape=(n,n))
    ncomp,lab=connected_components(g,directed=False)
    sizes=np.bincount(lab)
    out=pd.Series(index=obs.index[idx],dtype=object)
    keep=sizes[lab]>=2
    out[keep]=[f"{prefix}{l}" for l in lab[keep]]
    log(f"  {prefix}: pairs={len(rr)} clones(>=2)={int((sizes>=2).sum())} cells={int(keep.sum())}")
    return out

def direct_labels(idx,prefix):
    """barcode_overlap: each qualifying pair is its own 2-cell unit; a cell may appear in many pairs, so we store the
    PAIR LIST separately and additionally emit a representative label for schema compatibility."""
    rr,cc,n=jac_pairs(idx)
    names=obs.index[idx]
    pairs=pd.DataFrame({'cell_a':names[rr],'cell_b':names[cc]})
    fname=f"pairs_{prefix}{SUFFIX}.csv"
    pairs.to_csv(os.path.join(HERE,"..","exports","legacy_reproductions", "pairing", "celltag_clones",fname),index=False)
    log(f"  {prefix}: pairs={len(rr)} (written to {fname})")
    return pairs

idx_all=np.arange(len(obs))
idx_225=np.where((nt>=TAG_MIN)&(nt<=TAG_MAX))[0]
log(f"source: {DS}")
log(f"eligible {TAG_MIN}-{TAG_MAX} tags: {len(idx_225)} | >=1 tag: {len(idx_all)}")
# The published clone-calling code keeps cells with strictly more than 1 and strictly fewer than 25
# tags (2..24); the paper text says 2-25. Whether that distinction can matter here is a property of
# THIS cell set, so it is measured rather than argued: report the count at the upper bound.
log(f"cells at exactly {TAG_MAX} tags: {int((nt==TAG_MAX).sum())} | above {TAG_MAX}: {int((nt>TAG_MAX).sum())}")

log("connected-component methods:")
obs['jaccard_overlap']=components_labels(idx_225,"J")
obs['no_tag_filter']=components_labels(idx_all,"NF")
log("exact:")
ex=pd.Series(obs['lineage_barcodes'].values,index=obs.index)
vc=ex.value_counts()
obs['exact_tag_match']=np.where(ex.map(vc)>=2,"E"+ex.astype(str),None)
log(f"  exact_tag_match: cells={int(obs['exact_tag_match'].notna().sum())} clones={obs['exact_tag_match'].nunique()}")
log("direct-pair methods (barcode_overlap):")
# The name carries the bounds actually used. At the defaults it is the existing literal, so default
# runs write the same filenames as before; under other bounds the file cannot claim a rule it did
# not apply.
PREFIX_225=("barcode_overlap_2to25tags" if (TAG_MIN,TAG_MAX)==(2,25)
            else f"barcode_overlap_{TAG_MIN}to{TAG_MAX}tags")
direct_labels(idx_225,PREFIX_225)
direct_labels(idx_all,"barcode_overlap_min1tag")

# ---- master_clone_table: authors' published clone table, lane-aware ----
m=pd.read_csv(MASTER)
sp2=m['cell.bc'].str.rsplit('-',n=2,expand=True); m['sa'],m['bc'],m['suf']=sp2[0],sp2[1],sp2[2]
s2=m['sa'].str.rsplit('-',n=1,expand=True); m['msample'],m['assay']=s2[0],s2[1]
m=m[m['assay']=='rna']
lm={('d2_5','1'):'',('d5r1','1'):'s1',('d5r1','2'):'s2',('d5r1','3'):'s3',('d5r1','4'):'s4',
    ('d5r2','1'):'s5',('d5r2','2'):'s6',('d5r2','3'):'s7',('d5r2','4'):'s8'}
m['lane']=[lm.get((x,y)) for x,y in zip(m.msample,m.suf)]
k2c={(l,b):c for l,b,c in zip(m['lane'],m['bc'].astype(str),m['clone.id'])}
k2t={(l,b):c for l,b,c in zip(m['lane'],m['bc'].astype(str),m['cell_type2'])}
lane=obs['lane'].fillna('').values; cid=obs['cellID'].astype(str).values
obs['master_clone_table']=[k2c.get((l,c)) for l,c in zip(lane,cid)]
obs['cell_type2']=[k2t.get((l,c)) for l,c in zip(lane,cid)]
log(f"  master_clone_table: cells={int(obs['master_clone_table'].notna().sum())} clones={obs['master_clone_table'].nunique()}")
log(f"  cell_type2 annotation coverage: {int(obs['cell_type2'].notna().sum())}/{len(obs)}")

obs.to_csv(OUT)
log(f"wrote {OUT}")
log("BUILD_CORRECTED_CLONES_DONE")
