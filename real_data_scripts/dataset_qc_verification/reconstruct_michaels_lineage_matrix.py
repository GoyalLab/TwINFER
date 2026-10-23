import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, scipy.sparse as sp, gzip
from qc_common import read_h5ad_obs_var, read_h5ad_csr
SP="/gpfs/home/gzu5140/.claude-tmp/claude-2000104/-gpfs-projects-b1255-hzhang-TwINFER-KA/aceb1544-650b-49b8-928f-86e7e02952c7/scratchpad"
R="/gpfs/projects/b1255/yscher/Transcriptomic Distance/datasets/michaels2024/raw"
f="/home/gzu5140/TwINFER_KA/temp/T-lineage_T1_T2_combined_singlets (2).h5ad"
o,v=read_h5ad_obs_var(f); X=read_h5ad_csr(f,"X").tocsr()
def stream(g,keys):
    bc=pd.read_csv(f"{R}/{g}_barcodes.tsv.gz",header=None,sep="\t")[0].str.replace("-1$","",regex=True)
    feat=pd.read_csv(f"{R}/{g}_features.tsv.gz",header=None,sep="\t")
    pos=pd.Series(np.arange(len(bc)),index=bc.to_numpy()).reindex(keys)
    assert pos.notna().all(), "cells missing in raw"
    want=np.full(len(bc)+1,-1,dtype=np.int64); want[pos.to_numpy().astype(int)+1]=np.arange(len(keys))
    ng=len(feat); rr=[];cc=[];vv=[]; tot=np.zeros(len(keys))
    for ch in pd.read_csv(f"{R}/{g}_matrix.mtx.gz",sep=" ",header=None,skiprows=3,chunksize=5_000_000,dtype=np.int64):
        m=want[ch[1].to_numpy()]; k=m>=0
        rr.append(m[k]); cc.append(ch[0].to_numpy()[k]-1); vv.append(ch[2].to_numpy()[k])
    r=np.concatenate(rr); c=np.concatenate(cc); x=np.concatenate(vv)
    return sp.csr_matrix((x.astype(np.float32),(r,c)),shape=(len(keys),ng)), feat
parts=[];ids=None
for s,g in (("T1","GSM7916614_T1"),("T2","GSM7916615_T2")):
    idx=[i for i in o.index if o.loc[i,"batch"]==s]; keys=[i.rsplit("_",1)[0] for i in idx]
    M,feat=stream(g,keys); print(s,"streamed",M.shape,flush=True); parts.append((idx,M)); ids=feat[0].to_numpy()
rows=sum([p[0] for p in parts],[]); Mall=sp.vstack([p[1] for p in parts]).tocsr()
order=pd.Series(np.arange(len(rows)),index=rows).reindex(o.index).to_numpy(); Mall=Mall[order]
tot_all=np.asarray(Mall.sum(1)).ravel()
gp=pd.Series(np.arange(len(ids)),index=ids).reindex(v.gene_ids.to_numpy())
print("lineage genes unmapped to raw ids:",gp.isna().sum(),flush=True)
S=Mall[:,gp.to_numpy().astype(int)].tocsr(); tg=np.asarray(S.sum(1)).ravel()
sp.save_npz(f"{SP}/rec_counts.npz",S)
print("stored total_counts vs kept-gene sum maxdiff",np.abs(o.total_counts.to_numpy()-tg).max(),"; vs all-gene sum",np.abs(o.total_counts.to_numpy()-tot_all).max())
print("stored n_genes vs recomputed",np.abs(o.n_genes.to_numpy()-np.asarray((S>0).sum(1)).ravel()).max())
print("var n_cells vs recomputed maxdiff",np.abs(v.n_cells.to_numpy()-np.asarray((S>0).sum(0)).ravel()).max())
print("nonzero pattern differences vs X:",abs((X>0).astype(int)-(S>0).astype(int)).sum())
for k,tot in (("CP10k kept genes",tg),("CP10k all genes",tot_all)):
    A=S.multiply(1e4/tot[:,None]).tocsr(); A.data=np.log1p(A.data); print(k,"log1p max|diff| vs X:",abs(A-X).max())
print("X rowsum[:3]",X[:3].sum(1).A1,"X row0 max",X[0].data.max())
