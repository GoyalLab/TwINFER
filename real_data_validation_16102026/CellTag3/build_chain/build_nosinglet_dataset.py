"""Build the CellTag dataset WITHOUT singletCode filtering, so every method can be run with and without it.

Cells come straight from the CellRanger filtered matrices; CellTags come from the raw all_samples.csv with no
singlet/multiplet filtering. Day-2 has one lane so its barcodes map directly; day-5 barcodes are resolved to one
of 8 lanes by CellRanger membership and dropped when they match 0 or >1 lane (~3%).

Also reports, per method, how many twin pairs result -- needed to judge whether TwINFER is tractable at this scale.
"""
import os, itertools, numpy as np, pandas as pd, scanpy as sc, anndata as ad, warnings
from collections import defaultdict
warnings.filterwarnings("ignore")
SRC="/gpfs/projects/p32655/yscher/celltag_3_geo_data/"
HERE=os.path.dirname(os.path.abspath(__file__))
OUT=os.path.join(HERE,"..","datasets", "celltag", "celltag_nosinglet_counts.h5ad")
FILES={"s1":"GSM6681128_d5_1","s2":"GSM6681129_d5_2","s3":"GSM6681130_d5_3","s4":"GSM6681131_d5_4",
       "s5":"GSM6681132_d5_5","s6":"GSM6681133_d5_6","s7":"GSM6681134_d5_7","s8":"GSM6681135_d5_8",
       "":"GSM6681127_d2_5"}
def log(m): print(m,flush=True)

mats={}
for lane,stem in FILES.items():
    A=sc.read_10x_h5(SRC+stem+"_filtered_feature_bc_matrix.h5")
    # align on gene_ids: CellRanger lanes do NOT share column order, and var_names_make_unique can
    # suffix differently per lane. An assert caught this; aligning by stable ID is the fix.
    A.var['gid']=A.var['gene_ids'].astype(str) if 'gene_ids' in A.var else pd.Index(A.var_names).astype(str)
    A.obs_names=pd.Index(A.obs_names).str.replace(r"-\d+$","",regex=True)
    mats[lane]=A; log(f"  lane '{lane}': {A.shape}")

raw=pd.read_csv(SRC+"all_samples.csv")
raw=raw[raw['sample'].isin(['d2-RNA-5','d5-RNA-1','d5-RNA-2'])]
# PAPER'S RULE (Jindal et al. Methods): "binarized at a threshold of >1 UMI count per cell".
# Each row of all_samples.csv is one UMI observation, so a (cell,tag) pair needs >1 row to count.
# Skipping this keeps single-UMI noise: median 10 tags/cell instead of 3, which would fabricate
# Jaccard matches. Verified 2026-07-18.
_n=raw.groupby(['sample','cellID','barcode']).size()
_n=_n[_n>1].reset_index()
log(f"(cell,tag) pairs kept at >1 UMI: {len(_n)}")
tagsets=_n.groupby(['sample','cellID'])['barcode'].apply(lambda x: frozenset(x))
log(f"raw tagged cells: {len(tagsets)}")

d5=["s1","s2","s3","s4","s5","s6","s7","s8"]
sets={L:set(mats[L].obs_names) for L in mats}
rows=[]
for (samp,cid),tg in tagsets.items():
    if samp=='d2-RNA-5':
        if cid in sets[""]: rows.append(("",cid,'day2',tg))
    else:
        hit=[L for L in d5 if cid in sets[L]]
        if len(hit)==1: rows.append((hit[0],cid,'day5',tg))
log(f"resolved cells: {len(rows)}")

lanes=[r[0] for r in rows]; cids=[r[1] for r in rows]
names=[f"{l}:{c}" if l else c for l,c in zip(lanes,cids)]
import scipy.sparse as sp
blocks=[]; order=defaultdict(list)
for i,(l,c,_,_) in enumerate(rows): order[l].append((i,c))
# common gene axis = gene_ids present in EVERY lane, ordered by the day-2 lane
common=None
for l,A in mats.items():
    g=pd.Index(A.var['gid']); common=g if common is None else common.intersection(g)
ref=mats[""]; refg=pd.Index(ref.var['gid'])
common=[g for g in refg if g in set(common)]
log(f"common genes across all lanes: {len(common)}")
colpos={l:{g:k for k,g in enumerate(pd.Index(mats[l].var['gid']))} for l in mats}
X=sp.lil_matrix((len(rows), len(common)), dtype=np.float32)
for l,items in order.items():
    A=mats[l]; cp=colpos[l]
    take=np.array([cp[g] for g in common])          # reorder this lane's columns to the common axis
    idx={b:k for k,b in enumerate(A.obs_names)}
    ri=[idx[c] for _,c in items]; ti=[i for i,_ in items]
    X[ti,:]=A.X[ri,:][:,take]
X=X.tocsr()
var_df=ref.var.set_index('gid').loc[common].copy()
var_df.index=pd.Index([ref.var_names[colpos[""][g]] for g in common])
obs=pd.DataFrame({'lane':lanes,'cellID':cids,'timepoint':[r[2] for r in rows],
                  'lineage_barcodes':[",".join(sorted(r[3])) for r in rows],
                  'ntags':[len(r[3]) for r in rows]}, index=pd.Index(names))
B=ad.AnnData(X=X, obs=obs, var=var_df)
B.write_h5ad(OUT)
log(f"wrote {OUT} {B.shape}")
log(f"timepoints: {dict(obs.timepoint.value_counts())}")
log(f"ntags: median={int(obs.ntags.median())} mean={obs.ntags.mean():.2f} max={int(obs.ntags.max())} | >=2 tags={int((obs.ntags>=2).sum())}")

# ---- pair counts per method, to judge tractability ----
tags=pd.Series([frozenset(s.split(',')) for s in obs['lineage_barcodes']], index=obs.index)
def jpairs(min_t,max_t,cap=60_000_000):
    el=tags[(obs.ntags>=min_t)&(obs.ntags<=max_t)]
    inv=defaultdict(list)
    for x in el.index:
        for t in el[x]: inv[t].append(x)
    n=0; seen=set()
    for t,g in inv.items():
        if len(g)>3000: log(f"    (tag shared by {len(g)} cells)")
        for a,b in itertools.combinations(g,2):
            k=(a,b) if a<b else (b,a)
            if k in seen: continue
            seen.add(k)
            ta,tb=el[k[0]],el[k[1]]
            if len(ta&tb)/len(ta|tb)>=0.6: n+=1
            if len(seen)>cap: log("    CAP hit"); return n,len(seen)
    return n,len(seen)
n,c=jpairs(2,25); log(f"barcode_overlap(2-25 tags) pairs = {n} (candidates examined {c})")
ex=obs['lineage_barcodes']; vc=ex.value_counts()
m3=int(sum(v*(v-1)//2 for v in vc[vc>=2]))
log(f"exact_tag_match pairs = {m3}")
log(f"tags/cell after >1 UMI: median={int(obs.ntags.median())} mean={obs.ntags.mean():.2f} (paper mean 4.2)")
log(f"cells with 2-25 tags: {int(((obs.ntags>=2)&(obs.ntags<=25)).sum())}")
log("BUILD_NOSINGLET_DONE")
