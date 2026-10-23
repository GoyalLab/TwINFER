"""Fig 2A/2B variant: Pearson correlation of cell pairs, same clone vs different clone, for EVERY sample (paper shows hESCs p2 (d = 3.51) and p3 (Fig 2B)).
four inputs as in reproduce_paper_fig2.py: SCT data on 3,000 variable genes; SCT data all genes; log1p(CP10k) of raw counts all genes; stored SCT PCA (50 PCs)."""
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
EXP="/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/export"; OUT="/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/paper_reproduction"; rng=np.random.default_rng(0)
obs=pd.read_csv(f"{EXP}/obs.csv",index_col=0,low_memory=False)
def load_dense(prefix, cols, nr):
    p=np.fromfile(f"{EXP}/{prefix}_p.bin",dtype=np.float64).astype(np.int64); I=np.memmap(f"{EXP}/{prefix}_i.bin",dtype=np.int32,mode="r"); X=np.memmap(f"{EXP}/{prefix}_x.bin",dtype=np.float64,mode="r")
    D=np.zeros((len(cols),nr),np.float32)
    for s in range(0,len(cols),2000):
        c=cols[s:s+2000]; l=p[c+1]-p[c]; idx=np.repeat(p[c]-np.r_[0,np.cumsum(l)[:-1]],l)+np.arange(int(l.sum())); D[s+np.repeat(np.arange(len(c)),l),np.asarray(I[idx])]=np.asarray(X[idx])
    return D
def pearson_rows(D): D-=D.mean(1,keepdims=True); D/=np.sqrt(np.einsum("ij,ij->i",D,D)/D.shape[1])[:,None]; return (D@D.T)/D.shape[1]
def cohen_d(a,b): return (a.mean()-b.mean())/np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
def pairs_within(l):
    o=np.argsort(l,kind="stable"); ls=l[o]; st=np.flatnonzero(np.r_[True,ls[1:]!=ls[:-1]]); en=np.r_[st[1:],len(ls)]; A,B=[],[]
    for s,e in zip(st,en):
        if e-s>=2: i,j=np.triu_indices(e-s,1); A.append(o[s+i]); B.append(o[s+j])
    return np.concatenate(A),np.concatenate(B)
sct=pd.read_csv(f"{EXP}/SCT_counts_features.csv").feature.astype(str).values; var=set(pd.read_csv(f"{EXP}/SCT_variable_features.csv").feature.astype(str)); keep=np.array([g in var for g in sct])
nrna=int(open(f"{EXP}/RNA_counts_dim.txt").read().split()[0]); pcall=pd.read_csv(f"{EXP}/red_pca.csv",index_col=0).iloc[:,:50].to_numpy(np.float64)
rows=[]
for s in ["hESCs_p2_parent","hESCs_p3_same","hESCs_p3_sis","PS_sis","DE_same","DE_sis"]:
    m=((obs.SampleID.astype(str)==s)&obs["clone.id"].notna()).to_numpy(); cols=np.flatnonzero(m); clone=obs["clone.id"][m].astype(int).to_numpy()
    a,b=pairs_within(clone); ra=rng.integers(0,len(cols),600_000); rb=rng.integers(0,len(cols),600_000); ok=clone[ra]!=clone[rb]; ra,rb=ra[ok][:300_000],rb[ok][:300_000]
    def rep(name,R):
        w,bt=R[a,b],R[ra,rb]; rows.append(dict(sample=s,variant=name,mean_within=float(w.mean()),mean_between=float(bt.mean()),cohens_d=float(cohen_d(w,bt)),AUC=float(mannwhitneyu(w,bt,alternative="greater").statistic/(len(w)*len(bt))))); print(rows[-1],flush=True)
    D=load_dense("SCT_data",cols,len(sct)); R=pearson_rows(D[:,keep].copy()); rep("SCT data, 3,000 variable genes",R); del R
    R=pearson_rows(D); rep("SCT data, all genes",R); del R,D
    D=load_dense("RNA_counts",cols,nrna); tot=D.sum(1,keepdims=True); D*=(1e4/tot).astype(np.float32); np.log1p(D,out=D); R=pearson_rows(D); rep("log1p(CP10k) raw counts, all genes",R); del R,D
    rep("stored SCT PCA, 50 PCs",np.corrcoef(pcall[m]))
T=pd.DataFrame(rows); T.to_csv(f"{OUT}/figA_pearson_all_samples.csv",index=False); pd.set_option("display.width",220)
print("\nCohen's d by sample and input (paper: hESCs p2 d = 3.51)"); print(T.pivot(index="sample",columns="variant",values="cohens_d").round(3).reindex(["hESCs_p2_parent","hESCs_p3_same","hESCs_p3_sis","PS_sis","DE_same","DE_sis"]).to_string())
