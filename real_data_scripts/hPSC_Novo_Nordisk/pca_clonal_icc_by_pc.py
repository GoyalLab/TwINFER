"""Where does the clonal signal sit in PCA space? Per sample (per-sample log1p(CP10k) PCA): intraclass-style correlation per PC, 1 - E[(x_a - x_b)^2]_clone-pairs / (2 var(x)),
and the clone-vs-random AUC (equal pair numbers) using only the first k PCs. unweighted pairs. Writes CSVs to analysis_data/hPSC_Novo_Nordisk/pca_clonal_distance/."""
import numpy as np, pandas as pd
EXP="/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/export"; OUT="/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/pca_clonal_distance"
rng=np.random.default_rng(0); obs=pd.read_csv(f"{EXP}/obs.csv",index_col=0,low_memory=False)
def pairs_within(l):
    od=np.argsort(l,kind="stable"); ls=l[od]; st=np.flatnonzero(np.r_[True,ls[1:]!=ls[:-1]]); en=np.r_[st[1:],len(ls)]; A,B=[],[]
    for s,e in zip(st,en):
        if e-s>=2: i,j=np.triu_indices(e-s,1); A.append(od[s+i]); B.append(od[s+j])
    return np.concatenate(A),np.concatenate(B)
def auc_k(Z,a,b,k,clone):
    N=len(a); cd=np.linalg.norm(Z[a,:k]-Z[b,:k],axis=1)
    ra=rng.integers(0,len(Z),3*N); rb=rng.integers(0,len(Z),3*N); ok=clone[ra]!=clone[rb]; rd=np.linalg.norm(Z[ra[ok][:N],:k]-Z[rb[ok][:N],:k],axis=1)
    rs=np.sort(rd); gt=len(rs)-np.searchsorted(rs,cd,side="right"); eq=np.searchsorted(rs,cd,side="right")-np.searchsorted(rs,cd,side="left"); return float(((gt+0.5*eq)/len(rs)).mean())
rows_pc,rows_k=[],[]
for s in ["hESCs_p2_parent","hESCs_p3_same","hESCs_p3_sis","PS_sis","DE_same","DE_sis"]:
    m=(obs.SampleID.astype(str)==s)&obs["clone.id"].notna(); clone=obs.loc[m,"clone.id"].astype(int).to_numpy()
    Z=pd.read_csv(f"{EXP}/pca_log1p_cp10k_hvg5000_{s}.csv",index_col=0).reindex(obs.index[m]).to_numpy(np.float64)
    a,b=pairs_within(clone); var=Z.var(0,ddof=1); icc=1-((Z[a]-Z[b])**2).mean(0)/(2*var)
    ve=pd.read_csv(f"{EXP}/pca_log1p_cp10k_hvg5000_{s}_variance.csv").explained_variance_ratio.to_numpy()
    for lo,hi,nm in ((0,1,"PC1"),(1,5,"PC2-5"),(5,20,"PC6-20"),(20,50,"PC21-50")):
        rows_pc.append(dict(sample=s,pcs=nm,mean_ICC=icc[lo:hi].mean(),variance_share_of_50PCs=ve[lo:hi].sum()/ve.sum()))
    rows_k.append(dict(sample=s,**{f"AUC_first_{k}_PCs":auc_k(Z,a,b,k,clone) for k in (2,5,10,20,50)}))
    d2c=((Z[a]-Z[b])**2).sum(1).mean(); d2r=2*var.sum(); rows_k[-1]["ICC_all_50_PCs"]=1-d2c/d2r
P=pd.DataFrame(rows_pc); K=pd.DataFrame(rows_k); P.to_csv(f"{OUT}/icc_by_pc_group.csv",index=False); K.to_csv(f"{OUT}/auc_by_number_of_pcs.csv",index=False)
pd.set_option("display.width",220)
print(P.pivot(index="sample",columns="pcs",values="mean_ICC")[["PC1","PC2-5","PC6-20","PC21-50"]].round(3).loc[K["sample"]].to_string())
print("\nshare of the 50-PC variance:"); print(P.pivot(index="sample",columns="pcs",values="variance_share_of_50PCs")[["PC1","PC2-5","PC6-20","PC21-50"]].round(2).loc[K["sample"]].to_string())
print("\nclone-vs-random AUC using only the first k PCs, and overall ICC:"); print(K.round(3).to_string(index=False))
