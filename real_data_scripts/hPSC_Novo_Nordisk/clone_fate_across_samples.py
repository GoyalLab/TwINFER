"""Fate concordance of clone members ACROSS samples (clone.id is shared between samples), observed vs a clone-label shuffle.
For two samples X,Y: all pairs (cell in X, cell in Y) with the same clone.id; share with the same clust_annot_02 label; null = clone labels shuffled among the cells of X and Y jointly
(sizes kept per sample... labels permuted within each sample), 2,000 shuffles. Also: share with the same 'coarse' fate for DE samples (DE vs mesoderm)."""
import numpy as np, pandas as pd
EXP="/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/export"; rng=np.random.default_rng(0); NP=2000
o=pd.read_csv(f"{EXP}/obs.csv",index_col=0,low_memory=False); o=o[o["clone.id"].notna()].copy(); o["clone"]=o["clone.id"].astype(int)
def cross_share(cx,tx,cy,ty):
    # pairs with same clone; returns same-type count and total, via per-clone type counts
    dx=pd.crosstab(cx,tx); dy=pd.crosstab(cy,ty); cl=dx.index.intersection(dy.index)
    if len(cl)==0: return 0,0,0
    A=dx.loc[cl]; B=dy.loc[cl].reindex(columns=A.columns,fill_value=0); A=A.reindex(columns=B.columns,fill_value=0)
    tot=(A.sum(1)*B.sum(1)).sum(); same=(A*B).sum().sum(); return same,tot,len(cl)
def run(sx,sy,label):
    X=o[o.SampleID.astype(str)==sx]; Y=o[o.SampleID.astype(str)==sy]
    same,tot,ncl=cross_share(X.clone.values,X.clust_annot_02.values,Y.clone.values,Y.clust_annot_02.values)
    if tot==0: print(sx,sy,"no shared clones"); return None
    null=[]
    for _ in range(NP):
        cx=rng.permutation(X.clone.values); cy=rng.permutation(Y.clone.values)
        s,t,_=cross_share(cx,X.clust_annot_02.values,cy,Y.clust_annot_02.values); null.append(s/t if t else np.nan)
    null=np.array(null); obs=same/tot
    return dict(comparison=label,shared_clones=ncl,cross_pairs=int(tot),observed_same_type=obs,chance=np.nanmean(null),null_sd=np.nanstd(null),enrichment=obs/np.nanmean(null),z=(obs-np.nanmean(null))/np.nanstd(null),p=(1+(null>=obs).sum())/(NP+1))
rows=[run("DE_same","DE_sis","DE_same x DE_sis (sister cultures, DE stage)"),run("hESCs_p3_same","hESCs_p3_sis","hESCs_p3_same x hESCs_p3_sis (sister cultures, hESC)"),
      run("PS_sis","DE_sis","PS_sis x DE_sis (earlier stage -> later, sister line)"),run("hESCs_p2_parent","DE_same","hESCs_p2_parent x DE_same"),run("hESCs_p2_parent","DE_sis","hESCs_p2_parent x DE_sis")]
T=pd.DataFrame([r for r in rows if r]); pd.set_option("display.width",250); print(T.round(4).to_string(index=False)); T.to_csv("/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/clone_fate_across_samples.csv",index=False)
# clone-level DE fate fractions across the two DE sisters (clones with >=3 cells in each)
de=o[o.clust_annot_02.isin(["Definitive endoderm","Early mesoderm"])]
f=lambda s: de[de.SampleID.astype(str)==s].groupby("clone").clust_annot_02.agg(n="size",fde=lambda x:(x=="Definitive endoderm").mean())
a,b=f("DE_same"),f("DE_sis"); j=a.join(b,lsuffix="_same",rsuffix="_sis",how="inner"); print("\nclones present in both DE cultures:",len(j))
for k in (1,2,3):
    jj=j[(j.n_same>=k)&(j.n_sis>=k)]
    if len(jj)>5:
        r=np.corrcoef(jj.fde_same,jj.fde_sis)[0,1]; nl=[np.corrcoef(jj.fde_same,rng.permutation(jj.fde_sis.values))[0,1] for _ in range(2000)]
        print(f"  clones with >={k} DE/meso cells in each culture: n={len(jj)}, correlation of DE fraction between sisters r={r:.3f} (shuffle sd {np.std(nl):.3f}, p={(1+(np.array(nl)>=r).sum())/2001:.4f})")
