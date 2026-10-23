import anndata as ad, numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
R="/gpfs/projects/b1255/hzhang/TwINFER_KA/"; O=R+"analysis_data/hPSC_undifferentiated/clone_de/"
a=ad.read_h5ad(R+"finalized_data/hPSC_undifferentiated_data/hPSC_undifferentiated.h5ad")
tfs=set(l.strip() for l in open(R+"analysis_data/hPSC_20260927/gene_modules/all_tfs.txt"))
a=a[a.obs.clone_id.notna()].copy()
X=a.X.tocsr() if hasattr(a.X,"tocsr") else a.X
obs=a.obs[["timepoint","clone_id"]].copy(); obs["i"]=np.arange(len(obs))
def pb(idx_groups):
    # mean log1pPF per clone x timepoint (rows) -> dense
    return np.vstack([np.asarray(X[ix].mean(0)).ravel() for ix in idx_groups])
for g1,g2 in [("T0","T15"),("T0","T10"),("T10","T15")]:
    s=obs[obs.timepoint.isin([g1,g2])]
    ct=pd.crosstab(s.clone_id,s.timepoint); cl=ct.index[(ct[g1]>0)&(ct[g2]>0)]
    s=s[s.clone_id.isin(cl)]
    grp=s.groupby(["clone_id","timepoint"],observed=True).i.apply(list)
    M1=pb([grp[(c,g1)] for c in cl]); M2=pb([grp[(c,g2)] for c in cl])
    n1=s[s.timepoint==g1].shape[0]; n2=s[s.timepoint==g2].shape[0]
    # % cells expressing
    Xs=X[s.i.values]; t=s.timepoint.values
    p1=np.asarray((Xs[t==g1]>0).mean(0)).ravel(); p2=np.asarray((Xs[t==g2]>0).mean(0)).ravel()
    keep=np.maximum(p1,p2)>=0.05
    d=(M2-M1)[:,keep]
    tt=stats.ttest_1samp(d,0,axis=0); 
    w=np.array([stats.wilcoxon(d[:,j],zero_method="zsplit").pvalue if np.any(d[:,j]!=0) else 1.0 for j in range(d.shape[1])])
    genes=a.var_names[keep]
    m1=np.expm1(M1[:,keep]).mean(0); m2=np.expm1(M2[:,keep]).mean(0)
    r=pd.DataFrame({"gene":genes,"is_TF":genes.isin(tfs),"log2FC":np.log2((m2+1e-3)/(m1+1e-3)),"mean_diff_log1p":d.mean(0),
        "t":tt.statistic,"p_t":tt.pvalue,"p_wilcoxon":w,"pct_"+g1:p1[keep],"pct_"+g2:p2[keep]})
    r["FDR_t"]=multipletests(r.p_t.fillna(1),method="fdr_bh")[1]; r["FDR_wilcoxon"]=multipletests(r.p_wilcoxon,method="fdr_bh")[1]
    r["FDR_t_TFonly"]=np.nan; mt=r.is_TF.values
    r.loc[mt,"FDR_t_TFonly"]=multipletests(r.loc[mt,"p_t"].fillna(1),method="fdr_bh")[1]
    r=r.sort_values("t",key=abs,ascending=False); r.to_csv(O+f"clonepaired_DE_{g1}_to_{g2}.csv",index=False)
    print(f"\n== {g1}->{g2}: {len(cl)} shared clones, {n1}/{n2} cells, {keep.sum()} genes tested, {mt.sum()} TFs")
    sg=r[(r.FDR_t<0.05)&(r.log2FC.abs()>0.25)]; print(" all genes FDR<0.05 & |log2FC|>0.25:",len(sg),"; TFs:",int(sg.is_TF.sum()))
    st=r[r.is_TF&(r.FDR_t_TFonly<0.05)&(r.log2FC.abs()>0.25)]
    print(" TFs (TF-only FDR<0.05,|log2FC|>0.25):",len(st))
    print(st[["gene","log2FC","t","FDR_t_TFonly","FDR_wilcoxon","pct_"+g1,"pct_"+g2]].round(3).head(30).to_string(index=False))
    if (g1,g2)==("T0","T15"):
        u=pd.read_csv(R+"analysis_data/hPSC_undifferentiated/de_tfs/DE_TFs_T0_to_T15.csv").set_index("gene")
        j=r[r.is_TF].set_index("gene").join(u[["log2FC_T15_vs_T0"]],how="inner")
        print(" corr of TF log2FC paired vs all-cell:",round(j.log2FC.corr(j.log2FC_T15_vs_T0),3),"n=",len(j))
        prev=pd.read_csv(R+"analysis_data/hPSC_undifferentiated/de_tfs/DE_TFs_T0_to_T15_min5pct.csv")
        print(" prev 22 TFs in clone-paired result:"); 
        q=r.set_index("gene").loc[[g for g in prev.gene if g in set(r.gene)],["log2FC","t","FDR_t_TFonly","pct_T0","pct_T15"]]
        q["prev_log2FC"]=prev.set_index("gene").loc[q.index,"log2FC_T15_vs_T0"]; print(q.round(3).to_string())
