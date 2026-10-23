import anndata as ad, numpy as np, pandas as pd, scanpy as sc
from scipy.sparse import issparse
R="/gpfs/projects/b1255/hzhang/TwINFER_KA/"
a=ad.read_h5ad(R+"finalized_data/hPSC_undifferentiated_data/hPSC_undifferentiated.h5ad",backed="r")
tfs=[l.strip() for l in open(R+"analysis_data/hPSC_20260927/gene_modules/all_tfs.txt") if l.strip()]
tfs=[g for g in dict.fromkeys(tfs) if g in a.var_names]
print(len(tfs),"TFs present;",a.uns["normalization"])
b=a[:,tfs].to_memory(); b.obs["tp"]=b.obs["timepoint"].astype(str)
mean_expr=b.to_df().groupby(b.obs["tp"].values).mean()
pct=(b.to_df()>0).groupby(b.obs["tp"].values).mean()
out=R+"analysis_data/hPSC_undifferentiated/de_tfs/"
for g1,g2 in [("T0","T10"),("T10","T15"),("T0","T15")]:
    s=b[b.obs.tp.isin([g1,g2])].copy()
    sc.tl.rank_genes_groups(s,"tp",groups=[g2],reference=g1,method="wilcoxon",use_raw=False,n_genes=len(tfs),tie_correct=True)
    r=sc.get.rank_genes_groups_df(s,group=g2).rename(columns={"names":"gene","scores":"wilcoxon_z","pvals":"pval","pvals_adj":"FDR"})
    # log2FC from means of log1pPF is not exact; report mean difference in log1p units + fold of expm1 means
    m1=np.expm1(b[b.obs.tp==g1].X).mean(0); m2=np.expm1(b[b.obs.tp==g2].X).mean(0)
    m1=np.asarray(m1).ravel(); m2=np.asarray(m2).ravel()
    fc=pd.Series(np.log2((m2+1e-3)/(m1+1e-3)),index=tfs)
    r["log2FC_"+g2+"_vs_"+g1]=fc.reindex(r.gene).values
    r["mean_"+g1]=mean_expr.loc[g1,r.gene].values; r["mean_"+g2]=mean_expr.loc[g2,r.gene].values
    r["pct_"+g1]=pct.loc[g1,r.gene].values; r["pct_"+g2]=pct.loc[g2,r.gene].values
    r=r.drop(columns=["logfoldchanges"],errors="ignore").sort_values("wilcoxon_z",key=abs,ascending=False)
    r.to_csv(out+f"DE_TFs_{g1}_to_{g2}.csv",index=False)
    sig=r[(r.FDR<0.05)&(r["log2FC_"+g2+"_vs_"+g1].abs()>0.5)]
    print(f"\n== {g1}->{g2}: {len(sig)} TFs FDR<0.05 & |log2FC|>0.5 (up {int((sig['log2FC_'+g2+'_vs_'+g1]>0).sum())}, down {int((sig['log2FC_'+g2+'_vs_'+g1]<0).sum())})")
    c=["gene","wilcoxon_z","log2FC_"+g2+"_vs_"+g1,"FDR","pct_"+g1,"pct_"+g2]
    print("UP\n",sig[sig["log2FC_"+g2+"_vs_"+g1]>0].head(15)[c].round(3).to_string(index=False))
    print("DOWN\n",sig[sig["log2FC_"+g2+"_vs_"+g1]<0].head(15)[c].round(3).to_string(index=False))
