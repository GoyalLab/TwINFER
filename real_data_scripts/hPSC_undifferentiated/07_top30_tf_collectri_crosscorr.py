# Top-30 most expressed TFs (mean log1pPF; >=5% of cells, >=20 CollecTRI targets expressed in >=5% of cells) vs their CollecTRI targets.
# Spearman across clones shared by two timepoints; TF->gene = TF earlier, gene later; gene->TF = reverse. Output: stress_tf_targets/top30_tf_collectri_crosscorr_wide.csv (+ top30_expressed_TFs.csv). Needs ~10 GB RAM.
import time, anndata as ad, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from scipy.stats import rankdata
t0=time.time(); P=lambda *x: print(f"[{time.time()-t0:.0f}s]",*x,flush=True)
R="/gpfs/projects/b1255/hzhang/TwINFER_KA/"; D=R+"analysis_data/hPSC_undifferentiated/"; O=D+"stress_tf_targets/"
tfs=[l.strip() for l in open(R+"analysis_data/hPSC_20260927/gene_modules/all_tfs.txt") if l.strip()]
c=pd.read_csv(R+"analysis_data/paper_analysis/larry_hematopoiesis_validation/resources/collectri_human.tsv",sep="\t",usecols=["source_genesymbol","target_genesymbol","is_stimulation","is_inhibition"])
c["sign"]=np.where(c.is_stimulation&~c.is_inhibition,"activation",np.where(c.is_inhibition&~c.is_stimulation,"inhibition","unknown"))
c=c.drop_duplicates(["source_genesymbol","target_genesymbol"])
A=ad.read_h5ad(R+"finalized_data/hPSC_undifferentiated_data/hPSC_undifferentiated.h5ad"); P("loaded",A.shape)
Xc=A.X.tocsr(); frac=pd.Series(np.bincount(Xc.indices,minlength=A.n_vars)/A.n_obs,index=A.var_names); mean=pd.Series(np.asarray(Xc.sum(0)).ravel()/A.n_obs,index=A.var_names); del Xc
ok=set(frac[frac>=0.05].index)
ntar=c[c.target_genesymbol.isin(ok)&(c.target_genesymbol!=c.source_genesymbol)].groupby("source_genesymbol").size()
cand=pd.DataFrame({"mean_expr":mean.reindex(tfs),"pct_cells":100*frac.reindex(tfs),"n_targets":ntar.reindex(tfs)}).dropna()
cand=cand[(cand.n_targets>=20)&(cand.pct_cells>=5)].sort_values("mean_expr",ascending=False)
top=cand.head(30); top.to_csv(O+"top30_expressed_TFs.csv"); print(top.round(3).to_string()); P("tfs chosen; TFs with >=20 targets:",len(cand))
sel=list(top.index); tg=c[c.source_genesymbol.isin(sel)&c.target_genesymbol.isin(ok)&(c.target_genesymbol!=c.source_genesymbol)]
genes=sorted(set(sel)|set(tg.target_genesymbol)); P("pairs",len(tg),"genes",len(genes))
B=A[A.obs.clone_id.notna(),genes].copy(); X=pd.DataFrame(B.X.toarray(),columns=genes); X["clone"]=B.obs.clone_id.astype(str).values; X["tp"]=B.obs.timepoint.astype(str).values; P("matrix",X.shape)
rk=lambda M:(lambda r:(r-r.mean(0))/np.where(r.std(0)==0,1,r.std(0)))(np.apply_along_axis(rankdata,0,M)); gi={x:i for i,x in enumerate(genes)}
out=tg.rename(columns={"source_genesymbol":"TF","target_genesymbol":"gene"})[["TF","gene","sign"]].reset_index(drop=True)
out.insert(1,"TF_mean_expr",out.TF.map(top.mean_expr).round(3)); jt=out.TF.map(gi).values; jg=out.gene.map(gi).values
for t1,t2 in [("T0","T15"),("T0","T10"),("T10","T15")]:
    m1=X[X.tp==t1].groupby("clone")[genes].mean(); m2=X[X.tp==t2].groupby("clone")[genes].mean(); cl=m1.index.intersection(m2.index)
    Ra,Rb=rk(m1.loc[cl].values),rk(m2.loc[cl].values); n=len(cl)
    out[f"{t1}_to_{t2}__TF_to_gene"]=((Ra[:,jt]*Rb[:,jg]).mean(0)).round(3); out[f"{t1}_to_{t2}__gene_to_TF"]=((Ra[:,jg]*Rb[:,jt]).mean(0)).round(3); P(t1,t2,n,"clones")
out["TF"]=pd.Categorical(out.TF,list(top.index)); out=out.sort_values(["TF","gene"]); out.to_csv(O+"top30_tf_collectri_crosscorr_wide.csv",index=False)
print(out.shape); print(out.head(6).to_string(index=False))
