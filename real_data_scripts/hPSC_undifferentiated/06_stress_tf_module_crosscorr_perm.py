# Stress TFs (JDP2,NFE2L2,ATF4 | CEBPB | DDIT3) vs all genes of the Hotspot module they sit in (3,4,17), both directions, across shared clones; null = NPERM scrambles of clone pairing.
# env: NPERM (default 10000), STRESS_OUT (output dir). Needs hotspot/gene_modules.csv (script 04).
import os, anndata as ad, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests
R="/gpfs/projects/b1255/hzhang/TwINFER_KA/"; D=R+"analysis_data/hPSC_undifferentiated/"; O=os.environ.get("STRESS_OUT",D+"stress_tf_targets/")
NP=int(os.environ.get("NPERM","10000")); rng=np.random.default_rng(0)
mod=pd.read_csv(D+"hotspot/gene_modules.csv",index_col=0).module
TFS={"JDP2":3,"NFE2L2":3,"ATF4":3,"CEBPB":4,"DDIT3":17}; MODS=[3,4,17]
A=ad.read_h5ad(R+"finalized_data/hPSC_undifferentiated_data/hPSC_undifferentiated.h5ad"); A=A[A.obs.clone_id.notna()].copy()
rk=lambda M:(lambda r:(r-r.mean(0))/r.std(0))(np.apply_along_axis(rankdata,0,M))
rows=[]; ext=[]; genes_out=[]
for t1,t2 in [("T0","T10"),("T10","T15"),("T0","T15")]:
    a=A[A.obs.timepoint.isin([t1,t2])].copy(); ct=pd.crosstab(a.obs.clone_id,a.obs.timepoint); cl=ct.index[(ct[t1]>0)&(ct[t2]>0)]
    a=a[a.obs.clone_id.isin(cl)].copy(); fr=np.diff(a.X.tocsc().indptr)/a.n_obs; a=a[:,fr>=0.05].copy(); g=a.var_names; gi={x:i for i,x in enumerate(g)}
    def cm(tp):
        s=a[a.obs.timepoint==tp]; df=pd.DataFrame(s.X.toarray(),columns=g); df["c"]=s.obs.clone_id.values; return df.groupby("c").mean().loc[cl].values
    Ra,Rb=rk(cm(t1)),rk(cm(t2)); n=len(cl)
    tfl=[t for t in TFS if t in gi]; tf_i=[gi[t] for t in tfl]
    sets={(t,m):np.array([gi[x] for x in mod[mod==m].index if x in gi and x!=t]) for t in tfl for m in MODS}
    def stat(Rb_):
        fwd=(Ra[:,tf_i].T@Rb_)/n            # nTF x genes: TF(t1) -> gene(t2)
        rev=(Ra.T@Rb_[:,tf_i])/n            # genes x nTF: gene(t1) -> TF(t2)
        res={}
        for (t,m),ix in sets.items():
            k=tfl.index(t); res[("fwd",t,m)]=fwd[k,ix]; res[("rev",t,m)]=rev[ix,k]
        return res
    obs=stat(Rb); keys=list(obs); NM=np.zeros((NP,len(keys))); GMX={d:np.zeros(NP) for d in("fwd","rev")}; GMN={d:np.zeros(NP) for d in("fwd","rev")}
    for i in range(NP):
        r=stat(Rb[rng.permutation(n)])
        NM[i]=[np.median(r[k]) for k in keys]
        for d in("fwd","rev"):
            v=np.concatenate([r[k] for k in keys if k[0]==d]); GMX[d][i]=v.max(); GMN[d][i]=v.min()
    for j,k in enumerate(keys):
        d,t,m=k; om=np.median(obs[k]); nm=NM[:,j]
        rows.append(dict(pair=f"{t1}-{t2}",direction="TF->gene" if d=="fwd" else "gene->TF",TF=t,module=m,TF_in_module=(TFS[t]==m),n_genes=len(obs[k]),obs_median_rho=om,null_mean=nm.mean(),null_sd=nm.std(),
            null_p2_5=np.percentile(nm,2.5),null_p97_5=np.percentile(nm,97.5),z=(om-nm.mean())/nm.std(),p_perm=(1+np.sum(np.abs(nm)>=abs(om)))/(NP+1)))
        gn=[x for x in mod[mod==m].index if x in gi and x!=t]
        genes_out.append(pd.DataFrame({"pair":f"{t1}-{t2}","direction":rows[-1]["direction"],"TF":t,"module":m,"gene":gn,"rho":obs[k]}))
    for d in("fwd","rev"):
        v=np.concatenate([obs[k] for k in keys if k[0]==d])
        ext.append(dict(pair=f"{t1}-{t2}",direction="TF->gene" if d=="fwd" else "gene->TF",n_pairs=len(v),obs_max=v.max(),null_max_95=np.percentile(GMX[d],95),p_max=(1+np.sum(GMX[d]>=v.max()))/(NP+1),
            obs_min=v.min(),null_min_5=np.percentile(GMN[d],5),p_min=(1+np.sum(GMN[d]<=v.min()))/(NP+1)))
    print(t1,t2,"done",flush=True)
r=pd.DataFrame(rows); r["FDR"]=multipletests(r.p_perm,method="fdr_bh")[1]; r.to_csv(O+"stress_tf_module_perm_null.csv",index=False)
pd.DataFrame(ext).to_csv(O+"stress_tf_module_perm_null_extremes.csv",index=False); pd.concat(genes_out).to_csv(O+"stress_tf_module_per_gene_rho.csv",index=False)
pd.set_option("display.width",230)
print(pd.DataFrame(ext).round(3).to_string(index=False)); print()
print(r.sort_values("p_perm").head(12)[["pair","direction","TF","module","n_genes","obs_median_rho","null_p2_5","null_p97_5","z","p_perm","FDR"]].round(4).to_string(index=False))
print("combos p_perm<0.05:",(r.p_perm<0.05).sum(),"of",len(r),"(expected ~",round(0.05*len(r),1),"by chance); FDR<0.05:",(r.FDR<0.05).sum())
