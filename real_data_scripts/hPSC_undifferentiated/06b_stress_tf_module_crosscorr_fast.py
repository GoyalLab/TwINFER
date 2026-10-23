# Stress TFs (JDP2,NFE2L2,ATF4 | CEBPB | DDIT3) vs all genes of the Hotspot module they sit in (3,4,17), both directions, across clones shared by two timepoints.
# Reads ONLY the ~100 module genes (not all genes). Null = NPERM scrambles of the clone pairing. Supersedes 06_stress_tf_module_crosscorr_perm.py (all-gene version, slow).
import os, time, anndata as ad, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests
t0=time.time(); P=lambda *x: print(f"[{time.time()-t0:.0f}s]",*x,flush=True)
R="/gpfs/projects/b1255/hzhang/TwINFER_KA/"; D=R+"analysis_data/hPSC_undifferentiated/"; O=os.environ.get("STRESS_OUT",D+"stress_tf_targets/")
NP=int(os.environ.get("NPERM","10000")); rng=np.random.default_rng(0)
mod=pd.read_csv(D+"hotspot/gene_modules.csv",index_col=0).module
TFS={"JDP2":3,"NFE2L2":3,"ATF4":3,"CEBPB":4,"DDIT3":17}; MODS=[3,4,17]
genes=sorted(set(mod[mod.isin(MODS)].index)|set(TFS))
b=ad.read_h5ad(R+"finalized_data/hPSC_undifferentiated_data/hPSC_undifferentiated.h5ad",backed="r")
A=b[:,genes].to_memory(); obs=b.obs[["timepoint","clone_id"]].copy(); A.obs=obs; A=A[A.obs.clone_id.notna()].copy(); P("loaded",A.shape)
X=pd.DataFrame(A.X.toarray() if hasattr(A.X,"toarray") else A.X,columns=genes,index=A.obs_names); X["clone"]=A.obs.clone_id.astype(str).values; X["tp"]=A.obs.timepoint.astype(str).values
rk=lambda M:(lambda r:(r-r.mean(0))/np.where(r.std(0)==0,1,r.std(0)))(np.apply_along_axis(rankdata,0,M))
rows=[];ext=[];per=[]
for t1,t2 in [("T0","T10"),("T10","T15"),("T0","T15")]:
    m1=X[X.tp==t1].groupby("clone")[genes].mean(); m2=X[X.tp==t2].groupby("clone")[genes].mean(); cl=m1.index.intersection(m2.index)
    Ra,Rb=rk(m1.loc[cl].values),rk(m2.loc[cl].values); n=len(cl); gi={x:i for i,x in enumerate(genes)}
    sets={(t,m):np.array([gi[x] for x in mod[mod==m].index if x!=t]) for t in TFS for m in MODS}; tfi={t:gi[t] for t in TFS}
    def stat(Rb_):
        res={}
        for (t,m),ix in sets.items():
            j=tfi[t]; res[("fwd",t,m)]=(Ra[:,j][:,None]*Rb_[:,ix]).mean(0); res[("rev",t,m)]=(Ra[:,ix]*Rb_[:,j][:,None]).mean(0)
        return res
    ob=stat(Rb); keys=list(ob); NM=np.zeros((NP,len(keys))); GX={d:np.zeros(NP) for d in("fwd","rev")}; GN={d:np.zeros(NP) for d in("fwd","rev")}
    for i in range(NP):
        r=stat(Rb[rng.permutation(n)]); NM[i]=[np.median(r[k]) for k in keys]
        for d in("fwd","rev"):
            v=np.concatenate([r[k] for k in keys if k[0]==d]); GX[d][i]=v.max(); GN[d][i]=v.min()
    for j,k in enumerate(keys):
        d,t,m=k; om=np.median(ob[k]); nm=NM[:,j]; dn="TF->gene" if d=="fwd" else "gene->TF"; gn=[x for x in mod[mod==m].index if x!=t]; r=ob[k]
        rows.append(dict(pair=f"{t1}-{t2}",clones=n,direction=dn,TF=t,module=m,own_module=TFS[t]==m,n_genes=len(gn),median_rho=om,frac_pos=(r>0).mean(),max_rho=r.max(),max_gene=gn[int(r.argmax())],min_rho=r.min(),min_gene=gn[int(r.argmin())],
            null_mean=nm.mean(),null_sd=nm.std(),null_p2_5=np.percentile(nm,2.5),null_p97_5=np.percentile(nm,97.5),z=(om-nm.mean())/nm.std(),p_perm=(1+np.sum(np.abs(nm)>=abs(om)))/(NP+1)))
        per.append(pd.DataFrame({"pair":f"{t1}-{t2}","direction":dn,"TF":t,"module":m,"gene":gn,"rho":r}))
    for d in("fwd","rev"):
        v=np.concatenate([ob[k] for k in keys if k[0]==d]); ext.append(dict(pair=f"{t1}-{t2}",direction="TF->gene" if d=="fwd" else "gene->TF",n_pairs=len(v),obs_max=v.max(),null_max_95=np.percentile(GX[d],95),p_max=(1+np.sum(GX[d]>=v.max()))/(NP+1),obs_min=v.min(),null_min_5=np.percentile(GN[d],5),p_min=(1+np.sum(GN[d]<=v.min()))/(NP+1)))
    P(t1,t2,n,"clones done")
r=pd.DataFrame(rows); r["FDR"]=multipletests(r.p_perm,method="fdr_bh")[1]
r.to_csv(O+"stress_tf_module_crosscorr_summary.csv",index=False); pd.DataFrame(ext).to_csv(O+"stress_tf_module_crosscorr_extremes.csv",index=False); pd.concat(per).to_csv(O+"stress_tf_module_crosscorr_per_gene.csv",index=False)
pd.set_option("display.width",250)
for p in ["T0-T15","T0-T10","T10-T15"]:
    x=r[r.pair==p]; print("\n== observed median rho,",p,int(x.clones.iloc[0]),"clones (TF->gene = TF earlier, gene later)"); print(x.pivot_table(index=["TF","module"],columns="direction",values="median_rho").round(3).to_string())
print("\nglobal extremes vs null"); print(pd.DataFrame(ext).round(3).to_string(index=False))
print("\nsmallest scramble p"); print(r.sort_values("p_perm").head(8)[["pair","direction","TF","module","n_genes","median_rho","null_p2_5","null_p97_5","z","p_perm","FDR"]].round(4).to_string(index=False))
print("combos p<0.05:",(r.p_perm<0.05).sum(),"of",len(r),"(chance ~",round(.05*len(r),1),"); FDR<0.05:",(r.FDR<0.05).sum())
