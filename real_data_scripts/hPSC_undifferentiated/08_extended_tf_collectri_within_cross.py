# All TFs with >=5% cells and >=20 CollecTRI targets (>=5% cells) [env NTF = top-N by mean expression, default all].
# (1) within-sample Spearman across cells per timepoint (T0,T10,T15); (2) cross-timepoint Spearman across shared clones, TF->gene (TF earlier) and gene->TF (gene earlier);
# (3) asymmetry = |x|-|y| (|TF->gene| minus |gene->TF|), flagged beyond 2 x empirical SD of |x|-|y|. Env OUTDIR overrides the output folder.
import os, time, anndata as ad, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from scipy.stats import rankdata, t as tdist
from statsmodels.stats.multitest import multipletests
t0=time.time(); P=lambda *x: print(f"[{time.time()-t0:.0f}s]",*x,flush=True)
R="/gpfs/projects/b1255/hzhang/TwINFER_KA/"; O=os.environ.get("OUTDIR",R+"analysis_data/hPSC_undifferentiated/tf_collectri_extended/"); os.makedirs(O,exist_ok=True)
NTF=os.environ.get("NTF","all"); NPERM=int(os.environ.get("NPERM","0")); NULLQ={}; rng=np.random.default_rng(0)
tfl=[l.strip() for l in open(R+"analysis_data/hPSC_20260927/gene_modules/all_tfs.txt") if l.strip()]; tfset=set(tfl)
c=pd.read_csv(R+"analysis_data/paper_analysis/larry_hematopoiesis_validation/resources/collectri_human.tsv",sep="\t",usecols=["source_genesymbol","target_genesymbol","is_stimulation","is_inhibition"])
c["sign"]=np.where(c.is_stimulation&~c.is_inhibition,"activation",np.where(c.is_inhibition&~c.is_stimulation,"inhibition","unknown")); c=c.drop_duplicates(["source_genesymbol","target_genesymbol"])
b=ad.read_h5ad(R+"finalized_data/hPSC_undifferentiated_data/hPSC_undifferentiated.h5ad",backed="r"); var=b.var_names
# per-gene % cells and mean from a chunked pass (no full-matrix copy)
nnz=np.zeros(b.n_vars); sm=np.zeros(b.n_vars)
for s in range(0,b.n_obs,5000):
    X=b.X[s:s+5000]; nnz+=np.bincount(X.indices,minlength=b.n_vars); sm+=np.asarray(X.sum(0)).ravel()
frac=pd.Series(nnz/b.n_obs,index=var); mean=pd.Series(sm/b.n_obs,index=var); ok=set(frac[frac>=.05].index); P("expression summary done")
c=c[c.target_genesymbol.isin(ok)&(c.target_genesymbol!=c.source_genesymbol)]; ntar=c.groupby("source_genesymbol").size()
cand=pd.DataFrame({"mean_expr":mean.reindex(tfl),"pct_cells":100*frac.reindex(tfl),"n_targets":ntar.reindex(tfl)}).dropna(); cand=cand[(cand.n_targets>=20)&(cand.pct_cells>=5)].sort_values("mean_expr",ascending=False)
if NTF!="all": cand=cand.head(int(NTF))
cand.to_csv(O+"selected_TFs.csv"); sel=list(cand.index); tg=c[c.source_genesymbol.isin(sel)]; genes=sorted(set(sel)|set(tg.target_genesymbol)); P(len(sel),"TFs;",len(tg),"pairs;",len(genes),"genes")
A=b[:,genes].to_memory(); tp=b.obs.timepoint.astype(str).values; clone=b.obs.clone_id.astype(object).values
X=np.asarray(A.X.toarray() if hasattr(A.X,"toarray") else A.X,dtype=np.float32); del A; P("loaded",X.shape)
gi={g:i for i,g in enumerate(genes)}; out=tg.rename(columns={"source_genesymbol":"TF","target_genesymbol":"gene"})[["TF","gene","sign"]].reset_index(drop=True)
out.insert(1,"TF_mean_expr",out.TF.map(cand.mean_expr).round(3)); out["gene_is_TF"]=out.gene.isin(tfset); jt=out.TF.map(gi).values; jg=out.gene.map(gi).values
rk=lambda M:(lambda r:(r-r.mean(0))/np.where(r.std(0)==0,1,r.std(0)))(np.apply_along_axis(rankdata,0,M))
def pr(A1,A2,ia,ib,ch=1500):  # mean over rows of A1[:,ia]*A2[:,ib], in chunks (keeps memory small)
    o=np.empty(len(ia))
    for s in range(0,len(ia),ch): o[s:s+ch]=(A1[:,ia[s:s+ch]]*A2[:,ib[s:s+ch]]).mean(0)
    return o
tfidx=np.array([gi[t] for t in sel]); tgt={t:set(gi[g] for g in d.gene) for t,d in out.groupby("TF")}; bgp=[]
for t in ["T0","T10","T15"]:   # within-sample, across cells
    Rk=rk(X[tp==t]); n=Rk.shape[0]; rho=pr(Rk,Rk,jt,jg); out[f"rho_{t}"]=rho.round(4); tt=rho*np.sqrt((n-2)/(1-rho**2)); out[f"FDR_{t}"]=multipletests(2*tdist.sf(np.abs(tt),n-2),method="fdr_bh")[1]
    bt=rng.choice(tfidx,40000); bg=rng.integers(0,len(genes),40000); keep=np.array([(g!=a) and (g not in tgt[genes[a]]) for a,g in zip(bt,bg)]); bgp.append(dict(tp=t,cells=n,bg_pairs=int(keep.sum()),bg_abs_gt_0_05=float((np.abs(pr(Rk,Rk,bt[keep],bg[keep]))>.05).mean())))
    P("within",t,n); del Rk
Xd=pd.DataFrame(X,columns=genes); Xd["clone"]=clone; Xd["tp"]=tp; Xd=Xd[Xd.clone.notna()]; del X; N={}
for t1,t2 in [("T0","T15"),("T0","T10"),("T10","T15")]:  # cross-timepoint, across shared clones
    m1=Xd[Xd.tp==t1].groupby("clone")[genes].mean(); m2=Xd[Xd.tp==t2].groupby("clone")[genes].mean(); cl=m1.index.intersection(m2.index); Ra,Rb=rk(m1.loc[cl].values),rk(m2.loc[cl].values); p=f"{t1}_to_{t2}"; N[p]=len(cl)
    x=pr(Ra,Rb,jt,jg); y=pr(Ra,Rb,jg,jt); da=np.abs(x)-np.abs(y); out[f"{p}__TF_to_gene"]=x.round(4); out[f"{p}__gene_to_TF"]=y.round(4); out[f"absdiff_{p}"]=da.round(4); out[f"asym_{p}"]=np.abs(da)>2*da.std(); P("cross",p,len(cl),"clones; SD |x|-|y| =",round(da.std(),3))
    if NPERM>0:   # scramble the clone pairing NPERM times (Rb rows shuffled) -> empirical null for x, y and |x|-|y| for every pair
        cx=np.zeros(len(x)); cy=np.zeros(len(x)); cd=np.zeros(len(x)); smp=[]
        for _ in range(NPERM):
            Rbp=Rb[rng.permutation(len(cl))]; xp=pr(Ra,Rbp,jt,jg); yp=pr(Ra,Rbp,jg,jt); dp=np.abs(xp)-np.abs(yp)
            cx+=np.abs(xp)>=np.abs(x); cy+=np.abs(yp)>=np.abs(y); cd+=np.abs(dp)>=np.abs(da); smp.append(np.abs(dp[rng.integers(0,len(dp),300)]))
        for nm,cc in (("x",cx),("y",cy),("absdiff",cd)):
            pv=(1+cc)/(NPERM+1); out[f"perm_p_{nm}_{p}"]=pv; out[f"perm_FDR_{nm}_{p}"]=multipletests(pv,method="fdr_bh")[1]
        smp=np.concatenate(smp); NULLQ[p]=dict(null_absdiff_q95=float(np.quantile(smp,.95)),null_absdiff_q99=float(np.quantile(smp,.99)),analytic_cutoff_2SD=float(2*da.std())); P("scrambles done",p,NPERM)
out["TF"]=pd.Categorical(out.TF,sel); out=out.sort_values(["TF","gene"]); out.to_csv(O+"extended_tf_collectri_within_cross_asym.csv",index=False); pd.DataFrame(bgp).to_csv(O+"within_sample_background.csv",index=False)
pd.set_option("display.width",250); print(pd.DataFrame(bgp).to_string(index=False)); print("pairs:",len(out),"; TF-TF targets:",int(out.gene_is_TF.sum()))
for t in ["T0","T10","T15"]: print(f"within {t}: |rho|>0.05 {(out[f'rho_{t}'].abs()>.05).sum()}, >0.1 {(out[f'rho_{t}'].abs()>.1).sum()}, >0.2 {(out[f'rho_{t}'].abs()>.2).sum()}, FDR<0.05 {(out[f'FDR_{t}']<.05).sum()}")
for p,(t1,t2) in {"T10_to_T15":("T10","T15"),"T0_to_T15":("T0","T15"),"T0_to_T10":("T0","T10")}.items():
    g1,g2=out[f"rho_{t1}"].abs()>.05,out[f"rho_{t2}"].abs()>.05; f=out[f"asym_{p}"]; da=out[f"absdiff_{p}"]; rows=[]
    for name,m in [("|rho|>0.05 at both",g1&g2),("|rho|>0.05 at either",g1|g2),("neither",~(g1|g2))]:
        ff=f&m; rows.append(dict(group=name,pairs=int(m.sum()),asym=int(ff.sum()),pct=round(100*ff.sum()/max(m.sum(),1),1),TF_stronger=int((ff&(da>0)).sum()),gene_stronger=int((ff&(da<0)).sum())))
    print(f"\n== {p} ({N[p]} clones, cutoff {2*da.std():.3f})"); print(pd.DataFrame(rows).to_string(index=False))

if NPERM>0:
    pd.DataFrame(NULLQ).T.to_csv(O+"scramble_null_cutoffs.csv"); print("\nscramble null (",NPERM,"scrambles): 95th/99th percentile of |(|x|-|y|)| under scrambled clone pairing vs the 2-SD cutoff used"); print(pd.DataFrame(NULLQ).T.round(3).to_string())
    for p,(t1,t2) in {"T10_to_T15":("T10","T15"),"T0_to_T15":("T0","T15"),"T0_to_T10":("T0","T10")}.items():
        g1,g2=out[f"rho_{t1}"].abs()>.05,out[f"rho_{t2}"].abs()>.05; rows=[]
        for name,m in [("all pairs",g1|True),("|rho|>0.05 at both",g1&g2),("neither",~(g1|g2))]:
            r=dict(group=name,pairs=int(m.sum()))
            for nm in ("x","y","absdiff"): r[f"{nm}_p<0.05"]=int((out[f"perm_p_{nm}_{p}"][m]<.05).sum()); r[f"{nm}_FDR<0.05"]=int((out[f"perm_FDR_{nm}_{p}"][m]<.05).sum())
            r["expected_p<0.05"]=round(.05*m.sum(),1); rows.append(r)
        print(f"\n== scramble p-values, {p}"); print(pd.DataFrame(rows).to_string(index=False))
    out.to_csv(O+"extended_tf_collectri_within_cross_asym.csv",index=False)
