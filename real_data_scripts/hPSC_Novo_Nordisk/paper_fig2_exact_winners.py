"""Fig 2I / 2J on the EXACT winner set (filter reproduced in paper_clone_filter_counts.py: detected p2&p3 (>=1 cell), >=4 cells, contiguous, detected at PS and DE).
Scans the minimum cells per clone (per replicate) to find the n that matches the paper (purity n=495; agreement p3 n=343, DE n=67) and tests observed vs weighted-random.
null (Methods): clone-size-matched cell-type assignments sampled from the observed cluster distribution at the stage, 1,000 permutations; dominant cluster ties broken at random."""
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, binomtest
EXP="/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/export"; OUT="/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/paper_reproduction"; rng=np.random.default_rng(0)
o=pd.read_csv(f"{EXP}/obs.csv",index_col=0,low_memory=False); allo=o.copy(); o=o[o["clone.id"].notna()].copy(); o["clone"]=o["clone.id"].astype(int); o["cl"]=o.clust_annot_02.astype(str)
TP={"hESCs_p2_parent":"p2","hESCs_p3_same":"p3","hESCs_p3_sis":"p3","PS_sis":"PS","DE_same":"DE","DE_sis":"DE"}; o["tp"]=o.SampleID.astype(str).map(TP); order=["p2","p3","PS","DE"]
cnt=o.groupby(["clone","tp"]).size().unstack(fill_value=0).reindex(columns=order,fill_value=0); det=cnt>0
contig=cnt.apply(lambda r: (lambda d: len(d)>0 and d==list(range(d[0],d[-1]+1)))([i for i,t in enumerate(order) if r[t]>0]),axis=1)
win=set(cnt.index[det.p2&det.p3&det.PS&det.DE&(cnt.sum(1)>=4)&contig&(det.sum(1)>=2)]); print("winners:",len(win))
w=o[o.clone.isin(win)]
def stage_dist(tp,pool):
    d=(w if pool=="winner cells" else o)[lambda x:x.tp==tp]; cats=sorted(o[o.tp==tp].cl.unique()); return cats,d.cl.value_counts(normalize=True).reindex(cats).fillna(0).to_numpy()
def dom(k): return rng.choice(np.flatnonzero(k==k.max()))
def purity(mincells,pool,nperm=1000):
    cats,pd_=stage_dist("DE",pool); g=w[w.tp=="DE"].groupby("clone").cl.agg(list); ob,nu=[],[]
    for c,v in g.items():
        if len(v)<mincells: continue
        k=pd.Series(v).value_counts().to_numpy(); ob.append(k.max()/len(v)); nu.append((rng.multinomial(len(v),pd_,size=nperm).max(1)/len(v)).mean())
    ob,nu=np.array(ob),np.array(nu); return dict(n=len(ob),observed=ob.mean() if len(ob) else np.nan,weighted_random=nu.mean() if len(ob) else np.nan,wilcoxon_p=wilcoxon(ob-nu).pvalue if len(ob)>5 else np.nan)
def agree(tp,same,sis,mincells,pool,nsim=1000):
    cats,pd_=stage_dist(tp,pool); idx={c:i for i,c in enumerate(cats)}; d=w[w.tp==tp]
    cs=d[d.SampleID.astype(str)==same].groupby("clone").cl.agg(list); ci=d[d.SampleID.astype(str)==sis].groupby("clone").cl.agg(list); ob,ex=[],[]
    for c in cs.index.intersection(ci.index):
        x,y=cs[c],ci[c]
        if len(x)<mincells or len(y)<mincells: continue
        kx=np.bincount([idx[v] for v in x],minlength=len(cats)); ky=np.bincount([idx[v] for v in y],minlength=len(cats)); ob.append(dom(kx)==dom(ky))
        sx=rng.multinomial(len(x),pd_,size=nsim); sy=rng.multinomial(len(y),pd_,size=nsim)
        ex.append((np.array([dom(r) for r in sx])==np.array([dom(r) for r in sy])).mean())
    ob,ex=np.array(ob),np.array(ex)
    if len(ob)==0: return dict(n=0)
    return dict(n=len(ob),observed=ob.mean(),weighted_random=ex.mean(),binom_p=binomtest(int(ob.sum()),len(ob),ex.mean(),alternative="greater").pvalue)
rows=[]
for pool in ("winner cells","all clone-labelled cells"):
    for k in range(1,9):
        r=purity(k,pool); r.update(test="purity DE",min_cells=k,null_distribution_from=pool); rows.append(r)
        for tp,s,t in (("p3","hESCs_p3_same","hESCs_p3_sis"),("DE","DE_same","DE_sis")):
            r=agree(tp,s,t,k,pool); r.update(test=f"agreement {tp}",min_cells=k,null_distribution_from=pool); rows.append(r)
R=pd.DataFrame(rows)[["test","null_distribution_from","min_cells","n","observed","weighted_random","wilcoxon_p","binom_p"]]; R.to_csv(f"{OUT}/fig2_exact_winners.csv",index=False)
pd.set_option("display.width",250); print("paper: purity DE n=495 0.68 vs 0.646 | agreement p3 n=343 0.77 vs 0.545 | agreement DE n=67 0.687 vs 0.50")
for t in ("purity DE","agreement p3","agreement DE"): print("\n"+t); print(R[R.test==t].drop(columns="test").round(4).to_string(index=False))
