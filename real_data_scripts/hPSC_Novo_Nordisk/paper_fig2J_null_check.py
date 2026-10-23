"""Fig 2J null check on the exact winner set (>=5 cells per clone in each replicate; deterministic first-index tie-break = reproduces the paper's observed 0.770 / 0.687).
Compares observed agreement to (a) the paper's null sum p_k^2, (b) a null that matches the clone sizes (multinomial sample per clone, same tie-break), and reports how many clones have ties."""
import numpy as np, pandas as pd
from scipy.stats import binomtest
EXP="/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/export"; OUT="/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/paper_reproduction"; rng=np.random.default_rng(1)
o=pd.read_csv(f"{EXP}/obs.csv",index_col=0,low_memory=False); o=o[o["clone.id"].notna()].copy(); o["clone"]=o["clone.id"].astype(int); o["cl"]=o.clust_annot_02.astype(str)
TP={"hESCs_p2_parent":"p2","hESCs_p3_same":"p3","hESCs_p3_sis":"p3","PS_sis":"PS","DE_same":"DE","DE_sis":"DE"}; o["tp"]=o.SampleID.astype(str).map(TP); order=["p2","p3","PS","DE"]
cnt=o.groupby(["clone","tp"]).size().unstack(fill_value=0).reindex(columns=order,fill_value=0); det=cnt>0
contig=cnt.apply(lambda r:(lambda d:len(d)>0 and d==list(range(d[0],d[-1]+1)))([i for i,t in enumerate(order) if r[t]>0]),axis=1)
win=set(cnt.index[det.p2&det.p3&det.PS&det.DE&(cnt.sum(1)>=4)&contig]); w=o[o.clone.isin(win)]
rows=[]
for tp,same,sis in (("p3","hESCs_p3_same","hESCs_p3_sis"),("DE","DE_same","DE_sis")):
    dd=w[w.tp==tp]; cats=sorted(o[o.tp==tp].cl.unique()); idx={c:i for i,c in enumerate(cats)}; pk=dd.cl.value_counts(normalize=True).reindex(cats).fillna(0).to_numpy()
    cs=dd[dd.SampleID.astype(str)==same].groupby("clone").cl.agg(list); ci=dd[dd.SampleID.astype(str)==sis].groupby("clone").cl.agg(list)
    ob,tie,ex=[],[],[]; tied_agree=[]
    for c in cs.index.intersection(ci.index):
        x,y=cs[c],ci[c]
        if len(x)<5 or len(y)<5: continue
        kx=np.bincount([idx[v] for v in x],minlength=len(cats)); ky=np.bincount([idx[v] for v in y],minlength=len(cats))
        a=int(np.argmax(kx))==int(np.argmax(ky)); t=((kx==kx.max()).sum()>1) or ((ky==ky.max()).sum()>1); ob.append(a); tie.append(t)
        sx=rng.multinomial(len(x),pk,size=2000); sy=rng.multinomial(len(y),pk,size=2000); ex.append((sx.argmax(1)==sy.argmax(1)).mean())
    ob,tie,ex=np.array(ob),np.array(tie),np.array(ex); n=len(ob)
    rows.append(dict(stage=tp,n_clones=n,observed=ob.mean(),paper_null_sum_pk2=float((pk**2).sum()),size_matched_null=ex.mean(),
                     p_vs_paper_null=binomtest(int(ob.sum()),n,float((pk**2).sum()),alternative="greater").pvalue,p_vs_size_matched_null=binomtest(int(ob.sum()),n,ex.mean(),alternative="greater").pvalue,
                     clones_with_a_tie=int(tie.sum()),agree_among_tied=ob[tie].mean() if tie.any() else np.nan,agree_among_untied=ob[~tie].mean()))
R=pd.DataFrame(rows); R.to_csv(f"{OUT}/fig2J_null_check.csv",index=False); pd.set_option("display.width",250); print(R.round(4).T.to_string())
