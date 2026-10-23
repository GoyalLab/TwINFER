"""Which definition of 'dominant cluster agreement' and of the weighted-random null reproduces the paper's 0.77 vs 0.545 (hESCs p3, n=343) and 0.687 vs 0.50 (DE, n=67)?
clones: exact winner set, >=5 cells per clone in EACH replicate (this gives n=343 / 67 exactly).
observed variants: dominant cluster over all clusters | over the two major clusters only | deterministic tie-break (first in cluster order) vs random tie-break
null variants   : (a) sum_k p_k^2 of the stage's cluster frequencies (random cluster label per clone, 'weighted random')  (b) clone-size-matched multinomial dominant-cluster simulation"""
import numpy as np, pandas as pd
EXP="/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/export"; OUT="/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/paper_reproduction"; rng=np.random.default_rng(0)
o=pd.read_csv(f"{EXP}/obs.csv",index_col=0,low_memory=False); o=o[o["clone.id"].notna()].copy(); o["clone"]=o["clone.id"].astype(int); o["cl"]=o.clust_annot_02.astype(str)
TP={"hESCs_p2_parent":"p2","hESCs_p3_same":"p3","hESCs_p3_sis":"p3","PS_sis":"PS","DE_same":"DE","DE_sis":"DE"}; o["tp"]=o.SampleID.astype(str).map(TP); order=["p2","p3","PS","DE"]
cnt=o.groupby(["clone","tp"]).size().unstack(fill_value=0).reindex(columns=order,fill_value=0); det=cnt>0
contig=cnt.apply(lambda r:(lambda d:len(d)>0 and d==list(range(d[0],d[-1]+1)))([i for i,t in enumerate(order) if r[t]>0]),axis=1)
win=set(cnt.index[det.p2&det.p3&det.PS&det.DE&(cnt.sum(1)>=4)&contig]); w=o[o.clone.isin(win)]
MAJOR={"p3":["Epiblast 1","Epiblast 2"],"DE":["Definitive endoderm","Early mesoderm"]}
def run(tp,same,sis,major,tie,cells_for_dist):
    base=(w if cells_for_dist=="winner" else o); dd=base[base.tp==tp]; cats=sorted(dd.cl.unique()); 
    if major: cats=[c for c in cats if c in MAJOR[tp]]
    idx={c:i for i,c in enumerate(cats)}; pk=dd.cl.value_counts(normalize=True).reindex(cats).fillna(0).to_numpy(); pk=pk/pk.sum()
    d=w[w.tp==tp]; cs=d[d.SampleID.astype(str)==same].groupby("clone").cl.agg(list); ci=d[d.SampleID.astype(str)==sis].groupby("clone").cl.agg(list)
    def dom(k): 
        if tie=="first": return int(np.argmax(k))
        return rng.choice(np.flatnonzero(k==k.max()))
    ob,sim=[],[]
    for c in cs.index.intersection(ci.index):
        x,y=cs[c],ci[c]
        if len(x)<5 or len(y)<5: continue                                   # n=343 / 67 selection uses ALL cells of the replicate
        kx=np.bincount([idx[v] for v in x if v in idx],minlength=len(cats)); ky=np.bincount([idx[v] for v in y if v in idx],minlength=len(cats))
        if kx.sum()==0 or ky.sum()==0: continue
        ob.append(dom(kx)==dom(ky))
        sx=rng.multinomial(kx.sum(),pk,size=400); sy=rng.multinomial(ky.sum(),pk,size=400); sim.append((np.array([dom(r) for r in sx])==np.array([dom(r) for r in sy])).mean())
    return dict(n=len(ob),observed=np.mean(ob),null_sum_pk2=float((pk**2).sum()),null_size_matched=np.mean(sim))
rows=[]
for tp,s,t in (("p3","hESCs_p3_same","hESCs_p3_sis"),("DE","DE_same","DE_sis")):
    for major in (False,True):
        for tie in ("random","first"):
            for dist in ("winner","all"):
                r=run(tp,s,t,major,tie,dist); r.update(stage=tp,clusters="two major only" if major else "all clusters",tie_break=tie,null_distribution_from=dist); rows.append(r)
R=pd.DataFrame(rows)[["stage","clusters","tie_break","null_distribution_from","n","observed","null_sum_pk2","null_size_matched"]]; R.to_csv(f"{OUT}/fig2J_definition_variants.csv",index=False)
pd.set_option("display.width",250); print("paper: hESCs p3 n=343 observed 0.77 vs 0.545 | DE n=67 observed 0.687 vs 0.50\n"); print(R.round(3).to_string(index=False))
