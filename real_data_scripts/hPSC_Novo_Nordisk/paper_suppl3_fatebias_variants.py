"""Suppl 3B/3C: which clone set and definition gives endoderm-biased 582, mesoderm-biased 285, ectoderm clones 74 (66+6+2), ~961 clones with >=5 cells at DE (7.7% with ectoderm)?"""
import numpy as np, pandas as pd
EXP="/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/export"
o=pd.read_csv(f"{EXP}/obs.csv",index_col=0,low_memory=False); o=o[o["clone.id"].notna()].copy(); o["clone"]=o["clone.id"].astype(int); o["cl"]=o.clust_annot_02.astype(str)
TP={"hESCs_p2_parent":"p2","hESCs_p3_same":"p3","hESCs_p3_sis":"p3","PS_sis":"PS","DE_same":"DE","DE_sis":"DE"}; o["tp"]=o.SampleID.astype(str).map(TP); order=["p2","p3","PS","DE"]
cnt=o.groupby(["clone","tp"]).size().unstack(fill_value=0).reindex(columns=order,fill_value=0); det=cnt>0; tot=cnt.sum(1)
contig=cnt.apply(lambda r:(lambda d:len(d)>0 and d==list(range(d[0],d[-1]+1)))([i for i,t in enumerate(order) if r[t]>0]),axis=1)
win=set(cnt.index[det.p2&det.p3&det.PS&det.DE&(tot>=4)&contig]); de=o[o.tp=="DE"]
def stats(groups,thr=0.65):
    f=lambda lab:groups.map(lambda x:np.mean([v==lab for v in x]))
    fe,fm,fx=f("Definitive endoderm"),f("Early mesoderm"),f("Neuroectoderm"); return len(groups),int((fe>=thr).sum()),int((fm>=thr).sum()),int((fx>0).sum()),100*(fx>0).mean()
rows=[]
for setname,sel in (("winner clones",de[de.clone.isin(win)]),("all clones",de)):
    for unit,key in (("clone (DE_same+DE_sis pooled)",["clone"]),("clone x well (each DE well separately)",["clone","SampleID"])):
        g=sel.groupby(key).cl.agg(list)
        for mc in (3,5):
            gg=g[g.map(len)>=mc]; n,e,m,x,pc=stats(gg); rows.append(dict(clone_set=setname,unit=unit,min_cells_at_DE=mc,clones=n,endoderm_biased=e,mesoderm_biased=m,with_ectoderm=x,pct_ectoderm=round(pc,1)))
R=pd.DataFrame(rows); pd.set_option("display.width",250); print("paper: clones ~961 | endoderm-biased 582 | mesoderm-biased 285 | with ectoderm 74 (7.7%)\n"); print(R.to_string(index=False))
