"""Checks against the paper's supplementary numbers (Suppl 2B, 2C, 2I/J, 3B, 3C). Winner clones = filter reproduced in paper_clone_filter_counts.py (detected p2&p3&PS&DE, >=4 cells, contiguous)."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
EXP="/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/export"; OUT="/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/paper_reproduction"
o=pd.read_csv(f"{EXP}/obs.csv",index_col=0,low_memory=False); o=o[o["clone.id"].notna()].copy(); o["clone"]=o["clone.id"].astype(int); o["cl"]=o.clust_annot_02.astype(str)
TP={"hESCs_p2_parent":"p2","hESCs_p3_same":"p3","hESCs_p3_sis":"p3","PS_sis":"PS","DE_same":"DE","DE_sis":"DE"}; o["tp"]=o.SampleID.astype(str).map(TP); order=["p2","p3","PS","DE"]
cnt=o.groupby(["clone","tp"]).size().unstack(fill_value=0).reindex(columns=order,fill_value=0); det=cnt>0; tot=cnt.sum(1)
contig=cnt.apply(lambda r:(lambda d:len(d)>0 and d==list(range(d[0],d[-1]+1)))([i for i,t in enumerate(order) if r[t]>0]),axis=1)
ok=det.p2&det.p3&(tot>=4)&contig&(det.sum(1)>=2); win=ok&det.PS&det.DE
print("== Suppl 2B (J=0.7): paper total 3,410 | winners 1,635 | cells in winners 32,043 | losers 558 | cells in losers 3,435 | median winner size 16 | clones with >=10 cells 1,336")
print(f"ours: total clones {len(cnt)} | winners {int(win.sum())} | cells in winners {int(tot[win].sum())} | median winner size {tot[win].median():.0f} | clones with >=10 cells {(tot>=10).sum()}")
lose_variants={"passes filter, not detected at DE":ok&~det.DE,
               "detected p2&p3, >=4 cells, not a winner":det.p2&det.p3&(tot>=4)&~win,
               "detected p2&p3, >=4 cells, not detected at DE":det.p2&det.p3&(tot>=4)&~det.DE,
               "detected p2&p3, >=2 timepoints, >=4 cells, no DE (any gaps)":det.p2&det.p3&(tot>=4)&(det.sum(1)>=2)&~det.DE}
for k,v in lose_variants.items(): print(f"  loser = {k}: {int(v.sum())} clones, {int(tot[v].sum())} cells")
print("\n== Suppl 2I/2J (clone size vs purity; paper: hESCs p3 n=1,127 r=-0.091 p=0.0022 | DE n=494 r=-0.158 p=0.00041)")
w=o[o.clone.isin(win[win].index)]
for tp,mc in (("p3",5),("DE",5)):
    rows=[]
    for c,g in w[w.tp==tp].groupby("clone"):
        n=len(g); rows.append((n,g.cl.value_counts().iloc[0]/n))
    d=pd.DataFrame(rows,columns=["size","purity"])
    for m in (4,5,6):
        dd=d[d["size"]>=m]; r,p=spearmanr(dd["size"],dd.purity); print(f"  {tp}, clones with >= {m} cells at the stage: n={len(dd)}, Spearman r={r:.3f}, p={p:.2g}")
print("\n== Suppl 3B/3C (clones with >=5 cells at DE; paper: endoderm-biased 582, mesoderm-biased 285, with ectoderm 66+6+2 = 74 of ~961 (7.7%))")
d=w[w.tp=="DE"].groupby("clone").cl.agg(list); d=d[d.map(len)>=5]
fe=d.map(lambda x:np.mean([v=="Definitive endoderm" for v in x])); fm=d.map(lambda x:np.mean([v=="Early mesoderm" for v in x])); fx=d.map(lambda x:np.mean([v=="Neuroectoderm" for v in x]))
print(f"ours: clones with >=5 cells at DE: {len(d)} | endoderm-biased (>=65% endoderm): {(fe>=0.65).sum()} | mesoderm-biased (>=65% mesoderm): {(fm>=0.65).sum()} | clones with any ectoderm cell: {(fx>0).sum()} ({100*(fx>0).mean():.1f}%)")
for thr in (0.65,0.7,0.75):
    print(f"   threshold {thr}: endoderm-biased {(fe>=thr).sum()}, mesoderm-biased {(fm>=thr).sum()}")
