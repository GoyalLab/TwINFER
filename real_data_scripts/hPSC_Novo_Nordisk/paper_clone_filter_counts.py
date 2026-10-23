"""Apply the clone filter described in the text of Linneberg-Agerholm et al. 2026 (Results p.7 + Suppl 2A as described) to the object's clone.id and compare the counts to the paper.
paper: 3,814 clones; 1,635 winners; 397 losers; 1,782 filtered out; winners' replicate overlap hESCs p3 117 | 1,383 | 136 (same-only | both | sister-only), DE 326 | 981 | 329.
filter (text): detected at hESC p2 AND hESC p3; >=4 cells per clone; detected at >=2 timepoints; continuous coverage without gaps (order p2,p3,PS,DE); >=2 cells per detected timepoint.
winner = detected at all timepoints (p2,p3,PS,DE) [+ optional 'sustained growth']; loser = passes the filter but lost during differentiation (not detected at DE)."""
import numpy as np, pandas as pd
EXP="/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/export"; OUT="/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/paper_reproduction"
o=pd.read_csv(f"{EXP}/obs.csv",index_col=0,low_memory=False); o=o[o["clone.id"].notna()].copy(); o["clone"]=o["clone.id"].astype(int)
TP={"hESCs_p2_parent":"p2","hESCs_p3_same":"p3","hESCs_p3_sis":"p3","PS_sis":"PS","DE_same":"DE","DE_sis":"DE"}; o["tp"]=o.SampleID.astype(str).map(TP)
order=["p2","p3","PS","DE"]; cnt=o.groupby(["clone","tp"]).size().unstack(fill_value=0).reindex(columns=order,fill_value=0); wells=o.groupby(["clone","SampleID"]).size().unstack(fill_value=0)
total=len(cnt); print(f"clones with clone.id in the object: {total} (paper 3,814); multicellular: {(cnt.sum(1)>=2).sum()} (paper 3,808)")
def contiguous(row):
    det=[i for i,t in enumerate(order) if row[t]>0]; return len(det)>0 and det==list(range(det[0],det[-1]+1))
def classify(min_tp, min_total, growth):
    det=cnt>0; ok=det.p2&det.p3&(cnt.sum(1)>=min_total)&(det.sum(1)>=2)&cnt.apply(contiguous,axis=1)
    for t in order: ok&=(~det[t])|(cnt[t]>=min_tp)
    win=ok&det.PS&det.DE; 
    if growth=="DE>=p3": win&=(cnt.DE>=cnt.p3)
    if growth=="DE>=p2": win&=(cnt.DE>=cnt.p2)
    lose=ok&~det.DE
    return ok,win,lose
rows=[]
for min_tp in (1,2):
    for min_total in (4,):
        for growth in ("none","DE>=p3","DE>=p2"):
            ok,win,lose=classify(min_tp,min_total,growth)
            w=wells[win]; p3=((w.hESCs_p3_same>0).astype(int)*2+(w.hESCs_p3_sis>0)); de=((w.DE_same>0).astype(int)*2+(w.DE_sis>0))
            rows.append(dict(min_cells_per_timepoint=min_tp,sustained_growth=growth,pass_filter=int(ok.sum()),winners=int(win.sum()),losers=int(lose.sum()),filtered_out=total-int(ok.sum()),
                             p3_same_only=int((p3==2).sum()),p3_both=int((p3==3).sum()),p3_sister_only=int((p3==1).sum()),DE_same_only=int((de==2).sum()),DE_both=int((de==3).sum()),DE_sister_only=int((de==1).sum())))
R=pd.DataFrame(rows); pd.set_option("display.width",250); print(R.to_string(index=False)); R.to_csv(f"{OUT}/clone_filter_counts_vs_paper.csv",index=False)
print("\npaper: winners 1,635 | losers 397 | filtered out 1,782 | p3 117/1,383/136 | DE 326/981/329 | purity n=495 | agreement p3 n=343, DE n=67")
