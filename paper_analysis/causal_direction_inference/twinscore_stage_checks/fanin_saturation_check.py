"""Why is the correlation low in Mutual_regulation? Test: genes with TWO activating inputs (fan-in) have their promoter closer to saturation, so they respond less to gene_1.
(1) model prediction (same parameters as the simulations: k_on .66, k_off 8.6, k_add 6, Hill n=2, regulators at K): relative sensitivity dln(p_on)/dlnP1, 1 input vs 2 inputs.
(2) data (t=20 h, first 20 common replicates): Spearman of each target's mRNA with gene_1 mRNA / gene_1 protein, and the log-log slope of target mean mRNA on gene_1 protein (deciles)."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from motif_replicates_zscores import files, MOT
from _paths import DATA
kon,koff,kadd,n=0.66,8.6,6.0,2.0
def pon1(l): h=np.exp(n*l); k=kon+kadd*h/(1+h); return k/(k+koff)
def pon2(l,h3=1.0): h1=np.exp(n*l); k=kon+kadd*(h1+h3+2*h1*h3)/(1+h1+h3+h1*h3); return k/(k+koff)
e=1e-4; rel=lambda f:((f(e)-f(-e))/(2*e))/f(0.0)
print(f"model: p_on 1 input={pon1(0):.3f}, 2 inputs={pon2(0):.3f}; relative sensitivity 1 input={rel(pon1):.3f}, 2 inputs={rel(pon2):.3f}; ratio={rel(pon1)/rel(pon2):.2f}")
F=files(); common=sorted(set.intersection(*[{r for (mm,r) in F if mm==m} for m in MOT]))[:20]; rows=[]
for m in MOT:
    for r in common:
        d=pd.read_csv(F[(m,r)],usecols=["time_step","gene_1_mRNA","gene_2_mRNA","gene_3_mRNA","gene_1_protein"]); d=d[d.time_step==20]
        P=np.log(d.gene_1_protein.values+1); q=pd.qcut(P,10,labels=False,duplicates="drop")
        for tgt in ("gene_2_mRNA","gene_3_mRNA"):
            mean=d.groupby(q)[tgt].mean().values; lnP=pd.Series(P).groupby(q).mean().values
            rows.append(dict(motif=m,target=tgt[:6],rep=r,rho_g1_mRNA=spearmanr(d.gene_1_mRNA,d[tgt])[0],rho_g1_protein=spearmanr(d.gene_1_protein,d[tgt])[0],slope_lnmean_vs_lnP1=np.polyfit(lnP,np.log(mean),1)[0]))
t=pd.DataFrame(rows); t.to_csv(DATA/"fanin_saturation_check.csv",index=False)
print(t.groupby(["motif","target"])[["rho_g1_mRNA","rho_g1_protein","slope_lnmean_vs_lnP1"]].mean().round(3).to_string())
