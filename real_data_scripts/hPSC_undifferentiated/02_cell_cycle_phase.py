import anndata as ad, numpy as np, pandas as pd, scanpy as sc
from scipy.stats import chi2_contingency
R="/gpfs/projects/b1255/hzhang/TwINFER_KA/"
s_genes="MCM5 PCNA TYMS FEN1 MCM2 MCM4 RRM1 UNG GINS2 MCM6 CDCA7 DTL PRIM1 UHRF1 CENPU HELLS RFC2 RPA2 NASP RAD51AP1 GMNN WDR76 SLBP CCNE2 UBR7 POLD3 MSH2 ATAD2 RAD51 RRM2 CDC45 CDC6 EXO1 TIPIN DSCC1 BLM CASP8AP2 USP1 CLSPN POLA1 CHAF1B BRIP1 E2F8".split()
g2m="HMGB2 CDK1 NUSAP1 UBE2C BIRC5 TPX2 TOP2A NDC80 CKS2 NUF2 CKS1B MKI67 TMPO CENPF TACC3 PIMREG SMC4 CCNB2 CKAP2L CKAP2 AURKB BUB1 KIF11 ANP32E TUBB4B GTSE1 KIF20B HJURP CDCA3 JPT1 CDC20 TTK CDC25C KIF2C RANGAP1 NCAPD2 DLGAP5 CDCA2 CDCA8 ECT2 KIF23 HMMR AURKA PSRC1 ANLN LBR CKAP5 CENPE CTCF NEK2 G2E3 GAS2L3 CBX5 CENPA".split()
a=ad.read_h5ad(R+"finalized_data/hPSC_undifferentiated_data/hPSC_undifferentiated.h5ad")
s=[g for g in s_genes if g in a.var_names]; g=[x for x in g2m if x in a.var_names]
print(f"S {len(s)}/{len(s_genes)}, G2M {len(g)}/{len(g2m)}; missing:",[x for x in s_genes+g2m if x not in a.var_names])
sc.tl.score_genes_cell_cycle(a,s_genes=s,g2m_genes=g)
a.obs["cycling"]=np.where(a.obs.phase=="G1","non-cycling (G1)","cycling (S/G2M)")
o=a.obs[["timepoint","S_score","G2M_score","phase","cycling"]]
o.to_csv(R+"analysis_data/hPSC_undifferentiated/cell_cycle_phase_per_cell.csv")
t=pd.crosstab(o.timepoint,o.phase); print(t); print((t.div(t.sum(1),axis=0)*100).round(1))
c=pd.crosstab(o.timepoint,o.cycling); print((c.div(c.sum(1),axis=0)*100).round(1))
for x,y in [("T0","T10"),("T10","T15"),("T0","T15")]:
    p=chi2_contingency(c.loc[[x,y]])[1]; print(x,y,"cycling vs non chi2 p=%.2e"%p)
print(o.groupby("timepoint")[["S_score","G2M_score"]].mean().round(3))
print("overall phase-vs-timepoint chi2 p=%.2e"%chi2_contingency(t)[1])
