"""Test the fan-in/saturation model on the Mutual_regulation parameter variants (mutual_param_variants/). Per variant (all finished simulations):
 measured Spearman(target mRNA, gene_1 protein) and (gene_1 mRNA) at t=20 h, log-log slope of target mean mRNA on gene_1 protein, mean mRNA, bimodality (2-component GMM BIC),
 and the TwinScore results on the 3 pairs (1,500 twins + 3,000 singletons, t=1 h/20 h). Model predictions from the crude fan-in model (V=0.094 calibrated on one-input r=0.31)."""
import glob, os, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.mixture import GaussianMixture
import figure2_1000_zscores as F
from _paths import DATA
M="/projects/b1255/hzhang/TwINFER_KA/simulation_data/figure_4/mutual_param_variants/"
VAR=["control","n3","n4","kadd12","koff17","combo","kprod2x","mrnahl2x"]
PRED={"control":(5.0,0.169),"n3":(5.0,0.253),"n4":(5.0,0.337),"kadd12":(6.9,0.151),"koff17":(3.2,0.173),"combo":(4.9,0.363),"kprod2x":(10.1,0.231),"mrnahl2x":(10.1,0.238)}
rows=[]; ts=[]
for v in VAR:
    for f in sorted(glob.glob(M+f"{v}/df_rows_*.csv")):
        d=pd.read_csv(f,usecols=["time_step","clone_id","replicate","gene_1_mRNA","gene_2_mRNA","gene_3_mRNA","gene_1_protein","gene_2_protein","gene_3_protein"])
        s=d[d.time_step==20]; P=np.log(s.gene_1_protein.values+1); q=pd.qcut(P,10,labels=False,duplicates="drop"); r={}
        for g in (2,3):
            t=s[f"gene_{g}_mRNA"].values; r[f"rho_g1prot_g{g}"]=spearmanr(s.gene_1_protein,t)[0]; r[f"rho_g1mRNA_g{g}"]=spearmanr(s.gene_1_mRNA,t)[0]
            mean=s.groupby(q)[f"gene_{g}_mRNA"].mean().values; lnP=pd.Series(P).groupby(q).mean().values; r[f"slope_g{g}"]=np.polyfit(lnP,np.log(mean),1)[0]; r[f"mean_g{g}"]=t.mean()
            x=np.log1p(s[f"gene_{g}_protein"].values).reshape(-1,1); b1=GaussianMixture(1,random_state=0).fit(x).bic(x); b2=GaussianMixture(2,random_state=0).fit(x).bic(x); r[f"dBIC_2v1_g{g}"]=b1-b2
        r["rho_g2g3"]=spearmanr(s.gene_2_mRNA,s.gene_3_mRNA)[0]
        rows.append(dict(variant=v,file=os.path.basename(f)[-30:],**r))
        # TwinScore, three pairs, 1500 twins + 3000 singletons
        Dd={t:{g:d[d.time_step==t].pivot(index="clone_id",columns="replicate",values=f"gene_{g}_mRNA").values for g in (1,2,3)} for t in (1,20)}
        for k,(y,z) in enumerate([(1,2),(1,3),(2,3)]):
            o=F.analyse({"gene_1":Dd[1][y],"gene_2":Dd[1][z]},{"gene_1":Dd[20][y],"gene_2":Dd[20][z]},np.random.RandomState(hash(f)%10000+k),200)
            ts.append(dict(variant=v,pair=f"{y}-{z}",z_rho=o["z_rho"],z_C=o["z_C"],z_U=o["z_U"],delta=o["delta"],p_delta=o["p_delta"]))
R=pd.DataFrame(rows); R.to_csv(DATA/"variant_measurements.csv",index=False); T=pd.DataFrame(ts); T.to_csv(DATA/"variant_twinscore.csv",index=False)
pd.set_option("display.width",230)
g=R.groupby("variant"); out=pd.DataFrame({"n_sims":g.size(),"mean_mRNA_g2g3":(g.mean_g2.mean()+g.mean_g3.mean())/2,"model_mean":[PRED[v][0] for v in g.size().index],
    "rho_g1prot (g2,g3 avg)":(g.rho_g1prot_g2.mean()+g.rho_g1prot_g3.mean())/2,"model_r":[PRED[v][1] for v in g.size().index],"slope (avg)":(g.slope_g2.mean()+g.slope_g3.mean())/2,
    "rho_g1mRNA (avg)":(g.rho_g1mRNA_g2.mean()+g.rho_g1mRNA_g3.mean())/2,"rho_g2g3":g.rho_g2g3.mean(),"dBIC(2 vs 1 comp) g2":g.dBIC_2v1_g2.mean()}).reindex([v for v in VAR if v in g.size().index]).round(3)
print(out.to_string())
print(); T["directed"]=T.p_delta<=.05; T["dir_ok"]=T.directed&(T.delta>0)
print(T.groupby(["variant","pair"]).agg(z_rho=("z_rho","median"),z_C=("z_C","median"),z_U=("z_U","median"),delta=("delta","median"),dir_ok=("dir_ok","mean"),n=("pair","size")).round(2).to_string())
