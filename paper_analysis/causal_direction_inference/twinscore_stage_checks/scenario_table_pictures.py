"""Scenario call from the shared table (images 1.png): test the shared part rho_s first (= test of C), then z_rho or z_rho_hat_Delta.
 shared ns & z_rho ns -> 1 nothing | shared ns & z_rho sig -> 2 regulation | shared sig & z_U sig -> 3 regulation + heterogeneity | shared sig & z_U ns -> 4 heterogeneity
All tests two-sided |z|>1.96 (single pair, no BH); the shared test is also shown at 2.326."""
import glob, numpy as np, pandas as pd
from _paths import DATA
NAMES={1:"1 nothing",2:"2 regulation",3:"3 reg+heterogeneity",4:"4 heterogeneity"}
def scen(d,thr_shared=1.96):
    sh=d.z_C.abs()>thr_shared; rho=d.z_rho.abs()>1.96; dl=d.z_U.abs()>1.96
    return pd.Series(np.where(~sh&~rho,1,np.where(~sh&rho,2,np.where(sh&dl,3,4))),index=d.index).map(NAMES)
pd.set_option("display.width",250); pd.set_option("display.max_columns",20)
r=pd.concat([pd.read_csv(f) for f in glob.glob(str(DATA/"results/rep_*.csv"))+glob.glob(str(DATA/"results_drift/*_rep_*.csv"))]).reset_index(drop=True)
order=["A_B (single state)","A_B (two states)","A_B (drift)","A_to_B (single state)","A_to_B (two states)","A_to_B (drift)"]
r["call"]=scen(r,1.96); r["call_2326"]=scen(r,2.326)
print("FIGURE-2 + DRIFT types (100 replicates each), shared part tested at 1.96:"); print(pd.crosstab(r.scenario,r.call,normalize="index").loc[order].round(2).to_string())
print("\nshared part tested at 2.326:"); print(pd.crosstab(r.scenario,r.call_2326,normalize="index").loc[order].round(2).to_string())
M=pd.concat([pd.read_csv(f) for f in glob.glob(str(DATA/"results_motif_design1500/*.csv"))]).reset_index(drop=True); M["call"]=scen(M,1.96)
print("\nMOTIF pairs (1,500 twins + 3,000 singletons, 50 replicates):"); print(pd.crosstab([M.motif,M.pair,M.truth],M.call,normalize="index").round(2).to_string())
r.to_csv(DATA/"scenario_table_pictures_figure2.csv",index=False); M.to_csv(DATA/"scenario_table_pictures_motifs.csv",index=False)
