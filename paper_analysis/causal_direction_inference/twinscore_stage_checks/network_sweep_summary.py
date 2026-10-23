"""Summarise network_sweep_scenarios.py output: scenario table (pictures) by pair class, direction vs the true edge sign, PDF-style calls, AUPRC x base rate."""
import sys, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
from _paths import DATA
f=sys.argv[1] if len(sys.argv)>1 else "network_sweep_final_e5_pairs.csv"
R=pd.read_csv(DATA/f); NAMES={1:"1 nothing",2:"2 regulation",3:"3 reg+het",4:"4 het"}; cols=list(NAMES.values()); order=["direct edge","indirect","co-regulated","unrelated"]
def scen(d,thr=1.96):
    sh=d.z_C.abs()>thr; rho=d.z_rho.abs()>1.96; dl=d.z_U.abs()>1.96
    return pd.Series(np.where(~sh&~rho,1,np.where(~sh&rho,2,np.where(sh&dl,3,4))),index=d.index).map(NAMES)
R["call"]=scen(R); R["called"]=(R.z_rho.abs()>1.96)&(R.z_U.abs()>1.96); R["directed"]=R.p_delta<=.05
n=R.groupby("cls").size().reindex(order); t=pd.crosstab(R.cls,R.call).reindex(index=order,columns=cols,fill_value=0)
print(f"file {f}: {R.groupby(['network','sim']).ngroups} simulation files, {len(R)} pairs"); print("\nSCENARIO TABLE (pictures): count / pairs in class")
for c in order: t.loc[c]=t.loc[c].astype(int)
print(pd.DataFrame({k:[f"{t.loc[c,k]}/{n[c]}" for c in order] for k in cols},index=order).to_string())
e=R[R.cls=="direct edge"]; per=e.groupby("network").size().iloc[0]
print(f"\nDirect edges by network (out of {per} each):"); print(pd.crosstab(e.network,e.call).reindex(columns=cols,fill_value=0).astype(str).add(f"/{per}").to_string())
e=e.assign(sign_ok=e.directed&(np.sign(e.delta)==e.sign),pdf_x=e.directed&(np.sign(e.delta)==np.sign(e.U)),reg=e.called&e.directed)
e["reg_pdf_ok"]=e.reg&e.pdf_x
print(f"\nDIRECT EDGES ({len(e)}; sign counts {e.sign.value_counts().to_dict()}): co-varies={int((e.z_rho.abs()>1.96).sum())} inherited(z_C>2.326)={int((e.z_C>2.326).sum())} called={int(e.called.sum())} directed={int(e.directed.sum())} "
      f"delta sign = edge sign={int(e.sign_ok.sum())} PDF rule names x={int(e.pdf_x.sum())} regulation(called&directed)={int(e.reg.sum())} regulation with right regulator={int(e.reg_pdf_ok.sum())}")
ne=R[R.cls!="direct edge"]; print("NON-EDGES directed (p_delta<=.05):",{c:f"{int(g.directed.sum())}/{len(g)}" for c,g in ne.groupby("cls")},"| called&directed:",int((ne.called&ne.directed).sum()),"of",len(ne),
      "| scenario 2 or 3 (regulation):",int(ne.call.isin([NAMES[2],NAMES[3]]).sum()),"of",len(ne))
res=[]
for (net,sim),d in R.groupby(["network","sim"]):
    y=(d.cls=="direct edge").astype(int).values; b=y.mean()
    for k,v in (("|z_U|",d.z_U.abs()),("|z_rho|",d.z_rho.abs()),("|delta|",d.delta.abs())): res.append(dict(key=k,v=average_precision_score(y,v)/b))
print("\nEdge recovery AUPRC x base rate (1.0 = chance), mean [min-max] over files:"); print(pd.DataFrame(res).groupby("key").v.agg(["mean","min","max"]).round(2).to_string())
