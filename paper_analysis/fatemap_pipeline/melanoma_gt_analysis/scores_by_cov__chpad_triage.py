# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import sys, numpy as np, pandas as pd
exec(open(sys.argv[1]+"/iso.py").read().split("G=list(genes)")[0])   # reuse data/covariate setup
G=list(genes); cl=clone[m]; cid=df.cell_id.to_numpy()[m]
variants={"none":[],"logtot":["logtot"],"S":["S"],"G2M":["G2M"],"IEG":["IEG"],"all4":["logtot","S","G2M","IEG"]}
rows=[(x,y) for x in range(len(G)) for y in range(len(G)) if x!=y]
out=pd.DataFrame(dict(gene_1=[G[x] for x,_ in rows],gene_2=[G[y] for _,y in rows]))
xi,yi=np.array([x for x,_ in rows]),np.array([y for _,y in rows])
for name,ns in variants.items():
    smp=T.Sample(resid(ns),cl,cid,np.random.default_rng(0)); r=T.steps(smp,smp.null_sds("analytic"))
    ok=r["stage1"]&r["called"]; out["R_"+name]=np.where(ok,r["R"],np.nan)[xi,yi]; out["lam_"+name]=r["lam"][xi,yi]; out["zstar_"+name]=r["zstar"][xi,yi]
    print(name,"called",int(ok[~np.eye(len(G),dtype=bool)].sum()),flush=True)
out.to_csv(sys.argv[1]+"/scores_by_cov.csv",index=False)
