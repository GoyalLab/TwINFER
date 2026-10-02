# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import sys, numpy as np, pandas as pd
exec(open(sys.argv[1]+"/iso.py").read().split("G=list(genes)")[0])
G=list(genes); cl=clone[m]; cid=df.cell_id.to_numpy()[m]
MINDET=100
def resid_det(names):
    out=Vp.copy(); Z=np.column_stack([np.ones(len(Vp))]+[cov[n] for n in names]); nreg=0
    for j in range(Vp.shape[1]):
        det=Vp[:,j]>0
        if det.sum()<MINDET: continue
        b=np.linalg.lstsq(Z[det],Vp[det,j],rcond=None)[0]
        adj=Vp[det,j]-Z[det]@b; out[det,j]=adj-adj.min()+1e-3; nreg+=1   # zeros stay 0 and remain below every detected value
    print("   genes regressed (>= %d detected cells): %d of %d"%(MINDET,nreg,Vp.shape[1]),flush=True); return out
variants={"none":[],"IEG":["IEG"],"all4":["logtot","S","G2M","IEG"]}
rows=[(x,y) for x in range(len(G)) for y in range(len(G)) if x!=y]
out=pd.DataFrame(dict(gene_1=[G[x] for x,_ in rows],gene_2=[G[y] for _,y in rows]))
xi,yi=np.array([x for x,_ in rows]),np.array([y for _,y in rows])
ji,ii=G.index("JUN"),G.index("IL11")
for name,ns in variants.items():
    if name=="none": continue
    smp=T.Sample(resid_det(ns),cl,cid,np.random.default_rng(0)); r=T.steps(smp,smp.null_sds("analytic"))
    ok=r["stage1"]&r["called"]; out["R_det_"+name]=np.where(ok,r["R"],np.nan)[xi,yi]; out["lam_det_"+name]=r["lam"][xi,yi]
    print(name,"JUN->IL11 rho %.3f h %.2f lam %.2f | called %d"%(smp.RHO[ji,ii],r["h"][ji,ii],r["lam"][ji,ii],int(ok[~np.eye(len(G),dtype=bool)].sum())),flush=True)
out.to_csv(sys.argv[1]+"/scores_det.csv",index=False)
