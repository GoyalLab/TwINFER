import os
import sys, numpy as np, pandas as pd
exec(open(os.path.dirname(os.path.abspath(__file__))+"/iso_mel.py").read().split("G=list(genes)")[0])
G=list(genes); cl=clone[m]; cid=df.cell_id.to_numpy()[m]; MINDET=100
def resid_det(names):
    out=Vp.copy(); Z=np.column_stack([np.ones(len(Vp))]+[cov[n] for n in names]); n=0
    for j in range(Vp.shape[1]):
        det=Vp[:,j]>0
        if det.sum()<MINDET: continue
        b=np.linalg.lstsq(Z[det],Vp[det,j],rcond=None)[0]; adj=Vp[det,j]-Z[det]@b; out[det,j]=adj-adj.min()+1e-3; n+=1
    print("   det-regressed genes",n,"of",Vp.shape[1],flush=True); return out
ALL=["logtot","S","G2M","IEG"]
variants={"none":lambda:Vp,"all4":lambda:resid(ALL),"det_all4":lambda:resid_det(ALL)}
rows=[(x,y) for x in range(len(G)) for y in range(len(G)) if x!=y]
out=pd.DataFrame(dict(gene_1=[G[x] for x,_ in rows],gene_2=[G[y] for _,y in rows]))
xi,yi=np.array([x for x,_ in rows]),np.array([y for _,y in rows])
for name,f in variants.items():
    smp=T.Sample(f(),cl,cid,np.random.default_rng(0)); r=T.steps(smp,smp.null_sds("analytic"))
    ok=r["stage1"]&r["called"]; out["R_"+name]=np.where(ok,r["R"],np.nan)[xi,yi]; out["lam_"+name]=r["lam"][xi,yi]
    print(name,"called",int(ok[~np.eye(len(G),dtype=bool)].sum()),"of",len(rows),flush=True)
out.to_csv(sys.argv[1]+"/scores_mel.csv",index=False); print("done")
