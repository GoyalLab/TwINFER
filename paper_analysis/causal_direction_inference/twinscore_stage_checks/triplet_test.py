"""Triplet test of triplet_test_2.pdf (Fan-out and multiple-state false positives in a twin ranked list) on the figure_4 3-gene motif simulations.
Per replicate and age t: genes 1,2,3 -> twin pairs (clone, 2 sister cells at time_step t), levels = midranks over all twin cells standardised to mean 0, var 1 ("rank-normalised counts").
For each choice of Z (the other two genes are X,Y): orient X,Y so that S_ZX, S_ZY > 0; s_XY=(xy+x'y')/2, c_XY=(xy'+x'y)/2; R=C/S; Rbar=(C_ZX+C_ZY)/(S_ZX+S_ZY); omega=|S_XY|/(S_ZX+S_ZY);
g=sgn(S_XY)[(Rbar s_XY - c_XY) - omega((Rbar s_ZX - c_ZX)+(Rbar s_ZY - c_ZY))]; z_T=mean(g)/(sd(g)/sqrt(N_t)); OR=(R_XY/(1-R_XY))/(Rbar/(1-Rbar)); R*=Rbar/(2-Rbar); z_pow=(Rbar-R*)|S_XY|/(sd(g)/sqrt(N_t)).
p=2(1-Phi(|z_T|)), Benjamini-Hochberg over the 3 triplets of the replicate (all Z choices), q<=0.05. Verdict (Sec. 5): no co-expression (any S not significant at |z|>1.96, R_XY or Rbar outside (0,1));
fan-out (significant, z_T<0); regulation (significant, z_T>0, OR<1/2); heterogeneity (not significant and z_pow>=3); else no verdict."""
import argparse, glob, os, re, numpy as np, pandas as pd, multiprocessing as mp
from scipy.stats import norm
from u_time_pool import rk
from motif_replicates_zscores import files, MOT
from _paths import DATA
# [2026-10-13 was AGES=(12,20); the user wants 1 h and 20 h]
AGES=(1,20)
def lev(x):
    r=rk(x.ravel()).reshape(x.shape); r=(r-r.mean())/r.std(); return r
def sc(a,b): return 0.5*(a[:,0]*b[:,0]+a[:,1]*b[:,1]), 0.5*(a[:,0]*b[:,1]+a[:,1]*b[:,0])
def triplet(L,Z,X,Y):
    z,x,y=L[Z].copy(),L[X].copy(),L[Y].copy(); n=len(z)
    sZX,cZX=sc(z,x); sZY,cZY=sc(z,y)
    if sZX.mean()<0: x=-x
    if sZY.mean()<0: y=-y
    sZX,cZX=sc(z,x); sZY,cZY=sc(z,y); sXY,cXY=sc(x,y)
    SZX,SZY,SXY,CZX,CZY,CXY=sZX.mean(),sZY.mean(),sXY.mean(),cZX.mean(),cZY.mean(),cXY.mean()
    Rb=(CZX+CZY)/(SZX+SZY); om=abs(SXY)/(SZX+SZY); Rxy=CXY/SXY
    g=np.sign(SXY)*((Rb*sXY-cXY)-om*((Rb*sZX-cZX)+(Rb*sZY-cZY))); sd=g.std(ddof=1)/np.sqrt(n); zT=g.mean()/sd
    Rs=Rb/(2-Rb); zp=(Rb-Rs)*abs(SXY)/sd
    zS=[a.mean()/(a.std(ddof=1)/np.sqrt(n)) for a in (sXY,sZX,sZY)]
    OR=(Rxy/(1-Rxy))/(Rb/(1-Rb)) if 0<Rxy<1 and 0<Rb<1 else np.nan
    return dict(S_XY=SXY,S_ZX=SZX,S_ZY=SZY,R_XY=Rxy,R_ZX=CZX/SZX,R_ZY=CZY/SZY,Rbar=Rb,zT=zT,OR=OR,zpow=zp,p=2*(1-norm.cdf(abs(zT))),zS_min=min(abs(v) for v in zS),coexp=(min(abs(v) for v in zS)>1.96) and 0<Rxy<1 and 0<Rb<1)
def bh(p):
    p=np.asarray(p); m=len(p); o=np.argsort(p); q=np.empty(m); q[o]=np.minimum.accumulate((p[o]*m/np.arange(1,m+1))[::-1])[::-1]; return np.minimum(q,1)
def verdict(t,q):
    if not t["coexp"]: return "no co-expression"
    if q<=0.05: return "fan-out" if t["zT"]<0 else ("regulation" if t["OR"]<0.5 else "no verdict")
    return "heterogeneity" if t["zpow"]>=3 else "no verdict"
def run(a):
    m,r,fn,out=a; dest=f"{out}/{m}_rep_{r:04d}.csv"
    if os.path.exists(dest): return
    d=pd.read_csv(fn,usecols=["time_step","clone_id","replicate","gene_1_mRNA","gene_2_mRNA","gene_3_mRNA"]); rows=[]
    for age in AGES:
        s=d[d.time_step==age]; L={g:lev(s.pivot(index="clone_id",columns="replicate",values=f"gene_{g}_mRNA").values.astype(float)) for g in (1,2,3)}
        ts=[]
        for Z in (1,2,3):
            X,Y=sorted({1,2,3}-{Z}); t=triplet(L,Z,X,Y); t.update(Z=Z,X=X,Y=Y); ts.append(t)
        q=bh([t["p"] for t in ts])
        for t,qq in zip(ts,q): t.update(motif=m,rep=r,age=age,q=qq,verdict=verdict(t,qq),verdict_z3=("fan-out" if t["zT"]<-3 else "regulation" if (t["zT"]>3 and t["OR"]<0.5) else "other") if t["coexp"] else "no co-expression"); rows.append(t)
    pd.DataFrame(rows).to_csv(dest,index=False)
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--nreps",type=int,default=100); ap.add_argument("--out",required=True); ap.add_argument("--jobs",type=int,default=12); a=ap.parse_args()
    F=files(); jobs=[(m,r,F[(m,r)],a.out) for m in MOT for r in range(a.nreps) if (m,r) in F]; os.makedirs(a.out,exist_ok=True); print(len(jobs),"files",flush=True)
    with mp.Pool(a.jobs) as p: p.map(run,jobs)
    print("done")
