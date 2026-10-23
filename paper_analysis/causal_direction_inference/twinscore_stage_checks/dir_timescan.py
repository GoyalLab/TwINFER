"""Direction signal (rho_dagger x->y, y->x, delta, permutation p) vs the pair of sampling times (t1 in sample A, t2 in sample B).
3,000 shared clones: one cell to A at t1, the sister cell to B at t2. Ranks over the 3,000 shared cells in each sample."""
import argparse, glob, os, numpy as np, pandas as pd
from figure2_1000_zscores import NCL,G,SCEN,find
PAIRS=[(1,20),(5,20),(10,20),(15,20),(1,48),(10,48)]; TIMES=sorted({t for p in PAIRS for t in p}); NSH=3000; D=500
from u_time_pool import loadt as _l
import u_time_pool; u_time_pool.TIMES[:]=TIMES
def rk(v):
    u,inv,c=np.unique(v,return_inverse=True,return_counts=True); return (np.cumsum(c)-c+(c+1)/2)[inv]
def std(r): r=r-r.mean(); return r/np.sqrt((r*r).sum())
def run_rep(a):
    rep,out=a; dest=f"{out}/rep_{rep:03d}.csv"
    if os.path.exists(dest): return rep
    rows=[]; jobs=[(name,cfgs,None,si) for si,(name,cfgs) in enumerate(SCEN)]+[(f"{n} (drift)",None,n,7) for n in ("A_B","A_to_B")]
    for name,cfgs,net,si in jobs:
        rng=np.random.RandomState(rep*10+si)
        if net: Dd=_l(glob.glob(f"/projects/b1255/hzhang/TwINFER_KA/simulation_data/drift_lowhigh_Kramp/{net}/df_*_ncells_6000_{net}_drift_lowhigh_Kramp_rep_{rep}_*.csv")[0])
        elif len(cfgs)==1: Dd=_l(find(cfgs[0],rep)[0])
        else:
            S=[_l(find(c,rep)[0]) for c in cfgs]; sa,sb=rng.choice(NCL,3000,replace=False),rng.choice(NCL,3000,replace=False)
            Dd={t:{g:np.vstack([S[0][t][g][sa],S[1][t][g][sb]]) for g in G} for t in TIMES}
        sh=rng.permutation(NCL)[:NSH]; pick=rng.randint(0,2,NSH); ix=np.arange(NSH)
        for t1,t2 in PAIRS:
            ax,ay=[std(rk(Dd[t1][g][sh,pick])) for g in G]; bx,by=[std(rk(Dd[t2][g][sh,1-pick])) for g in G]
            f=ax@by; r=ay@bx; delta=f-r
            nd=np.array([ax@by[p]-ay@bx[p] for p in (rng.permutation(NSH) for _ in range(D))])
            rows.append(dict(rep=rep,scenario=name,t1=t1,t2=t2,rho_xy=f,rho_yx=r,delta=delta,p_delta=(1+np.sum(abs(nd)>=abs(delta)))/(D+1),z_xy=f*np.sqrt(NSH-1),z_yx=r*np.sqrt(NSH-1)))
    pd.DataFrame(rows).to_csv(dest,index=False); return rep
if __name__=="__main__":
    import multiprocessing as mp
    ap=argparse.ArgumentParser();ap.add_argument("--reps",default="0-99");ap.add_argument("--out",required=True);ap.add_argument("--jobs",type=int,default=12);a=ap.parse_args()
    lo,hi=map(int,a.reps.split("-"));os.makedirs(a.out,exist_ok=True)
    with mp.Pool(a.jobs) as p:
        for r in p.imap_unordered(run_rep,[(r,a.out) for r in range(lo,hi+1)]): print(r,flush=True)
