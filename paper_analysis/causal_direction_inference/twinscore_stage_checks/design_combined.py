"""One clone set serves both tests: per clone 1 early cell (t1) + 2 late cells (t2). z_U from the late sister pair (N clones as twins),
direction (delta permutation test) from early cell vs late sister. N = 1500, 3000, 6000 clones."""
import argparse, glob, os, numpy as np, pandas as pd
from figure2_1000_zscores import NCL,G,SCEN,find
import u_time_pool; u_time_pool.TIMES[:]=[1,20]
from u_time_pool import loadt, zU
from dir_timescan import rk, std
T1,T2,DP=1,20,500
def run_rep(a):
    rep,out=a; dest=f"{out}/rep_{rep:03d}.csv"
    if os.path.exists(dest): return rep
    rows=[]; jobs=[(name,cfgs,None,si) for si,(name,cfgs) in enumerate(SCEN)]+[(f"{n} (drift)",None,n,7) for n in ("A_B","A_to_B")]
    for name,cfgs,net,si in jobs:
        rng=np.random.RandomState(rep*10+si)
        if net: Dd=loadt(glob.glob(f"/projects/b1255/hzhang/TwINFER_KA/simulation_data/drift_lowhigh_Kramp/{net}/df_*_ncells_6000_{net}_drift_lowhigh_Kramp_rep_{rep}_*.csv")[0])
        elif len(cfgs)==1: Dd=loadt(find(cfgs[0],rep)[0])
        else:
            S=[loadt(find(c,rep)[0]) for c in cfgs]; sa,sb=rng.choice(NCL,3000,replace=False),rng.choice(NCL,3000,replace=False)
            Dd={t:{g:np.vstack([S[0][t][g][sa],S[1][t][g][sb]]) for g in G} for t in (T1,T2)}
        perm=rng.permutation(NCL); pickall=rng.randint(0,2,NCL)
        for N in (1500,3000,6000):
            c=perm[:N]; pk=pickall[:N]; zu=zU(Dd[T2]["gene_1"][c],Dd[T2]["gene_2"][c])
            ax,ay=[std(rk(Dd[T1][g][c,pk])) for g in G]; bx,by=[std(rk(Dd[T2][g][c,1-pk])) for g in G]
            f=ax@by; r=ay@bx; delta=f-r
            nd=np.array([ax@by[p]-ay@bx[p] for p in (rng.permutation(N) for _ in range(DP))]); p=(1+np.sum(abs(nd)>=abs(delta)))/(DP+1)
            rows.append(dict(rep=rep,scenario=name,N=N,z_U=zu,delta=delta,p_delta=p,z_xy=f*np.sqrt(N-1),z_yx=r*np.sqrt(N-1)))
    pd.DataFrame(rows).to_csv(dest,index=False); return rep
if __name__=="__main__":
    import multiprocessing as mp
    ap=argparse.ArgumentParser();ap.add_argument("--reps",default="0-99");ap.add_argument("--out",required=True);ap.add_argument("--jobs",type=int,default=12);a=ap.parse_args()
    lo,hi=map(int,a.reps.split("-"));os.makedirs(a.out,exist_ok=True)
    with mp.Pool(a.jobs) as p:
        for r in p.imap_unordered(run_rep,[(r,a.out) for r in range(lo,hi+1)]): print(r,flush=True)
