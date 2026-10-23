"""z_U (rho_hat_delta*sqrt(N_t), twin ranks) vs sampling time, and Stouffer pooling over time points using DISJOINT twin clones.
Per time t: 1500 twin clones (ranks over the 3000 twin cells). Also N_t=6000 (all clones as twins) as the sqrt(N) reference."""
import argparse, glob, os, numpy as np, pandas as pd
from figure2_1000_zscores import NCL,G,SCEN,find
TIMES=[1,2,4,6,8,10,15,20,30,40,48]; NT=1500
def loadt(path):
    d=pd.read_csv(path,usecols=["time_step","gene_1_mRNA","gene_2_mRNA","replicate","clone_id"]); d=d[d.time_step.isin(TIMES)]
    return {t:{g:s.pivot(index="clone_id",columns="replicate",values=f"{g}_mRNA").values for g in G} for t,s in d.groupby("time_step")}
def rk(v):
    u,inv,c=np.unique(v,return_inverse=True,return_counts=True); return (np.cumsum(c)-c+(c+1)/2)[inv]
def zU(X,Y):
    n=len(X); rx=rk(X.ravel()).reshape(n,2); ry=rk(Y.ravel()).reshape(n,2); rx=rx-rx.mean(); ry=ry-ry.mean()
    dx=rx[:,0]-rx[:,1]; dy=ry[:,0]-ry[:,1]; return (dx*dy).sum()/np.sqrt((dx**2).sum()*(dy**2).sum())*np.sqrt(n)
def run_rep(a):
    rep,out=a; dest=f"{out}/rep_{rep:03d}.csv"
    if os.path.exists(dest): return rep
    rows=[]; jobs=[(name,cfgs,None,si) for si,(name,cfgs) in enumerate(SCEN)]+[(f"{n} (drift)",None,n,7) for n in ("A_B","A_to_B")]
    for name,cfgs,net,si in jobs:
        rng=np.random.RandomState(rep*10+si)
        if net: D=loadt(glob.glob(f"/projects/b1255/hzhang/TwINFER_KA/simulation_data/drift_lowhigh_Kramp/{net}/df_*_ncells_6000_{net}_drift_lowhigh_Kramp_rep_{rep}_*.csv")[0])
        elif len(cfgs)==1: D=loadt(find(cfgs[0],rep)[0])
        else:
            S=[loadt(find(c,rep)[0]) for c in cfgs]; sa,sb=rng.choice(NCL,3000,replace=False),rng.choice(NCL,3000,replace=False)
            D={t:{g:np.vstack([S[0][t][g][sa],S[1][t][g][sb]]) for g in G} for t in TIMES}
        perm=rng.permutation(NCL); o=dict(rep=rep,scenario=name)
        for t in TIMES:
            o[f"z_t{t}"]=zU(D[t]["gene_1"][perm[:NT]],D[t]["gene_2"][perm[:NT]]); o[f"z6000_t{t}"]=zU(D[t]["gene_1"],D[t]["gene_2"])
        grp=lambda k:perm[k*NT:(k+1)*NT]
        zs=lambda t,k:zU(D[t]["gene_1"][grp(k)],D[t]["gene_2"][grp(k)])
        o["pool_1_20"]=(zs(1,0)+zs(20,1))/np.sqrt(2)
        o["pool_10_20_30_48"]=sum(zs(t,k) for k,t in enumerate((10,20,30,48)))/2
        o["pool_4x20"]=sum(zs(20,k) for k in range(4))/2   # same time, 4 disjoint clone groups (= 6000 twins), reference
        rows.append(o)
    pd.DataFrame(rows).to_csv(dest,index=False); return rep
if __name__=="__main__":
    import multiprocessing as mp
    ap=argparse.ArgumentParser();ap.add_argument("--reps",default="0-99");ap.add_argument("--out",required=True);ap.add_argument("--jobs",type=int,default=12);a=ap.parse_args()
    lo,hi=map(int,a.reps.split("-"));os.makedirs(a.out,exist_ok=True)
    with mp.Pool(a.jobs) as p:
        for r in p.imap_unordered(run_rep,[(r,a.out) for r in range(lo,hi+1)]): print(r,flush=True)
