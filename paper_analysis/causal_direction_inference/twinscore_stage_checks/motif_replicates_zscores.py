"""TwinScore z-scores on the figure_4 3-gene motif replicates (Fan_out, Feed_forward, Mutual_regulation; 1k_per_motif_v2).
Truth: Fan_out 1->2,1->3 | Feed_forward 1->2,1->3,2->3 | Mutual 1->2,1->3,2->3,3->2. Pair (2,3) in Fan_out = shared driver (gene_1), no edge.
Design per replicate (same clones for both tests): 1 early cell (t1=1 h) + 2 late cells (t2=20 h); z_U from the late sister pair, delta permutation test from early cell vs late sister.
Uses the first --nreps replicates (lowest indices) that are finished for ALL motifs; N clones = 3000 and 6000."""
import argparse, glob, os, re, numpy as np, pandas as pd
from u_time_pool import zU
from dir_timescan import rk, std
from motif_shared_driver import zC, TRUTH, PAIRS
SIM="/projects/b1255/hzhang/TwINFER_KA/simulation_data/figure_4/1k_per_motif_v2/"
MOT=["Fan_out","Feed_forward","Mutual_regulation"]; T1,T2,DP=1,20,500
def files():
    out={}
    for c,m in enumerate(MOT):
        for f in glob.glob(SIM+f"df_rows_*_ncells_6000_{m}_{c}_*_*.csv"):
            r=int(re.search(rf"_{m}_{c}_(\d+)_[0-9a-f]{{8}}\.csv$",f).group(1)); out[(m,r)]=f
    return out
def load(f):
    d=pd.read_csv(f,usecols=["time_step","clone_id","replicate","gene_1_mRNA","gene_2_mRNA","gene_3_mRNA"]); o={}
    for t in (T1,T2):
        s=d[d.time_step==t]; o[t]={g:s.pivot(index="clone_id",columns="replicate",values=f"gene_{g}_mRNA").values for g in (1,2,3)}
    return o
def zRho(X,Y):
    """Stage I on twin pairs only (n_c=2, no singletons): weighted Pearson of midranks; sigma^2=(0.5+0.5*hx*hy)/N, h floored at 0."""
    n=len(X); rx=rk(X.ravel()).reshape(n,2); ry=rk(Y.ravel()).reshape(n,2)
    from u_time_pool import rk as rkm
    rx=rkm(X.ravel()).reshape(n,2); ry=rkm(Y.ravel()).reshape(n,2); a=rx-rx.mean(); b=ry-ry.mean()
    rho=(a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum())
    h=lambda r:(lambda M:max(0.,(r[:,0]*r[:,1]).mean()/M))(((r**2).sum(1)/2).mean())
    return rho/np.sqrt((0.5+0.5*h(a)*h(b))/n)
def run(a):
    m,r,f,out=a; dest=f"{out}/{m}_rep_{r:04d}.csv"
    if os.path.exists(dest): return m,r
    D=load(f); rng=np.random.RandomState(r); NCL=D[T1][1].shape[0]; perm=rng.permutation(NCL); pickall=rng.randint(0,2,NCL); rows=[]
    for N in (3000,6000):
        c=perm[:N]; pk=pickall[:N]
        for (i,j) in PAIRS:
            z_u=zU(D[T2][i][c],D[T2][j][c]); S,C,z_c=zC(D[T2][i][c],D[T2][j][c]); z_r=zRho(D[T2][i][c],D[T2][j][c])  # [2026-10-04 added z_rho]
            ax,ay=std(rk(D[T1][i][c,pk])),std(rk(D[T1][j][c,pk])); bx,by=std(rk(D[T2][i][c,1-pk])),std(rk(D[T2][j][c,1-pk]))
            fw=ax@by; rv=ay@bx; delta=fw-rv; nd=np.array([ax@by[p]-ay@bx[p] for p in (rng.permutation(N) for _ in range(DP))])
            rows.append(dict(motif=m,rep=r,pair=f"{i}-{j}",truth=TRUTH[m][(i,j)],N=N,z_rho=z_r,S=S,C=C,z_C=z_c,z_U=z_u,delta=delta,p_delta=(1+np.sum(abs(nd)>=abs(delta)))/(DP+1),z_xy=fw*np.sqrt(N-1),z_yx=rv*np.sqrt(N-1)))
    pd.DataFrame(rows).to_csv(dest,index=False); return m,r
if __name__=="__main__":
    import multiprocessing as mp
    ap=argparse.ArgumentParser();ap.add_argument("--nreps",type=int,default=50);ap.add_argument("--out",required=True);ap.add_argument("--jobs",type=int,default=12);a=ap.parse_args()
    F=files(); common=sorted(set.intersection(*[{r for (mm,r) in F if mm==m} for m in MOT]))[:a.nreps]
    print("finished per motif:",{m:sum(1 for (mm,r) in F if mm==m) for m in MOT},"| common replicates used:",len(common),common[:3],"...",common[-1:],flush=True)
    os.makedirs(a.out,exist_ok=True)
    with mp.Pool(a.jobs) as p:
        for x in p.imap_unordered(run,[(m,r,F[(m,r)],a.out) for m in MOT for r in common]): pass
    print("done",flush=True)
