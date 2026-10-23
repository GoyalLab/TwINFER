"""Shared-driver check on the figure_4 3-gene motifs (Fan_out, Feed_forward, Mutual_regulation), one replicate per motif.
Truth (connectivity_matrix): Fan_out 1->2,1->3 | Feed_forward 1->2,1->3,2->3 | Mutual 1->2,1->3,2->3,3->2.
Fan_out pair (2,3) = no edge, shared driver gene_1. Design: per clone 1 cell at t1=1h + 2 cells at t2=20h; z_U from the late sister pair,
delta test from early cell vs late sister. N random clones per redraw (resampling only -> subsampling variability, not replicate variability)."""
import argparse, glob, os, numpy as np, pandas as pd
from u_time_pool import rk as rk_mid, zU
from dir_timescan import rk, std
SIM="/projects/b1255/hzhang/TwINFER_KA/simulation_data/figure_4/1k_per_motif/"
MOT=["Fan_out","Feed_forward","Mutual_regulation"]; PAIRS=[(1,2),(1,3),(2,3)]
TRUTH={"Fan_out":{(1,2):"1->2",(1,3):"1->3",(2,3):"none (shared driver 1)"},"Feed_forward":{(1,2):"1->2",(1,3):"1->3",(2,3):"2->3 (+shared driver 1)"},
       "Mutual_regulation":{(1,2):"1->2",(1,3):"1->3",(2,3):"2<->3 (+shared driver 1)"}}
T1,T2,DP=1,20,500
def load(m):
    f=[x for x in glob.glob(SIM+f"df_rows_*_ncells_6000_{m}_*.csv") if "before_division" not in x][0]
    d=pd.read_csv(f,usecols=["time_step","clone_id","replicate","gene_1_mRNA","gene_2_mRNA","gene_3_mRNA"]); out={}
    for t in (T1,T2):
        s=d[d.time_step==t]; out[t]={g:s.pivot(index="clone_id",columns="replicate",values=f"gene_{g}_mRNA").values for g in (1,2,3)}
    return out
def zC(X,Y):
    n=len(X); rx=rk_mid(X.ravel()).reshape(n,2); ry=rk_mid(Y.ravel()).reshape(n,2); rx=rx-rx.mean(); ry=ry-ry.mean()
    M=lambda a,b:((a*b).sum(1)/2).mean(); nrm=np.sqrt(M(rx,rx)*M(ry,ry))
    S=((rx[:,0]*ry[:,0]+rx[:,1]*ry[:,1])/2).mean()/nrm; C=((rx[:,0]*ry[:,1]+rx[:,1]*ry[:,0])/2).mean()/nrm
    hx=max(0,(rx[:,0]*rx[:,1]).mean()/M(rx,rx)); hy=max(0,(ry[:,0]*ry[:,1]).mean()/M(ry,ry))
    return S,C,np.sign(S)*C/np.sqrt((1+S*S+hx*hy)/(2*n))
def run(a):
    m,seed,out=a; dest=f"{out}/{m}_redraw_{seed:03d}.csv"
    if os.path.exists(dest): return m,seed
    D=load(m); rng=np.random.RandomState(1000+seed); NCL=D[T1][1].shape[0]; perm=rng.permutation(NCL); pickall=rng.randint(0,2,NCL); rows=[]
    for N in (3000,6000):
        c=perm[:N]; pk=pickall[:N]
        for (i,j) in PAIRS:
            z_u=zU(D[T2][i][c],D[T2][j][c]); S,C,z_c=zC(D[T2][i][c],D[T2][j][c])
            ax,ay=std(rk(D[T1][i][c,pk])),std(rk(D[T1][j][c,pk])); bx,by=std(rk(D[T2][i][c,1-pk])),std(rk(D[T2][j][c,1-pk]))
            f=ax@by; r=ay@bx; delta=f-r; nd=np.array([ax@by[p]-ay@bx[p] for p in (rng.permutation(N) for _ in range(DP))])
            rows.append(dict(motif=m,pair=f"{i}-{j}",truth=TRUTH[m][(i,j)],N=N,redraw=seed,S=S,C=C,z_C=z_c,z_U=z_u,delta=delta,p_delta=(1+np.sum(abs(nd)>=abs(delta)))/(DP+1),z_xy=f*np.sqrt(N-1),z_yx=r*np.sqrt(N-1)))
    pd.DataFrame(rows).to_csv(dest,index=False); return m,seed
if __name__=="__main__":
    import multiprocessing as mp
    ap=argparse.ArgumentParser();ap.add_argument("--redraws",type=int,default=100);ap.add_argument("--out",required=True);ap.add_argument("--jobs",type=int,default=12);a=ap.parse_args()
    os.makedirs(a.out,exist_ok=True)
    with mp.Pool(a.jobs) as p:
        for r in p.imap_unordered(run,[(m,s,a.out) for m in MOT for s in range(a.redraws)]): print(r,flush=True)
