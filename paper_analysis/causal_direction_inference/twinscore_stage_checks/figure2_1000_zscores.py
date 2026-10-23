"""Per-replicate TwinScore z-scores (PDF Sec. 4,7,8,11) for four Figure-2 scenarios.
Design (same as the interactive runs): t1=1h sample A, t2=20h sample B (scored).
B: 1,500 twin clones + 3,000 singletons; A: a different 1,500 twins + 3,000 singletons;
the other 3,000 clones give one cell to A and the other to B (cross-sample twins).
Two-state scenarios: 3,000 random clones from the low_k_on file + 3,000 from the high_k_on file
(replicates of matching number), no regard to source afterwards."""
import argparse, glob, os, re, numpy as np, pandas as pd
ROOT="/projects/b1255/yscher/TranscriptomicDistance/simulation_data/figure_2_simulations_1000/"
T1,T2=1,20; NT,NSH=1500,3000; NB=NT+NSH; NCL=6000; G=["gene_1","gene_2"]
W=np.concatenate([np.full(2*NT,.5),np.ones(NSH)])
def wrank(v,w):
    u,inv=np.unique(v,return_inverse=True); sw=np.bincount(inv,weights=w); return (np.cumsum(sw)-sw+.5*sw)[inv]
def wcorr(a,b,w=None):
    if w is None: w=np.ones(len(a))
    a=a-(w*a).sum()/w.sum(); b=b-(w*b).sum()/w.sum(); return (w*a*b).sum()/np.sqrt((w*a*a).sum()*(w*b*b).sum())
def s2(hx,hy): return (NT*(.5+.5*hx*hy)+NSH)/NB**2
def h_of(r2):
    r=r2-r2.mean(); M=((r**2).sum(1)/2).mean(); return max(0.,(r[:,0]*r[:,1]).mean()/M)
def find(cfg,rep):
    pat=re.compile(rf"^df_rows_.*_ncells_6000_{cfg}_rep_{rep}_[0-9a-f]{{8}}\.csv$")
    fs=[f for f in os.listdir(ROOT+cfg) if pat.match(f)]
    if not fs: raise FileNotFoundError(f"{cfg} rep {rep}")
    fs.sort(key=lambda f: os.path.getmtime(ROOT+cfg+"/"+f)); return ROOT+cfg+"/"+fs[-1],len(fs)
def load(path):
    d=pd.read_csv(path,usecols=["time_step","gene_1_mRNA","gene_2_mRNA","replicate","clone_id"])
    out={}
    for t in (T1,T2):
        s=d[d.time_step==t]; out[t]={g:s.pivot(index='clone_id',columns='replicate',values=f"{g}_mRNA").values for g in G}
    return out
def analyse(P1,P2,rng,D):
    perm=rng.permutation(NCL); at,bt,sh=perm[:NT],perm[NT:2*NT],perm[2*NT:]; pick=rng.randint(0,2,NSH)
    RA={};RB={}
    for g in G:
        RA[g]=wrank(np.concatenate([P1[g][at].ravel(),P1[g][sh,pick]]),W)
        RB[g]=wrank(np.concatenate([P2[g][bt].ravel(),P2[g][sh,1-pick]]),W)
    hx=h_of(RB["gene_1"][:2*NT].reshape(NT,2)); hy=h_of(RB["gene_2"][:2*NT].reshape(NT,2))
    rho=wcorr(RB["gene_1"],RB["gene_2"],W); z_rho=rho/np.sqrt(s2(hx,hy))
    rx=RB["gene_1"][:2*NT].reshape(NT,2); ry=RB["gene_2"][:2*NT].reshape(NT,2); rx=rx-rx.mean(); ry=ry-ry.mean()
    nrm=np.sqrt(((rx**2).sum(1)/2).mean()*((ry**2).sum(1)/2).mean())
    Sv=((rx[:,0]*ry[:,0]+rx[:,1]*ry[:,1])/2).mean()/nrm; Cv=((rx[:,0]*ry[:,1]+rx[:,1]*ry[:,0])/2).mean()/nrm; Uv=Sv-Cv
    sC=np.sqrt((1+Sv**2+hx*hy)/(2*NT)); z_C=np.sign(Sv)*Cv/sC
    dx=rx[:,0]-rx[:,1]; dy=ry[:,0]-ry[:,1]; rd=(dx*dy).sum()/np.sqrt((dx**2).sum()*(dy**2).sum()); zU=rd*np.sqrt(NT)
    ax,ay=RA["gene_1"][2*NT:],RA["gene_2"][2*NT:]; bx,by=RB["gene_1"][2*NT:],RB["gene_2"][2*NT:]
    f=wcorr(ax,by); r=wcorr(ay,bx); delta=f-r
    nd=np.array([(lambda p: wcorr(ax,by[p])-wcorr(ay,bx[p]))(rng.permutation(NSH)) for _ in range(D)])
    p_delta=(1+np.sum(abs(nd)>=abs(delta)))/(D+1)
    return dict(h_x=hx,h_y=hy,rho=rho,z_rho=z_rho,S=Sv,C=Cv,U=Uv,z_C=z_C,rho_delta=rd,z_U=zU,
                rho_dag_xy=f,rho_dag_yx=r,delta=delta,p_delta=p_delta,
                z_dag_xy=f*np.sqrt(NSH-1),z_dag_yx=r*np.sqrt(NSH-1))
SCEN=[("A_B (single state)",["A_B"]),("A_to_B (single state)",["A_to_B"]),
      ("A_B (two states)",["A_B_low_k_on","A_B_high_k_on"]),("A_to_B (two states)",["A_to_B_low_k_on","A_to_B_high_k_on"])]
def run_rep(args):
    rep,out,D=args; dest=f"{out}/rep_{rep:03d}.csv"
    if os.path.exists(dest): return rep,"skipped (exists)"
    rows=[]
    for si,(name,cfgs) in enumerate(SCEN):
        rng=np.random.RandomState(rep*10+si); srcs=[]; nf=[]
        for c in cfgs:
            p,n=find(c,rep); srcs.append(load(p)); nf.append(n)
        if len(srcs)==1: P1,P2=srcs[0][T1],srcs[0][T2]
        else:
            sa,sb=rng.choice(NCL,3000,replace=False),rng.choice(NCL,3000,replace=False)
            P1={g:np.vstack([srcs[0][T1][g][sa],srcs[1][T1][g][sb]]) for g in G}; P2={g:np.vstack([srcs[0][T2][g][sa],srcs[1][T2][g][sb]]) for g in G}
        o=analyse(P1,P2,rng,D); o.update(rep=rep,scenario=name,n_files_found=max(nf)); rows.append(o)
    pd.DataFrame(rows).to_csv(dest+".tmp",index=False); os.replace(dest+".tmp",dest); return rep,"done"
if __name__=="__main__":
    import multiprocessing as mp
    ap=argparse.ArgumentParser(); ap.add_argument("--reps",required=True,help="e.g. 0-99 or 5"); ap.add_argument("--out",required=True)
    ap.add_argument("--D",type=int,default=500); ap.add_argument("--jobs",type=int,default=1); a=ap.parse_args()
    lo,hi=(a.reps.split("-")+[a.reps])[:2] if "-" in a.reps else (a.reps,a.reps)
    reps=list(range(int(lo),int(hi)+1)); print(f"{len(reps)} replicates, {a.jobs} workers",flush=True)
    with mp.Pool(a.jobs) as pool:
        for rep,status in pool.imap_unordered(run_rep,[(r,a.out,a.D) for r in reps]): print("rep",rep,status,flush=True)
