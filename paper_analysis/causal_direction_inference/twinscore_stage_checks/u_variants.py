"""Alternative z_U estimators on sample B (20 h): U = S - C with S from (a) twin cells only or (b) all cells (twins + singletons);
values = clone-weighted midranks or log1p(mRNA). z = U / bootstrap-over-clones sd. Same sampling as figure2_1000_zscores.analyse."""
import argparse, glob, os, numpy as np, pandas as pd
import figure2_1000_zscores as F
from figure2_1000_zscores import NT,NSH,NCL,G,W,T1,T2,SCEN,find,load,wrank
NB_=200
def stats(X1,X2,Y1,Y2,xs,ys):
    # twin clones: (x1,x2),(y1,y2); singletons xs,ys ; weights: twin cell .5, singleton 1 ; standardise over all cells
    w=np.concatenate([np.full(2*len(X1),.5),np.ones(len(xs))])
    x=np.concatenate([X1,X2,xs]);y=np.concatenate([Y1,Y2,ys]); mx=(w*x).sum()/w.sum();my=(w*y).sum()/w.sum()
    sx=np.sqrt((w*(x-mx)**2).sum()/w.sum());sy=np.sqrt((w*(y-my)**2).sum()/w.sum())
    a1,a2,b1,b2,a,b=(X1-mx)/sx,(X2-mx)/sx,(Y1-my)/sy,(Y2-my)/sy,(xs-mx)/sx,(ys-my)/sy
    Stw=((a1*b1+a2*b2)/2).mean(); C=((a1*b2+a2*b1)/2).mean(); Sall=(len(X1)*Stw+(a*b).sum())/(len(X1)+len(xs))
    return Stw-C,Sall-C
def zs(xt,yt,xs,ys,rng):
    u=stats(xt[:,0],xt[:,1],yt[:,0],yt[:,1],xs,ys); bs=[]
    for _ in range(NB_):
        i=rng.randint(0,len(xt),len(xt)); j=rng.randint(0,len(xs),len(xs))
        bs.append(stats(xt[i,0],xt[i,1],yt[i,0],yt[i,1],xs[j],ys[j]))
    sd=np.array(bs).std(0); return u[0]/sd[0],u[1]/sd[1],u[0],u[1]
def variants(P2,rng):
    perm=rng.permutation(NCL); bt=perm[NT:2*NT]; sh=perm[2*NT:]; pick=rng.randint(0,2,NSH)
    raw={g:(P2[g][bt].astype(float),P2[g][sh,1-pick].astype(float)) for g in G}
    out={}
    # ranks (clone-weighted midranks over B)
    R={g:wrank(np.concatenate([raw[g][0].ravel(),raw[g][1]]),W) for g in G}
    rk={g:(R[g][:2*NT].reshape(NT,2),R[g][2*NT:]) for g in G}
    lg={g:(np.log1p(raw[g][0]),np.log1p(raw[g][1])) for g in G}
    for nm,V in (("rank",rk),("log",lg)):
        a,b,c,d=zs(V["gene_1"][0],V["gene_2"][0],V["gene_1"][1],V["gene_2"][1],rng)
        out.update({f"z_{nm}_Stw":a,f"z_{nm}_Sall":b,f"U_{nm}_Stw":c,f"U_{nm}_Sall":d})
    return out
def run_rep(a):
    rep,out=a; dest=f"{out}/rep_{rep:03d}.csv"
    if os.path.exists(dest): return rep
    rows=[]
    for si,(name,cfgs) in enumerate(SCEN):
        rng=np.random.RandomState(rep*10+si); srcs=[load(find(c,rep)[0]) for c in cfgs]
        if len(srcs)==1: P2=srcs[0][T2]
        else:
            sa,sb=rng.choice(NCL,3000,replace=False),rng.choice(NCL,3000,replace=False)
            P2={g:np.vstack([srcs[0][T2][g][sa],srcs[1][T2][g][sb]]) for g in G}
        o=variants(P2,rng);o.update(rep=rep,scenario=name);rows.append(o)
    for net in ("A_B","A_to_B"):
        fs=glob.glob(f"/projects/b1255/hzhang/TwINFER_KA/simulation_data/drift_lowhigh_Kramp/{net}/df_*_ncells_6000_{net}_drift_lowhigh_Kramp_rep_{rep}_*.csv")
        o=variants(load(fs[0])[T2],np.random.RandomState(rep*10+7));o.update(rep=rep,scenario=f"{net} (drift)");rows.append(o)
    pd.DataFrame(rows).to_csv(dest,index=False);return rep
if __name__=="__main__":
    import multiprocessing as mp
    ap=argparse.ArgumentParser();ap.add_argument("--reps",default="0-99");ap.add_argument("--out",required=True);ap.add_argument("--jobs",type=int,default=8);a=ap.parse_args()
    lo,hi=map(int,a.reps.split("-"));os.makedirs(a.out,exist_ok=True)
    with mp.Pool(a.jobs) as p:
        for r in p.imap_unordered(run_rep,[(r,a.out) for r in range(lo,hi+1)]): print(r,flush=True)
