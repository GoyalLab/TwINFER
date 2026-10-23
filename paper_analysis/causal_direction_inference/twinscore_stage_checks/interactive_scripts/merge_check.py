import pandas as pd, numpy as np
S="/home/gzu5140/TwINFER_KA/simulation_data/"
files={"A_B":S+"figure_2/A_B/df_rows_0_1_08082025_093223_ncells_6000_A_B_rep_0.csv",
"A_to_B":S+"figure_2/A_to_B/df_rows_0_1_05082025_160041_ncells_6000_A_to_B_rep_0.csv",
"A_rep_B":S+"figure_3_1k/A_rep_B/df_rows_4_4_30082026_190919_ncells_6000_A_rep_B_rep_0_4b054658.csv"}
T1,T2=1,20; NT,NSH=1500,3000; NB=NT+NSH; NCL=6000
W=np.concatenate([np.full(2*NT,.5),np.ones(NSH)])
def wrank(v,w):
    u,inv=np.unique(v,return_inverse=True); sw=np.bincount(inv,weights=w); return (np.cumsum(sw)-sw+.5*sw)[inv]
def wcorr(a,b,w=None):
    if w is None: w=np.ones(len(a))
    a=a-(w*a).sum()/w.sum(); b=b-(w*b).sum()/w.sum(); return (w*a*b).sum()/np.sqrt((w*a*a).sum()*(w*b*b).sum())
def s2(hx,hy): return (NT*(.5+.5*hx*hy)+NSH)/NB**2
def h_of(r2):
    r=r2-r2.mean(); M=((r**2).sum(1)/2).mean(); return max(0.,(r[:,0]*r[:,1]).mean()/M)
def piv(d,t,g):
    s=d[d.time_step==t]; return s.pivot(index='clone_id',columns='replicate',values=f"{g}_mRNA").values

def tw(R,g): return R[g][:2*NT].reshape(NT,2)
def draw(P1,P2,rng):
    perm=rng.permutation(NCL); at,bt,sh=perm[:NT],perm[NT:2*NT],perm[2*NT:]; pick=rng.randint(0,2,NSH)
    D={}
    for k,P,idx,pk in (("A",P1,at,pick),("B",P2,bt,1-pick)):
        out=[]
        for g in ["gene_1","gene_2"]:
            R=wrank(np.concatenate([P[g][idx].ravel(),P[g][sh,pk]]),W); t=tw({g:R},g); out.append(t[:,0]-t[:,1])
        D[k]=out
    return D
def zraw(D,sy=None):  # pooled, as before
    dx=np.concatenate([D["A"][0],D["B"][0]]); dy=np.concatenate([D["A"][1],D["B"][1]]) if sy is None else sy
    return (dx*dy).sum()/np.sqrt((dx**2).sum()*(dy**2).sum())*np.sqrt(len(dx))
def zstd(D,sy=None):  # each sample's differences scaled to unit rms first
    f=lambda a: a/np.sqrt((a**2).mean())
    dx=np.concatenate([f(D["A"][0]),f(D["B"][0])]); dy=np.concatenate([f(D["A"][1]),f(D["B"][1])]) if sy is None else sy
    return (dx*dy).sum()/np.sqrt((dx**2).sum()*(dy**2).sum())*np.sqrt(len(dx))
rng=np.random.RandomState(0)
for name,fp in files.items():
    d=pd.read_csv(fp); P1={g:piv(d,T1,g) for g in ["gene_1","gene_2"]}; P2={g:piv(d,T2,g) for g in ["gene_1","gene_2"]}
    D=draw(P1,P2,rng)
    ra=np.sqrt((D["A"][0]**2).mean()*(D["A"][1]**2).mean()); rb=np.sqrt((D["B"][0]**2).mean()*(D["B"][1]**2).mean())
    # shuffle null: permute gene_2 differences across pairs within each sample
    nr=[];ns=[]
    for _ in range(2000):
        E={"A":(D["A"][0],D["A"][1][rng.permutation(NT)]),"B":(D["B"][0],D["B"][1][rng.permutation(NT)])}
        nr.append(zraw(E)); ns.append(zstd(E))
    R=[draw(P1,P2,rng) for _ in range(200)]
    zr=np.array([zraw(x) for x in R]); zs=np.array([zstd(x) for x in R])
    print(f"{name}: rms(delta) product A={ra:.0f} B={rb:.0f} (ratio {rb/ra:.2f}) | shuffle-null sd: raw pooling={np.std(nr):.2f}  standardised={np.std(ns):.2f}")
    print(f"    main draw z_U raw={zraw(D):+.1f} standardised={zstd(D):+.1f} | 200 redraws frac|z|>1.96: raw={np.mean(abs(zr)>1.96):.2f}  standardised={np.mean(abs(zs)>1.96):.2f}  (mean z raw={zr.mean():+.2f}, std={zs.mean():+.2f})")
