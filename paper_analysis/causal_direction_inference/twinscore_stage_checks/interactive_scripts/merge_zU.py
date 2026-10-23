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
def rdelta(a,b): dx=a[0][:,0]-a[0][:,1]; dy=b[0][:,0]-b[0][:,1]; return dx,dy
def analyse(P1,P2,rng):
    perm=rng.permutation(NCL); at,bt,sh=perm[:NT],perm[NT:2*NT],perm[2*NT:]; pick=rng.randint(0,2,NSH)
    RA={};RB={}
    for g in ["gene_1","gene_2"]:
        RA[g]=wrank(np.concatenate([P1[g][at].ravel(),P1[g][sh,pick]]),W)
        RB[g]=wrank(np.concatenate([P2[g][bt].ravel(),P2[g][sh,1-pick]]),W)
    hxB=h_of(tw(RB,"gene_1")); hyB=h_of(tw(RB,"gene_2")); z_rho=wcorr(RB["gene_1"],RB["gene_2"],W)/np.sqrt(s2(hxB,hyB))
    out={'z_rho':z_rho}
    cen=lambda r: r-r.mean()
    X={k:(cen(tw(R,"gene_1")),cen(tw(R,"gene_2"))) for k,R in (("A",RA),("B",RB))}
    def stats(parts):
        rx=np.vstack([X[k][0] for k in parts]); ry=np.vstack([X[k][1] for k in parts]); n=len(rx)
        nrm=np.sqrt(((rx**2).sum(1)/2).mean()*((ry**2).sum(1)/2).mean())
        Sv=((rx[:,0]*ry[:,0]+rx[:,1]*ry[:,1])/2).mean()/nrm; Cv=((rx[:,0]*ry[:,1]+rx[:,1]*ry[:,0])/2).mean()/nrm
        dx=rx[:,0]-rx[:,1]; dy=ry[:,0]-ry[:,1]; rd=(dx*dy).sum()/np.sqrt((dx**2).sum()*(dy**2).sum())
        return Sv,Cv,Sv-Cv,rd,rd*np.sqrt(n)
    for k,p in (("A","A"),("B","B"),("M","AB")): out[k]=stats(p)
    return out
rng=np.random.RandomState(0)
for name,fp in files.items():
    d=pd.read_csv(fp); P1={g:piv(d,T1,g) for g in ["gene_1","gene_2"]}; P2={g:piv(d,T2,g) for g in ["gene_1","gene_2"]}
    o=analyse(P1,P2,rng)
    print(f"{name}: z_rho(B)={o['z_rho']:+.1f}")
    for k,l,n in (("A","A only (1h, 1,500 pairs)",1500),("B","B only (20h, 1,500 pairs)",1500),("M","MERGED (3,000 pairs)",3000)):
        S_,C_,U_,rd,zu=o[k]; print(f"    {l:27s} S={S_:+.3f} C={C_:+.3f} U={U_:+.3f} rhoDelta={rd:+.3f} z_U={zu:+.1f}")
    R=[analyse(P1,P2,rng) for _ in range(200)]
    f=lambda k: np.mean([abs(x[k][4])>1.96 for x in R]); g_=lambda: np.mean([abs(x['M'][4])>1.96 and abs(x['z_rho'])>1.96 for x in R])
    print(f"    200 redraws, frac |z_U|>1.96: A={f('A'):.2f} B={f('B'):.2f} merged={f('M'):.2f} ; merged & Stage I(B) significant={g_():.2f}; mean z_U merged={np.mean([x['M'][4] for x in R]):+.2f} (sd {np.std([x['M'][4] for x in R]):.2f})")
