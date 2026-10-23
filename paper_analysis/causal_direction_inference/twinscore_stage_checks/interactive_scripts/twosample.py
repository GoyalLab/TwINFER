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
def analyse(P1,P2,rng,D):
    perm=rng.permutation(NCL); at,bt,sh=perm[:NT],perm[NT:2*NT],perm[2*NT:]; pick=rng.randint(0,2,NSH)
    RA={};RB={}
    for g in ["gene_1","gene_2"]:
        RA[g]=wrank(np.concatenate([P1[g][at].ravel(),P1[g][sh,pick]]),W)
        RB[g]=wrank(np.concatenate([P2[g][bt].ravel(),P2[g][sh,1-pick]]),W)
    hx=h_of(RB["gene_1"][:2*NT].reshape(NT,2)); hy=h_of(RB["gene_2"][:2*NT].reshape(NT,2))
    rho=wcorr(RB["gene_1"],RB["gene_2"],W); z_rho=rho/np.sqrt(s2(hx,hy))
    rx=RB["gene_1"][:2*NT].reshape(NT,2); ry=RB["gene_2"][:2*NT].reshape(NT,2); rx=rx-rx.mean(); ry=ry-ry.mean()
    nrm=np.sqrt(((rx**2).sum(1)/2).mean()*((ry**2).sum(1)/2).mean())
    Sv=((rx[:,0]*ry[:,0]+rx[:,1]*ry[:,1])/2).mean()/nrm; Cv=((rx[:,0]*ry[:,1]+rx[:,1]*ry[:,0])/2).mean()/nrm; Uv=Sv-Cv
    sC=np.sqrt((1+Sv**2+hx*hy)/(2*NT)); hst=np.sign(Sv)*Cv/sC
    dx=rx[:,0]-rx[:,1]; dy=ry[:,0]-ry[:,1]; rd=(dx*dy).sum()/np.sqrt((dx**2).sum()*(dy**2).sum()); zU=rd*np.sqrt(NT)
    ax,ay=RA["gene_1"][2*NT:],RA["gene_2"][2*NT:]; bx,by=RB["gene_1"][2*NT:],RB["gene_2"][2*NT:]
    f=wcorr(ax,by); r=wcorr(ay,bx); delta=f-r
    nd=np.array([ (lambda p: wcorr(ax,by[p])-wcorr(ay,bx[p]))(rng.permutation(NSH)) for _ in range(D)])
    pd_=(1+np.sum(abs(nd)>=abs(delta)))/(D+1)
    zf,zr=f*np.sqrt(NSH-1),r*np.sqrt(NSH-1)
    return dict(rho=rho,z_rho=z_rho,S=Sv,C=Cv,U=Uv,hst=hst,zU=zU,f=f,r=r,delta=delta,p=pd_,zf=zf,zr=zr,
                covary=abs(z_rho)>1.96,called=abs(z_rho)>1.96 and abs(zU)>1.96,directed=pd_<=.05,coupled=max(abs(zf),abs(zr))>2.576)
rng=np.random.RandomState(0)
for name,fp in files.items():
    d=pd.read_csv(fp); P1={g:piv(d,T1,g) for g in ["gene_1","gene_2"]}; P2={g:piv(d,T2,g) for g in ["gene_1","gene_2"]}
    o=analyse(P1,P2,rng,2000)
    cls="no co-variation" if not o['covary'] else ("called" if o['called'] else "co-varies, not called")
    dirn=("directed: regulator "+("gene_1" if np.sign(o['delta'])==np.sign(o['U']) else "gene_2")+(", activation" if o['U']>0 else ", repression")) if o['directed'] else ("symmetric (coupled, not directed)" if o['coupled'] else "not coupled")
    print(f"{name} (A={T1}h, B={T2}h)  rho={o['rho']:+.3f} z_rho={o['z_rho']:+.1f} | S={o['S']:+.3f} C={o['C']:+.3f} U={o['U']:+.3f} h-stat={o['hst']:+.1f} zU={o['zU']:+.1f}  => {cls}")
    print(f"    rho_dag(1->2)={o['f']:+.3f} (z={o['zf']:+.1f})  rho_dag(2->1)={o['r']:+.3f} (z={o['zr']:+.1f})  delta={o['delta']:+.3f} p_delta={o['p']:.4f}  => {dirn}")
    R=[analyse(P1,P2,rng,500) for _ in range(100)]
    fr=lambda k: np.mean([x[k] for x in R])
    both=np.mean([x['called'] and x['directed'] for x in R]); 
    dr=np.mean([x['directed'] and x['delta']>0 for x in R]); drn=np.mean([x['directed'] and x['delta']<0 for x in R])
    print(f"    100 redraws: covary={fr('covary'):.2f} called={fr('called'):.2f} coupled={fr('coupled'):.2f} directed={fr('directed'):.2f} (delta>0: {dr:.2f}, delta<0: {drn:.2f}) called&directed={both:.2f}")
