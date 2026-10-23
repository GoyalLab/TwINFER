import pandas as pd, numpy as np
S="/home/gzu5140/TwINFER_KA/simulation_data/"
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
    return dict(hx=hx,hy=hy,rho=rho,z_rho=z_rho,S=Sv,C=Cv,U=Uv,hst=hst,zU=zU,f=f,r=r,delta=delta,p=pd_,zf=zf,zr=zr,
                covary=abs(z_rho)>1.96,called=abs(z_rho)>1.96 and abs(zU)>1.96,directed=pd_<=.05,coupled=max(abs(zf),abs(zr))>2.576)

F="/home/gzu5140/TwINFER_KA/simulation_data/figure_2/"
files={"A_B":(F+"A_B_high_k_on/df_rows_3_3_09082025_201359_ncells_6000_A_B_high_k_on_rep_0.csv",F+"A_B_low_k_on/df_rows_2_2_09082025_210256_ncells_6000_A_B_low_k_on_rep_0.csv"),
"A_to_B":(F+"A_to_B_high_k_on/df_rows_3_3_09082025_220257_ncells_6000_A_to_B_high_k_on_rep_0.csv",F+"A_to_B_low_k_on/df_rows_2_2_09082025_212311_ncells_6000_A_to_B_low_k_on_rep_0.csv")}
G=["gene_1","gene_2"]
def mix(src,rng):
    sh_=rng.choice(NCL,3000,replace=False); sl_=rng.choice(NCL,3000,replace=False)
    P1={g:np.vstack([src[0][T1][g][sh_],src[1][T1][g][sl_]]) for g in G}; P2={g:np.vstack([src[0][T2][g][sh_],src[1][T2][g][sl_]]) for g in G}
    return P1,P2
def calib(P1,P2,rng,nperm=1000):
    perm=rng.permutation(NCL); bt,sh=perm[NT:2*NT],perm[2*NT:]; pick=rng.randint(0,2,NSH)
    RB={g:wrank(np.concatenate([P2[g][bt].ravel(),P2[g][sh,1-pick]]),W) for g in G}
    hx=h_of(RB["gene_1"][:2*NT].reshape(NT,2)); hy=h_of(RB["gene_2"][:2*NT].reshape(NT,2))
    a=RB["gene_2"][:2*NT].reshape(NT,2); b=RB["gene_2"][2*NT:]; x=RB["gene_1"]
    dx=RB["gene_1"][:2*NT].reshape(NT,2); dx=dx[:,0]-dx[:,1]; dy=a[:,0]-a[:,1]
    zr=[];zn=[];zu=[]
    for _ in range(nperm):
        pa=rng.permutation(NT); r2=np.concatenate([a[pa].ravel(),b[rng.permutation(NSH)]]); rr=wcorr(x,r2,W)
        zr.append(rr/np.sqrt(s2(hx,hy))); zn.append(rr/np.sqrt(s2(0,0)))
        d2=dy[pa]; zu.append((dx*d2).sum()/np.sqrt((dx**2).sum()*(d2**2).sum())*np.sqrt(NT))
    zr,zn,zu=map(np.array,(zr,zn,zu)); return hx,hy,zr.std(),zn.std(),np.mean(abs(zr)>1.96),np.mean(abs(zn)>1.96),zu.std(),np.mean(abs(zu)>1.96)
rng=np.random.RandomState(0)
for name,(fh,fl) in files.items():
    src=[]
    for fp in (fh,fl):
        d=pd.read_csv(fp); src.append({t:{g:piv(d,t,g) for g in G} for t in (T1,T2)})
    P1,P2=mix(src,rng)
    o=analyse(P1,P2,rng,2000)
    cls="no co-variation" if not o['covary'] else ("called" if o['called'] else "co-varies, not called")
    dirn=("directed: regulator "+("gene_1" if np.sign(o['delta'])==np.sign(o['U']) else "gene_2")+(", activation" if o['U']>0 else ", repression")) if o['directed'] else ("symmetric (coupled, not directed)" if o['coupled'] else "not coupled")
    st="inherited" if o['hst']>2.326 else "single"
    print(f"== {name} 50/50 mix (A={T1}h, B={T2}h): h1={o['hx']:.2f} h2={o['hy']:.2f}  rho={o['rho']:+.3f} z_rho={o['z_rho']:+.1f} | S={o['S']:+.3f} C={o['C']:+.3f} U={o['U']:+.3f} h-stat={o['hst']:+.1f} ({st}) zU={o['zU']:+.1f} => {cls}")
    print(f"    rho_dag(1->2)={o['f']:+.3f} (z={o['zf']:+.1f}) rho_dag(2->1)={o['r']:+.3f} (z={o['zr']:+.1f}) delta={o['delta']:+.3f} p_delta={o['p']:.4f} => {dirn}")
    R=[]
    for sidx in range(20):
        Q1,Q2=mix(src,rng); R+= [analyse(Q1,Q2,rng,300) for _ in range(5)]
    fr=lambda k: np.mean([x[k] for x in R])
    print(f"    100 draws (20 a-priori mixes x 5): covary={fr('covary'):.2f} inherited={np.mean([x['hst']>2.326 for x in R]):.2f} called={fr('called'):.2f} coupled={fr('coupled'):.2f} directed={fr('directed'):.2f} called&directed={np.mean([x['called'] and x['directed'] for x in R]):.2f} ; mean h1={np.mean([x['hx'] for x in R]):.2f} h2={np.mean([x['hy'] for x in R]):.2f}; mean z_rho={np.mean([x['z_rho'] for x in R]):+.1f}, mean z_U={np.mean([x['zU'] for x in R]):+.1f}")
    C=np.array([calib(*mix(src,rng),rng) for _ in range(5)])
    print(f"    null calibration (shuffle gene_2 across all clones, 5 mixes x 1000): z_rho sd h-adj={C[:,2].mean():.2f} naive={C[:,3].mean():.2f} | FP h-adj={C[:,4].mean():.3f} naive={C[:,5].mean():.3f} | z_U sd={C[:,6].mean():.2f} FP={C[:,7].mean():.3f} (h1,h2 ~ {C[:,0].mean():.2f},{C[:,1].mean():.2f})")
