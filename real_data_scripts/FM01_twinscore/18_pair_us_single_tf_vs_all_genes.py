import numpy as np, pandas as pd, h5py, sys, gc
from scipy.stats import norm
R0="/gpfs/projects/b1255/hzhang/TwINFER_KA/"
def load(path):
    f=h5py.File(path,"r"); vi=f["var"].attrs["_index"]; genes=np.array([s.decode() if isinstance(s,bytes) else s for s in f["var"][vi][:]]); o=f["obs"]
    g=o["fatemap_clone_singletcode"]; cats=np.array([x.decode() if isinstance(x,bytes) else x for x in g["categories"][:]],dtype=object); clone=cats[g["codes"][:]]
    L=f["layers"]["counts"]; ip=L["indptr"][:]; ind=L["indices"]; dat=L["data"]; n=len(ip)-1; G=len(genes); nnz=np.zeros(G); tot=np.zeros(n)
    for s in range(0,n,2000):
        e=min(s+2000,n); a,b=ip[s],ip[e]; ii=ind[a:b]; dd=dat[a:b]; nnz+=np.bincount(ii,minlength=G); np.add.at(tot,np.repeat(np.arange(s,e),np.diff(ip[s:e+1])),dd)
    keep=np.where(nnz/n>0.05)[0]; pos=-np.ones(G,int); pos[keep]=np.arange(len(keep)); X=np.zeros((len(keep),n),np.float32)
    for s in range(0,n,2000):
        e=min(s+2000,n); a,b=ip[s],ip[e]; ii=ind[a:b]; dd=dat[a:b]; rows=np.repeat(np.arange(s,e),np.diff(ip[s:e+1])); m=pos[ii]>=0; X[pos[ii[m]],rows[m]]=np.log1p(dd[m]/tot[rows[m]]*1e4)
    return genes[keep],X,clone
def wranks(v,w):
    o=np.argsort(v,kind="stable"); vs=v[o]; ws=w[o]; cw=np.cumsum(ws); st=np.r_[0,np.flatnonzero(np.diff(vs))+1]; en=np.r_[st[1:]-1,len(vs)-1]
    before=np.where(st>0,cw[st-1],0.0); gw=cw[en]-before; mid=before+gw/2; out=np.empty(len(v)); out[o]=mid[np.repeat(np.arange(len(st)),en-st+1)]; return out
def run(path,label,OUT,tf="FOSL1",tg="SERPINE1"):
    genes,X,clone=load(path); n=X.shape[1]; G=len(genes); gi={g:i for i,g in enumerate(genes)}; cs=pd.Series(clone); vcs=cs.value_counts(); nc=cs.map(vcs).values.astype(float); w=1.0/nc
    R=np.empty(X.shape,np.float32)
    for k in range(G): R[k]=wranks(X[k].astype(np.float64),w)
    del X; gc.collect()
    twin=nc>=2; idx=np.where(twin)[0]; codes,uniq=pd.factorize(cs[twin]); nct=np.bincount(codes).astype(float); Pc=nct*(nct-1)/2; Nt=len(uniq); Wt=2.0/nct[codes]
    onehotP=np.zeros((len(idx),Nt),np.float32); onehotP[np.arange(len(idx)),codes]=1.0/Pc[codes]*0+1.0; oh=onehotP; ohP=oh/Pc[None,:].astype(np.float32)
    x=gi[tf]; mux=(R[x,idx].astype(np.float64)*Wt).sum()/(2*Nt); rx=R[x,idx].astype(np.float64)-mux; Mx=0.5*(Wt*rx**2).sum(); sx=oh.T@rx
    S=np.zeros(G); Cx=np.zeros(G); hg=np.zeros(G); rho=np.zeros(G); M=np.zeros(G)
    wsum=w.sum(); Rx=R[x].astype(np.float64); mwx=(Rx*w).sum()/wsum; dx=Rx-mwx; vx=(w*dx**2).sum()
    for b in range(0,G,1000):
        sl=slice(b,min(b+1000,G)); Rb=R[sl][:,idx].astype(np.float64); mu=(Rb*Wt).sum(1)/(2*Nt); r=Rb-mu[:,None]; Mb=0.5*(Wt*r**2).sum(1); M[sl]=Mb
        S[sl]=0.5*(r*(Wt*rx)).sum(1)/np.sqrt(Mx*Mb)
        sg=r@oh; cross=(sg*sx[None,:]/Pc[None,:]).sum(1); same=((r*rx)@ohP).sum(1); Cx[sl]=0.5*(cross-same)/np.sqrt(Mx*Mb)
        hg[sl]=0.5*((sg**2/Pc[None,:]).sum(1)-((r*r)@ohP).sum(1))/Mb
        Rw=R[sl].astype(np.float64); dg=Rw-((Rw*w).sum(1)/wsum)[:,None]; rho[sl]=(dg*(w*dx)).sum(1)/np.sqrt(vx*(w*dg**2).sum(1))
    U=S-Cx; sd=np.sqrt((1.0/(nct-1)).sum())/Nt; rhoD=U/np.sqrt(np.clip((1-hg[x])*(1-hg),1e-9,None)); zU=rhoD/sd
    sizes=vcs.values.astype(float); N1=len(vcs); hh=np.clip(hg,0,None); sig=np.sqrt(((1/sizes).sum()+((1-1/sizes)).sum()*hh[x]*hh))/N1; zrho=rho/sig
    ty=gi[tg]; oth=np.array([i for i in range(G) if i!=x]); rk=lambda v:int((np.abs(v[oth])>abs(v[ty])).sum())+1
    print(f"\n######## {label}: {n:,} cells; genes detected in >5% of cells: {G:,}; clones {N1:,}; twin clones Nt={Nt:,}; twin pairs {int(Pc.sum()):,}")
    print(f"  heritability: {tf} h = {hg[x]:+.3f}, {tg} h = {hg[ty]:+.3f}")
    print(f"  {tf} -> {tg}: clone-weighted Spearman rho {rho[ty]:+.3f} (z {zrho[ty]:+.1f}); S {S[ty]:+.3f}; C {Cx[ty]:+.3f}; U = S-C {U[ty]:+.3f}; twin-difference rho {rhoD[ty]:+.3f}; z_U {zU[ty]:+.2f} (p {2*norm.sf(abs(zU[ty])):.2g})")
    print(f"  rank of {tg} among {len(oth):,} genes as a target of {tf}: by |rho| {rk(rho)}, |S| {rk(S)}, |U| {rk(U)}, |z_U| {rk(zU)}")
    t=[i for i in np.argsort(-np.abs(zU)) if i!=x][:10]; print(f"  top 10 genes for {tf} by |z_U|: "+", ".join(f"{genes[i]} ({zU[i]:+.1f})" for i in t)); t=[i for i in np.argsort(-np.abs(S)) if i!=x][:10]; print(f"  top 10 by |S|: "+", ".join(f"{genes[i]} ({S[i]:+.2f})" for i in t))
    pd.DataFrame({"gene":genes,"rho":rho,"S":S,"C":Cx,"U":U,"z_U":zU,"h":hg}).to_csv(f"{OUT}/{label}_{tf}_vs_all_genes_US.csv",index=False)
if __name__=="__main__": run(R0+sys.argv[2],sys.argv[3],sys.argv[1])
