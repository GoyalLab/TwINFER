import numpy as np, pandas as pd, h5py, sys, gc
from scipy.stats import norm
import importlib.util, os
_s=importlib.util.spec_from_file_location("pair_us",os.path.join(os.path.dirname(os.path.abspath(__file__)),"18_pair_us_single_tf_vs_all_genes.py")); pair_us=importlib.util.module_from_spec(_s); _s.loader.exec_module(pair_us); load,wranks,R0=pair_us.load,pair_us.wranks,pair_us.R0
S_=("MCM5 PCNA TYMS FEN1 MCM2 MCM4 RRM1 UNG GINS2 MCM6 CDCA7 DTL PRIM1 UHRF1 MLF1IP CENPU HELLS RFC2 RPA2 NASP RAD51AP1 GMNN WDR76 SLBP CCNE2 UBR7 POLD3 MSH2 ATAD2 RAD51 RRM2 CDC45 CDC6 EXO1 TIPIN DSCC1 BLM CASP8AP2 USP1 CLSPN POLA1 CHAF1B BRIP1 E2F8").split()
G2M=("HMGB2 CDK1 NUSAP1 UBE2C BIRC5 TPX2 TOP2A NDC80 CKS2 NUF2 CKS1B MKI67 TMPO CENPF TACC3 FAM64A PIMREG SMC4 CCNB2 CKAP2L CKAP2 AURKB BUB1 KIF11 ANP32E TUBB4B GTSE1 KIF20B HJURP CDCA3 HN1 JPT1 CDC20 TTK CDC25C KIF2C RANGAP1 NCAPD2 DLGAP5 CDCA2 CDCA8 ECT2 KIF23 HMMR AURKA PSRC1 ANLN LBR CKAP5 CENPE CTCF NEK2 G2E3 GAS2L3 CBX5 CENPA").split()
path=R0+sys.argv[1]; label=sys.argv[2]; tf,tg="FOSL1","SERPINE1"
genes,X,clone=load(path); n=X.shape[1]; G=len(genes); gi={g:i for i,g in enumerate(genes)}
f=h5py.File(path,"r"); o=f["obs"]; oi=o.attrs["_index"]
def cat(nm): g=o[nm]; c=np.array([x.decode() if isinstance(x,bytes) else x for x in g["categories"][:]],dtype=object); return c[g["codes"][:]]
cov=pd.DataFrame({"n_genes":o["n_genes"][:].astype(float),"mt":o["pct_counts_mt"][:].astype(float),"rep":cat("replicate")})
z=lambda ix:((X[ix]-X[ix].mean(1,keepdims=True))/np.where(X[ix].std(1,keepdims=True)==0,1,X[ix].std(1,keepdims=True))).mean(0)
sS=[gi[g] for g in dict.fromkeys(S_) if g in gi]; sG=[gi[g] for g in dict.fromkeys(G2M) if g in gi]; cov["S"]=z(sS); cov["G2M"]=z(sG); cov["lgn"]=np.log(cov.n_genes)
sc=cov[["S","G2M"]].values; cov["phase"]=np.where(sc.max(1)<=0,"G1",np.where(sc[:,1]>=sc[:,0],"G2M","S")); lead=np.where(cov.phase=="G2M",cov.G2M,np.where(cov.phase=="S",cov.S,-cov.G2M-cov.S)); cov["lead"]=lead
q=lambda s,k:pd.qcut(s.rank(method="first"),k,labels=False)
cov["strat_cc"]=cov.phase+"_"+cov.groupby("phase").lead.transform(lambda s:q(s,3)).astype(str); cov["strat_cx"]="nG"+q(cov.lgn,10).astype(str)
cov["strat_all"]=cov.phase+"_nG"+cov.groupby("phase").lgn.transform(lambda s:q(s,5)).astype(str)+"_rep"+cov.rep+"_mt"+q(cov.mt,3).astype(str)
print(f"{label}: {n:,} cells; phase fractions {cov.phase.value_counts(normalize=True).round(2).to_dict()}; corr(log n_genes, G2M) {cov.lgn.corr(cov.G2M):+.2f}, S {cov.lgn.corr(cov.S):+.2f}",flush=True)
cs=pd.Series(clone); vcs=cs.value_counts(); nc=cs.map(vcs).values.astype(float); w=1.0/nc; Wtot=w.sum()
twin=nc>=2; idx=np.where(twin)[0]; codes,uniq=pd.factorize(cs[twin]); nct=np.bincount(codes).astype(float); Pc=nct*(nct-1)/2; Nt=len(uniq); Wt=2.0/nct[codes]
oh=np.zeros((len(idx),Nt),np.float32); oh[np.arange(len(idx)),codes]=1.0; ohP=oh/Pc[None,:].astype(np.float32); sd=np.sqrt((1.0/(nct-1)).sum())/Nt; sizes=vcs.values.astype(float); N1=len(vcs)
x=gi[tf]; ty=gi[tg]; oth=np.array([i for i in range(G) if i!=x])
def compute(R,tag):
    mux=(R[x,idx].astype(np.float64)*Wt).sum()/(2*Nt); rx=R[x,idx].astype(np.float64)-mux; Mx=0.5*(Wt*rx**2).sum(); sx=oh.T@rx
    S=np.zeros(G); Cx=np.zeros(G); hg=np.zeros(G); rho=np.zeros(G); wsum=w.sum(); Rx=R[x].astype(np.float64); dx=Rx-(Rx*w).sum()/wsum; vx=(w*dx**2).sum()
    for b in range(0,G,1000):
        sl=slice(b,min(b+1000,G)); Rb=R[sl][:,idx].astype(np.float64); mu=(Rb*Wt).sum(1)/(2*Nt); r=Rb-mu[:,None]; Mb=0.5*(Wt*r**2).sum(1)
        S[sl]=0.5*(r*(Wt*rx)).sum(1)/np.sqrt(Mx*Mb); sg=r@oh; cross=(sg*sx[None,:]/Pc[None,:]).sum(1); same=((r*rx)@ohP).sum(1); Cx[sl]=0.5*(cross-same)/np.sqrt(Mx*Mb); hg[sl]=0.5*((sg**2/Pc[None,:]).sum(1)-((r*r)@ohP).sum(1))/Mb
        Rw=R[sl].astype(np.float64); dg=Rw-((Rw*w).sum(1)/wsum)[:,None]; rho[sl]=(dg*(w*dx)).sum(1)/np.sqrt(vx*(w*dg**2).sum(1))
    U=S-Cx; rhoD=U/np.sqrt(np.clip((1-hg[x])*(1-hg),1e-9,None)); zU=rhoD/sd; rk=lambda v:int((np.abs(v[oth])>abs(v[ty])).sum())+1
    top=[i for i in np.argsort(-np.abs(zU)) if i!=x][:8]
    print(f"[{tag}] h(FOSL1) {hg[x]:+.2f}, h(SERPINE1) {hg[ty]:+.2f} | rho {rho[ty]:+.3f}  S {S[ty]:+.3f}  C {Cx[ty]:+.3f}  U {U[ty]:+.3f}  z_U {zU[ty]:+.2f} | rank of SERPINE1 by |S| {rk(S)}, |U| {rk(U)}, |z_U| {rk(zU)} of {len(oth):,} | top by |z_U|: "+", ".join(f"{genes[i]} {zU[i]:+.1f}" for i in top),flush=True)
    return dict(tag=tag,rho=rho[ty],S=S[ty],C=Cx[ty],U=U[ty],zU=zU[ty],rankS=rk(S),rankU=rk(U),rankZ=rk(zU),hx=hg[x],hy=hg[ty])
res=[]
R=np.empty(X.shape,np.float32)
for k in range(G): R[k]=wranks(X[k].astype(np.float64),w)
res.append(compute(R,"unadjusted"))
for name,col in (("cell cycle only","strat_cc"),("library complexity only","strat_cx"),("everything (cycle+size+replicate+mt)","strat_all")):
    groups=[np.where(cov[col].values==u)[0] for u in pd.unique(cov[col])]; Wg=[w[g].sum() for g in groups]
    for k in range(G):
        v=X[k].astype(np.float64); r=np.empty(n)
        for g,wg in zip(groups,Wg): r[g]=wranks(v[g],w[g])*(Wtot/wg)
        R[k]=r
    res.append(compute(R,f"{name}; {len(groups)} strata"))
pd.DataFrame(res).to_csv(f"/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/fm01/pair_checks/{label}_FOSL1_SERPINE1_adjusted.csv",index=False); print("done",flush=True)
