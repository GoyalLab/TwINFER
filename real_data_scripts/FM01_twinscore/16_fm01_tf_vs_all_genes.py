# TwinScore statistics (PDF procedure; same formulas as twinscore_pdf_candidates.py) for chosen genes against ALL genes of FM01, no ChIP restriction.
# CP10k (log1p) values, clone weights 1/n_c, weighted mid-ranks, twin pairs within clone. rho, S, C, U, h and z_U are symmetric in the two genes, so
# running the genes KDM5A/KDM5B against all genes also gives every TF x KDM5A/B pair (TF rows are selected afterwards by 16_fm01_tf_kdm5_table.py).
# usage: python 16_fm01_tf_vs_all_genes.py OUT_DIR GENE1,GENE2,...
import numpy as np, pandas as pd, h5py, sys, gc, json
R0="/gpfs/projects/b1255/hzhang/TwINFER_KA/"; H5=R0+"finalized_data/FM01_data/FM01_integrated.h5ad"
EXCL=R0+"analysis_data/fm01/data/twinscore_pdf_candidates/exclusion_list_B.json"
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
    o=np.argsort(v,kind="stable"); vs=v[o]; cw=np.cumsum(w[o]); st=np.r_[0,np.flatnonzero(np.diff(vs))+1]; en=np.r_[st[1:]-1,len(vs)-1]
    before=np.where(st>0,cw[st-1],0.0); gw=cw[en]-before; mid=before+gw/2; out=np.empty(len(v)); out[o]=mid[np.repeat(np.arange(len(st)),en-st+1)]; return out
OUT=sys.argv[1]; TFS=sys.argv[2].split(",")
genes,X,clone=load(H5)
if len(sys.argv)>3 and sys.argv[3]=="TFONLY":   # keep only TF rows (+ the named genes): ranks, h and S/C/U are per gene, so nothing else changes, memory is ~1/6
    _tf=set(open(R0+"real_data/humanTFs/TF_names_v_1.01.txt").read().split())|set(TFS); _k=np.array([g in _tf for g in genes]); genes=genes[_k]; X=X[_k]
n=X.shape[1]; G=len(genes); gi={g:i for i,g in enumerate(genes)}; cs=pd.Series(clone); vcs=cs.value_counts(); nc=cs.map(vcs).values.astype(float); w=1.0/nc
R=np.empty(X.shape,np.float32)
for k in range(G): R[k]=wranks(X[k].astype(np.float64),w)
del X; gc.collect()
twin=nc>=2; idx=np.where(twin)[0]; codes,uniq=pd.factorize(cs[twin]); nct=np.bincount(codes).astype(float); Pc=nct*(nct-1)/2; Nt=len(uniq); Wt=2.0/nct[codes]
oh=np.zeros((len(idx),Nt),np.float32); oh[np.arange(len(idx)),codes]=1.0; ohP=oh/Pc[None,:].astype(np.float32)
sd=np.sqrt((1.0/(nct-1)).sum())/Nt; sizes=vcs.values.astype(float); N1=len(vcs); wsum=w.sum()
excl=set(json.load(open(EXCL))["_union_all"]); tech=lambda g: g in {"MALAT1","NEAT1","XIST","FIRRE"} or g.startswith("MT-")
hg=np.zeros(G)   # heritability of every gene, h_g = C(g,g)
for b in range(0,G,1000):
    sl=slice(b,min(b+1000,G)); Rb=R[sl][:,idx].astype(np.float64); m=(Rb*Wt).sum(1)/(2*Nt); r=Rb-m[:,None]; Mb=0.5*(Wt*r**2).sum(1); sg=r@oh
    hg[sl]=0.5*((sg**2/Pc[None,:]).sum(1)-((r*r)@ohP).sum(1))/Mb
print(f"FM01: {n:,} cells, {G:,} genes, clones {N1:,}, twin clones {Nt:,}",flush=True)
for tf in TFS:
    if tf not in gi: print(tf,"not detected in >5% of cells; skipped",flush=True); continue
    x=gi[tf]; mux=(R[x,idx].astype(np.float64)*Wt).sum()/(2*Nt); rx=R[x,idx].astype(np.float64)-mux; Mx=0.5*(Wt*rx**2).sum(); sx=oh.T@rx
    S=np.zeros(G); Cx=np.zeros(G); rho=np.zeros(G); Rx=R[x].astype(np.float64); dx=Rx-(Rx*w).sum()/wsum; vx=(w*dx**2).sum()
    for b in range(0,G,1000):
        sl=slice(b,min(b+1000,G)); Rb=R[sl][:,idx].astype(np.float64); m=(Rb*Wt).sum(1)/(2*Nt); r=Rb-m[:,None]; Mb=0.5*(Wt*r**2).sum(1)
        S[sl]=0.5*(r*(Wt*rx)).sum(1)/np.sqrt(Mx*Mb); sg=r@oh; cross=(sg*sx[None,:]/Pc[None,:]).sum(1); same=((r*rx)@ohP).sum(1); Cx[sl]=0.5*(cross-same)/np.sqrt(Mx*Mb)
        Rw=R[sl].astype(np.float64); dg=Rw-((Rw*w).sum(1)/wsum)[:,None]; rho[sl]=(dg*(w*dx)).sum(1)/np.sqrt(vx*(w*dg**2).sum(1))
    U=S-Cx; rhoD=U/np.sqrt(np.clip((1-hg[x])*(1-hg),1e-9,None)); zU=rhoD/sd
    hh=np.clip(hg,0,None); sig=np.sqrt(((1/sizes).sum()+((1-1/sizes)).sum()*hh[x]*hh))/N1; zrho=rho/sig
    df=pd.DataFrame({"gene":genes,"rho":rho,"z_rho":zrho,"S":S,"C":Cx,"U":U,"h_x":hg[x],"h_y":hg,"z_U":zU,"filtered_out":[(g in excl) or tech(g) for g in genes]})
    df=df[df.gene!=tf]; df.to_csv(f"{OUT}/{tf}_vs_all_genes_US.csv",index=False)
    print(tf,"done: heritability",round(hg[x],3),flush=True)
