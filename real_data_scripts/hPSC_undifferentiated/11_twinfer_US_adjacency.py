# TwINFER S, C, U (TwinScore procedure sec. 2, 4-5) for every TF x gene pair, from within-sample twin pairs of one timepoint, written as pySCENIC adjacencies.
#   weights: cell i -> 1/n_c(i) (singletons / no clone id = 1); ranks R_gi = weighted mid-ranks (== CP10k ranks: per-cell scaling does not change within-gene order)
#   twin pairs p=(a,b): all unordered pairs of cells of a clone (n_c>=2) in this sample, w_p = 1/P_c; mu_g = sum_p w_p (R_ga+R_gb)/2 / N_t ; r = R - mu_g ; M_g = sum_p w_p (r_ga^2+r_gb^2)/2
#   S = sum_p w_p (r_xa r_ya + r_xb r_yb)/2 / sqrt(Mx My) ;  C = sum_p w_p (r_xa r_yb + r_xb r_ya)/2 / sqrt(Mx My) ;  U = S - C
# Adjacency edges: for each TF the top-K targets by |U| (variant U) or |S| (variant S), K = that TF's edge count in the GRNBoost2 adjacency (same budget per TF); importance = |value| (pySCENIC takes the TF-target sign from expression).
# env: TP, EXPR (loom, genes x cells), CLONES (h5ad for clone ids), GRB (GRNBoost2 adjacencies tsv), OUT (dir). Run with the pyscenic venv python.
import os, time, h5py, loompy, numpy as np, pandas as pd
t0=time.time(); P=lambda *x: print(f"[{time.time()-t0:.0f}s]",*x,flush=True)
R="/gpfs/projects/b1255/hzhang/TwINFER_KA/"; TP=os.environ["TP"]; O=os.environ.get("OUT",R+f"analysis_data/hPSC_undifferentiated/pyscenic/{TP}/"); os.makedirs(O,exist_ok=True)
EXPR=os.environ.get("EXPR",O+f"expr_{TP}.loom"); GRB=os.environ.get("GRB",O+f"adjacencies_{TP}.tsv"); H5=os.environ.get("CLONES",R+"finalized_data/hPSC_undifferentiated_data/hPSC_undifferentiated.h5ad")
with loompy.connect(EXPR,mode="r") as ds: genes=np.array(ds.ra.Gene); cells=np.array(ds.ca.CellID); X=ds[:,:].astype(np.float32)   # genes x cells
P("expr",X.shape)
TAG=os.environ.get("TAG","")   # suffix for every output file (e.g. _resid)
if os.environ.get("RESID"):   # regress out library complexity: per gene remove the linear effect of log(genes detected) and log(total counts) of the cell (env COV_COLS, COV_H5)
    if os.environ.get("COV_CSV"):   # ready-made covariate table (CellID index, already-transformed columns named in COV_COLS), e.g. cell-cycle scores + complexity + replicate
        _cols=os.environ["COV_COLS"].split(","); Cv=pd.read_csv(os.environ["COV_CSV"],index_col=0).reindex(cells)[_cols].values.astype(float)
    else:
        _h=h5py.File(os.environ.get("COV_H5",os.environ.get("CLONES",R+"finalized_data/hPSC_undifferentiated_data/hPSC_undifferentiated.h5ad")),"r"); _o=_h["obs"]; _oi=_o.attrs["_index"]
        _nm=np.array([s.decode() if isinstance(s,bytes) else s for s in _o[_oi][:]]); _cols=os.environ.get("COV_COLS","n_genes_by_counts,total_counts").split(",")
        Cv=np.column_stack([np.log(pd.Series(_o[c][:],index=_nm).reindex(cells).values.astype(float)) for c in _cols])
    Cv=Cv-Cv.mean(0); Ci=np.linalg.inv(Cv.T@Cv).astype(np.float32); Cv=Cv.astype(np.float32)
    for s0 in range(0,X.shape[0],1000):
        B=X[s0:s0+1000]; B-=B.mean(1,keepdims=True); B-=(B@Cv)@Ci@Cv.T; X[s0:s0+1000]=B
    P("regressed out",_cols,"; corr of residual expression with log n_genes (median |r|):",round(float(np.median(np.abs(np.corrcoef(X[:300],Cv[:,0])[-1,:-1]))),4))
if os.environ.get("CLONES_CSV"):   # clone table (CellID, clone) instead of the hPSC h5ad column
    cl=pd.read_csv(os.environ["CLONES_CSV"]).set_index("CellID")["clone"].reindex(cells).values; n=len(cells)
else:
    f=h5py.File(H5,"r"); o=f["obs"]; oi=o.attrs["_index"]; names=np.array([s.decode() if isinstance(s,bytes) else s for s in o[oi][:]]); g=o["clone_id"]
    cats=np.array([c.decode() if isinstance(c,bytes) else c for c in g["categories"][:]],dtype=object); codes=g["codes"][:]; cl_all=np.where(codes<0,None,cats[np.clip(codes,0,None)])
    cl=pd.Series(cl_all,index=names).reindex(cells).values; n=len(cells)

lab=pd.Series(cl); nc=lab.map(lab.value_counts()).fillna(1).values.astype(float); nc[pd.isna(cl)]=1; w=1.0/nc   # cell weights
P("cells",n,"with clone id",int((~pd.isna(cl)).sum()))
# clone-weighted mid-ranks, per gene
def wranks(v,w):
    o=np.argsort(v,kind="stable"); vs=v[o]; ws=w[o]; cw=np.cumsum(ws); st=np.r_[0,np.flatnonzero(np.diff(vs))+1]; en=np.r_[st[1:]-1,len(vs)-1]
    before=np.where(st>0,cw[st-1],0.0); gw=cw[en]-before; mid=before+gw/2; out=np.empty(len(v)); grp=np.repeat(np.arange(len(st)),en-st+1); out[o]=mid[grp]; return out
# twin pairs
pos=pd.Series(np.arange(n)).groupby(pd.Series(cl)).apply(list); ia=[];ib=[];wp=[];Nt=0
for c,idx in pos.items():
    if len(idx)<2: continue
    Nt+=1; Pc=len(idx)*(len(idx)-1)//2
    for i in range(len(idx)):
        for j in range(i+1,len(idx)): ia.append(idx[i]); ib.append(idx[j]); wp.append(1.0/Pc)
ia=np.array(ia); ib=np.array(ib); wp=np.array(wp); P("Nt (clones >=2 cells)",Nt,"twin pairs",len(ia))
tw=np.unique(np.r_[ia,ib]); Rk=np.empty((len(genes),len(tw)),np.float32)
if os.environ.get("STRATA_COL"):   # stratified ranks: rank within strata of a confounder (keeps zeros tied; unlike linear residuals this does not distort sparse genes), rescaled to the full-sample rank scale
    _st=pd.read_csv(os.environ["COV_CSV"],index_col=0).reindex(cells)[os.environ["STRATA_COL"]].values; _groups=[np.where(_st==u)[0] for u in pd.unique(_st)]; Wtot=w.sum(); Wg=[w[g].sum() for g in _groups]
    for k in range(len(genes)):
        v=X[k].astype(np.float64); r=np.empty(n)
        for g,wg in zip(_groups,Wg): r[g]=wranks(v[g],w[g])*(Wtot/wg)
        Rk[k]=r[tw]
    P("stratified ranks done:",len(_groups),"strata of",os.environ["STRATA_COL"])
else:
    for k in range(len(genes)): Rk[k]=wranks(X[k].astype(np.float64),w)[tw]
P("ranks done"); loc={c:i for i,c in enumerate(tw)}; la=np.array([loc[i] for i in ia]); lb=np.array([loc[i] for i in ib])
CH=2000; npair=len(wp); sw=np.sqrt(wp); G_=len(genes)                       # memory: work on chunks of twin pairs, store float32
mu=np.zeros(G_)
for s in range(0,npair,CH): sl=slice(s,s+CH); mu+=(wp[sl][None,:]*(Rk[:,la[sl]].astype(np.float64)+Rk[:,lb[sl]])/2).sum(1)
mu/=Nt; Mg=np.zeros(G_); Aw=np.empty((G_,npair),np.float32); Bw=np.empty((G_,npair),np.float32)   # centred ranks (genes x pairs), scaled by sqrt(w_p)
for s in range(0,npair,CH):
    sl=slice(s,s+CH); ra=Rk[:,la[sl]].astype(np.float64)-mu[:,None]; rb=Rk[:,lb[sl]].astype(np.float64)-mu[:,None]
    Mg+=(wp[sl][None,:]*(ra**2+rb**2)/2).sum(1); Aw[:,sl]=ra*sw[sl]; Bw[:,sl]=rb*sw[sl]
del Rk; P("centred ranks done")
adj=pd.read_csv(GRB,sep="\t"); tfs=[t for t in pd.unique(adj.TF) if t in set(genes)]; gi={x:i for i,x in enumerate(genes)}; ti=np.array([gi[t] for t in tfs])
den=np.sqrt(np.outer(Mg[ti],Mg)).astype(np.float32); den[den==0]=np.inf   # genes with no variance among twin cells -> 0
S=((Aw[ti]@Aw.T+Bw[ti]@Bw.T)/2)/den; C=((Aw[ti]@Bw.T+Bw[ti]@Aw.T)/2)/den; U=S-C; P("S,C,U computed",S.shape)
if os.environ.get("SAVE_FULL"):   # full TF x gene matrices (for regulator-profile gene clustering); SKIP_ADJ=1 stops here without rewriting the adjacencies
    np.savez_compressed(O+f"twinfer_US_full_{TP}{TAG}.npz",U=U.astype(np.float32),S=S.astype(np.float32),tfs=np.array(tfs),genes=genes); P("saved full U,S matrices")
    if os.environ.get("SKIP_ADJ"): raise SystemExit(0)
K=adj.groupby("TF").size()
pd.DataFrame({"gene":genes,"M":Mg}).to_csv(O+f"twinfer_M_{TP}{TAG}.csv",index=False)
for nm,Mx in (("U",U),("S",S)):
    rows=[]
    for r,t in enumerate(tfs):
        v=np.abs(Mx[r]).copy(); v[gi[t]]=-1; k=int(K[t]); top=np.argpartition(-v,k)[:k]; top=top[np.argsort(-v[top])]
        rows.append(pd.DataFrame({"TF":t,"target":genes[top],"importance":v[top]}))
    out=pd.concat(rows); out=out[out.importance>0]; out.to_csv(O+f"adjacencies_{nm}{TAG}_{TP}.tsv",sep="\t",index=False); P(nm,"adjacency",len(out),"edges for",len(tfs),"TFs; |value| median",round(float(out.importance.median()),3))
# full-matrix summary (for the report)
dS=np.array([S[r,gi[t]] for r,t in enumerate(tfs)]); hC=np.array([C[r,gi[t]] for r,t in enumerate(tfs)])
with open(O+f"twinfer_US_summary_{TP}{TAG}.txt","w") as fh:
    fh.write(f"TP={TP} Nt={Nt} twin_pairs={len(ia)} cells={n} genes={len(genes)} TFs={len(tfs)}\nS(x,x) mean {dS.mean():.3f} (should be 1); h_x=C(x,x) median {np.median(hC):.3f} range {hC.min():.3f}..{hC.max():.3f}; |S|max {np.abs(S).max():.3f} |C|max {np.abs(C).max():.3f}\nmedian S={np.median(S):.4f} median C={np.median(C):.4f} median U={np.median(U):.4f}; SD S={S.std():.4f} U={U.std():.4f}; corr(S,U)={np.corrcoef(S.ravel()[::50],U.ravel()[::50])[0,1]:.3f}; zero-variance genes {int((Mg==0).sum())}\n")
print(open(O+f"twinfer_US_summary_{TP}{TAG}.txt").read())
