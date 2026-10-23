# Build the per-timepoint expression loom (genes x cells) + regulator list for pySCENIC. Run with the pyscenic venv python (no anndata needed: reads the h5ad with h5py).
# Cells: all cells of TP (env TP). Genes: expressed (>0) in >5% of ALL cells (shared across the 3 timepoints). Regulators: TFs from all_tfs.txt present in the gene set.
# Values are the stored log1pPF. Test knobs: NCELLS (first N cells of the timepoint), NGENES (first N genes passing the filter, TFs always kept).
import os, time, h5py, numpy as np, loompy
t0=time.time(); P=lambda *x: print(f"[{time.time()-t0:.0f}s]",*x,flush=True)
R="/gpfs/projects/b1255/hzhang/TwINFER_KA/"; TP=os.environ["TP"]; OUT=os.environ.get("OUTDIR",R+"analysis_data/hPSC_undifferentiated/pyscenic/")+TP+"/"; os.makedirs(OUT,exist_ok=True)
f=h5py.File(R+"finalized_data/hPSC_undifferentiated_data/hPSC_undifferentiated.h5ad","r"); X=f["X"]; ip=X["indptr"][:]; n=len(ip)-1
idx=f["var"].attrs["_index"]; names=np.array([s.decode() if isinstance(s,bytes) else s for s in f["var"][idx][:]]); G=len(names)
o=f["obs"]["timepoint"]; tp=np.array([c.decode() if isinstance(c,bytes) else c for c in o["categories"][:]])[o["codes"][:]]
oi=f["obs"].attrs["_index"]; cells=np.array([s.decode() if isinstance(s,bytes) else s for s in f["obs"][oi][:]])
nnz=np.zeros(G); S=5000
for s in range(0,n,S): nnz+=np.bincount(X["indices"][ip[s]:ip[min(s+S,n)]],minlength=G)
keep=nnz/n>0.05; P("genes >5% of all cells:",int(keep.sum()),"of",G)
tfs=[l.strip() for l in open(R+"analysis_data/hPSC_20260927/gene_modules/all_tfs.txt") if l.strip()]; gs=set(names[keep]); tfs=[t for t in dict.fromkeys(tfs) if t in gs]
sel=keep.copy()
if os.environ.get("NGENES"): 
    ng=int(os.environ["NGENES"]); cand=np.where(keep)[0][:ng]; sel=np.zeros(G,bool); sel[cand]=True; sel|=np.isin(names,tfs)&keep
colmap=-np.ones(G,int); colmap[sel]=np.arange(sel.sum()); genes=names[sel]; tfs=[t for t in tfs if t in set(genes)]
rows=np.where(tp==TP)[0]
if os.environ.get("NCELLS"): rows=rows[:int(os.environ["NCELLS"])]
inrow=np.zeros(n,bool); inrow[rows]=True; blocks=[]
for s in range(0,n,S):
    e=min(s+S,n); r=np.where(inrow[s:e])[0]
    if len(r)==0: continue
    a,b=ip[s],ip[e]; dat=X["data"][a:b]; ind=X["indices"][a:b]; ptr=ip[s:e+1]-a
    from scipy.sparse import csr_matrix
    M=csr_matrix((dat,ind,ptr),shape=(e-s,G))[r][:,np.where(sel)[0]]; blocks.append(M.toarray().astype(np.float32))
E_=np.vstack(blocks); cn=cells[rows]; P(TP,"cells",E_.shape[0],"genes",E_.shape[1],"regulators",len(tfs))
loompy.create(OUT+f"expr_{TP}.loom",E_.T.copy(),row_attrs={"Gene":genes},col_attrs={"CellID":cn}); open(OUT+"tfs_in_geneset.txt","w").write("\n".join(tfs)+"\n"); P("wrote",OUT+f"expr_{TP}.loom")
