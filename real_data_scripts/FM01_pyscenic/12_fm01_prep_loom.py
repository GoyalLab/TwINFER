# FM01 (melanoma) -> pySCENIC inputs. Raw integer counts layer -> counts per 10,000 (CP10k) -> log1p (standard scRNA-seq normalization; NOT the log1pPF used for hPSC). All 10,475 cells, one sample (replicates A/B pooled).
# Genes: expressed (>0) in >5% of cells. Regulators: analysis_data/fm01/gene_modules/all_tfs.txt within the gene set. Also writes the clone table (fatemap_clone_singletcode).
import os, time, h5py, loompy, numpy as np, pandas as pd
from scipy.sparse import csr_matrix
t0=time.time(); P=lambda *x: print(f"[{time.time()-t0:.0f}s]",*x,flush=True)
R="/gpfs/projects/b1255/hzhang/TwINFER_KA/"; OUT=os.environ.get("OUTDIR",R+"analysis_data/fm01/pyscenic/"); os.makedirs(OUT,exist_ok=True)
f=h5py.File(R+"finalized_data/FM01_data/FM01_integrated.h5ad","r"); L=f["layers"]["counts"]; ip=L["indptr"][:]; n=len(ip)-1
dec=lambda a:np.array([s.decode() if isinstance(s,bytes) else s for s in a],dtype=object)
vi=f["var"].attrs["_index"]; genes=dec(f["var"][vi][:]); G=len(genes); oi=f["obs"].attrs["_index"]; cells=dec(f["obs"][oi][:])
def ocol(nm):
    g=f["obs"][nm]
    if isinstance(g,h5py.Group): cats=dec(g["categories"][:]); c=g["codes"][:]; return np.where(c<0,None,cats[np.clip(c,0,None)])
    return dec(g[:])
clone=ocol("fatemap_clone_singletcode"); rep=ocol("replicate")
M=csr_matrix((L["data"][:],L["indices"][:],ip),shape=(n,G)); P("counts",M.shape,"nnz",M.nnz)
tot=np.asarray(M.sum(1)).ravel(); P("median depth",np.median(tot),"min depth",tot.min())
nz=np.diff(M.tocsc().indptr)/n; keep=nz>0.05; P("genes >5% of cells:",int(keep.sum()),"of",G)
Xn=csr_matrix(M.multiply((1e4/tot)[:,None])); Xn.data=np.log1p(Xn.data); X=Xn[:,np.where(keep)[0]].toarray().astype(np.float32); gk=genes[keep]
tfs=[l.strip() for l in open(R+"analysis_data/fm01/gene_modules/all_tfs.txt") if l.strip()]; gs=set(gk); tfs=[t for t in dict.fromkeys(tfs) if t in gs]; P("regulators in gene set:",len(tfs))
loompy.create(OUT+"expr_FM01.loom",X.T.copy(),row_attrs={"Gene":gk},col_attrs={"CellID":cells}); open(OUT+"tfs_in_geneset.txt","w").write("\n".join(tfs)+"\n")
pd.DataFrame({"CellID":cells,"clone":clone,"replicate":rep}).to_csv(OUT+"clones_FM01.csv",index=False); P("wrote loom, tfs, clones; values max",round(float(X.max()),2),"; normalization log1p(CP10k)")
