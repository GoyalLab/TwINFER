# GRNBoost2 (arboreto) on hPSC_undifferentiated, one timepoint per run (env TP = T0|T10|T15; or SLURM_ARRAY_TASK_ID 0,1,2).
# Cells: all cells of that timepoint. Genes: all genes expressed (>0) in >5% of ALL cells (shared gene set so the 3 networks are comparable).
# Regulators: TFs from analysis_data/hPSC_20260927/gene_modules/all_tfs.txt that are in the gene set. Input: log1pPF as stored in the h5ad.
# Targets are run in chunks (env CHUNK, default 500) and each chunk is saved, so an interrupted job resumes. Test knobs: NTARGETS (first N non-TF targets), NCELLS (random cell subsample).
import os, sys, time, gzip, anndata as ad, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from dask.distributed import Client, LocalCluster
from arboreto.core import create_graph, SGBM_KWARGS, EARLY_STOP_WINDOW_LENGTH
t0=time.time(); P=lambda *x: print(f"[{time.time()-t0:.0f}s]",*x,flush=True)
R="/gpfs/projects/b1255/hzhang/TwINFER_KA/"; TP=os.environ.get("TP") or ["T0","T10","T15"][int(os.environ["SLURM_ARRAY_TASK_ID"])]
OUT=os.environ.get("OUTDIR",R+"analysis_data/hPSC_undifferentiated/grnboost2/")+TP+"/"; os.makedirs(OUT,exist_ok=True)
CH=int(os.environ.get("CHUNK","500")); NW=int(os.environ.get("NWORKERS","16")); NT=os.environ.get("NTARGETS"); NC=os.environ.get("NCELLS"); SEED=666
b=ad.read_h5ad(R+"finalized_data/hPSC_undifferentiated_data/hPSC_undifferentiated.h5ad",backed="r"); var=np.array(b.var_names)
nnz=np.zeros(b.n_vars)
for s in range(0,b.n_obs,5000): nnz+=np.bincount(b.X[s:s+5000].indices,minlength=b.n_vars)
frac=nnz/b.n_obs; keep=frac>0.05; P("genes >5% of all cells:",int(keep.sum()),"of",b.n_vars)
tfs_all=[l.strip() for l in open(R+"analysis_data/hPSC_20260927/gene_modules/all_tfs.txt") if l.strip()]; gset=set(var[keep]); tfs=[t for t in dict.fromkeys(tfs_all) if t in gset]
genes=[g for g in var[keep]]
if NT: nontf=[g for g in genes if g not in set(tfs)][:int(NT)]; genes=sorted(set(tfs)|set(nontf),key=lambda g:list(var).index(g))
P(TP,"regulators (TFs in gene set):",len(tfs),"; genes:",len(genes))
tpv=b.obs.timepoint.astype(str).values; rows=np.where(tpv==TP)[0]
if NC: rows=np.sort(np.random.default_rng(0).choice(rows,int(NC),replace=False))
A=b[:,genes].to_memory(); X=A.X[rows]; X=np.asarray(X.toarray() if hasattr(X,"toarray") else X,dtype=np.float32); del A; P("cells",X.shape[0],"matrix",X.shape)
targets=[g for g in genes]; chunks=[targets[i:i+CH] for i in range(0,len(targets),CH)]; P(len(chunks),"chunks of <=",CH,"targets")
cluster=LocalCluster(n_workers=NW,threads_per_worker=1,memory_limit="0",processes=True); client=Client(cluster); P("dask workers:",len(client.scheduler_info()["workers"]))
for i,ch in enumerate(chunks):
    f=f"{OUT}chunk_{i:03d}.csv.gz"
    if os.path.exists(f): P("skip existing",f); continue
    t1=time.time(); g=create_graph(X,genes,tfs,"GBM",SGBM_KWARGS,client,target_genes=ch,early_stop_window_length=EARLY_STOP_WINDOW_LENGTH,seed=SEED)
    df=client.compute(g,sync=True); df.to_csv(f,index=False,compression="gzip"); P(f"chunk {i+1}/{len(chunks)}: {len(ch)} targets, {len(df)} links, {time.time()-t1:.0f}s")
client.close(); cluster.close()
allv=pd.concat([pd.read_csv(f"{OUT}chunk_{i:03d}.csv.gz") for i in range(len(chunks))],ignore_index=True).sort_values("importance",ascending=False)
allv.to_csv(OUT+f"grnboost2_{TP}_links.csv.gz",index=False,compression="gzip"); P("links",len(allv),"; TFs with links",allv.TF.nunique(),"; targets",allv.target.nunique()); print(allv.head(15).to_string(index=False))
