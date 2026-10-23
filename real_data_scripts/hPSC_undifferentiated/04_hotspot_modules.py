import anndata as ad, numpy as np, pandas as pd, scanpy as sc, hotspot, time, warnings
warnings.filterwarnings("ignore")
R="/gpfs/projects/b1255/hzhang/TwINFER_KA/"; O=R+"analysis_data/hPSC_undifferentiated/hotspot/"
t0=time.time(); P=lambda *x: print(f"[{time.time()-t0:.0f}s]",*x,flush=True)
a=ad.read_h5ad(R+"finalized_data/hPSC_undifferentiated_data/hPSC_undifferentiated.h5ad")
tfs=set(l.strip() for l in open(R+"analysis_data/hPSC_20260927/gene_modules/all_tfs.txt"))
keep=np.diff(a.X.tocsc().indptr)/a.n_obs>=0.05
a=a[:,keep].copy(); P(a.shape)
sc.pp.highly_variable_genes(a,n_top_genes=3000,flavor="seurat")
b=a[:,a.var.highly_variable].copy(); b.X=b.X.toarray(); sc.pp.scale(b,max_value=10); sc.tl.pca(b,n_comps=30)
hv=set(b.var_names)|set(g for g in tfs if g in a.var_names)
a.obsm["X_pca"]=b.obsm["X_pca"]; del b
a=a[:,[g in hv for g in a.var_names]].copy(); a.X=a.X.toarray().astype("float32"); P("pca",a.shape)
a.layers["lp"]=a.X
hs=hotspot.Hotspot(a,layer_key="lp",model="normal",latent_obsm_key="X_pca")
hs.create_knn_graph(weighted_graph=False,n_neighbors=30); P("knn")
res=hs.compute_autocorrelations(jobs=16); P("autocorr"); res.to_csv(O+"autocorrelations_all_genes.csv")
prev=set(pd.read_csv(R+"analysis_data/hPSC_undifferentiated/de_tfs/DE_TFs_T0_to_T15_min5pct.csv").gene)
cp=pd.read_csv(R+"analysis_data/hPSC_undifferentiated/clone_de/clonepaired_DE_T0_to_T15.csv")
prev|=set(cp[cp.is_TF&(cp.FDR_t_TFonly<0.05)&(cp.log2FC.abs()>0.25)].gene)
sig=res[res.FDR<0.05].sort_values("Z",ascending=False)
genes=list(dict.fromkeys(list(sig.head(400).index)+[g for g in prev if g in res.index and res.loc[g,"FDR"]<0.05]))
P(len(sig),"autocorr-significant;",len(genes),"genes for modules")
hs.compute_local_correlations(genes,jobs=16); P("local corr"); hs.local_correlation_z.to_csv(O+"local_corr_Z.csv")
mod=hs.create_modules(min_gene_threshold=10,core_only=False,fdr_threshold=0.05); P("modules"); print(mod.value_counts().sort_index().to_string())
mod.rename("module").to_csv(O+"gene_modules.csv")
ms=hs.calculate_module_scores(); ms.to_csv(O+"module_scores_per_cell.csv"); P("scores")
