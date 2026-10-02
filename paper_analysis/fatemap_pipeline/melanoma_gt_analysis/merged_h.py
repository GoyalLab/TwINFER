import sys, json, numpy as np, pandas as pd
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0,"/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline")
from paper_analysis.fatemap_pipeline import twinscore_spec_fm06 as T
supp=T.supp; L=T.L
from twinfer.inference.correlation_functions import assign_twin_id, split_twins, get_unit_weights
gs="correlation_high_clean_30x6"
df,Xc,tot,genes,pidx,gf=T.load_raw(gs)
extra=["SNAI2","FOSL1","TEAD1"]
gidx={g:i for i,g in enumerate(gf)}
genes=genes+[g for g in extra if g not in genes and g in gidx]
pidx=np.array([gidx[g] for g in genes])
clone=df.barcode.to_numpy(); m=T.clone_filter(pd.DataFrame(dict(c=clone)),"c")
V=T.panel_expression(Xc[np.flatnonzero(m)],tot[np.flatnonzero(m)],pidx,None,gf)
clone=clone[m]; cid=df.cell_id.to_numpy()[m]
print("cells",len(cid),"clones",len(set(clone)),"genes",len(genes),flush=True)
def hdiag(cl,rng=None):
    fr=pd.DataFrame(dict(cell_id=cid,clone_id=cl)); tw=assign_twin_id(fr); a0,b0=split_twins(tw)
    pos=pd.Series(np.arange(len(cid)),index=cid)
    ia,ib=pos[a0.cell_id.to_numpy()].to_numpy(),pos[b0.cell_id.to_numpy()].to_numpy()
    w=get_unit_weights(a0,unit="clone")
    S,C=L.full_depth_layers(V,ia,ib,w); return np.diag(C).copy()
h=hdiag(clone); rng=np.random.default_rng(0)
nul=np.array([hdiag(rng.permutation(clone)) for _ in range(30)])
nf=nul.std(0,ddof=1)
out=pd.DataFrame(dict(gene=genes,h_merged=h,nf=nf,ratio=h/nf,pass2=h>2*nf))
out.to_csv(f"{sys.argv[1]}/merged_h.csv",index=False)
print(out.set_index("gene").loc[["JUN","JUND","FOSB","FOS","SOX10","SNAI2","MITF","FOSL1"]].round(3).to_string())
print("median h %.3f; JUN rank %d/%d"%(np.median(h),(out.h_merged.rank(ascending=False)[out.gene=="JUN"]).iloc[0],len(out)))
