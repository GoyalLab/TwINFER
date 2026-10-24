"""Build the hPSC_diff_endo matrices with EXACTLY the hPSC recipe (exports/_probe_tmp/fanout_delta/HPSC_REPRODUCTION.md).

1. Cell QC: cellranger filtered_feature_bc_matrix.h5, pct_counts_mt < 20 and n_genes_by_counts >= 200.
2. clone_c5: singletCode on the per-read table (cellID, BC50StarcodeD8), min_umi_cutoff 5, umi_cutoff_ratio 0,
   umi_dominant_threshold 30, umi_diff_threshold 0; barcode = most unique UMIs, ties to the alphabetically first.
   clone_id: the same with singletCode defaults.
3. Spliced layer: kb count --workflow=nac (cell.mtx = mature + ambiguous), cell set = the QC cells.
Outputs:
  datasets/twinfer_diff_endo/endo_counts_mito20_min200genes.h5ad
  datasets/twinfer_diff_endo/endo_kbnac_spliced_unspliced.h5ad
"""
import os,sys,warnings
import numpy as np,pandas as pd,scanpy as sc,anndata as ad,scipy.sparse as ss,scipy.io as sio
warnings.filterwarnings("ignore")
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))      # fig5_paths.py lives in preprocessing_code/
from fig5_paths import HPSC_HANXIAO,REBUILD,YSCHER_ENDO      # noqa: E402     (singletCode below = the installed package, identical to her pylibs copy)
from singletCode.get_singlets import get_singlets
CR=f"{HPSC_HANXIAO}/cellranger_out"
S3=os.environ.get("ENDO_S3",os.path.join(YSCHER_ENDO,"sidereaction_d8","stepThree"))   # starcode D8 read table per sample
KB=os.path.join(YSCHER_ENDO,"kb_nac")
OUTD=os.environ.get("ENDO_OUT",os.path.join(REBUILD,"hPSC_endo_yscher"))      # rebuilds never write into yscher's tree or final_data
SAMPLES={"endo_T0":"T0","endo_T15":"T15"}
CRDIR={"endo_T0":"endo_T0","endo_T15":"endo_T1"}   # cellranger folder names on disk (the T15 sample is filed as endo_T1)

def clones(sample,**kw):
    raw=pd.read_csv(f"{S3}/{sample}/stepThreeStarcodeShavedReads.txt",sep="\t",usecols=["cellID","UMI","BC50StarcodeD8"])
    raw.columns=["cellID","UMI","barcode"]
    sh=raw[["cellID","barcode"]].copy(); sh["sample"]=sample
    sl,_=get_singlets(sh,**kw)
    cells=set(sl.cellID[sl.label=="Singlet"])
    u=raw[raw.cellID.isin(cells)].drop_duplicates(["cellID","UMI","barcode"]).groupby(["cellID","barcode"]).size().rename("n").reset_index()
    return u.sort_values(["cellID","n","barcode"],ascending=[True,False,True]).groupby("cellID").barcode.first()

blocks=[]
for s,tp in SAMPLES.items():
    A=sc.read_10x_h5(f"{CR}/{CRDIR[s]}/outs/filtered_feature_bc_matrix.h5"); A.var_names_make_unique()
    A.var["mt"]=A.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(A,qc_vars=["mt"],inplace=True,percent_top=None,log1p=False)
    A=A[(A.obs.pct_counts_mt<20)&(A.obs.n_genes_by_counts>=200)].copy()
    bc=A.obs_names.str.replace("-1","",regex=False)
    c5=clones(s,min_umi_cutoff=5,umi_cutoff_ratio=0.0,umi_dominant_threshold=30,umi_diff_threshold=0)
    cid=clones(s)                                             # singletCode defaults
    A.obs["cell_id"]=bc.values; A.obs["timepoint"]=tp
    A.obs["clone_c5"]=pd.Series(bc,index=A.obs_names).map(c5).values
    A.obs["clone_id"]=pd.Series(bc,index=A.obs_names).map(cid).values
    A.obs_names=[f"{tp}_{x}" for x in A.obs_names]
    A.obs=A.obs[["cell_id","clone_c5","clone_id","n_genes_by_counts","pct_counts_mt","timepoint","total_counts","total_counts_mt"]]
    v=A.obs.clone_c5.dropna().value_counts()
    print(f"{s} -> {tp}: QC cells {A.n_obs}; clone_c5 cells {int(A.obs.clone_c5.notna().sum())}, clone_id cells {int(A.obs.clone_id.notna().sum())}; "
          f"clones 2-50 {int(((v>=2)&(v<=50)).sum())} over {int(v[(v>=2)&(v<=50)].sum())} cells",flush=True)
    blocks.append(A)
C=ad.concat(blocks,join="outer"); C.X=C.X.tocsr() if ss.issparse(C.X) else ss.csr_matrix(C.X)
for c in ("clone_c5","clone_id","timepoint"): C.obs[c]=C.obs[c].astype("category")
C.write_h5ad(f"{OUTD}/endo_counts_mito20_min200genes.h5ad")
print("counts written:",C.shape,flush=True)

# spliced layer, cell set = the QC cells
genes=None; lay={k:[] for k in ("cell","nascent","mature","ambiguous")}; keep_rows=[]
for s,tp in SAMPLES.items():
    d=f"{KB}/{s}/counts_unfiltered"
    bcs=pd.Index([l.strip() for l in open(f"{d}/cells_x_genes.barcodes.txt")])
    ids=[l.strip() for l in open(f"{d}/cells_x_genes.genes.txt")]
    nms=[l.strip().split("\t")[-1] for l in open(f"{d}/cells_x_genes.genes.names.txt")]
    if genes is None: genes=(ids,nms)
    else: assert ids==genes[0], f"{s}: gene order differs"
    m=(C.obs.timepoint==tp).to_numpy(); want=pd.Index(C.obs.cell_id[m].astype(str))
    pos=bcs.get_indexer(want); miss=int((pos<0).sum())
    print(f"{s}: QC cells {m.sum()}, kb barcodes {len(bcs)}, missing in kb {miss}",flush=True)
    keep_rows.append(np.flatnonzero(m)[pos>=0])
    for k in lay:
        X=sio.mmread(f"{d}/cells_x_genes.{k}.mtx").tocsr()
        lay[k].append(X[pos[pos>=0]])
rows=np.concatenate(keep_rows)
obs=C.obs.iloc[rows].copy()
var=pd.DataFrame(index=pd.Index(genes[1]).astype(str)); var["gene_id"]=genes[0]; var["gene_symbol"]=genes[1]
S=ad.AnnData(X=ss.vstack(lay["cell"]).tocsr(),obs=obs,var=var)
S.var_names_make_unique()
S.layers["spliced"]=S.X.copy(); S.layers["unspliced"]=ss.vstack(lay["nascent"]).tocsr()
S.layers["mature"]=ss.vstack(lay["mature"]).tocsr(); S.layers["ambiguous"]=ss.vstack(lay["ambiguous"]).tocsr()
S.write_h5ad(f"{OUTD}/endo_kbnac_spliced_unspliced.h5ad")
print("spliced written:",S.shape,flush=True)
