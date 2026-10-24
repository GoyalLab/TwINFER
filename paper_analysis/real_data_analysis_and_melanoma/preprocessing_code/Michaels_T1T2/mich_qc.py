"""QC chain on the Michaels 2024 lineage-barcoding samples, by the project's own rule.

The chain, as helpers/qc.py states it and helpers/larry_unscrubbed_qc_chain.py runs it:
    mitochondrial content > 20% removed  ->  barcode merge  ->  singletCode  ->  fewer than 200
    genes discarded  ->  normalize.

Two differences from LARRY, both forced by the data rather than chosen:
  the mitochondrial prefix here is HUMAN, MT-, and qc.py derives and asserts it, so a wrong-case
  prefix halts instead of silently passing every cell;
  no cell type is filtered out -- this dataset's point is the T lineage, so the myeloid/mast/Ery
  cells stay in as the alternate fates the twins can take.

Barcode merge and singletCode need the cell<->barcode table, which GEO does not distribute for this
series (GSM7916616 is SRA only), so those two stages run separately once the reads are processed.
This stage does the mitochondrial and gene-count filters and records every count.

T1 = day 5 (48 h after transduction), T2 = day 10 (after the day-7 passage): the two timepoints of
one barcoded population, which is the twin structure rho_dagger and rho_hat_Delta need."""
import os, sys, json
import numpy as np, pandas as pd, scipy.io as sio, scipy.sparse as ss, anndata as ad
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))        # qc.py (copy of yscher's helpers/qc.py) next to this script
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # fig5_paths.py lives in preprocessing_code/
from qc import apply_qc, MITO_PCT_MAX, MIN_GENES      # noqa: E402
from fig5_paths import MICHAELS_GEO,REBUILD      # noqa: E402
RAW=MICHAELS_GEO                                                        # GEO matrices of T1 and T2 (raw_data link)
OUT=os.path.join(REBUILD,"Michaels_T1T2","processed"); os.makedirs(OUT,exist_ok=True)      # rebuilt QC h5ads; yscher's own are in final_data/Michaels_T1T2_counts_yscher
rec={}
for tag,gsm in (("T1","GSM7916614"),("T2","GSM7916615")):
    m=sio.mmread(f"{RAW}/{gsm}_{tag}_matrix.mtx.gz").T.tocsr()
    bc=pd.read_csv(f"{RAW}/{gsm}_{tag}_barcodes.tsv.gz",header=None)[0].to_numpy()
    ft=pd.read_csv(f"{RAW}/{gsm}_{tag}_features.tsv.gz",header=None,sep="\t")
    A=ad.AnnData(X=m.astype(np.float32),
                 obs=pd.DataFrame(index=[f"{tag}|{b}" for b in bc]),
                 var=pd.DataFrame(index=ft[1].astype(str).to_numpy()))
    A.var_names_make_unique()
    A.obs["timepoint"]=tag
    n0=A.n_obs
    B,qrec=apply_qc(A,label=tag)
    rec[tag]=dict(cells_in=int(n0),cells_out=int(B.n_obs),qc=qrec,
                  removed=int(n0-B.n_obs),genes=int(B.n_vars))
    B.write(f"{OUT}/michaels_{tag}_qc.h5ad")
    print("%-4s in %6d  out %6d  removed %6d  genes %6d"%(tag,n0,B.n_obs,n0-B.n_obs,B.n_vars),flush=True)
json.dump(dict(rule=dict(mito_pct_max=MITO_PCT_MAX,min_genes=MIN_GENES),per_sample=rec),
          open(f"{OUT}/qc_record.json","w"),indent=1)
print("\nrule applied: mitochondrial <= %g%%, genes >= %d"%(MITO_PCT_MAX,MIN_GENES),flush=True)
