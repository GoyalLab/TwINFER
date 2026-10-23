"""Per-sample TwinScore inputs for the Novo Nordisk hPSC endoderm dataset (same QC-folder convention as finalized_data/FM01_data/qc_filtered).
For each SampleID: finalized_data/hPSC_Novo_Nordisk_data/qc_filtered/<SampleID>/{counts.npz (cells x genes, int32 raw RNA counts, csr), genes.txt, obs_metadata.csv}
obs_metadata.csv has fatemap_clone_singletcode = the authors' clone.id (as a string). Only cells with a clone.id are kept (cells without a clone are not scored by twinscore_pdf_candidates.py anyway).
QC as in the hPSC_20260927 pipeline (handoff 2026-10-01): n_genes >= 1000, mito <= 10% (Seurat nFeature_RNA / percent.mt; 0 cells fail here).
Counts are read from the raw binary CSC export (export_seurat.R) with memory maps, so memory stays small.
"""
import os, sys
import numpy as np, pandas as pd, scipy.sparse as sp

EXP = "/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/export"
OUT = "/home/gzu5140/TwINFER_KA/finalized_data/hPSC_Novo_Nordisk_data/qc_filtered"
only = sys.argv[1:]                                                    # optional: SampleIDs to build

nr, nc = [int(v) for v in open(f"{EXP}/RNA_counts_dim.txt").read().split()]
genes = pd.read_csv(f"{EXP}/RNA_counts_features.csv")["feature"].astype(str).values; assert len(genes) == nr
cells = pd.read_csv(f"{EXP}/cells.csv")["cell"].astype(str).values; assert len(cells) == nc
p = np.fromfile(f"{EXP}/RNA_counts_p.bin", dtype=np.float64).astype(np.int64)
I = np.memmap(f"{EXP}/RNA_counts_i.bin", dtype=np.int32, mode="r"); Xv = np.memmap(f"{EXP}/RNA_counts_x.bin", dtype=np.float64, mode="r")
obs = pd.read_csv(f"{EXP}/obs.csv", index_col=0, low_memory=False); assert (obs.index.astype(str).values == cells).all()

for s in sorted(obs["SampleID"].astype(str).unique()):
    if only and s not in only: continue
    m = (obs["SampleID"].astype(str) == s) & obs["clone.id"].notna() & (obs["nFeature_RNA"] >= 1000) & (obs["percent.mt"] <= 10)
    cols = np.flatnonzero(m.values)
    n_drop_qc = int(((obs["SampleID"].astype(str) == s) & obs["clone.id"].notna()).sum() - len(cols))
    nnz = p[cols + 1] - p[cols]; indptr = np.r_[0, np.cumsum(nnz)]
    ii = np.empty(indptr[-1], dtype=np.int32); xx = np.empty(indptr[-1], dtype=np.int32)
    for k, j in enumerate(cols):
        a, b = p[j], p[j + 1]; v = Xv[a:b]
        assert np.array_equal(v, np.rint(v)); ii[indptr[k]:indptr[k + 1]] = I[a:b]; xx[indptr[k]:indptr[k + 1]] = v
    X = sp.csc_matrix((xx, ii, indptr), shape=(nr, len(cols))).T.tocsr()     # cells x genes
    d = f"{OUT}/{s}"; os.makedirs(d, exist_ok=True)
    sp.save_npz(f"{d}/counts.npz", X, compressed=False)
    open(f"{d}/genes.txt", "w").write("\n".join(genes) + "\n")
    o = obs.iloc[cols][["SampleID", "StageID", "clust_annot_02", "clone.id", "cell.barcode.ct"]].copy()
    o.insert(0, "fatemap_clone_singletcode", o["clone.id"].astype(int).astype(str)); o = o.drop(columns="clone.id")
    o.index.name = "cell"; o.to_csv(f"{d}/obs_metadata.csv")
    cs = o["fatemap_clone_singletcode"].value_counts()
    print(f"{s}: {X.shape[0]} cells x {X.shape[1]} genes, nnz {X.nnz}, QC-dropped {n_drop_qc}, clones {len(cs)}, N_t {int((cs >= 2).sum())}", flush=True)
