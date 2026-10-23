#!/usr/bin/env python3
"""Stage the Schaffer 2026 naive melanoma data (GSE279162 Naive1-3 merged into ONE sample) for twinscore_pdf_candidates.py.
Cells = singletCode singlets (analysis_data/schaffer_lineage_table/singletcode/cell_clone.csv); clone = the singletCode barcode combination (pooled over the 3 samples).
Counts = raw GEO gene counts (human). Candidates = union of the FM01 and FM06 stage1_tf_ge20 TF x target pair lists (the script itself then keeps pairs whose genes are detected in >=5% of these cells).
Writes analysis_data/stage1_inputs/schaffer/{counts.npz,genes.txt,obs_metadata.csv} and analysis_data/schaffer_naive/data/stage1_tf_ge20/candidates.tsv"""
import tarfile, gzip, io, os
import numpy as np, pandas as pd, scipy.sparse as sp
ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"; T = f"{ROOT}/real_data/schaffer_2026_cell_genomics/GSE279162_RAW.tar"
ST = f"{ROOT}/analysis_data/stage1_inputs/schaffer"; os.makedirs(ST, exist_ok=True)
CC = pd.read_csv(f"{ROOT}/analysis_data/schaffer_lineage_table/singletcode/cell_clone.csv")
tf = tarfile.open(T); rd = lambda n: gzip.open(io.BytesIO(tf.extractfile(n).read()))
Xs, obs, genes0 = [], [], None
for s, g in [("Naive1", "GSM8562999"), ("Naive2", "GSM8563000"), ("Naive3", "GSM8563001")]:
    p = f"{g}_{s}_filtered_"; feat = pd.read_csv(rd(p + "features.tsv.gz"), sep="\t", header=None); bc = pd.read_csv(rd(p + "barcodes.tsv.gz"), header=None)[0].to_numpy()
    gene_rows = np.where(feat[2] == "Gene Expression")[0]; ng = len(gene_rows); assert (gene_rows == np.arange(ng)).all()
    f = rd(p + "matrix.mtx.gz")
    while True:
        l = f.readline()
        if not l.startswith(b"%"): break
    nr, nc, nn = map(int, l.split()); a = pd.read_csv(f, sep=" ", header=None, names=["r", "c", "v"], dtype=np.int64); a = a[a.r <= ng]
    M = sp.csr_matrix((a.v, (a.c - 1, a.r - 1)), shape=(nc, ng)); del a
    sub = CC[CC["sample"] == s].set_index("cellID"); idx = pd.Series(np.arange(nc), index=bc).reindex(sub.index)
    assert idx.notna().all(); M = M[idx.to_numpy().astype(int)]
    Xs.append(M); obs.append(pd.DataFrame({"sample": s, "cell_barcode": sub.index, "fatemap_clone_singletcode": sub.clone.to_numpy()}, index=[f"{s}:{c}" for c in sub.index]))
    genes = feat[1].to_numpy()[:ng]
    if genes0 is None: genes0 = genes
    assert (genes == genes0).all(); print(s, M.shape, "integer:", bool((M.data == np.round(M.data)).all()), flush=True)
X = sp.vstack(Xs).tocsr(); obs = pd.concat(obs)
gu = pd.Series(genes0); dup = gu.duplicated(keep=False); gu[gu.duplicated()] = gu[gu.duplicated()] + "_dup" + gu[gu.duplicated()].groupby(gu).cumcount().astype(str) if dup.any() else gu
sp.save_npz(f"{ST}/counts.npz", X.astype(np.float32)); open(f"{ST}/genes.txt", "w").write("\n".join(gu) + "\n"); obs.to_csv(f"{ST}/obs_metadata.csv")
print("merged", X.shape, "cells", len(obs), "clones", obs.fatemap_clone_singletcode.nunique(), "duplicated gene symbols", int(dup.sum()))
c = pd.concat([pd.read_csv(f"{ROOT}/analysis_data/{d}/data/stage1_tf_ge20/candidates.tsv", sep="\t", header=None) for d in ("fm01", "fm06")]).drop_duplicates()
od = f"{ROOT}/analysis_data/schaffer_naive/data/stage1_tf_ge20"; os.makedirs(od, exist_ok=True); c.to_csv(f"{od}/candidates.tsv", sep="\t", header=False, index=False)
print("candidates: FM01 U FM06 =", len(c), "pairs;", c[0].nunique(), "TFs;", c[1].nunique(), "targets")
