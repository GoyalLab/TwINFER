"""Step 4: pull raw gene counts for the analysed cells out of one GEO matrix (run one task per GSM file).
Usage: python 04_extract_counts.py <GSM_prefix e.g. GSM9914171>
Keeps rows of cells in inputs/wanted_cells_with_cousins.csv (sister cells + random pool + cousins).
Matrix = cells x 78,258 genes, MatrixMarket 1-based; column j = row j of support_data/mouse.v37.geneID.txt (checked on top genes).
Writes <DTT_OUT>/counts_wanted/<GSM>.npz (csr counts) + <GSM>_cells.txt
"""
import sys, glob, os, time, numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # fig5_paths.py lives in preprocessing_code/
from fig5_paths import DTT_GEO, DTT_REPO, FINAL, INPUT, REBUILD      # noqa: E402
DTT_OUT = os.environ.get("DTT_OUT", os.path.join(REBUILD, "DTT_mouse", "F"))      # rebuilt files; final_data is never overwritten
gsm = sys.argv[1]; OUT = DTT_OUT + '/counts_wanted/'; os.makedirs(OUT, exist_ok=True)
cellf = glob.glob(DTT_GEO + f'/{gsm}_*.df_cell.csv.gz')[0]; mtxf = cellf.replace('.df_cell.csv.gz', '.gene_count.mtx.gz')
want = pd.read_csv(INPUT + '/wanted_cells_with_cousins.csv').cell_id.tolist(); wset = {c: i for i, c in enumerate(want)}
ids = pd.read_csv(cellf, header=None).iloc[:, 0].astype(str).values
rowmap = np.array([wset.get(c, -1) for c in ids], np.int64)
print(gsm, len(ids), 'cells,', int((rowmap >= 0).sum()), 'wanted', flush=True); t0 = time.time()
hdr = pd.read_csv(mtxf, sep=' ', header=None, skiprows=1, nrows=1).iloc[0].tolist(); nr, ng, nnz = map(int, hdr); assert nr == len(ids)
R_, C_, V_ = [], [], []
for ch in pd.read_csv(mtxf, sep=' ', header=None, skiprows=2, dtype=np.int32, chunksize=10_000_000):
    r = ch[0].values - 1; m = rowmap[r] >= 0
    R_.append(rowmap[r[m]].astype(np.int32)); C_.append(ch[1].values[m] - 1); V_.append(ch[2].values[m].astype(np.float32))
R_, C_, V_ = map(np.concatenate, (R_, C_, V_))
X = sp.csr_matrix((V_, (R_, C_)), shape=(len(want), ng)); keep = np.flatnonzero(np.diff(X.indptr) > 0)
sp.save_npz(OUT + gsm + '.npz', X[keep]); open(OUT + gsm + '_cells.txt', 'w').write('\n'.join(np.array(want)[keep]) + '\n')
print('done', len(keep), 'cells with counts,', X.nnz, 'nnz,', round(time.time() - t0), 's', flush=True)
