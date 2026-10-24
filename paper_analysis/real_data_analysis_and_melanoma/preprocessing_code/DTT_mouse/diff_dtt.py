"""Element-wise comparison of the rebuilt DTT files with final_data/DTT_mouse_data."""
import glob, hashlib, json, os, sys
import numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # fig5_paths.py lives in preprocessing_code/
from fig5_paths import FINAL, REBUILD      # noqa: E402
FIN = os.path.join(FINAL, "DTT_mouse_data"); W = os.path.join(REBUILD, "DTT_mouse")      # final files vs the rebuilt ones in <W>/F
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
res = {}
for rel in ["sister_pairs_strict.csv", "cousin_cells.csv", "all_cherries_age_edits.csv", "counts_samecell_analysis_cells.csv"] + \
           [os.path.relpath(p, FIN) for p in sorted(glob.glob(f"{FIN}/counts_wanted/*_cells.txt"))]:
    a, b = f"{FIN}/{rel}", f"{W}/F/{rel}"
    r = dict(byte_identical=os.path.exists(b) and md5(a) == md5(b))
    if not r["byte_identical"] and os.path.exists(b) and rel.endswith(".csv"):
        A, B = pd.read_csv(a), pd.read_csv(b); r["shape"] = [A.shape, B.shape]; r["frame_equal"] = A.equals(B)
    res[rel] = r
for rel in ["counts_samecell_analysis.npz"] + [os.path.relpath(p, FIN) for p in sorted(glob.glob(f"{FIN}/counts_wanted/*.npz"))]:
    A = sp.load_npz(f"{FIN}/{rel}").tocsr(); B = sp.load_npz(f"{W}/F/{rel}").tocsr(); A.sort_indices(); B.sort_indices()
    r = dict(shape=[A.shape, B.shape], nnz=[A.nnz, B.nnz])
    r["values_identical"] = A.shape == B.shape and A.nnz == B.nnz and np.array_equal(A.indptr, B.indptr) and np.array_equal(A.indices, B.indices) and np.array_equal(A.data, B.data)
    res[rel] = r
res["ALL_IDENTICAL"] = all(v.get("byte_identical") or v.get("values_identical") or v.get("frame_equal") for v in res.values())
json.dump(res, open(f"{W}/diff.json", "w"), indent=1, default=str); print(json.dumps(res, indent=1, default=str))
