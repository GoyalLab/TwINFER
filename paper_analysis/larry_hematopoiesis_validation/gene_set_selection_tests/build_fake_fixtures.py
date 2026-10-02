# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-10-01 added: project root for the rescued scratchpad copies]
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.io as sio
import h5py

# [2026-10-01 commented out: ephemeral Claude scratchpad; the data was copied to clean_data/_rescued_scratchpad] D = "/gpfs/home/gzu5140/.claude-tmp/claude-2000104/-gpfs-projects-b1255-hzhang-TwINFER-KA/23eb9009-c544-4f9f-a9ae-3ed691000196/scratchpad/fake_source"
D = f"{TWINFER_PROJECT_ROOT}/clean_data/_rescued_scratchpad/23eb9009/fake_source"

n_cells = 100
n_genes = 5
rng = np.random.default_rng(0)
X = np.zeros((n_cells, n_genes), dtype=float)
X[:, 0] = rng.poisson(5, n_cells)          # expressed broadly -> keep
X[:10, 1] = rng.poisson(5, 10)             # expressed in 10% -> keep (boundary case, >=5%)
X[:3, 2] = rng.poisson(5, 3)               # expressed in 3% -> DROP
X[:, 3] = rng.poisson(5, n_cells)          # keep
X[:5, 4] = rng.poisson(5, 5)               # expressed in 5% exactly -> keep (>=5%)
sio.mmwrite(f"{D}/larry_qc_counts.mtx", sp.csr_matrix(X))

genes = [f"Gene{i}" for i in range(n_genes)]
open(f"{D}/genes.txt", "w").write("\n".join(genes) + "\n")

libs = ["LSK_d2_1"] * 50 + ["LSK_d4_1_1"] * 50
def _acgt(k, n=8):
    bases = "ACGT"
    s = ""
    for _ in range(n):
        s += bases[k % 4]
        k //= 4
    return s
barcodes = [_acgt(i) + "-" + _acgt(i * 7 + 3) for i in range(n_cells)]
cell_ids = [f"{l}:{b}" for l, b in zip(libs, barcodes)]
open(f"{D}/cells.txt", "w").write("\n".join(cell_ids) + "\n")

obs = pd.DataFrame({
    "library": libs,
    "larry_clone": ["cloneA"] * n_cells,
    "total_counts": X.sum(axis=1),
    "pct_counts_mt": [1.0] * n_cells,
    "n_genes": (X > 0).sum(axis=1),
    "singlet_label": ["Singlet"] * n_cells,
    "larry_clone_singletcode": [f"clone{i%7}" for i in range(n_cells)],
    "criterion4_rescued": [False] * n_cells,
}, index=cell_ids)
obs.index.name = "cell_id"
obs.to_csv(f"{D}/obs_metadata.csv")

# fake REF h5ad matching real schema: dashless-after-transform barcodes, Library column, ann
with h5py.File(f"{D}/fake_ref.h5ad", "w") as h:
    o = h.create_group("obs")
    o.attrs["_index"] = "index"
    names = np.array([b for b in barcodes], dtype="S")  # same dash-joined barcodes as source
    o.create_dataset("index", data=names)
    lib_arr = np.array(libs, dtype="S")
    o.create_dataset("Library", data=lib_arr)
    ann_arr = np.array(["TypeA" if i % 2 == 0 else "TypeB" for i in range(n_cells)], dtype="S")
    o.create_dataset("Cell type annotation", data=ann_arr)

print("fixtures built:", n_cells, "cells,", n_genes, "genes")
print("expected genes kept (>=5% expr): Gene0, Gene1, Gene3, Gene4 ; dropped: Gene2")
