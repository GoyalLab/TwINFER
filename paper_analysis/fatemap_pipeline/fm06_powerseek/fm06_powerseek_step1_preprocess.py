from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import time, gc
import numpy as np
import pandas as pd
import scipy.io as sio

t0 = time.time()
DATA = f'{TWINFER_PROJECT_ROOT}/finalized_data/FM06_data/qc_filtered'
OUT = "/gpfs/home/gzu5140/.claude-tmp/claude-2000104/-gpfs-projects-b1255-hzhang-TwINFER-KA/0e38220a-4f7f-4160-b243-03b8fa7848ac/scratchpad/fm06_powerseek"


def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)


log("loading mtx...")
counts = sio.mmread(f"{DATA}/fm06_qc_counts.mtx").tocsr()
counts.data = counts.data.astype(np.float32)
cells = [l.strip() for l in open(f"{DATA}/cells.txt")]
genes = [l.strip() for l in open(f"{DATA}/genes.txt")]
log(f"counts shape {counts.shape}, nnz {counts.nnz}")
assert counts.shape == (len(cells), len(genes))

obs = pd.read_csv(f"{DATA}/obs_metadata.csv", index_col=0)
obs = obs.loc[cells]
clone = obs["fatemap_clone_singletcode"].values
np.save(f"{OUT}/clone_labels.npy", np.array(clone))
np.save(f"{OUT}/cell_ids.npy", np.array(cells))
del obs
gc.collect()
log(f"n cells {len(cells)}, n unique clones {len(set(clone))}")

# CP10k normalize in place (no raw counts), then log1p in place
total = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
total[total == 0] = 1.0
row_idx = counts.indptr
row_counts = np.diff(row_idx)
row_scale = (1e4 / total).astype(np.float32)
scale_per_nz = np.repeat(row_scale, row_counts)
counts.data *= scale_per_nz
del scale_per_nz, row_scale, total, row_counts
gc.collect()
np.log1p(counts.data, out=counts.data)
log("normalized (CP10k + log1p) in place")

# gene filter: expressed in >=200 cells
counts_csc = counts.tocsc()
del counts
gc.collect()
gene_nnz = np.diff(counts_csc.indptr)
keep_mask = gene_nnz >= 200
log(f"genes passing >=200-cell filter: {keep_mask.sum()} / {len(genes)}")

logn_f = counts_csc[:, keep_mask]
del counts_csc
gc.collect()
genes_f = np.array(genes)[keep_mask]

n_cells = logn_f.shape[0]
mean_c = np.asarray(logn_f.mean(axis=0)).ravel()
sq = logn_f.copy()
sq.data **= 2
mean_sq = np.asarray(sq.mean(axis=0)).ravel()
del sq
gc.collect()
var = mean_sq - mean_c ** 2
var[var < 0] = 0
std = np.sqrt(var)
cv = np.divide(std, mean_c, out=np.zeros_like(std), where=mean_c > 0)
log("computed per-gene mean/std/CV")

N_HVG = 2000
top_idx = np.argsort(-cv)[:N_HVG]
top_idx.sort()
genes_hvg = genes_f[top_idx]
X_hvg = np.asarray(logn_f[:, top_idx].todense(), dtype=np.float32)
del logn_f
gc.collect()
log(f"X_hvg shape {X_hvg.shape}, size MB {X_hvg.nbytes/1e6:.1f}")

mu = X_hvg.mean(axis=0)
sd = X_hvg.std(axis=0)
sd[sd == 0] = 1.0
X = (X_hvg - mu) / sd
X = X.astype(np.float32)
del X_hvg
gc.collect()

np.save(f"{OUT}/X_hvg_zscored.npy", X)
with open(f"{OUT}/genes_hvg.txt", "w") as f:
    f.write("\n".join(genes_hvg))

log("done, saved outputs")
