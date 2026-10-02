# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""
Re-derive the z-scored HVG data matrix from the canonical FM06_integrated.h5ad
(lognorm layer = log1p(CP10k), already-flagged 2000-gene Seurat HVG panel),
instead of the ad hoc mtx-based preprocessing. Lineage barcode + cell-state
`cluster` columns are pulled out here but not touched until evaluation.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import time, gc
import numpy as np
import anndata as ad

t0 = time.time()
OUT = "/gpfs/home/gzu5140/.claude-tmp/claude-2000104/-gpfs-projects-b1255-hzhang-TwINFER-KA/0e38220a-4f7f-4160-b243-03b8fa7848ac/scratchpad/fm06_powerseek"


def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)


a = ad.read_h5ad(
    f'{TWINFER_PROJECT_ROOT}/finalized_data/FM06_data/FM06_integrated.h5ad',
    backed="r",
)
log(f"loaded backed AnnData {a.shape}")

hvg_mask = a.var["highly_variable"].values
hvg_genes = a.var_names[hvg_mask].to_numpy()
log(f"n HVG genes (Seurat flag) = {hvg_mask.sum()}")

# pull only the HVG columns of the lognorm layer into memory (keeps footprint small)
X_sp = a[:, hvg_mask].layers["lognorm"]
log(f"pulled HVG sparse block, nnz={X_sp.nnz}")

# Paper's Methods explicitly filters out genes "expressed in less than 200
# cells" (dropout-driven near-binary genes otherwise dominate the covariance
# eigenspectrum with spurious rank-1 structure, not lineage correlation).
# The h5ad's stock Seurat highly_variable flag does not apply this filter,
# so intersect it here.
det_count = np.asarray((X_sp > 0).sum(axis=0)).ravel()
detect_mask = det_count >= 200
log(f"HVG genes also detected in >=200 cells: {detect_mask.sum()} / {hvg_mask.sum()}")

X_sp = X_sp[:, detect_mask]
hvg_genes = hvg_genes[detect_mask]
X_hvg = np.asarray(X_sp.todense(), dtype=np.float32)
log(f"X_hvg shape {X_hvg.shape}, size MB {X_hvg.nbytes/1e6:.1f}")

cell_ids = a.obs_names.to_numpy()
clone = a.obs["fatemap_clone_singletcode"].to_numpy()
cluster = a.obs["cluster"].to_numpy()
del a
gc.collect()

mu = X_hvg.mean(axis=0)
sd = X_hvg.std(axis=0)
sd[sd == 0] = 1.0
X = (X_hvg - mu) / sd
X = X.astype(np.float32)
del X_hvg
gc.collect()

np.save(f"{OUT}/X_hvg_zscored_h5ad.npy", X)
with open(f"{OUT}/genes_hvg_h5ad.txt", "w") as f:
    f.write("\n".join(hvg_genes))
np.save(f"{OUT}/clone_labels_h5ad.npy", clone)
np.save(f"{OUT}/cell_ids_h5ad.npy", cell_ids)
np.save(f"{OUT}/cellstate_cluster_h5ad.npy", cluster)

log("saved. done")
