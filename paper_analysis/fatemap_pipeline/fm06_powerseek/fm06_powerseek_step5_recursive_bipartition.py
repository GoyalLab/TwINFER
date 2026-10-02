# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""
Push the memory-gene spectral clustering to finer resolution via recursive
(divisive) spectral bipartition: at each node, re-center/re-scale the cell
subset on the same blind top-M memory-gene panel, split on the sign of the
median-thresholded leading eigenvector of the node-local covariance (standard
PDDP-style bisection -- each split is one "generation" of the Eq. 4 binary
lineage tree). A permutation test at each node (shuffle each gene's values
within just that node's cells) decides whether the split is a real signal or
noise, so the tree stops growing once we run out of detectable structure.

No lineage/clone information is used to build the tree. Ground truth is only
used afterward, per depth, to see where recovery holds up vs collapses.
"""
import sys, time
import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator

t0 = time.time()
OUT = "/gpfs/home/gzu5140/.claude-tmp/claude-2000104/-gpfs-projects-b1255-hzhang-TwINFER-KA/0e38220a-4f7f-4160-b243-03b8fa7848ac/scratchpad/fm06_powerseek"


def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)


SRC = sys.argv[1] if len(sys.argv) > 1 else "h5ad"  # "h5ad" or "mtx"
M = int(sys.argv[2]) if len(sys.argv) > 2 else 200
MIN_SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else 20
MAX_DEPTH = 16
N_PERM = 8
PERM_PCTL = 90  # observed top-eig must exceed this percentile of the null to keep splitting

suffix = "_h5ad" if SRC == "h5ad" else ""
X_full = np.load(f"{OUT}/X_hvg_zscored{suffix}.npy").astype(np.float64)
genes = np.array([l.strip() for l in open(f"{OUT}/genes_hvg{suffix}.txt")])
ranked = [l.split("\t")[0] for l in open(f"{OUT}/powerseek_ranked_genes{suffix}.txt")]
gene_to_col = {g: i for i, g in enumerate(genes)}
cols = [gene_to_col[g] for g in ranked[:M]]
X = X_full[:, cols].copy()
n_c_total = X.shape[0]
log(f"using top {M} Power-Seek genes, X shape {X.shape}")

rng = np.random.default_rng(0)


def top_eig(Xnode):
    """Top eigenpair of the node-local (re-centered/scaled) covariance."""
    n_node, n_g = Xnode.shape
    mu = Xnode.mean(axis=0)
    sd = Xnode.std(axis=0)
    sd[sd == 0] = 1.0
    Xz = (Xnode - mu) / sd
    if n_node <= 400:
        C = Xz @ Xz.T / n_g
        evals, evecs = np.linalg.eigh(C)
        return evals[-1], evecs[:, -1], Xz
    def matvec(v):
        return (Xz @ (Xz.T @ v)) / n_g
    Cop = LinearOperator((n_node, n_node), matvec=matvec, dtype=np.float64)
    evals, evecs = eigsh(Cop, k=2, which="LA")
    order = np.argsort(-evals)
    return evals[order][0], evecs[:, order][:, 0], Xz


def permutation_null(Xnode, n_perm=N_PERM):
    n_node, n_g = Xnode.shape
    null_tops = []
    for _ in range(n_perm):
        Xp = Xnode.copy()
        for j in range(n_g):
            Xp[:, j] = rng.permutation(Xp[:, j])
        lam1, _, _ = top_eig(Xp)
        null_tops.append(lam1)
    return np.array(null_tops)


# --- recursive tree build ---
# each cell gets a growing bit-path string; we track, per node, its cell indices
node_queue = [("", np.arange(n_c_total))]
path_of_cell = np.array([""] * n_c_total, dtype=object)
leaf_info = []  # (path, idx, reason_stopped)
depth_log = []

max_reached_depth = 0
while node_queue:
    path, idx = node_queue.pop(0)
    depth = len(path)
    max_reached_depth = max(max_reached_depth, depth)
    n_node = len(idx)

    if n_node < 2 * MIN_SIZE or depth >= MAX_DEPTH:
        leaf_info.append((path, idx, "min_size_or_depth"))
        path_of_cell[idx] = path
        continue

    Xnode = X[idx, :]
    lam1_obs, v1, Xz = top_eig(Xnode)
    null_tops = permutation_null(Xnode)
    thresh = np.percentile(null_tops, PERM_PCTL)

    if lam1_obs <= thresh:
        leaf_info.append((path, idx, "not_significant"))
        path_of_cell[idx] = path
        continue

    med = np.median(v1)
    left_mask = v1 <= med
    idx_left = idx[left_mask]
    idx_right = idx[~left_mask]
    if len(idx_left) < MIN_SIZE or len(idx_right) < MIN_SIZE:
        leaf_info.append((path, idx, "unbalanced_split"))
        path_of_cell[idx] = path
        continue

    depth_log.append(
        f"depth={depth} path={path!r} n={n_node} lam1={lam1_obs:.2f} "
        f"null_p{PERM_PCTL}={thresh:.2f} -> split {len(idx_left)}/{len(idx_right)}"
    )
    node_queue.append((path + "0", idx_left))
    node_queue.append((path + "1", idx_right))

log(f"tree built: {len(leaf_info)} leaves, max depth reached = {max_reached_depth}")
for line in depth_log[:40]:
    log("  " + line)
if len(depth_log) > 40:
    log(f"  ... ({len(depth_log)-40} more internal splits)")

tagx = f"{SRC}_M{M}_min{MIN_SIZE}"
np.save(f"{OUT}/bipartition_paths_{tagx}.npy", path_of_cell.astype(str))
with open(f"{OUT}/bipartition_leafinfo_{tagx}.txt", "w") as f:
    for path, idx, reason in leaf_info:
        f.write(f"{path}\t{len(idx)}\t{reason}\n")

log("done")
