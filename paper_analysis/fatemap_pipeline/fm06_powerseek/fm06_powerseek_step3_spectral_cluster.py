# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""
Build a spectral embedding of cells from the cell covariance matrix restricted
to the top Power-Seek-ranked "memory genes" (blind to lineage barcodes),
then run k-means on the eigenvector rows (Ng-Jordan-Weiss style spectral
clustering). No lineage/clone information is used anywhere in this script.
"""
import sys, time
import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

t0 = time.time()
OUT = "/gpfs/home/gzu5140/.claude-tmp/claude-2000104/-gpfs-projects-b1255-hzhang-TwINFER-KA/0e38220a-4f7f-4160-b243-03b8fa7848ac/scratchpad/fm06_powerseek"


def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)


M = int(sys.argv[1]) if len(sys.argv) > 1 else 200
SHUFFLE = "--shuffle" in sys.argv

X_full = np.load(f"{OUT}/X_hvg_zscored.npy").astype(np.float64)
genes = np.array([l.strip() for l in open(f"{OUT}/genes_hvg.txt")])
ranked = [l.split("\t") for l in open(f"{OUT}/powerseek_ranked_genes.txt")]
ranked_genes = [r[0] for r in ranked]

gene_to_col = {g: i for i, g in enumerate(genes)}
top_genes = ranked_genes[:M]
cols = [gene_to_col[g] for g in top_genes]
X = X_full[:, cols].copy()
n_c, n_g = X.shape
log(f"restricted matrix to top {M} Power-Seek genes: X shape {X.shape}")

if SHUFFLE:
    # Full independent per-gene permutation of cell identity: preserves each
    # gene's marginal distribution but destroys any across-gene, per-cell
    # (i.e. lineage-driven) coordination -- the rigorous version of the
    # paper's per-gene random-swap null control (Methods: "Shuffling").
    rng = np.random.default_rng(0)
    for j in range(n_g):
        X[:, j] = rng.permutation(X[:, j])
    log("shuffled data matrix (null control, full per-gene permutation)")

def matvec(v):
    return (X @ (X.T @ v)) / n_g

Cop = LinearOperator((n_c, n_c), matvec=matvec, dtype=np.float64)

K = 60
evals, evecs = eigsh(Cop, k=K, which="LA")
order = np.argsort(-evals)
evals, evecs = evals[order], evecs[:, order]
log(f"top 15 eigenvalues: {evals[:15]}")

ranks = np.arange(2, K + 1)
log_r = np.log(ranks)
log_l = np.log(evals[1:K])


def r2_fit(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return (1 - ss_res / ss_tot if ss_tot > 0 else 1.0), coef[0]


best_m = 3
for m in range(3, K - 1):
    r2, slope = r2_fit(log_r[:m], log_l[:m])
    if r2 >= 0.90:
        best_m = m
    else:
        break
n_pl = max(best_m, 3)
log(f"power-law regime on restricted matrix: n_pl={n_pl}")

# embedding = eigenvectors ranked 2..n_pl+1 (exclude the trivial top eigenvalue,
# analogous to excluding the non-power-law leading eigenvalue in Eq. 4)
embed_dim = n_pl
V = evecs[:, 1:1 + embed_dim]
V = normalize(V, axis=1)  # row-normalize (Ng-Jordan-Weiss)
log(f"embedding shape {V.shape}")

tag = f"M{M}" + ("_shuffled" if SHUFFLE else "")
np.save(f"{OUT}/embedding_{tag}.npy", V)
np.save(f"{OUT}/evals_{tag}.npy", evals)

# k-means over a range of K, pick via silhouette (blind — no lineage used)
from sklearn.metrics import silhouette_score

best_k, best_sil, best_labels = None, -2, None
for k in range(2, 30):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(V)
    if len(set(km.labels_)) < 2:
        continue
    sil = silhouette_score(V, km.labels_)
    if sil > best_sil:
        best_sil, best_k, best_labels = sil, k, km.labels_

log(f"chosen K={best_k} (silhouette={best_sil:.4f})")
np.save(f"{OUT}/labels_{tag}.npy", best_labels)
with open(f"{OUT}/summary_{tag}.txt", "w") as f:
    f.write(f"M={M}\nshuffle={SHUFFLE}\nn_pl={n_pl}\nembed_dim={embed_dim}\nbest_k={best_k}\nbest_sil={best_sil}\n")

log("done")
