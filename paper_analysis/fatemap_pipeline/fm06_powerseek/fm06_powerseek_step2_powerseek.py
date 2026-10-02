# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""
Power-Seek gene ranking (blind to lineage) + efficient delta_g via Rayleigh-Ritz
approximation, following Ghosh/Chakrabarti/Raju (2025) Methods:
  - C_S = X X^T / n_g   (cell covariance matrix, X = z-scored log1p(CP10k) data matrix)
  - delta_g = sum_{j=1}^{n} (lambda_j - lambda_j^g), n = # eigenvalues in power-law regime
  - exact recomputation of top eigenvalues after removing gene g is a rank-1 downdate:
      C_S^g = (n_g/(n_g-1)) C_S - (1/(n_g-1)) x_g x_g^T
    We approximate its top eigenvalues via Rayleigh-Ritz projection onto the
    leading k eigenvectors of C_S (valid since the power-law eigenvalues are the
    well-separated top of the spectrum -- exactly the regime delta_g sums over).
"""
import time
import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator

t0 = time.time()
OUT = "/gpfs/home/gzu5140/.claude-tmp/claude-2000104/-gpfs-projects-b1255-hzhang-TwINFER-KA/0e38220a-4f7f-4160-b243-03b8fa7848ac/scratchpad/fm06_powerseek"


def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)


X = np.load(f"{OUT}/X_hvg_zscored.npy").astype(np.float64)
genes = [l.strip() for l in open(f"{OUT}/genes_hvg.txt")]
n_c, n_g = X.shape
log(f"X shape {X.shape}")

# implicit C = X X^T / n_g, applied via matvec (never form n_c x n_c dense matrix)
def matvec(v):
    return (X @ (X.T @ v)) / n_g

Cop = LinearOperator((n_c, n_c), matvec=matvec, dtype=np.float64)

K = 120  # leading eigenpairs to track (power-law regime is expected to be << K)
log(f"running eigsh for top {K} eigenpairs (implicit operator)...")
evals, evecs = eigsh(Cop, k=K, which="LA")
order = np.argsort(-evals)
evals = evals[order]
evecs = evecs[:, order]
log(f"top eigenvalues: {evals[:10]}")
np.save(f"{OUT}/top_evals_full.npy", evals)
np.save(f"{OUT}/top_evecs_full.npy", evecs)

# --- estimate size of power-law regime (excluding the top, non-power-law eigenvalue) ---
# fit log(lambda) vs log(rank) on ranks 2..m, find largest m with R^2 >= threshold
ranks = np.arange(2, K + 1)
log_r = np.log(ranks)
log_l = np.log(evals[1:K])  # exclude rank-1 (top) eigenvalue, matches paper Methods


def r2_fit(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return r2, coef[0]


best_m = 5
for m in range(5, K - 1):
    r2, slope = r2_fit(log_r[:m], log_l[:m])
    if r2 >= 0.92:
        best_m = m
    else:
        break
n_pl = best_m
r2_final, slope_final = r2_fit(log_r[:n_pl], log_l[:n_pl])
log(f"power-law regime: n={n_pl} eigenvalues (ranks 2..{n_pl+1}), R2={r2_final:.4f}, slope={slope_final:.3f}")

# --- Rayleigh-Ritz delta_g for every gene (vectorized) ---
V = evecs[:, :K]           # n_c x K
Lam = evals[:K]            # K
proj = V.T @ X             # K x n_g  (projection of every gene's cell-vector onto top-K eigvecs)
log(f"computed projections, shape {proj.shape}")

a = n_g / (n_g - 1)
b = 1.0 / (n_g - 1)

delta_g = np.zeros(n_g)
top_n = n_pl
orig_sum = Lam[:top_n].sum()

# batch eigen-decompose the K x K rank-1-perturbed matrices
# M_g = a*diag(Lam) - b * p_g p_g^T   (symmetric, K x K)
base = a * np.diag(Lam)
for g in range(n_g):
    p = proj[:, g]
    M = base - b * np.outer(p, p)
    ev = np.linalg.eigvalsh(M)  # ascending
    ev = ev[::-1]               # descending
    new_sum = ev[:top_n].sum()
    delta_g[g] = orig_sum - new_sum
    if g % 500 == 0:
        log(f"  gene {g}/{n_g}")

log("delta_g computed for all genes")
order_g = np.argsort(-delta_g)
ranked_genes = np.array(genes)[order_g]
ranked_delta = delta_g[order_g]

np.save(f"{OUT}/delta_g.npy", delta_g)
with open(f"{OUT}/powerseek_ranked_genes.txt", "w") as f:
    for gname, d in zip(ranked_genes, ranked_delta):
        f.write(f"{gname}\t{d:.6f}\n")

log(f"top 20 memory genes (highest delta_g): {list(ranked_genes[:20])}")
log(f"n_pl (eigs in power-law regime) = {n_pl}")
with open(f"{OUT}/n_pl.txt", "w") as f:
    f.write(str(n_pl))

log("done")
