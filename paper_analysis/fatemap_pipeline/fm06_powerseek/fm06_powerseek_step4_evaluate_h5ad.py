# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""
Evaluate the blind spectral clustering / embedding against the held-out
ground-truth lineage barcodes (fatemap_clone_singletcode). Lineage info is
used ONLY here, for scoring -- never as input to steps 1-3.
"""
import sys, time
import numpy as np
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, roc_auc_score

t0 = time.time()
OUT = "/gpfs/home/gzu5140/.claude-tmp/claude-2000104/-gpfs-projects-b1255-hzhang-TwINFER-KA/0e38220a-4f7f-4160-b243-03b8fa7848ac/scratchpad/fm06_powerseek"


def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)


tag = sys.argv[1]  # e.g. "M200" or "M200_shuffled"

clone = np.load(f"{OUT}/clone_labels_h5ad.npy", allow_pickle=True)
V = np.load(f"{OUT}/embedding_{tag}.npy")
labels = np.load(f"{OUT}/labels_{tag}.npy")
n_c = len(clone)
assert V.shape[0] == n_c == len(labels)

# --- 1. ARI / AMI restricted to cells in clones with >=2 members ---
from collections import Counter

clone_counts = Counter(clone)
multi_mask = np.array([clone_counts[c] >= 2 for c in clone])
log(f"cells in clones with >=2 members: {multi_mask.sum()} / {n_c}")

ari = adjusted_rand_score(clone[multi_mask], labels[multi_mask])
ami = adjusted_mutual_info_score(clone[multi_mask], labels[multi_mask])
log(f"ARI (multi-member clones vs predicted clusters) = {ari:.5f}")
log(f"AMI (multi-member clones vs predicted clusters) = {ami:.5f}")

# --- 2. pairwise cohesion: cosine similarity of embedding for true clone-mate
#        pairs vs random background pairs -> AUC ---
rng = np.random.default_rng(0)

# build positive pairs: all within-clone pairs for clones with 2-15 members
clone_to_idx = {}
for i, c in enumerate(clone):
    clone_to_idx.setdefault(c, []).append(i)

pos_pairs = []
for c, idxs in clone_to_idx.items():
    if len(idxs) < 2:
        continue
    for a in range(len(idxs)):
        for b in range(a + 1, len(idxs)):
            pos_pairs.append((idxs[a], idxs[b]))
pos_pairs = np.array(pos_pairs)
log(f"n positive (true clone-mate) pairs = {len(pos_pairs)}")

n_neg = len(pos_pairs) * 5
neg_i = rng.integers(0, n_c, size=n_neg)
neg_j = rng.integers(0, n_c, size=n_neg)
# drop accidental same-clone / self pairs in the negative set
keep = (clone[neg_i] != clone[neg_j])
neg_i, neg_j = neg_i[keep], neg_j[keep]
log(f"n negative (background) pairs = {len(neg_i)}")


def cos_sim(idx_a, idx_b):
    A = V[idx_a]
    B = V[idx_b]
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return np.sum(An * Bn, axis=1)


pos_score = cos_sim(pos_pairs[:, 0], pos_pairs[:, 1])
neg_score = cos_sim(neg_i, neg_j)

y = np.concatenate([np.ones(len(pos_score)), np.zeros(len(neg_score))])
s = np.concatenate([pos_score, neg_score])
auc = roc_auc_score(y, s)
log(f"pairwise cohesion AUC (cosine sim, true clone-mates vs background) = {auc:.5f}")
log(f"  mean cos-sim clone-mates = {pos_score.mean():.4f}, background = {neg_score.mean():.4f}")

# --- 3. same-hard-cluster rate: clone-mates vs background ---
same_pos = (labels[pos_pairs[:, 0]] == labels[pos_pairs[:, 1]]).mean()
same_neg = (labels[neg_i] == labels[neg_j]).mean()
log(f"same-cluster rate: clone-mates = {same_pos:.4f}, background = {same_neg:.4f}  (enrichment x{same_pos/max(same_neg,1e-9):.2f})")

with open(f"{OUT}/eval_{tag}.txt", "w") as f:
    f.write(f"tag={tag}\n")
    f.write(f"n_multi_member_cells={multi_mask.sum()}\n")
    f.write(f"ARI={ari}\nAMI={ami}\n")
    f.write(f"n_pos_pairs={len(pos_pairs)}\nn_neg_pairs={len(neg_i)}\n")
    f.write(f"AUC_cohesion={auc}\n")
    f.write(f"mean_cos_pos={pos_score.mean()}\nmean_cos_neg={neg_score.mean()}\n")
    f.write(f"same_cluster_rate_pos={same_pos}\nsame_cluster_rate_neg={same_neg}\n")

log("done")
