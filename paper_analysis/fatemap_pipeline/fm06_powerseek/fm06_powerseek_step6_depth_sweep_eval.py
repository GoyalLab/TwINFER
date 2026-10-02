# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""
Evaluate the recursive bipartition tree against held-out ground-truth clones
at every resolution (depth) from K=2 up to the finest leaves, to see where
true-clone enrichment peaks / holds up vs where it collapses into noise.
"""
import sys, time
import numpy as np
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from collections import Counter

t0 = time.time()
OUT = "/gpfs/home/gzu5140/.claude-tmp/claude-2000104/-gpfs-projects-b1255-hzhang-TwINFER-KA/0e38220a-4f7f-4160-b243-03b8fa7848ac/scratchpad/fm06_powerseek"


def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)


SRC = sys.argv[1] if len(sys.argv) > 1 else "h5ad"
M = int(sys.argv[2]) if len(sys.argv) > 2 else 200
MIN_SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else 20
suffix = "_h5ad" if SRC == "h5ad" else ""
tagx = f"{SRC}_M{M}_min{MIN_SIZE}"

clone = np.load(f"{OUT}/clone_labels{suffix}.npy", allow_pickle=True)
paths = np.load(f"{OUT}/bipartition_paths_{tagx}.npy", allow_pickle=True).astype(str)
n_c = len(clone)
max_depth = max(len(p) for p in paths)
log(f"max depth in tree = {max_depth}, n_cells = {n_c}")

clone_counts = Counter(clone)
multi_mask = np.array([clone_counts[c] >= 2 for c in clone])

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

rng = np.random.default_rng(0)
n_neg = len(pos_pairs) * 5
neg_i = rng.integers(0, n_c, size=n_neg)
neg_j = rng.integers(0, n_c, size=n_neg)
keep = clone[neg_i] != clone[neg_j]
neg_i, neg_j = neg_i[keep], neg_j[keep]
log(f"n_pos_pairs={len(pos_pairs)}, n_neg_pairs={len(neg_i)}")

print(f"{'depth':>5} {'n_groups':>9} {'med_size':>9} {'ARI(multi)':>11} "
      f"{'same-rate pos':>14} {'same-rate neg':>14} {'enrichment':>11}")

rows = []
for d in range(1, max_depth + 1):
    labels_d = np.array([p[:d] if len(p) >= d else p for p in paths])
    n_groups = len(set(labels_d))
    sizes = np.array(list(Counter(labels_d).values()))
    med_size = np.median(sizes)

    ari = adjusted_rand_score(clone[multi_mask], labels_d[multi_mask])

    same_pos = (labels_d[pos_pairs[:, 0]] == labels_d[pos_pairs[:, 1]]).mean()
    same_neg = (labels_d[neg_i] == labels_d[neg_j]).mean()
    enrich = same_pos / max(same_neg, 1e-9)

    print(f"{d:>5} {n_groups:>9} {med_size:>9.0f} {ari:>11.5f} "
          f"{same_pos:>14.5f} {same_neg:>14.5f} {enrich:>11.2f}")
    rows.append((d, n_groups, med_size, ari, same_pos, same_neg, enrich))

with open(f"{OUT}/depth_sweep_{tagx}.tsv", "w") as f:
    f.write("depth\tn_groups\tmed_size\tARI_multi\tsame_rate_pos\tsame_rate_neg\tenrichment\n")
    for r in rows:
        f.write("\t".join(str(x) for x in r) + "\n")

log("done")
