# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

def balanced_fit_indices(y, cap, seed):
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        c_idx = np.flatnonzero(y == c)
        if len(c_idx) > cap:
            c_idx = rng.choice(c_idx, size=cap, replace=False)
        idx.append(c_idx)
    return np.sort(np.concatenate(idx))

rng = np.random.default_rng(1)
# 2D toy space: majority class is a big diffuse blob; rare class is a small TIGHT cluster
# fully embedded inside the majority blob's extent (worst case: no separation margin).
n_maj = 3000
n_rare = 15
maj = rng.normal(loc=[0, 0], scale=2.0, size=(n_maj, 2))
rare = rng.normal(loc=[0.3, 0.3], scale=0.15, size=(n_rare, 2))  # tight, inside the blob
X = np.vstack([maj, rare])
y = np.array(["MAJ"] * n_maj + ["RARE"] * n_rare)

K = 15
SEED = 0

print("=== baseline: uniform vote, full imbalanced fit pool (the OLD behavior) ===")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
pred = np.empty(len(y), dtype=object)
for tr, te in skf.split(X, y):
    m = KNeighborsClassifier(n_neighbors=K, n_jobs=-1).fit(X[tr], y[tr])
    pred[te] = m.predict(X[te])
rare_acc_old = (pred[y == "RARE"] == "RARE").mean()
print(f"RARE class CV accuracy (old): {rare_acc_old:.3f}  (n={n_rare})")

print("=== fix: class-balanced fit pool + distance-weighted vote (the NEW behavior) ===")
CAP = 500
pred2 = np.empty(len(y), dtype=object)
for tr, te in skf.split(X, y):
    tr_bal = tr[balanced_fit_indices(y[tr], CAP, SEED)]
    m = KNeighborsClassifier(n_neighbors=K, weights="distance", n_jobs=-1).fit(X[tr_bal], y[tr_bal])
    pred2[te] = m.predict(X[te])
rare_acc_new = (pred2[y == "RARE"] == "RARE").mean()
print(f"RARE class CV accuracy (new): {rare_acc_new:.3f}  (n={n_rare})")

assert rare_acc_new > rare_acc_old, "the fix should improve rare-class recall on this toy case"
print("ASSERTION PASSED: balanced+distance-weighted k-NN recovers the rare class better than plain uniform k-NN")
