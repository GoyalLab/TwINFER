"""Sweep BALANCE_CAP to find a tradeoff point between rare-class recall and majority-class
(Undifferentiated etc.) accuracy, instead of committing to one cap value blind.

Reuses larry_raw_annotate.py's Input + classifier-space (HVG/PCA) logic verbatim, then runs
the 5-fold CV at each candidate cap and reports per-class + overall accuracy side by side.
Does NOT write any output files (no OUT/REC) -- this is exploration only.

Run directly: python3 explore_balance_cap.py
Override the sweep with e.g.: CAPS=300,800,1500,3000 python3 explore_balance_cap.py
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import os
import re

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io as sio

SOURCE = os.environ.get(
    "SOURCE", f'{TWINFER_PROJECT_ROOT}/finalized_data/LARRY_data/filtered').strip()
assert os.path.isdir(SOURCE), f"missing input dir: {SOURCE}"
REF = os.environ.get(
    # [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] "REF", f'{TWINFER_PROJECT_ROOT}/code/TwINFER/real_data/LSK_d2_d4_d6.h5ad')
    "REF", f'{TWINFER_PROJECT_ROOT}/clean_data/real_data/LSK_d2_d4_d6.h5ad')
assert os.path.exists(REF), f"missing annotation reference: {REF}"

K = int(os.environ.get("KNN", "15"))
N_HVG = int(os.environ.get("N_HVG", "2000"))
N_PC = int(os.environ.get("N_PC", "50"))
SEED = int(os.environ.get("SEED", "0"))


def log(m):
    print(m, flush=True)


def _col(h, name):
    g = h["obs"][name]
    if isinstance(g, h5py.Group):
        return np.asarray(g["categories"]).astype(str)[np.asarray(g["codes"])]
    return np.asarray(g).astype(str)


with h5py.File(REF, "r") as h:
    oi = h["obs"].attrs["_index"]
    ref_names = np.asarray(h["obs"][oi]).astype(str)
    for need in ("Library", "Cell type annotation"):
        assert need in h["obs"], f"reference lacks {need!r}"
    ref_lib = _col(h, "Library")
    ref_ann = _col(h, "Cell type annotation")

published = {(L, re.sub(r"-\d+$", "", n).replace("-", "")): a
             for L, n, a in zip(ref_lib, ref_names, ref_ann)}
LABELS = sorted(set(ref_ann))
log(f"reference {os.path.basename(REF)}: {len(published):,} keyed cells, {len(LABELS)} labels")

X = sio.mmread(os.path.join(SOURCE, "larry_qc_counts.mtx")).tocsr()
var_names = pd.Index(open(os.path.join(SOURCE, "genes.txt")).read().split())
obs_df = pd.read_csv(os.path.join(SOURCE, "obs_metadata.csv"), index_col=0)
assert X.shape == (len(obs_df), len(var_names)), \
    f"{SOURCE}: matrix/obs_metadata/genes shape mismatch"
A = ad.AnnData(X=X, obs=obs_df, var=pd.DataFrame(index=var_names))

frac_expr = np.asarray((A.X > 0).sum(axis=0)).ravel() / A.n_obs
gene_mask = frac_expr >= 0.05
log(f"gene filter: keeping {int(gene_mask.sum()):,} / {A.n_vars:,} genes expressed in >=5% of cells")
A = A[:, gene_mask].copy()

for need in ("library", "larry_clone_singletcode"):
    assert need in A.obs.columns, f"{SOURCE} lacks obs[{need!r}]"
A.obs["sample"] = A.obs["library"].astype(str)
A.obs["clone_id"] = A.obs["larry_clone_singletcode"]
_day = A.obs["library"].astype(str).str.extract(r"_d(\d+)_", expand=False)
assert _day.notna().all(), "could not parse day/timepoint out of every library name"
A.obs["folder_timepoint"] = _day
smp = A.obs["sample"].astype(str).values
bc = (A.obs_names.to_series().str.split(":", n=1).str[1]
      .str.replace("-", "", regex=False).to_numpy().astype(str))
hit = np.array([published.get((s, b)) for s, b in zip(smp, bc)], dtype=object)
is_pub = np.array([x is not None for x in hit])
assert is_pub.any(), "no cell joined to the reference -- the join key is wrong"
log(f"{os.path.basename(SOURCE)}: {A.n_obs:,} cells | published {int(is_pub.sum()):,} | "
    f"to predict {int((~is_pub).sum()):,}")

# ---------------------------------------------------------------- classifier space (never written out)
W = A.copy()
W.X = W.X.astype(np.float32)
sc.pp.normalize_total(W, target_sum=1e4)
sc.pp.log1p(W)
sc.pp.highly_variable_genes(W, n_top_genes=N_HVG, batch_key="sample")
n_hvg = int(W.var["highly_variable"].sum())
assert n_hvg > 0, "no highly variable genes selected"
W = W[:, W.var["highly_variable"]].copy()
sc.pp.scale(W, max_value=10)
sc.tl.pca(W, n_comps=min(N_PC, W.n_vars - 1, W.n_obs - 1), svd_solver="arpack", random_state=SEED)
P = np.asarray(W.obsm["X_pca"])
log(f"classifier space: {n_hvg:,} HVGs (batch_key='sample'), {P.shape[1]} PCs")
del W

# ---------------------------------------------------------------- BALANCE_CAP sweep
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

Xp, yp = P[is_pub], np.asarray(hit[is_pub]).astype(str)


def balanced_fit_indices(y, cap, seed):
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        c_idx = np.flatnonzero(y == c)
        if len(c_idx) > cap:
            c_idx = rng.choice(c_idx, size=cap, replace=False)
        idx.append(c_idx)
    return np.sort(np.concatenate(idx))


vc = pd.Series(yp).value_counts()
usable = vc[vc >= 5].index
mask = np.isin(yp, usable)
Xc, yc = Xp[mask], yp[mask]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
classes = sorted(set(yc))

CAPS = [int(x) for x in os.environ.get("CAPS", "300,500,1000,2000,3000,5000,10000,999999").split(",")]
log(f"\nsweeping BALANCE_CAP over {CAPS} (999999 == effectively uncapped)")
log(f"class sizes: {dict(vc.sort_values(ascending=False))}\n")

# The actual downstream use: remove TARGET (non-myeloid) cells. Per-class diagonal accuracy
# can't tell us the thing that matters for that -- where do MYELOID cells' errors land? If a
# true myeloid cell is misclassified as another myeloid class, that's harmless (still kept).
# If it's misclassified as one of TARGET, it gets wrongly removed. So report, per cap:
#   recall_target_set : true TARGET cell correctly predicted as *some* TARGET label (not
#                        necessarily the exact right one of the 3 -- still correctly removed)
#   fpr_myeloid_to_target : true MYELOID cell wrongly predicted into the TARGET set
#                            (the actual cost of a lower cap: real myeloid cells lost)
TARGET = {"Lymphoid", "pDC", "Ccr7_DC"}
MYELOID = set(classes) - TARGET
log(f"TARGET (to remove): {sorted(TARGET)}")
log(f"MYELOID (to keep): {sorted(MYELOID)}\n")

rows = []
for cap in CAPS:
    pred_cv = np.empty(len(yc), dtype=object)
    for tr, te in skf.split(Xc, yc):
        tr_bal = tr[balanced_fit_indices(yc[tr], cap, SEED)]
        m = KNeighborsClassifier(n_neighbors=K, weights="distance", n_jobs=-1).fit(Xc[tr_bal], yc[tr_bal])
        pred_cv[te] = m.predict(Xc[te])
    per_class = {c: float((pred_cv[yc == c] == c).mean()) for c in classes}
    overall = float((pred_cv == yc).mean())

    is_true_target = np.isin(yc, list(TARGET))
    is_true_myeloid = np.isin(yc, list(MYELOID))
    pred_in_target = np.isin(pred_cv, list(TARGET))
    recall_target_set = float(pred_in_target[is_true_target].mean()) if is_true_target.any() else float("nan")
    fpr_myeloid_to_target = float(pred_in_target[is_true_myeloid].mean()) if is_true_myeloid.any() else float("nan")
    n_myeloid_lost = int(pred_in_target[is_true_myeloid].sum())

    rows.append((cap, overall, per_class, recall_target_set, fpr_myeloid_to_target, n_myeloid_lost))
    log(f"CAP={cap:>7,}  overall={overall:.4f}  recall_target_set={recall_target_set:.4f}  "
        f"fpr_myeloid_to_target={fpr_myeloid_to_target:.4f}  ({n_myeloid_lost}/{int(is_true_myeloid.sum())} "
        f"myeloid cells wrongly flagged for removal in CV)  " +
        "  ".join(f"{c}={per_class[c]:.2f}" for c in classes))

log("\n=== summary table (per-class) ===")
header = "CAP".rjust(8) + "  overall".rjust(9) + "".join(f"  {c[:10]:>10}" for c in classes)
log(header)
for cap, overall, per_class, *_ in rows:
    line = f"{cap:>8,}  {overall:>7.4f}" + "".join(f"  {per_class[c]:>10.4f}" for c in classes)
    log(line)

log("\n=== summary table (removal-task metrics: TARGET = Lymphoid/pDC/Ccr7_DC) ===")
log(f"{'CAP':>8}  {'recall_target':>13}  {'fpr_myeloid->target':>20}  {'n_myeloid_lost':>14}")
for cap, overall, per_class, recall_target_set, fpr_myeloid_to_target, n_myeloid_lost in rows:
    log(f"{cap:>8,}  {recall_target_set:>13.4f}  {fpr_myeloid_to_target:>20.4f}  {n_myeloid_lost:>14,}")
