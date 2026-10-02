"""Attach cell-type annotations to the raw-count LARRY matrices, published where they exist and
predicted where they do not, and record which is which.

WHY THIS EXISTS
  SOURCE is our own qc_filtered/-style output (build_final_matrix_umi3.py: larry_qc_counts.mtx +
  genes.txt + cells.txt + obs_metadata.csv), built from the inDrops outputs, and it keeps cells the
  published object might not have. The published object is the only source of the
  cell-type labels, so those extra cells CANNOT be looked up anywhere: they were not in the object
  that carries the annotation. Every cell in this matrix carries a clone identifier, so the
  unannotated ones would otherwise be excluded from twin pairs, which is exactly what the new data
  is meant to avoid.

  So the labels for those cells are assigned here, by transfer from the annotated cells of the same
  matrix. That is a modelling step and not a measurement: `annotation_source` marks every cell as
  `published` or `predicted`, and nothing downstream may treat the two as the same evidence.

WHAT IS MEASURED BEFORE IT IS USED
  Held-out accuracy by stratified 5-fold cross-validation over the PUBLISHED cells only, reported
  overall, per class and per timepoint, and written to the record JSON. A transfer whose accuracy is
  not reported is not usable, so the numbers are produced whether or not anyone reads them.

METHOD
  Genes expressed (count > 0) in fewer than 5% of cells are dropped before anything else is
  computed. Then: log1pCP10k on the raw counts -> highly variable genes selected per sample
  (batch_key='sample', so no gene is chosen because one sample varies) -> PCA -> k-nearest-neighbour
  vote among published cells. The classifier sees only expression; the clone identifier is never an
  input, so a predicted label cannot be induced by the lineage label that the analysis is about to
  test.

  Published class sizes span 3 orders of magnitude (Undifferentiated ~19k vs pDC ~13), which
  drowns rare classes under plain majority-vote k-NN. The k-NN fit pool is class-balanced
  (BALANCE_CAP downsamples any class above the cap; smaller classes are kept whole) and voting
  is distance-weighted, both applied identically in the CV loop and the final transfer.

  HVG and PCA here serve the classifier ONLY. They are not written to the output matrix and never
  reach TwINFER, which takes the full gene set.

The join key, the label set, k and the gene count are derived from the data at runtime and asserted.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
import os
import re
import sys

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")

# SOURCE is a directory in our qc_filtered/-style layout (larry_qc_counts.mtx, genes.txt,
# cells.txt, obs_metadata.csv), not a single .h5ad -- see build_final_matrix_umi3.py.
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

_stem = os.path.basename(os.path.normpath(SOURCE))
OUTDIR = os.path.join(os.path.dirname(os.path.normpath(SOURCE)), "annotated")
os.makedirs(OUTDIR, exist_ok=True)
OUT = os.path.join(OUTDIR, f"{_stem}_annotated_k{K}.h5ad")
REC = os.path.join(OUTDIR, f"annotation_transfer_{_stem}_k{K}.json")


def log(m):
    print(m, flush=True)


# ---------------------------------------------------------------- published labels, by measured key
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

# The reference's obs_names carry a dash the sample sheet does not; the key is (Library, dashless
# barcode). This is the join already used elsewhere in the project, re-derived rather than assumed.
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

# before any calculation: drop genes expressed (count > 0) in fewer than 5% of cells
frac_expr = np.asarray((A.X > 0).sum(axis=0)).ravel() / A.n_obs
gene_mask = frac_expr >= 0.05
log(f"gene filter: keeping {int(gene_mask.sum()):,} / {A.n_vars:,} genes "
    f"expressed in >=5% of cells")
A = A[:, gene_mask].copy()

for need in ("library", "larry_clone_singletcode"):
    assert need in A.obs.columns, f"{SOURCE} lacks obs[{need!r}]"
A.obs["sample"] = A.obs["library"].astype(str)
A.obs["clone_id"] = A.obs["larry_clone_singletcode"]
_day = A.obs["library"].astype(str).str.extract(r"_d(\d+)_", expand=False)
assert _day.notna().all(), "could not parse day/timepoint out of every library name"
A.obs["folder_timepoint"] = _day
smp = A.obs["sample"].astype(str).values
# our obs_names are "library:XXXXXXXX-YYYYYYYY" (two-part inDrops barcode); the join key
# below is the bare dashless 16-mer, same transform REF's own obs_names get further down.
bc = (A.obs_names.to_series().str.split(":", n=1).str[1]
      .str.replace("-", "", regex=False).to_numpy().astype(str))
hit = np.array([published.get((s, b)) for s, b in zip(smp, bc)], dtype=object)
is_pub = np.array([x is not None for x in hit])
assert is_pub.any(), "no cell joined to the reference -- the join key is wrong"
log(f"{os.path.basename(SOURCE)}: {A.n_obs:,} cells | published {int(is_pub.sum()):,} | "
    f"to predict {int((~is_pub).sum()):,}")

# 'Time point' is what the analysis notebook reads; folder_timepoint is the same quantity renamed.
tp = A.obs["folder_timepoint"].astype(str).astype(int).values
A.obs["Time point"] = tp

# The barcode alone is NOT a cell identifier here -- it recurs across samples, which is why the join
# above keys on (sample, barcode). Downstream the analysis indexes cells by obs_names, so a repeated
# name would make a cell id ambiguous rather than fail. Make the name the key that is actually
# unique, and assert it: measured on these matrices, (sample, barcode) is unique for every row.
A.obs["barcode"] = bc
A.obs["sample_barcode"] = np.array([f"{s}|{b}" for s, b in zip(smp, bc)])
assert A.obs["sample_barcode"].is_unique, \
    f"(sample, barcode) is not unique: {A.n_obs - A.obs['sample_barcode'].nunique()} repeats"
A.obs_names = pd.Index(A.obs["sample_barcode"].values)
assert A.obs_names.is_unique, "obs_names are still not unique after rekeying"

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

from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

Xp, yp = P[is_pub], np.asarray(hit[is_pub]).astype(str)

# Class sizes here span 3 orders of magnitude (Undifferentiated ~19k vs pDC ~13). Plain
# majority-vote k-NN structurally drowns the rare classes: a query cell's k=15 nearest
# neighbours are overwhelmingly likely to be non-rare-class members even when the query
# itself is one, so the vote can never go the rare class's way. Countering this needs two
# independent things: (1) BALANCE_CAP downsamples the *fit* pool so no one class can flood
# the neighbour candidates, and (2) weights="distance" so that if a genuinely close
# same-class neighbour exists, it isn't outweighed by a crowd of more-distant majority-class
# points. Neither alone reliably fixes this class of failure; both together are the standard
# combination for k-NN under severe class imbalance.
BALANCE_CAP = int(os.environ.get("BALANCE_CAP", "500"))


def balanced_fit_indices(y, cap, seed):
    """Indices into y capping every class at `cap` members (random downsample without
    replacement); classes at or under `cap` are kept whole."""
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        c_idx = np.flatnonzero(y == c)
        if len(c_idx) > cap:
            c_idx = rng.choice(c_idx, size=cap, replace=False)
        idx.append(c_idx)
    return np.sort(np.concatenate(idx))


# ---------------------------------------------------------------- measured accuracy, before any use
rec = {"source": os.path.basename(SOURCE), "reference": os.path.basename(REF), "k": K,
       "n_hvg": n_hvg, "n_pc": int(P.shape[1]), "n_cells": int(A.n_obs),
       "n_published": int(is_pub.sum()), "n_predicted": int((~is_pub).sum()),
       "balance_cap": BALANCE_CAP}

vc = pd.Series(yp).value_counts()
usable = vc[vc >= 5].index          # a class with fewer members than folds cannot be cross-validated
mask = np.isin(yp, usable)
log(f"cross-validation on {int(mask.sum()):,} published cells, {len(usable)} classes "
    f"(classes with <5 cells excluded from CV: {sorted(set(yp) - set(usable))})")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
pred_cv = np.empty(int(mask.sum()), dtype=object)
Xc, yc = Xp[mask], yp[mask]
for tr, te in skf.split(Xc, yc):
    tr_bal = tr[balanced_fit_indices(yc[tr], BALANCE_CAP, SEED)]
    m = KNeighborsClassifier(n_neighbors=K, weights="distance", n_jobs=-1).fit(Xc[tr_bal], yc[tr_bal])
    pred_cv[te] = m.predict(Xc[te])
acc = float((pred_cv == yc).mean())
rec["cv_accuracy_overall"] = round(acc, 4)
rec["cv_accuracy_per_class"] = {c: round(float((pred_cv[yc == c] == c).mean()), 4)
                                for c in sorted(set(yc))}
rec["cv_support_per_class"] = {c: int((yc == c).sum()) for c in sorted(set(yc))}
tp_c = tp[is_pub][mask]
rec["cv_accuracy_per_timepoint"] = {str(t): round(float((pred_cv[tp_c == t] == yc[tp_c == t]).mean()), 4)
                                    for t in sorted(set(tp_c))}
log(f"CV accuracy overall: {acc:.4f}")
for c in sorted(set(yc)):
    log(f"    {c:<18}{rec['cv_accuracy_per_class'][c]:>8.4f}  (n={rec['cv_support_per_class'][c]:,})")
for t in sorted(set(tp_c)):
    log(f"    day {t}: {rec['cv_accuracy_per_timepoint'][str(t)]:.4f}")

# ---------------------------------------------------------------- transfer
final = np.array(["" for _ in range(A.n_obs)], dtype=object)
final[is_pub] = yp
if (~is_pub).sum():
    bal = balanced_fit_indices(yp, BALANCE_CAP, SEED)
    knn = KNeighborsClassifier(n_neighbors=K, weights="distance", n_jobs=-1).fit(Xp[bal], yp[bal])
    final[~is_pub] = knn.predict(P[~is_pub])
assert all(x != "" for x in final), "a cell was left without a label"
A.obs["Cell type annotation"] = pd.Categorical(final)
A.obs["annotation_source"] = pd.Categorical(np.where(is_pub, "published", "predicted"))
assert (A.obs.loc[is_pub, "Cell type annotation"].astype(str).values == yp).all(), \
    "a published label was overwritten by a prediction"

pt = pd.crosstab(A.obs["Cell type annotation"].astype(str), A.obs["annotation_source"].astype(str))
log("\nlabels by source:\n" + pt.to_string())
rec["predicted_label_counts"] = {k: int(v) for k, v in
                                 pd.Series(final[~is_pub]).value_counts().items()} if (~is_pub).sum() else {}

prev = dict(A.uns.get("preprocessing", {})) if hasattr(A, "uns") else {}
prev.update({"annotation": f"published labels joined on (sample, barcode) from "
                           f"{os.path.basename(REF)}; cells absent from it labelled by {K}-NN "
                           f"transfer in a {n_hvg}-HVG / {P.shape[1]}-PC space built from this "
                           f"matrix. See obs['annotation_source'].",
             "annotation_cv_accuracy": rec["cv_accuracy_overall"]})
A.uns["preprocessing"] = prev
A.write_h5ad(OUT)
log(f"\nwrote {OUT}: {A.n_obs:,} cells x {A.n_vars:,} genes")
chk = ad.read_h5ad(OUT, backed="r")
assert chk.n_obs == A.n_obs and "annotation_source" in chk.obs.columns, "output is not as written"
json.dump(rec, open(REC, "w"), indent=2, sort_keys=True)
log(f"wrote {REC}")
log("done")
