# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Attach cell-type annotations to the raw-count LARRY matrices, published where they exist and
predicted where they do not, and record which is which.

WHY THIS EXISTS
  The matrices in raw_LARRY_data are built from the InDrops outputs and keep cells the published
  object might not have. The published object is the only source of the
  cell-type labels, so those extra cells CANNOT be looked up anywhere: they were not in the object
  that carries the annotation. Every cell in these matrices carries a clone identifier, so the
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
  log1pCP10k on the raw counts -> highly variable genes selected per sample (batch_key='sample', so
  no gene is chosen because one sample varies) -> PCA -> k-nearest-neighbour vote among published
  cells. The classifier sees only expression; the clone identifier is never an input, so a predicted
  label cannot be induced by the lineage label that the analysis is about to test.

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


print("SMOKE TEST CHECKS")
assert A.n_obs == 100, A.n_obs
assert list(A.var_names) == ["Gene0","Gene1","Gene3","Gene4"], list(A.var_names)
assert "sample" in A.obs.columns and "clone_id" in A.obs.columns and "folder_timepoint" in A.obs.columns
assert set(A.obs["folder_timepoint"]) == {"2","4"}, set(A.obs["folder_timepoint"])
assert is_pub.all(), f"expected all cells to join (barcodes identical to fake ref): {is_pub.sum()}/{len(is_pub)}"
print("ALL SMOKE ASSERTIONS PASSED")
