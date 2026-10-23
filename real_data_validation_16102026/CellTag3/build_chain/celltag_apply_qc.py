"""Apply quality control to the CellTag collection output and write the filtered matrix.

THE RULE, as agreed for CellTag: remove cells above 20% mitochondrial content; discard cells
expressing fewer than 200 genes. That is exactly `helpers/qc.py`, which asserts its own
postconditions, so this cannot complete while leaving the stated rule false of the data.

WHY THIS EXISTS AS A SEPARATE STEP
  `qc.py` was written and imported by the normalization/measurement scripts, but NOTHING on the
  TwINFER path imports it: `run_celltag.py` reads a schema artifact built by
  `build_corrected_schema_barcode_overlap.py` from `celltag_nosinglet_counts.h5ad`, and no step in
  that chain filters cells. So the pairing, the schema and every network run downstream were
  computed on unfiltered cells. This step inserts quality control between collection and pairing,
  which is where the ordering puts it, and everything after it is rebuilt from the output.

WHAT IS DELIBERATELY NOT HERE
  - No gene-level filter. Genes are not filtered anywhere in this chain and are not filtered here.
    The counts artifact written further downstream drops all-zero genes as part of its own
    construction; that behaviour is left exactly as it is.
  - No UMI/total-count threshold. The source paper's stated quality control for this dataset is
    "Low-quality cells with high mitochondrial reads, low UMIs and features per cell were removed"
    with no numbers given, and no low-UMI floor appears in the published clone-calling code. There
    is nothing to reproduce, so nothing is invented.
  - No doublet removal.

Counts are computed from the matrix being filtered, never read from obs, and the output path
carries the rule in its name so a run under a different rule cannot overwrite this one.
"""
import json
from dataset_paths import dpath  # datasets/ is grouped by arm; this resolves the group
import os

import anndata as ad
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
import sys
sys.path.insert(0, HERE)
import qc

SOURCE = os.environ.get("CT_SOURCE",
                        dpath("celltag", "celltag_nosinglet_counts.h5ad"))
MITO = float(os.environ.get("MITO_PCT_MAX", qc.MITO_PCT_MAX))
MING = int(os.environ.get("MIN_GENES", qc.MIN_GENES))

stem = os.path.basename(SOURCE)[:-len(".h5ad")] if SOURCE.endswith(".h5ad") else os.path.basename(SOURCE)
TAG = f"_mito{MITO:g}_min{MING}genes"
OUT = dpath(f"{stem}{TAG}.h5ad")
REC = os.path.join(ROOT, "exports", "legacy_reproductions", "pairing", "celltag_clones", f"qc_record{TAG}.json")

assert os.path.exists(SOURCE), f"missing input: {SOURCE}"
assert os.path.abspath(OUT) != os.path.abspath(SOURCE), "refusing to overwrite the input"


def log(m):
    print(m, flush=True)


if os.path.exists(OUT):
    chk = ad.read_h5ad(OUT, backed="r")
    log(f"{os.path.basename(OUT)} exists: {chk.n_obs:,} cells; leaving it")
    raise SystemExit(0)

log(f"reading {SOURCE}")
A = ad.read_h5ad(SOURCE)
log(f"in: {A.n_obs:,} cells x {A.n_vars:,} genes")

B, rec = qc.apply_qc(A, min_genes=MING, mito_pct_max=MITO, label=os.path.basename(SOURCE))

# Per-timepoint and per-lane bookkeeping, derived from whatever those columns are called here rather
# than assumed to exist: a rebuild that silently loses a lane must be visible in the record.
for key in ("timepoint", "lane"):
    if key in A.obs.columns:
        before = A.obs[key].astype(str).value_counts().to_dict()
        after = B.obs[key].astype(str).value_counts().to_dict()
        rec[f"{key}_before"] = {str(k): int(v) for k, v in before.items()}
        rec[f"{key}_after"] = {str(k): int(v) for k, v in after.items()}
        assert set(after) <= set(before), f"{key}: filtering invented a category"
        log(f"  {key}: {len(before)} categories in, {len(after)} out")

# Tag eligibility, if the collection output carries it. The pairing step re-derives its own counts
# from lineage_barcodes; this is a report, not an input to anything.
if "ntags" in A.obs.columns:
    nt_in = np.asarray(A.obs["ntags"]).astype(int)
    nt_out = np.asarray(B.obs["ntags"]).astype(int)
    rec["cells_2to24tags_before"] = int(((nt_in >= 2) & (nt_in <= 24)).sum())
    rec["cells_2to24tags_after"] = int(((nt_out >= 2) & (nt_out <= 24)).sum())
    rec["cells_with_25plus_tags"] = int((nt_in >= 25).sum())
    log(f"  cells with 2-24 tags: {rec['cells_2to24tags_before']:,} -> "
        f"{rec['cells_2to24tags_after']:,}  (cells at 25+ tags: {rec['cells_with_25plus_tags']:,})")

prev = dict(A.uns.get("preprocessing", {})) if hasattr(A, "uns") else {}
prev.update({"quality_control": f"cells above {MITO:g}% mitochondrial removed; cells with fewer "
                                f"than {MING} genes removed; metrics computed from the count matrix",
             "quality_control_source": os.path.basename(SOURCE)})
B.uns["preprocessing"] = prev

B.write_h5ad(OUT)
log(f"wrote {OUT}: {B.n_obs:,} cells x {B.n_vars:,} genes")

chk = ad.read_h5ad(OUT, backed="r")
assert chk.n_obs == B.n_obs, "the file on disk does not hold the expected cell count"
assert chk.n_vars == A.n_vars, "the gene axis changed; no gene filter is applied by this step"

os.makedirs(os.path.dirname(REC), exist_ok=True)
with open(REC, "w") as fh:
    json.dump(rec, fh, indent=2, sort_keys=True)
log(f"wrote {REC}")
log("done")
