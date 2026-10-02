from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_175bdfed_2026-09-17_todo4v2; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import json
import h5py
import numpy as np
from scipy.sparse import csr_matrix

PATH = f'{TWINFER_PROJECT_ROOT}/real_data/cellTag3_data/celltag_clones_repaired_nopad.h5ad'
f = h5py.File(PATH, "r")

n_obs = len(f["X/indptr"]) - 1
var_names = np.array([g.decode() for g in f["var/_index"][:]])
n_var = len(var_names)
print(f"n_obs={n_obs}  n_var={n_var}")

indptr = f["X/indptr"][:]
indices = f["X/indices"][:]
data = f["X/data"][:]
X = csr_matrix((data, indices, indptr), shape=(n_obs, n_var))
print("loaded CSR:", X.shape, X.nnz)

# per-gene detection fraction: fraction of cells with a nonzero (explicit) entry
gene_nnz = np.diff(X.tocsc().indptr)
frac_expr = gene_nnz / n_obs
gene_mask = frac_expr >= 0.05
print(f"gene filter: keeping {int(gene_mask.sum()):,} / {n_var:,} genes expressed in >=5% of cells")
kept_genes = set(var_names[gene_mask])

# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] gs = json.load(open(f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation/resources/gene_sets_yscher.json'))
gs = json.load(open(f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/larry_hematopoiesis_validation/resources/gene_sets_yscher.json'))
panel = gs["correlation_high"]
print(f"\ncorrelation_high panel: {len(panel)} genes")
all_var = set(var_names)
missing_entirely = [g for g in panel if g not in all_var]
below_5pct = [g for g in panel if g in all_var and g not in kept_genes]
survive = [g for g in panel if g in kept_genes]
print(f"  not in celltag3 var at all: {len(missing_entirely)} {missing_entirely}")
print(f"  present but <5% detection: {len(below_5pct)} {below_5pct}")
print(f"  survive (present, >=5% detection): {len(survive)}")

# twin/clone structure
tp = f["obs/Time point"][:]
rc = f["obs/repaired_clone"][:]
print(f"\nTime point counts: t2={np.sum(tp==2)}  t4={np.sum(tp==4)}")
has_clone = ~np.isnan(rc)
print(f"cells with a repaired_clone: {has_clone.sum()} / {n_obs}")
clones_t2 = set(rc[(tp == 2) & has_clone])
clones_t4 = set(rc[(tp == 4) & has_clone])
twin_clones = clones_t2 & clones_t4
print(f"clones present at t2: {len(clones_t2)}  at t4: {len(clones_t4)}  cross-time twin clones: {len(twin_clones)}")

ct_codes = f["obs/Cell type annotation/codes"][:]
ct_cats = np.array([c.decode() for c in f["obs/Cell type annotation/categories"][:]])
print("\nCell type annotation value counts:")
vals, counts = np.unique(ct_codes, return_counts=True)
for v, c in zip(vals, counts):
    label = ct_cats[v] if v >= 0 else "(NaN/unset)"
    print(f"  {label}: {c}")
