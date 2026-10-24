"""CellTag schema on CLONES: one row per cell, carrying the clone that cell belongs to.

WHY THIS EXISTS. build_corrected_schema_barcode_overlap.py turns each Jaccard PAIR into its own
synthetic two-cell clone and duplicates both cells into it, so a cell in k pairs becomes k rows named
`{cell}|P{i}`. The notebook does not need that: cell 34 already groups by clone_id and enumerates
`itertools.combinations(cells, 2)` itself, exactly as it does for LARRY. What the duplication does
instead is break the notebook's own de-duplication -- `all_t*_measurements` is built with
`drop_duplicates(subset="cell_id")`, and under the pair schema cell_id carries the `|P{i}` suffix, so
the same physical cell survives once per pair it belongs to. The existence correlation is then taken
across those repeated rows: 124,654 rows for 22,023 cells at day 4, and the day-4 median |rho| moves
from 0.0501 to 0.0755 when each cell counts once.

This file writes the schema the notebook was written for: one row per cell, clone_id = the clone.

THE CLONE CALL IS NOT MADE HERE. It is read from the clone table released with the CellTag-multi
study (clone_tables/hsc.rna&atac.r1&2_master_v2.csv), whose RNA rows are that study's own clone
assignment. The join from its `{library}-rna-{barcode}-{n}` ids to this matrix's ids is derived --
each matrix timepoint is matched to the table day it shares the most barcodes with, and each row
prefix to the one library it overlaps -- and asserted, never assumed.

Cells with no clone are written with a NaN label, which run_celltag's factorize turns into
clone_id = -1, the value the notebook filters out.

  CT_SOURCE   cell-level counts matrix (default: the quality-controlled no-singlet set)
  OUT_STEM    output stem under datasets/ (default: celltag_clones_published)
"""
import os
from dataset_paths import dpath  # datasets/ is grouped by arm; this resolves the group
import re
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
DS = os.environ.get("CT_SOURCE", dpath("celltag", "celltag_nosinglet_counts_mito20_min200genes.h5ad"))
TABLE = os.path.join(HERE, "CellTag-multi-2023", "clone_tables", "hsc.rna&atac.r1&2_master_v2.csv")
STEM = os.environ.get("OUT_STEM", "celltag_clones_published")
OUT_S = dpath(f"{STEM}.h5ad")
OUT_C = dpath(f"{STEM}_counts.h5ad")
CLONE_COL = "published_clone"


def log(*a):
    print(*a, flush=True)


assert os.path.exists(DS), f"missing input: {DS}"
assert os.path.exists(TABLE), f"missing clone table: {TABLE}"

A = ad.read_h5ad(DS)
log(f"source {os.path.basename(DS)}: {A.shape[0]} cells x {A.shape[1]} genes")
log(f"obs columns: {list(A.obs.columns)}")
assert A.obs_names.is_unique, "the source matrix has duplicate cell ids"

# ---- the clone table, RNA rows only -------------------------------------------------------------
ct = pd.read_csv(TABLE)
rna = ct[ct["assay"].astype(str).str.lower() == "rna"].copy()
rna["lib"] = rna["cell.bc"].str.split("-rna-").str[0]
rna["bare"] = rna["cell.bc"].str.extract(r"-rna-([ACGT]{16})")[0]
assert rna["bare"].notna().all(), "a clone-table id does not carry a 16bp barcode"
day_of_lib = {lib: sorted(set(g["day"])) for lib, g in rna.groupby("lib")}
assert all(len(v) == 1 for v in day_of_lib.values()), f"a library spans days: {day_of_lib}"
day_of_lib = {k: v[0] for k, v in day_of_lib.items()}
log(f"clone table RNA: {len(rna)} rows, {rna['clone.id'].nunique()} clones, libraries {day_of_lib}")

# ---- the matrix side: timepoint, row prefix, bare barcode ---------------------------------------
tp_col = next((c for c in A.obs.columns if c.lower() in ("time point", "timepoint", "time_point")), None)
assert tp_col is not None, f"no timepoint column in {list(A.obs.columns)}"
obs = pd.DataFrame(index=A.obs_names)
obs["tp"] = np.asarray(A.obs[tp_col])
obs["pref"] = obs.index.str.extract(r"^([^:]+):")[0].fillna("").to_numpy()
obs["bare"] = obs.index.str.split(":").str[-1]
assert (obs["bare"].str.len() == 16).all(), "a matrix id does not end in a 16bp barcode"
log(f"timepoints: {obs['tp'].value_counts().to_dict()}")

# Which table day each matrix timepoint is, decided by shared barcodes.
by_day = {d: set(g["bare"]) for d, g in rna.groupby("day")}
tday = {}
for mday, g in obs.groupby("tp"):
    b = set(g["bare"])
    hits = sorted(((d, len(b & s)) for d, s in by_day.items()), key=lambda kv: -kv[1])
    assert hits[0][1] > 10 * max(1, hits[1][1] if len(hits) > 1 else 0), \
        f"timepoint {mday} does not pick out one table day: {hits}"
    tday[mday] = hits[0][0]
    log(f"  timepoint {mday} -> table day {hits[0][0]}  ({hits[0][1]} barcodes; runner-up {hits[1:2]})")

# Which library each row prefix is, among that timepoint's libraries only.
by_lib = {lib: set(g["bare"]) for lib, g in rna.groupby("lib")}
pref_lib = {}
for (mday, pref), g in obs.groupby(["tp", "pref"]):
    b = set(g["bare"])
    cand = {lib: len(b & s) for lib, s in by_lib.items() if day_of_lib[lib] == tday[mday]}
    ranked = sorted(cand.items(), key=lambda kv: -kv[1])
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else ("-", 0)
    assert second[1] == 0, f"prefix {pref!r} at tp {mday} matches several libraries: {cand}"
    assert best[1] > 0, f"prefix {pref!r} at tp {mday} matches no library: {cand}"
    pref_lib[(mday, pref)] = best[0]
    log(f"  tp {mday} prefix {pref or '(none)':>6s} -> {best[0]:6s}  {best[1]}/{len(b)} barcodes")

# A barcode reused inside one library cannot be resolved to a clone; drop those keys rather than
# picking one arbitrarily.
key = rna.set_index(["lib", "bare"])["clone.id"]
dupes = int(key.index.duplicated(keep=False).sum())
key = key[~key.index.duplicated(keep=False)]
log(f"clone key: {len(key)} (library, barcode) pairs; {dupes} dropped as ambiguous")

obs["lib"] = [pref_lib.get((d, p)) for d, p in zip(obs["tp"], obs["pref"])]
obs[CLONE_COL] = pd.MultiIndex.from_arrays([obs["lib"], obs["bare"]]).map(key)
mapped = obs[CLONE_COL].notna()
log(f"\nmapped {int(mapped.sum())} of {len(obs)} cells ({100*mapped.mean():.2f}%); "
    f"{obs[CLONE_COL].nunique()} clones")
sz = obs.loc[mapped].groupby([CLONE_COL, "tp"]).size()
for t in sorted(obs["tp"].unique()):
    s = sz.xs(t, level="tp")
    log(f"  tp {t}: {int(s.sum())} cells in {len(s)} clones; clone size min {s.min()} "
        f"median {int(s.median())} max {s.max()}; within-timepoint pairs {int((s*(s-1)//2).sum())}")
_lv = sorted(set(obs["tp"].astype(str)))
if len(_lv) == 2:
    _e = min(_lv, key=lambda v: int(re.search(r"\d+", v).group()))
    _l = [v for v in _lv if v != _e][0]
    ce, cla = sz.xs(_e, level="tp"), sz.xs(_l, level="tp")
    both = ce.index.intersection(cla.index)
    log(f"  clones present at both collections: {len(both)}; cross pairs "
        f"{int((ce[both] * cla[both]).sum())}")

# ---- annotation and timepoint, from the same places the pair schema reads them ------------------
# The counts matrix carries neither a cell-type annotation nor the notebook's integer timepoints.
# build_corrected_schema_barcode_overlap.py takes the annotation from the clone-assignment table's
# cell_type2 column and applies one mapping; the same table and the same mapping are used here so the
# two schemas differ only in their pairing structure, never in their labels.
CLONE_ASSIGN = os.environ.get("CLONE_ASSIGN", os.path.join(
    ROOT, "exports", "legacy_reproductions", "pairing", "celltag_clones",
    "clone_assignments_corrected_mito20_min200genes.csv"))
assert os.path.exists(CLONE_ASSIGN), f"missing annotation table: {CLONE_ASSIGN}"
cl = pd.read_csv(CLONE_ASSIGN, index_col=0)
assert (cl.index.values == A.obs_names.values).all(), \
    "the annotation table is not indexed by this matrix's cells"
LYMPH = {"Lym", "Ccr7_DC"}


def map_ct(t, s):
    if t == "day2":
        return "Undifferentiated"
    if pd.isna(s):
        return "Unknown"
    if s in LYMPH:
        return "Lymphoid"
    return str(s)


tp_raw = obs["tp"].astype(str).to_numpy()
ann = np.array([map_ct(t, s) for t, s in zip(tp_raw, cl["cell_type2"].values)], dtype=object)

# The notebook requires integer timepoints t1 < t2 < t3. The pair schema encodes the two CellTag
# collections as 2 and 4; the same encoding is used here so nothing downstream has to branch. Neither
# is a CellTag day -- the collections are 60 h (day 2.5) and day 5, carried in experimental_day.
levels = sorted(set(tp_raw))
assert len(levels) == 2, f"expected two collections, found {levels}"
early = min(levels, key=lambda v: int(re.search(r"\d+", v).group()))
late = [v for v in levels if v != early][0]
log(f"timepoint encoding: {early} -> 2, {late} -> 4")
is_early = tp_raw == early

# ---- assemble the schema: ONE ROW PER CELL ------------------------------------------------------
S = pd.DataFrame(index=A.obs_names.copy())
S["Time point"] = np.where(is_early, 2, 4)
S["experimental_day"] = np.where(is_early, 2.5, 5.0)
S["Cell type annotation"] = ann
S[CLONE_COL] = obs[CLONE_COL].to_numpy()
S["is_padding"] = False

# NO PADDED THIRD TIMEPOINT. CellTag has two collections, 60 h and day 5, and the earlier schema
# copied the day-5 rows a second time under label 6 so the notebook's t1/t2/t3 split would find three.
# Nothing downstream may read it: run_celltag already empties adata_t3 in the dominant-cell-type cell
# and zeroes distances_t3, so the copy was structural only -- and a copy that must be neutralised
# everywhere it is touched is a copy that should not exist. With only the two real timepoints present,
# adata_t3 comes out empty by construction rather than by patch.
row_src = np.arange(A.shape[0])
assert S.index.is_unique, "non-unique schema obs names"
assert len(S) == len(row_src) == A.shape[0], "one row per cell, no copies"
log(f"\nschema rows {len(S)}: "
    + ", ".join(f"tp{t}={int((S['Time point'] == t).sum())}" for t in sorted(S['Time point'].unique())))
log(f"cell-type annotation counts:\n{S['Cell type annotation'].value_counts().to_string()}")

X = A.X.tocsr()
pred = int(np.diff(X.indptr)[row_src].astype(np.int64).sum())
log(f"predicted nonzeros: {pred:,}")
if pred >= 2**31 - 1:
    X.indptr = X.indptr.astype(np.int64)
    X.indices = X.indices.astype(np.int64)
    log("cast sparse indices to int64")
C = X[row_src]
var = A.var.copy()
del X, A

nz = np.asarray((C > 0).sum(0)).ravel() > 0
ad.AnnData(X=C[:, nz].astype(np.float32), obs=S.copy(), var=var.iloc[nz].copy()).write_h5ad(OUT_C)
log(f"wrote {OUT_C} (genes with a count: {int(nz.sum())})")

s = np.asarray(C.sum(1)).ravel()
N = (sp.diags(np.where(s > 0, 1e4 / s, 0.0)) @ C).tocsr()
N.data = np.log1p(N.data)
del C
ad.AnnData(X=N.astype(np.float32), obs=S.copy(), var=var.copy()).write_h5ad(OUT_S)
log(f"wrote {OUT_S}")

TF = ['Gata1', 'Gata2', 'Gfi1', 'Fli1', 'Spi1', 'Tal1', 'Cebpa', 'Jun', 'Egr1', 'Nab2', 'Klf1', 'Zfpm1']
log(f"12 panel genes present: {all(g in var.index for g in TF)}")
log("BUILD_CELLTAG_CLONE_SCHEMA_DONE")
