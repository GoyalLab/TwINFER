"""CellTag clones by the authors' FULL three-step caller, including the sparse-clone repair we skip.

WHY. sf_utils_morris.call_clones is only step 1 of their procedure. It thresholds Jaccard at 0.6,
builds the graph, and takes CONNECTED COMPONENTS -- which is transitive, so A-B and B-C chain A to C
even when Jaccard(A,C) < 0.6. Their published answer to that chaining is not a different clone
definition; it is a repair:

    call_clones(..., return_graph=True) -> (graph, clone_table with a per-clone `edge.den`)
    ident_sparse_clones(clone_info, n_largest=10, density_th=0.2)
        -> among the N largest clones, those whose induced subgraph density is below 0.2 are
           CHAINS, not clones
    fix_sparse_clones(graph, sparse_ids)
        -> each flagged clone is decomposed by repeatedly extracting its LARGEST CLIQUE; every
           clique becomes a clone of density 1, and cells left over once the subgraph runs out of
           edges are dropped ("wasted")

helpers/run_morris_clonecalling.py calls call_clones with return_graph=FALSE, which throws the graph
away, and nothing in the project calls ident_sparse_clones or fix_sparse_clones. So we ship step 1
of 3, and the clones their own QC would reject are exactly the large ones that dominate any
clone-weighted statistic.

Every computational step here is THEIR function. This file only assembles the input and reports.

INPUT. clone_assignments_corrected_mito20_min200genes.csv, whose `lineage_barcodes` column holds each
cell's CellTag set and whose index is the bare barcode that joins to the analysis matrix. Rebuilding
the graph from the tags (rather than from the stored pair list) is what makes the graph object
available, and the graph is what fix_sparse_clones needs.

POSITIVE CONTROL (G11), both must pass or nothing is reported:
  1. the Jaccard>0.6 edge set rebuilt here must reproduce the pair list the analysis matrix was built
     from (exports/legacy_reproductions/pairing/celltag_clones/pairs_barcode_overlap_2to24tags_mito20_min200genes.csv);
  2. call_clones' components must reproduce the partition already stored in the `jaccard_overlap`
     column of the input table.
If the rebuild does not reproduce what is shipped, the repair below would be a repair of a different
graph and the numbers would be meaningless.

fix_sparse_clones returns a bare None on either of its internal assertion failures (it prints and
`return()`s), so its result is checked for None rather than trusted.

OUTPUT  exports/legacy_reproductions/pairing/celltag_clones/clone_table_repaired.csv
        exports/qc/celltag_clones_repaired.json
"""
import json
import os
import sys
import types

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
for m in ["cassiopeia", "cassiopeia.critique", "cassiopeia.critique.critique_utilities",
          "shap", "upsetplot", "adjustText"]:
    if m not in sys.modules:
        try:
            __import__(m)
        except ImportError:
            mod = types.ModuleType(m)
            mod.get_outgroup = lambda *a, **k: None
            mod.__getattr__ = lambda n: None
            sys.modules[m] = mod
import sf_utils_morris as sf                                              # THEIR functions

CL = os.path.join(ROOT, "exports", "legacy_reproductions", "pairing", "celltag_clones",
                  "clone_assignments_corrected_mito20_min200genes.csv")
PAIRS = os.path.join(ROOT, "exports", "legacy_reproductions", "pairing", "celltag_clones",
                     "pairs_barcode_overlap_2to24tags_mito20_min200genes.csv")
OUT_T = os.path.join(ROOT, "exports", "legacy_reproductions", "pairing", "celltag_clones", "clone_table_repaired.csv")
OUT_J = os.path.join(ROOT, "exports", "qc", "celltag_clones_repaired.json")
os.makedirs(os.path.dirname(OUT_J), exist_ok=True)

JAC_TH = 0.6            # their call_clones default, and the project's settled value
BIN_TH = 0              # the table already stores called tags, one per cell-tag pair
LO, HI = 1, 26          # their 2-25 tag window, as (>LO) & (<HI)
N_LARGEST = 10          # ident_sparse_clones default
DENSITY_TH = 0.2        # ident_sparse_clones default


def log(m):
    print(m, flush=True)


# ---- assemble the (cell, tag) table their code expects -------------------------------------
c = pd.read_csv(CL, index_col=0)
c.index = c.index.astype(str)
have_tags = c["lineage_barcodes"].notna()
log(f"{len(c):,} cells in the clone table; {int(have_tags.sum()):,} carry lineage_barcodes")
cells_tags = (c.loc[have_tags, "lineage_barcodes"].astype(str).str.split(",")
              .explode().str.strip())
tab = pd.DataFrame({"CB": cells_tags.index.values, "tag": cells_tags.values})
tab = tab[tab.tag.astype(bool)]
log(f"(cell, tag) rows: {len(tab):,} | distinct tags: {tab.tag.nunique():,}")

# ---- THEIR functions from here -------------------------------------------------------------
mtx, cells, celltags = sf.table_to_spmtx(tab["CB"].values, tab["tag"].values,
                                         np.ones(len(tab), dtype=int))
mbin = mtx > BIN_TH
row_fil = ((mbin.sum(axis=1) > LO) & (mbin.sum(axis=1) < HI))
temp = mbin[row_fil.nonzero()[0], ]
col_fil = temp.sum(axis=0) > 0
met = temp[:, col_fil.nonzero()[1]].copy() * 1
cells_met = np.array(cells)[row_fil.nonzero()[0]]
log(f"after the {LO+1}-{HI-1} tag window: {met.shape[0]:,} cells, {met.shape[1]:,} tags")

jac = sf.jaccard_similarities(met.tocsc().astype(np.float64).transpose())
jac.setdiag(0)
jac.eliminate_zeros()
log(f"Jaccard nonzeros: {jac.nnz:,} | pairs > {JAC_TH}: {int((jac > JAC_TH).nnz):,}")

# DIAGNOSTIC: call_clones thresholds with a STRICT `>`; report how many pairs sit exactly at the
# cutoff, because a `>=` build and a `>` build differ by exactly that set.
_eq = int(((jac.data > JAC_TH - 1e-12) & (jac.data < JAC_TH + 1e-12)).sum() // 2)
log(f"  pairs with Jaccard EXACTLY {JAC_TH}: {_eq:,}  "
    f"(> {JAC_TH}: {int((jac > JAC_TH).nnz)//2:,}, >= {JAC_TH}: {int((jac >= JAC_TH - 1e-12).nnz)//2:,})")

# THEIR NOTEBOOK PASSES THE LOWER TRIANGLE. cloneCalling_scripts/celltag_analysis_single_assay.ipynb:
#     g, clones = sf_utils.call_clones(scipy.sparse.tril(jac_mat), cells_met)
# and call_clones' own parameter is named `jac_mat_low`. Feeding the FULL SYMMETRIC matrix -- which
# helpers/run_morris_clonecalling.py:61 does -- enters every undirected edge twice, because call_clones
# builds its edge list from jac_mat.nonzero(). igraph then reports 2x the true density, so edge.den
# maxes at 2.0, ident_sparse_clones(density_th=0.2) can never fire, and fix_sparse_clones' internal
# `assert new_gr.density() == 1` on a maximum clique reads 2.0 and makes it return None. Their code is
# correct; ours dropped the tril.
#
# SECOND DIVERGENCE, measured not assumed: call_clones thresholds STRICTLY (`jac_mat_low > jac_th`),
# but the shipped CellTag pair list was built INCLUSIVELY at >= 0.6 -- they differ by exactly the
# 12,849 pairs sitting at 0.6, which is exactly the control-1 deficit. The repair has to run on the
# graph our results actually use, so the threshold is moved one epsilon below 0.6 to make their strict
# `>` reproduce our `>=`. Their function is unmodified; only the argument moves.
import scipy.sparse as _sp
_TH_EFF = JAC_TH - 1e-9
g, ct = sf.call_clones(_sp.tril(jac), cells_met, jac_th=_TH_EFF, return_graph=True)
log(f"call_clones fed the LOWER TRIANGLE, as their notebook does: {g.ecount():,} edges, "
    f"multi-edges {int(sum(g.is_multiple())):,}")
ct = pd.DataFrame(ct)
ct.columns = [str(x) for x in ct.columns]
log(f"call_clones: {ct['clone.id'].nunique():,} clones over {ct['cell.bc'].nunique():,} cells")

# ---- POSITIVE CONTROL 1: the rebuilt edge set must be the shipped one ----------------------
bad = 0
mine = set()
gi = {v.index: v["cb"] for v in g.vs}
for e in g.es:
    mine.add(frozenset((str(gi[e.source]), str(gi[e.target]))))
P = pd.read_csv(PAIRS)
cc = [x for x in P.columns if P[x].dtype == object][:2]
theirs = {frozenset((str(a), str(b))) for a, b in zip(P[cc[0]], P[cc[1]])}
ok = mine == theirs
bad += 0 if ok else 1
log(f"  control 1  rebuilt edges {len(mine):,}  shipped edges {len(theirs):,}  "
    f"identical: {ok}" + ("" if ok else f"  (only-mine {len(mine - theirs):,}, "
                                        f"only-theirs {len(theirs - mine):,})"))

# ---- POSITIVE CONTROL 2: components must reproduce the stored jaccard_overlap partition ----
stored = c.loc[c["jaccard_overlap"].notna(), "jaccard_overlap"].astype(str)
mine_p = ct.groupby("clone.id")["cell.bc"].apply(lambda s: frozenset(map(str, s)))
mine_set = set(mine_p)
their_set = {frozenset(map(str, v.index)) for _, v in stored.groupby(stored)}
ok2 = mine_set == their_set
bad += 0 if ok2 else 1
log(f"  control 2  rebuilt partition {len(mine_set):,} groups  stored partition "
    f"{len(their_set):,} groups  identical: {ok2}"
    + ("" if ok2 else f"  (only-mine {len(mine_set - their_set):,}, "
                      f"only-stored {len(their_set - mine_set):,})"))
if bad:
    sys.exit(f"positive control failed ({bad}); the repair would be applied to the wrong graph. "
             f"Reporting nothing.")

# ---- their sparse-clone identification and repair -------------------------------------------
ct["edge.den"] = pd.to_numeric(ct["edge.den"]).astype(float)
_dmin, _dmax = float(ct["edge.den"].min()), float(ct["edge.den"].max())
log(f"  edge.den range {_dmin:.4f} .. {_dmax:.4f}")
assert _dmax <= 1.0 + 1e-9, (
    f"edge.den reaches {_dmax}; a density cannot exceed 1, so the graph still carries duplicate "
    f"edges -- call_clones was not fed a triangular matrix")
info = (ct.groupby("clone.id")
        .agg(size=("cell.bc", "size"), **{"edge.den": ("edge.den", "first")})
        .reset_index())
log(f"\nclone sizes: max {int(info['size'].max())}, "
    f"{int((info['size'] > 2).sum()):,} clones larger than 2 cells")

sparse = sf.ident_sparse_clones(info, n_largest=N_LARGEST, density_th=DENSITY_TH)
if sparse is None or not len(sparse):
    log(f"ident_sparse_clones (n_largest={N_LARGEST}, density_th={DENSITY_TH}): none flagged; "
        f"the repair is a no-op on this graph")
    rep = ct.copy()
    n_sparse = 0
else:
    n_sparse = len(sparse)
    log(f"ident_sparse_clones (n_largest={N_LARGEST}, density_th={DENSITY_TH}): "
        f"{n_sparse} flagged")
    for _, r in sparse.iterrows():
        log(f"    clone {int(r['clone.id']):5d}  size {int(r['size']):4d}  "
            f"edge density {float(r['edge.den']):.3f}")
    rep = sf.fix_sparse_clones(g, sparse["clone.id"].values.astype(int))
    if rep is None or not len(rep):
        sys.exit("fix_sparse_clones returned None (one of its internal assertions failed); "
                 "reporting nothing.")
    rep = pd.DataFrame(rep)
    rep.columns = [str(x) for x in rep.columns]

# how many clones would the same cutoff flag if it looked past the N largest?
all_sparse = int(((info["size"] > 2) & (info["edge.den"] < DENSITY_TH)).sum())
log(f"  (for reference: {all_sparse:,} clones of >2 cells fall below density {DENSITY_TH} in total; "
    f"their default only inspects the {N_LARGEST} largest)")

before_cells = int(ct["cell.bc"].nunique())
after_cells = int(rep["cell.bc"].nunique())
log(f"\nclones  {ct['clone.id'].nunique():,} -> {rep['clone.id'].nunique():,}")
log(f"cells   {before_cells:,} -> {after_cells:,}   (dropped as 'wasted': "
    f"{before_cells - after_cells:,})")
mx_b, mx_a = int(info["size"].max()), int(rep.groupby("clone.id").size().max())
log(f"largest clone  {mx_b} -> {mx_a} cells")

rep.to_csv(OUT_T, index=False)
json.dump({"jac_th": JAC_TH, "n_largest": N_LARGEST, "density_th": DENSITY_TH,
           "clones_before": int(ct["clone.id"].nunique()),
           "clones_after": int(rep["clone.id"].nunique()),
           "cells_before": before_cells, "cells_after": after_cells,
           "largest_before": mx_b, "largest_after": mx_a,
           "n_flagged_sparse": n_sparse, "n_below_density_all_sizes": all_sparse},
          open(OUT_J, "w"), indent=1, default=float)
log(f"\nwrote {os.path.relpath(OUT_T, ROOT)}")
log(f"wrote {os.path.relpath(OUT_J, ROOT)}")
