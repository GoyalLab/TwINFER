#!/usr/bin/env python3
"""Comparison variants of the CURRENT LARRY file (finalized_data/LARRY_data, 33,362 cells), written to finalized_data/rebuild_fullsheet_2026-10-16/variants/<name>/LARRY_data/{filtered,annotated}:
  opt2 : Weinreb et al. rules (ii)+(iii) applied as written (build_final_matrix_umi3.apply_weinreb_rule_ii_iii): every cell flagged by either rule is removed, both copies of a duplicated (cell barcode, clone) pair
  opt3 : one copy kept of each (cell barcode, clone) pair that occurs in more than one library (the copy with the highest total counts; ties -> first library name); rule (iii) not applied
The finalized files are not modified. usage: larry_variants.py opt2|opt3"""
import json
import os
import shutil
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io as sio

ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"
sys.path[:0] = [f"{ROOT}/code/TwINFER", f"{ROOT}/code/TwINFER/package"]
import paper_analysis.larry_pipeline.filtering.build_final_matrix_umi3 as m

F = f"{ROOT}/finalized_data/LARRY_data"
var = sys.argv[1]
OUT = f"{ROOT}/finalized_data/rebuild_fullsheet_2026-10-16/variants/{var}/LARRY_data"
obs = pd.read_csv(f"{F}/filtered/obs_metadata.csv", index_col=0)
cells = [l.rstrip("\n") for l in open(f"{F}/filtered/cells.txt")]
assert list(obs.index) == cells
clone = obs["larry_clone_singletcode"].astype(str)
cb = obs.index.str.split(":", n=1).str[1]
if var == "opt2":
    r2, r3 = m.apply_weinreb_rule_ii_iii(obs, pd.Series(obs.index))
    keep = ~(np.asarray(r2) | np.asarray(r3))
    note = f"Weinreb rules (ii)+(iii) as written: rule (ii) flagged {int(np.sum(r2))}, rule (iii) {int(np.sum(r3))}, either {int(np.sum(np.asarray(r2) | np.asarray(r3)))}"
elif var == "opt3":
    df = pd.DataFrame({"cb": cb, "clone": clone.to_numpy(), "library": obs["library"].to_numpy(), "tc": obs["total_counts"].to_numpy()}, index=obs.index)
    nl = df.groupby(["cb", "clone"])["library"].transform("nunique")
    dup = nl > 1
    order = df[dup].sort_values(["tc", "library"], ascending=[False, True])
    winners = set(order.drop_duplicates(["cb", "clone"]).index)
    keep = np.array([(not d) or (i in winners) for i, d in zip(df.index, dup)])
    note = f"one copy kept of each (cell barcode, clone) pair present in more than one library: {int(dup.sum())} cells in {df[dup].groupby(['cb', 'clone']).ngroups} pairs, {int((~keep).sum())} copies removed"
else:
    raise SystemExit(__doc__)
print(var, note, "| cells", int(keep.sum()), "of", len(keep), flush=True)
os.makedirs(f"{OUT}/filtered", exist_ok=True); os.makedirs(f"{OUT}/annotated", exist_ok=True)
X = sio.mmread(f"{F}/filtered/larry_qc_counts.mtx").tocsr()
sio.mmwrite(f"{OUT}/filtered/larry_qc_counts.mtx", X[np.flatnonzero(keep)])
shutil.copy(f"{F}/filtered/genes.txt", f"{OUT}/filtered/genes.txt")
open(f"{OUT}/filtered/cells.txt", "w").write("\n".join(np.array(cells)[keep]) + "\n")
obs[keep].to_csv(f"{OUT}/filtered/obs_metadata.csv")
j = json.load(open(f"{F}/filtered/preprocessing_summary.json")); j["variant"] = var; j["variant_note"] = note; j["n_variant_cells"] = int(keep.sum())
json.dump(j, open(f"{OUT}/filtered/preprocessing_summary.json", "w"), indent=2)
A = ad.read_h5ad(f"{F}/annotated/filtered_annotated_k15.h5ad")
ids = {f"{c.split(':', 1)[0]}|{c.split(':', 1)[1].replace('-', '')}" for c in np.array(cells)[keep]}
sel = A.obs_names.isin(ids)
assert int(sel.sum()) == int(keep.sum()), (int(sel.sum()), int(keep.sum()))
A[sel].copy().write_h5ad(f"{OUT}/annotated/filtered_annotated_k15.h5ad")
print(var, "written", OUT, flush=True)
