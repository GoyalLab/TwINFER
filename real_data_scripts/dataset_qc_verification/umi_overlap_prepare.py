#!/usr/bin/env python3
"""Step 1 of the duplicate-cell UMI-overlap test (LARRY day-2 libraries).

Writes, under OUT (analysis_data/larry_raw_sra/umi_overlap/):
  pairs.csv           one row per duplicate (cell barcode, clone) group present in >1 of the chosen day-2 libraries
  targets_<lib>.txt   cell barcodes whose reads are pulled from that library's SRA run
                      = duplicate cells (case 'dup'), plus a random control set of the same library (case 'ctrl'),
                        plus cells whose barcode sits in 2 libraries but with different clones (case 'samebc_diffclone')
  targets.csv         lib, cb, case

usage: umi_overlap_prepare.py [N_CTRL_PER_LIB]
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"
OUT = f"{ROOT}/analysis_data/larry_raw_sra/umi_overlap"
LIBS = ["LSK_d2_1", "LSK_d2_2", "LSK_d2_3"]
N_CTRL = int(sys.argv[1]) if len(sys.argv) > 1 else 400
os.makedirs(OUT, exist_ok=True)

f = pd.read_csv(f"{ROOT}/finalized_data/LARRY_data/filtered/obs_metadata.csv", index_col=0)
f["cb"] = f.index.str.split(":", n=1).str[1]
f["clone"] = f["larry_clone_singletcode"]
d = f[f.library.isin(LIBS)].copy()

dc = d[d.clone.notna()]
g = dc.groupby(["cb", "clone"]).library.nunique()
dup_keys = g[g > 1].index
dup = dc.set_index(["cb", "clone"]).loc[dup_keys].reset_index()
pairs = (dup.groupby(["cb", "clone"]).library.apply(lambda s: ",".join(sorted(s)))
         .rename("libraries").reset_index())
pairs.to_csv(f"{OUT}/pairs.csv", index=False)

dup_cells = set(zip(dup.library, dup.cb))
nl = d.groupby("cb").library.nunique()
multi = set(nl[nl > 1].index)
sbdc = d[d.cb.isin(multi) & ~d.apply(lambda r: (r.library, r.cb) in dup_cells, axis=1)]
rng = np.random.default_rng(0)
rows = [(l, c, "dup") for l, c in sorted(dup_cells)]
rows += [(r.library, r.cb, "samebc_diffclone") for r in sbdc.itertuples()]
taken = {(l, c) for l, c, _ in rows}
for lib in LIBS:
    pool = d[(d.library == lib) & ~d.cb.isin(multi)]
    pick = pool.sample(min(N_CTRL, len(pool)), random_state=int(rng.integers(1 << 30)))
    rows += [(lib, c, "ctrl") for c in pick.cb if (lib, c) not in taken]
t = pd.DataFrame(rows, columns=["lib", "cb", "case"]).drop_duplicates(["lib", "cb"])
t.to_csv(f"{OUT}/targets.csv", index=False)
for lib in LIBS:
    open(f"{OUT}/targets_{lib}.txt", "w").write("\n".join(sorted(set(t[t.lib == lib].cb))) + "\n")
print("pairs", len(pairs), "libraries per pair:", pairs.libraries.value_counts().to_dict())
print(t.groupby(["lib", "case"]).size().to_string())
