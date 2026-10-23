#!/usr/bin/env python3
"""Cell-type / tissue signature enrichment of the target genes of the top-100 unshared and shared lists (lists from build_top100_lists_alltf.py).
Reference libraries (Enrichr, downloaded once to analysis_data/reference_gene_sets/enrichr/): PanglaoDB_Augmented_2021, CellMarker_Augmented_2021, Azimuth_Cell_Types_2021, Descartes_Cell_Types_and_Tissue_2021.
Test: hypergeometric, the list's distinct target genes against all targets of the dataset's Stage I pairs (background); sets restricted to background genes, 10 to 1000 genes; BH over all sets of a library per list.
Old gene symbols in our tables that Enrichr has renamed (e.g. GNB2L1, CTGF, CYR61, KIAA0101, HIST1H*) do not match: reported as a limitation, no alias mapping is applied.
Output: analysis_data/fm_meta_programs_comparison/identity_signatures/<dataset>_<list>_<library>.csv and identity_signatures_top_terms.csv.   usage: list_identity_signatures.py [FM01 FM06]"""
import os, sys, urllib.request
import numpy as np, pandas as pd
from scipy.stats import hypergeom
ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"; A = f"{ROOT}/analysis_data"; LD = f"{A}/reference_gene_sets/enrichr"; OUT = f"{A}/fm_meta_programs_comparison/identity_signatures"; L = f"{A}/fm_meta_programs_comparison/top100_lists"
os.makedirs(LD, exist_ok=True); os.makedirs(OUT, exist_ok=True)
LIBS = ["PanglaoDB_Augmented_2021", "CellMarker_Augmented_2021", "Azimuth_Cell_Types_2021", "Descartes_Cell_Types_and_Tissue_2021"]
def lib(name):
    f = f"{LD}/{name}.txt"
    if not os.path.exists(f): urllib.request.urlretrieve(f"https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName={name}", f)
    S = {}
    for line in open(f):
        p = line.rstrip("\n").split("\t")
        if len(p) > 2: S[p[0]] = {g.split(",")[0].upper() for g in p[2:] if g.strip()}
    return S
def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n); q[o] = np.minimum.accumulate((n * p[o] / np.arange(1, n + 1))[::-1])[::-1]; return np.minimum(q, 1)
rows = []
for ds in (sys.argv[1:] or ["FM01", "FM06"]):
    bg = set(l.strip().upper() for l in open(f"{L}/{ds}_stage1_targets.txt") if l.strip())
    for lab in ("unshared", "shared"):
        tg = set(pd.read_csv(f"{L}/{ds}_{lab}.csv").gene.str.upper()) ; n = len(tg & bg)
        for ln in LIBS:
            S = lib(ln); res = []
            for t, g in S.items():
                g = g & bg
                if 10 <= len(g) <= 1000:
                    k = len(g & tg); res.append((t, len(g), k, hypergeom.sf(k - 1, len(bg), len(g), n) if k > 0 else 1.0, (k / n) / (len(g) / len(bg))))
            R = pd.DataFrame(res, columns=["term", "set_size_in_bg", "overlap", "p", "fold"]); R["q"] = bh(R.p); R.insert(0, "library", ln); R.insert(0, "list", lab); R.insert(0, "dataset", ds)
            R.sort_values("p").to_csv(f"{OUT}/{ds}_{lab}_{ln}.csv", index=False); rows.append(R.sort_values("p").head(6))
T = pd.concat(rows); T.to_csv(f"{OUT}/identity_signatures_top_terms.csv", index=False)
pd.set_option("display.width", 250, "display.max_colwidth", 60)
for (ds, lab, ln), d in T.groupby(["dataset", "list", "library"], sort=False):
    print(f"\n{ds} {lab} | {ln.replace('_Augmented_2021','').replace('_Cell_Types_2021','').replace('_Cell_Types_and_Tissue_2021','')}  (n targets in background: {len(set(pd.read_csv(f'{L}/{ds}_{lab}.csv').gene.str.upper()) & set(l.strip().upper() for l in open(f'{L}/{ds}_stage1_targets.txt')))})")
    print(d.head(4)[["term", "overlap", "set_size_in_bg", "fold", "q"]].assign(fold=lambda x: x.fold.round(1), q=lambda x: x.q.map(lambda v: f"{v:.0e}")).to_string(index=False, header=False))
