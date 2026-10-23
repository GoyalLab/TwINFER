#!/usr/bin/env python3
"""
module_enrichment.py

Formal over-representation analysis of gene modules against gene-set libraries (GMT).
Hypergeometric test (one-sided), background = every gene in the module set (the genes that passed the detection
filter), gene sets restricted to the background (min 5, max 500 genes, min 3 overlap), Benjamini-Hochberg FDR
within each module x library. Libraries are downloaded once from Enrichr as plain files (no gene list is sent out).
"""
import argparse, os
import numpy as np, pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests


def read_gmt(p):
    d = {}
    for line in open(p):
        f = line.rstrip("\n").split("\t")
        if len(f) > 2:
            d[f[0]] = {g for g in f[2:] if g}
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modules-csv", required=True)
    ap.add_argument("--libs-dir", required=True)
    ap.add_argument("--libs", nargs="+", default=["MSigDB_Hallmark_2020", "GO_Biological_Process_2023", "Reactome_Pathways_2024", "KEGG_2021_Human"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-size", type=int, default=5)
    ap.add_argument("--max-size", type=int, default=500)
    a = ap.parse_args()
    m = pd.read_csv(a.modules_csv, index_col=0).iloc[:, 0]
    bg = set(m.index); N = len(bg)
    rows = []
    for lib in a.libs:
        sets = {k: v & bg for k, v in read_gmt(f"{a.libs_dir}/{lib}.gmt").items()}
        sets = {k: v for k, v in sets.items() if a.min_size <= len(v) <= a.max_size}
        for k in sorted(m.unique()):
            genes = set(m.index[m == k]); n = len(genes); res = []
            for t, s in sets.items():
                o = genes & s
                if len(o) >= 3:
                    res.append((t, len(o), len(s), hypergeom.sf(len(o) - 1, N, len(s), n), ",".join(sorted(o)[:15])))
            if not res:
                continue
            d = pd.DataFrame(res, columns=["term", "overlap", "set_size", "p", "genes"])
            d["fdr"] = multipletests(d.p, method="fdr_bh")[1]
            d["fold"] = (d.overlap / n) / (d.set_size / N)
            d.insert(0, "module", k); d.insert(1, "module_size", n); d.insert(2, "library", lib)
            rows.append(d)
    R = pd.concat(rows).sort_values(["module", "library", "p"])
    R.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(R)} tests, {(R.fdr < 0.05).sum()} with FDR < 0.05")


if __name__ == "__main__":
    main()
