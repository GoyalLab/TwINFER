#!/usr/bin/env python3
"""GO Biological Process 2023 (Enrichr gmt, analysis_data/gene_set_libraries) over-representation of the TARGET genes in the top-1000 UNSHARED (|z_U|) and top-1000 SHARED (|h|=|z_C|) pairs of FM01 and FM06
(same lists as program_11_hallmarks_vertical_plot.py: ChIP-Atlas edges, Stage I, TF-only regulators, AEBP1/CENPA/HMGA1/YBX1 excluded).  Foreground = unique target genes of the list; background = all unique target genes of the dataset's
candidate pairs; terms with 10-500 genes in the background; one-sided hypergeometric, Benjamini-Hochberg per list.  Output: analysis_data/fm_meta_programs_comparison/go_{FM01,FM06}_{shared,unshared}_top1000.csv   usage: go_shared_unshared_enrichment.py"""
import re, numpy as np, pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests
ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"; A = f"{ROOT}/analysis_data"; N = 1000
lam = set(l.strip() for l in open(f"{ROOT}/real_data/humanTFs/TF_names_v_1.01.txt") if l.strip()); four = {"AEBP1", "CENPA", "HMGA1", "YBX1"}
gmt = {l.split("\t")[0]: set(x for x in l.rstrip("\n").split("\t")[2:] if x) for l in open(f"{A}/gene_set_libraries/GO_Biological_Process_2023.gmt")}
for ds in ("FM01", "FM06"):
    P = pd.read_csv(f"{A}/{ds.lower()}/data/twinscore_pdf_candidates/pdf_procedure_all_candidate_pairs.csv", usecols=["TF", "gene", "h", "z_U", "stage1"]); P = P[P.stage1]; P = P[~(P.TF.isin(four) | ~P.TF.isin(lam))]
    bg = set(P.gene); terms = {t: g & bg for t, g in gmt.items()}; terms = {t: g for t, g in terms.items() if 10 <= len(g) <= 500}
    for k, col in (("unshared", "z_U"), ("shared", "h")):
        fg = set(P.assign(s=P[col].abs()).nlargest(N, "s").gene); rows = []
        for t, g in terms.items():
            x = len(fg & g)
            if x >= 3: rows.append((t, x, len(g), x / len(fg) / (len(g) / len(bg)), hypergeom.sf(x - 1, len(bg), len(g), len(fg)), ",".join(sorted(fg & g)[:12])))
        R = pd.DataFrame(rows, columns=["term", "k", "term_size_bg", "fold", "p", "example_genes"]); R["q"] = multipletests(R.p, method="fdr_bh")[1]; R = R.sort_values("p")
        R.to_csv(f"{A}/fm_meta_programs_comparison/go_{ds}_{k}_top1000.csv", index=False); print(ds, k, "targets", len(fg), "bg", len(bg), "terms q<0.05:", int((R.q < 0.05).sum()))
