#!/usr/bin/env python3
"""GO Biological Process 2023 (Enrichr gmt, analysis_data/gene_set_libraries) over-representation of the TARGET genes of the top-100 UNSHARED (|z_U|) and top-100 SHARED (|z_C| = |h|) pairs, for FM01, FM06, Watermelon naive (existing
top100_lists/ from build_top100_lists_alltf.py), LARRY day 2, CellTag3 day 2.5 (t2) and Michaels t1 (Stage I all-TF tables, stage1 pairs, Lambert-TF regulators only, AEBP1/CENPA/HMGA1/YBX1 excluded, no gene filter, same recipe as build_top100_lists_alltf.py;
mouse symbols are upper-cased to match the human gmt).  Foreground = unique targets of the list; background = all unique targets of the dataset's Stage I pairs; terms with 10-500 background genes; k >= 2; one-sided hypergeometric; BH per list.
DTT (one cell type, Telencephalon) is NOT recomputed here: its own local MGI GO result (DTT/results/go_enrichment_top100_genes.csv, rankings z_C = shared, z_U = unshared, list 'all' = genes of both ends of each top-100 pair, universe = the type's panel genes) is read by go_dotplot.py.
Output: analysis_data/fm_meta_programs_comparison/go_top100/<dataset>_{shared,unshared}.csv   usage: go_top100_all_datasets.py"""
import os, numpy as np, pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests
ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"; A = f"{ROOT}/analysis_data"; T100 = f"{A}/fm_meta_programs_comparison/top100_lists"; OUT = f"{A}/fm_meta_programs_comparison/go_top100"; os.makedirs(OUT, exist_ok=True)
lam = set(l.strip() for l in open(f"{ROOT}/real_data/humanTFs/TF_names_v_1.01.txt") if l.strip()); four = {"AEBP1", "CENPA", "HMGA1", "YBX1"}
gmt = {l.split("\t")[0]: set(x for x in l.rstrip("\n").split("\t")[2:] if x) for l in open(f"{A}/gene_set_libraries/GO_Biological_Process_2023.gmt")}
LISTS = {}
for ds in ("FM01", "FM06", "Watermelon_naive"):
    LISTS[ds] = ({k: set(pd.read_csv(f"{T100}/{ds}_{k}.csv").gene) for k in ("unshared", "shared")}, set(open(f"{T100}/{ds}_stage1_targets.txt").read().split()))
for ds, d in (("LARRY_d2", "larry_d2"), ("CellTag3_d2.5", "celltag3_t2"), ("Michaels_t1", "michaels_t1")):
    P = pd.read_csv(f"{A}/{d}/data/stage1_tf_ge20/part_all/pdf_procedure_all_candidate_pairs.csv", usecols=["TF", "gene", "h", "z_U", "stage1"], dtype={"TF": "category", "gene": "category"}); P = P[P.stage1].copy(); P["TF"] = P.TF.astype(str).str.upper(); P["gene"] = P.gene.astype(str).str.upper()
    bg = set(P.gene); Q = P[~(P.TF.isin(four) | ~P.TF.isin(lam))]
    LISTS[ds] = ({k: set(Q.reindex(Q[c].abs().sort_values(ascending=False).index).head(100).gene) for k, c in (("unshared", "z_U"), ("shared", "h"))}, bg); print(ds, f"stage1 pairs {len(P):,}; TF pairs {len(Q):,}; bg targets {len(bg):,}", flush=True)
for ds, (fgs, bg) in LISTS.items():
    terms = {t: g & bg for t, g in gmt.items()}; terms = {t: g for t, g in terms.items() if 10 <= len(g) <= 500}
    for k, fg in fgs.items():
        fg = fg & bg; rows = []
        for t, g in terms.items():
            x = len(fg & g)
            if x >= 2: rows.append((t, x, len(g), x / len(fg) / (len(g) / len(bg)), hypergeom.sf(x - 1, len(bg), len(g), len(fg)), ",".join(sorted(fg & g)[:12])))
        R = pd.DataFrame(rows, columns=["term", "k", "term_size_bg", "fold", "p", "genes"]); R["q"] = multipletests(R.p, method="fdr_bh")[1]; R = R.sort_values("p"); R.to_csv(f"{OUT}/{ds}_{k}.csv", index=False)
        print(ds, k, "targets", len(fg), "bg", len(bg), "q<0.05:", int((R.q < .05).sum()), "q<0.1:", int((R.q < .1).sum()), "p<0.001:", int((R.p < .001).sum()), flush=True)
