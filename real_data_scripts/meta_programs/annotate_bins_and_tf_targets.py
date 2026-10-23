#!/usr/bin/env python3
"""Automated annotation of gene lists, run LOCALLY (nothing uploaded): over-representation of (a) each BIN = each region of the FM01/FM06 three-list top-100 Venn (venn_top100_proportional/<ds>_venn_top100_regions.csv, target genes of its pairs) and (b) each TF's targets
(TFs with >= MINT distinct targets in the union of the top-1000 unshared and top-1000 shared pairs; ChIP-Atlas candidate edges, Stage I, TF-only regulators, as in program_11_hallmarks_vertical_plot.py) against MSigDB Hallmark 2020, Reactome 2024, KEGG 2021 and GO BP 2023
(Enrichr gmt files in analysis_data/gene_set_libraries) with a CUSTOM BACKGROUND = all unique targets of the dataset's candidate pairs.  Sets with 5-500 background genes; k >= 2; one-sided hypergeometric; BH over all tests of one level (bins or TFs) and library.
Output: analysis_data/fm_meta_programs_comparison/annotation_<ds>_{bins,tfs}.csv (all q < 0.1 hits) and annotation_summary.csv   usage: annotate_bins_and_tf_targets.py [FM01 FM06 LARRY_d2 LARRY_d4 LARRY_d6 CellTag3_t2 CellTag3_t4]  (default FM01 FM06).  LARRY/CellTag3: all-TF Stage I table (part_all, stage1 pairs; mouse symbols are already upper case), summary written to annotation_summary_<datasets>.csv"""
import sys
import re, numpy as np, pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests
ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"; A = f"{ROOT}/analysis_data"; O = f"{A}/fm_meta_programs_comparison"; N = 1000; MINT = 15
lam = set(l.strip() for l in open(f"{ROOT}/real_data/humanTFs/TF_names_v_1.01.txt") if l.strip()); four = {"AEBP1", "CENPA", "HMGA1", "YBX1"}
LIBS = {"Hallmark": "MSigDB_Hallmark_2020", "Reactome": "Reactome_Pathways_2024", "KEGG": "KEGG_2021_Human", "GO_BP": "GO_Biological_Process_2023"}
lib = {k: {re.sub(r" \((GO:\d+|R-HSA-\d+)\)", "", l.split("\t")[0]).strip(): set(x for x in l.rstrip("\n").split("\t")[2:] if x) for l in open(f"{A}/gene_set_libraries/{v}.gmt") if l.strip()} for k, v in LIBS.items()}
def enrich(lists, bg, level, ds):
    out = []
    for L, sets in lib.items():
        ts = {t: g & bg for t, g in sets.items()}; ts = {t: g for t, g in ts.items() if 5 <= len(g) <= 500}; rows = []
        for nm, fg in lists.items():
            fg = fg & bg
            for t, g in ts.items():
                x = len(fg & g)
                if x >= 2: rows.append((level, nm, L, t, x, len(fg), len(g), x / len(fg) / (len(g) / len(bg)), hypergeom.sf(x - 1, len(bg), len(g), len(fg)), ",".join(sorted(fg & g)[:10])))
        if rows: R = pd.DataFrame(rows, columns=["level", "name", "library", "term", "k", "n_list", "term_size_bg", "fold", "p", "genes"]); R["q"] = multipletests(R.p, method="fdr_bh")[1]; out.append(R)
    R = pd.concat(out); return R[R.q < 0.1].sort_values(["name", "q"])
summ = []; DSL = sys.argv[1:] or ["FM01", "FM06"]; MOUSE = {"LARRY_d2": "larry_d2", "LARRY_d4": "larry_d4", "LARRY_d6": "larry_d6", "CellTag3_t2": "celltag3_t2", "CellTag3_t4": "celltag3_t4"}
for ds in DSL:
    if ds in MOUSE: P = pd.read_csv(f"{A}/{MOUSE[ds]}/data/stage1_tf_ge20/part_all/pdf_procedure_all_candidate_pairs.csv", usecols=["TF", "gene", "h", "z_U", "stage1"], dtype={"TF": "category", "gene": "category", "h": "float32", "z_U": "float32"}); P = P[P.stage1]; P["TF"] = P.TF.astype(str); P["gene"] = P.gene.astype(str)
    else: P = pd.read_csv(f"{A}/{ds.lower()}/data/twinscore_pdf_candidates/pdf_procedure_all_candidate_pairs.csv", usecols=["TF", "gene", "h", "z_U", "stage1"]); P = P[P.stage1]
    P = P[~(P.TF.isin(four) | ~P.TF.isin(lam))].reset_index(drop=True); bg = set(P.gene)
    V = pd.read_csv(f"{O}/venn_top100_proportional/{(MOUSE[ds].replace("celltag3_t", "celltag3_t") if ds in MOUSE else ds.lower())}_venn_top100_regions.csv", dtype={"region": str}); bins = {f"{r.region_name} (n={len(g)} pairs)": set(g.gene) for (r), g in ((x, y) for x, y in V.groupby(["region", "region_name"]).__iter__() if False)} if False else {}
    for (rg, rn), g in V.groupby(["region", "region_name"]): bins[f"{rn} (n={len(g)})"] = set(g.gene)
    u = P.assign(s=P.z_U.abs()).nlargest(N, "s"); s = P.assign(s=P.h.abs()).nlargest(N, "s"); D = P.loc[sorted(set(u.index) | set(s.index))]; tfl = {t: set(g.gene) for t, g in D.groupby("TF") if g.gene.nunique() >= MINT}
    for level, lists in (("bins", bins), ("tfs", tfl)):
        R = enrich(lists, bg, level, ds); R.to_csv(f"{O}/annotation_{ds}_{level}.csv", index=False)
        for nm in lists:
            r = R[R.name == nm]; sg = r[r.q < 0.05]; best = r.iloc[0] if len(r) else None
            summ.append(dict(dataset=ds, level=level, name=nm, n_genes=len(lists[nm] & bg), n_terms_q05=len(sg), top_term=(f"{best.term} [{best.library}] k={best.k} fold={best.fold:.1f} q={best.q:.3g}" if best is not None else "")))
S = pd.DataFrame(summ); S.to_csv(f"{O}/annotation_summary" + ("" if DSL == ["FM01", "FM06"] else "_" + "_".join(DSL)) + ".csv", index=False); pd.set_option("display.width", 250, "display.max_colwidth", 110); print(S.to_string(index=False))
