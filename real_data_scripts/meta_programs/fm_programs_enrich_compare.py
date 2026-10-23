#!/usr/bin/env python3
"""Enrichment labels for the robust NMF programs of every FatemapFM dataset, and a gene-list comparison across datasets.
Input : analysis_data/<fm>/data/mp_nmf/robust_programs.csv (top-50 genes per robust program, from fm01_meta_programs.py)
Step 1: per program, hypergeometric enrichment of the 50 genes against MSigDB Hallmark 2020, GO BP 2023, Reactome 2024, KEGG 2021
        (universe = genes detected in >= 5 % of the dataset's cells; terms restricted to the universe, 5 <= size <= 500;
        BH within each library). Flags: ribosomal (genes matching ^RP[LS]\\d), mitochondrial-encoded (MT-).
        Label = ribosomal if >= 15 ribosomal genes; mitochondrial if >= 8 MT- genes; else the top Hallmark term with q <= 0.05,
        else the top GO BP term with q <= 0.05, else 'no term q<=0.05'.
Step 2: the paper's MP clustering (fm01_meta_programs.make_mps) applied to ALL robust programs of all datasets as gene lists only
        (no expression data is combined). Program samples are named <dataset>_<replicate>.
Outputs under analysis_data/fm_meta_programs_comparison/.
"""
import glob, os, re, sys
import numpy as np, pandas as pd, scipy.io as sio
from scipy.stats import hypergeom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"; OUT = f"{ROOT}/analysis_data/fm_meta_programs_comparison"; os.makedirs(OUT, exist_ok=True)
DSS = ["FM01", "FM03", "FM05", "FM06", "FM07", "FM08"]
LIBS = {"Hallmark": "MSigDB_Hallmark_2020", "GO_BP": "GO_Biological_Process_2023", "Reactome": "Reactome_Pathways_2024", "KEGG": "KEGG_2021_Human"}


def gmt(n):
    d = {}
    for l in open(f"{ROOT}/analysis_data/gene_set_libraries/{n}.gmt"):
        p = l.rstrip("\n").split("\t"); d[p[0]] = set(g for g in p[2:] if g)
    return d


def bh(p):
    p = np.asarray(p); o = np.argsort(p); q = p[o] * len(p) / (np.arange(len(p)) + 1); q = np.minimum.accumulate(q[::-1])[::-1]; r = np.empty_like(q); r[o] = np.minimum(q, 1); return r


def universe(ds):
    f = f"{OUT}/universe_{ds}.txt"
    if os.path.exists(f): return set(l.strip() for l in open(f))
    Q = f"{ROOT}/finalized_data/{ds}_data/qc_filtered"
    genes = np.array([l.strip() for l in open(f"{Q}/genes.txt")]); X = sio.mmread(f"{Q}/{ds.lower()}_qc_counts.mtx").tocsc()
    U = genes[np.asarray((X > 0).mean(0)).ravel() >= 0.05]; open(f, "w").write("\n".join(U) + "\n"); return set(U)


L = {k: gmt(v) for k, v in LIBS.items()}
rows = []; allprog = []
for ds in DSS:
    U = universe(ds); N = len(U)
    r = pd.read_csv(f"{ROOT}/analysis_data/{ds.lower()}/data/mp_nmf/robust_programs.csv")
    LU = {k: {t: g & U for t, g in lib.items() if 5 <= len(g & U) <= 500} for k, lib in L.items()}
    for x in r.itertuples():
        genes = x.genes.split(); gs = set(genes) & U; n = len(gs)
        rec = dict(dataset=ds, replicate=x.sample, K=x.K, idx=x.idx, best_overlap_other_replicate=x.best_overlap_other_sample if hasattr(x, "best_overlap_other_sample") else np.nan,
                   n_ribosomal=sum(bool(re.match(r"RP[LS]\d", g)) for g in genes), n_mito=sum(g.startswith("MT-") for g in genes), top10=" ".join(genes[:10]))
        top = {}
        for k, lib in LU.items():
            terms = list(lib); p = [hypergeom.sf(len(gs & lib[t]) - 1, N, len(lib[t]), n) for t in terms]; q = bh(p); j = int(np.argmin(p))
            top[k] = (terms[j], q[j], len(gs & lib[terms[j]]))
            rec[f"{k}_top"] = terms[j]; rec[f"{k}_q"] = q[j]; rec[f"{k}_overlap"] = len(gs & lib[terms[j]])
        if rec["n_ribosomal"] >= 15: lab = "ribosomal"
        elif rec["n_mito"] >= 8: lab = "mitochondrial-encoded"
        elif top["Hallmark"][1] <= 0.05: lab = "Hallmark: " + top["Hallmark"][0]
        elif top["GO_BP"][1] <= 0.05: lab = "GO: " + top["GO_BP"][0]
        else: lab = "no term q<=0.05"
        rec["label"] = lab; rows.append(rec)
        allprog.append(dict(sample=f"{ds}_{x.sample}", K=int(x.K), idx=int(x.idx), genes=genes, score={g: 50 - i for i, g in enumerate(genes)}, ds=ds))
E = pd.DataFrame(rows); E.to_csv(f"{OUT}/program_enrichment_labels.csv", index=False)
print(E[["dataset", "replicate", "K", "idx", "n_ribosomal", "n_mito", "label", "Hallmark_q", "GO_BP_q"]].round(4).to_string())

import fm01_meta_programs as M
mps = M.make_mps(allprog)
out = []
for i, m in enumerate(mps):
    mem = m["members"]; ds = sorted({s.split("_")[0] for s, _, _ in mem})
    out.append(dict(MP=i + 1, n_programs=len(mem), n_datasets=len(ds), datasets=",".join(ds), members=";".join(f"{s}_K{k}_{j}" for s, k, j in mem), top20=" ".join(m["genes"][:20]), genes=" ".join(m["genes"])))
pd.DataFrame(out).to_csv(f"{OUT}/cross_dataset_mps.csv", index=False)
print("\ncross-dataset clusters:", len(out))
for o in out: print(o["MP"], o["n_programs"], "programs;", o["n_datasets"], "datasets", o["datasets"], "|", o["top20"])
