#!/usr/bin/env python3
"""Mouse version of annotate_bins_and_tf_targets.py for LARRY (d2, d4, d6) and CellTag3 (t2): every gene set is MOUSE-native, no human symbols or orthologs.
Gene sets: MSigDB mouse Hallmark (mh.all.v2024.1.Mm) and MSigDB mouse Reactome (m2.cp.reactome.v2024.1.Mm), both in analysis_data/gene_set_libraries_mouse/ (downloaded from data.broadinstitute.org, text gmt); GO Biological Process from the MGI annotations
(real_data/GO_annotation/mgi.gaf.gz, go-basic.obo; NOT-qualified lines dropped, annotations propagated over is_a/part_of, MGI primary symbols).  The Stage I tables carry upper-cased symbols; they are mapped back to MGI symbols through the symbols of the gene sets themselves (unmapped genes are counted and dropped).
Background (per library, as g:Profiler's 'annotated genes only') = the dataset's mapped Stage I targets that have >= 1 annotation in that library; sets with 5-500 background genes; k >= 2; one-sided hypergeometric; BH over all tests of one level (bins / TFs) and library.
Bins = regions of the three-list top-100 Venn (venn_top100_proportional/<ds>_venn_top100_regions.csv); TFs = TFs with >= MINT distinct targets in the union of the top-1000 |z_U| and top-1000 |h| pairs (Stage I pairs, AEBP1/CENPA/HMGA1/YBX1 excluded).
Output: analysis_data/fm_meta_programs_comparison/annotation_mouse_<ds>_{bins,tfs}.csv, annotation_mouse_summary.csv, annotation_mouse_mapping.csv   usage: annotate_bins_and_tf_targets_mouse.py [LARRY_d2 LARRY_d4 LARRY_d6 CellTag3_t2]"""
import sys, gzip, re, numpy as np, pandas as pd
from collections import defaultdict
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests
ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"; A = f"{ROOT}/analysis_data"; O = f"{A}/fm_meta_programs_comparison"; GL = f"{A}/gene_set_libraries_mouse"; GO = f"{ROOT}/real_data/GO_annotation"; N = 1000; MINT = 15
DS = {"LARRY_d2": "larry_d2", "LARRY_d4": "larry_d4", "LARRY_d6": "larry_d6", "CellTag3_t2": "celltag3_t2"}; DSL = sys.argv[1:] or list(DS); four = {"AEBP1", "CENPA", "HMGA1", "YBX1"}
def gmt(p, strip): return {re.sub(strip, "", l.split("\t")[0]).replace("_", " ").title() if strip else l.split("\t")[0]: set(x for x in l.rstrip("\n").split("\t")[2:] if x) for l in open(p) if l.strip()}
lib = {"Hallmark": gmt(f"{GL}/mh.all.v2024.1.Mm.symbols.gmt", r"^HALLMARK_"), "Reactome": gmt(f"{GL}/m2.cp.reactome.v2024.1.Mm.symbols.gmt", r"^REACTOME_")}
parents = defaultdict(set); name = {}; ns = {}; cur = None
for line in open(f"{GO}/go-basic.obo"):
    line = line.strip()
    if line == "[Term]": cur = {}
    elif line == "[Typedef]": cur = None
    elif cur is not None:
        if line.startswith("id: GO:"): cur["id"] = line[4:]
        elif line.startswith("name: "): name[cur["id"]] = line[6:]
        elif line.startswith("namespace: "): ns[cur["id"]] = line[11:]
        elif line.startswith("is_a: "): parents[cur["id"]].add(line[6:16])
        elif line.startswith("relationship: part_of "): parents[cur["id"]].add(line[22:32])
        elif line.startswith("is_obsolete: true"): ns[cur["id"]] = "obsolete"
anc = {}
def ancestors(t):
    if t in anc: return anc[t]
    s = {t}
    for p in parents.get(t, ()): s |= ancestors(p)
    anc[t] = s; return s
t2g = defaultdict(set); gaf_syms = set()
with gzip.open(f"{GO}/mgi.gaf.gz", "rt") as f:
    for line in f:
        if line[0] == "!": continue
        p = line.rstrip("\n").split("\t")
        if p[8] != "P" or "NOT" in p[3] or ns.get(p[4]) != "biological_process" or p[11] != "protein" and p[11] not in ("gene", "protein", "ncRNA", "lncRNA", "miRNA", "transcript", "RNA"): continue
        gaf_syms.add(p[2])
        for t in ancestors(p[4]):
            if ns.get(t) == "biological_process": t2g[t].add(p[2])
lib["GO_BP"] = {f"{name[t]} ({t})": g for t, g in t2g.items()}
canon = {}
for L in lib.values():
    for g in set().union(*L.values()): canon.setdefault(g.upper(), g)
for g in gaf_syms: canon.setdefault(g.upper(), g)
print({k: len(v) for k, v in lib.items()}, "canonical mouse symbols", len(canon), flush=True)
def enrich(lists, bg, level):
    out = []
    for Ln, sets in lib.items():
        ann = set().union(*sets.values()); bgL = bg & ann; ts = {t: g & bgL for t, g in sets.items()}; ts = {t: g for t, g in ts.items() if 5 <= len(g) <= 500}; rows = []
        for nm, fg in lists.items():
            fgL = fg & bgL
            if len(fgL) < 3: continue
            for t, g in ts.items():
                x = len(fgL & g)
                if x >= 2: rows.append((level, nm, Ln, t, x, len(fgL), len(g), x / len(fgL) / (len(g) / len(bgL)), hypergeom.sf(x - 1, len(bgL), len(g), len(fgL)), ",".join(sorted(fgL & g)[:10])))
        if rows: R = pd.DataFrame(rows, columns=["level", "name", "library", "term", "k", "n_list", "term_size_bg", "fold", "p", "genes"]); R["q"] = multipletests(R.p, method="fdr_bh")[1]; out.append(R)
    R = pd.concat(out); return R[R.q < 0.1].sort_values(["name", "q"])
summ, mapr = [], []
for ds in DSL:
    P = pd.read_csv(f"{A}/{DS[ds]}/data/stage1_tf_ge20/part_all/pdf_procedure_all_candidate_pairs.csv", usecols=["TF", "gene", "h", "z_U", "stage1"], dtype={"TF": "category", "gene": "category", "h": "float32", "z_U": "float32"}); P = P[P.stage1].copy()
    P["TF"] = P.TF.astype(str).str.upper(); P["gene"] = P.gene.astype(str); P = P[~P.TF.isin(four)]; raw = set(P.gene); P["gene"] = P.gene.str.upper().map(canon); mapped = set(P.gene.dropna())
    mapr.append(dict(dataset=ds, targets=len(raw), mapped_to_mouse_symbol=len(mapped), fraction=round(len(mapped) / len(raw), 3))); P = P.dropna(subset=["gene"]).reset_index(drop=True); bg = set(P.gene)
    V = pd.read_csv(f"{O}/venn_top100_proportional/{DS[ds]}_venn_top100_regions.csv", dtype={"region": str}); V["gene"] = V.gene.str.upper().map(canon); V = V.dropna(subset=["gene"]); bins = {f"{rn} (n={len(g)})": set(g.gene) for (rg, rn), g in V.groupby(["region", "region_name"])}
    u = P.assign(s=P.z_U.abs()).nlargest(N, "s"); s = P.assign(s=P.h.abs()).nlargest(N, "s"); D = P.loc[sorted(set(u.index) | set(s.index))]; tfl = {t: set(g.gene) for t, g in D.groupby("TF") if g.gene.nunique() >= MINT}
    for level, lists in (("bins", bins), ("tfs", tfl)):
        R = enrich(lists, bg, level); R.to_csv(f"{O}/annotation_mouse_{ds}_{level}.csv", index=False)
        for nm in lists:
            r = R[R.name == nm]; sg = r[r.q < 0.05]; best = r.iloc[0] if len(r) else None
            summ.append(dict(dataset=ds, level=level, name=nm, n_genes=len(lists[nm] & bg), n_terms_q05=len(sg), top_term=(f"{best.term} [{best.library}] k={best.k} fold={best.fold:.1f} q={best.q:.3g}" if best is not None else "")))
    print(ds, mapr[-1], flush=True)
pd.DataFrame(mapr).to_csv(f"{O}/annotation_mouse_mapping.csv", index=False); S = pd.DataFrame(summ); S.to_csv(f"{O}/annotation_mouse_summary.csv", index=False); pd.set_option("display.width", 250, "display.max_colwidth", 110); print(S.to_string(index=False))
