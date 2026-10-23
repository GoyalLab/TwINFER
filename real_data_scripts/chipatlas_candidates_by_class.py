#!/usr/bin/env python3
"""ChIP-Atlas TF x target candidate pairs for one dataset, same recipe as FM01's list (fm01_candidates_chipatlas_epidermis_176512.tsv):
  per TF: ChIP-Atlas per-TF target file (<genome>, +-5 kb of the TSS), keep the experiments of the chosen cell-type class (+ cell types), a gene is a target if its score
  in ANY of those experiments is >= 100 (reproduces FM01: AHR 31/31 targets, MITF 9045 vs 9047, JUN 7942 vs 7946 on the FM01-measured genes), and only genes measured in the dataset are kept.
Experiment -> class / cell type comes from ChIP-Atlas's experimentList.tab (columns: 1 id, 2 genome, 3 track class, 4 antigen, 5 cell type class, 6 cell type).
TFs  = union of the tf_list.csv of the given units (TFs detected in >= 20 % of clonal cells); genes = union of the targets of those units' candidates.tsv (detected in >= 5 %).
Mouse (mm10): TFs / genes are upper-cased to match the (upper-cased) Stage I tables.
Output: analysis_data/chipatlas_<tag>/{chip_candidates.tsv (TF<TAB>gene, upper-case), chipatlas_raw.json (per-TF cache), selection.json}
usage: chipatlas_candidates_by_class.py --tag T --genome hg38|mm10 --cls CLASS --celltypes "a;b;c" --units u1,u2 [--min-score 100] [--max-tfs N] [--meta path/to/experimentList_subset.tsv]"""
import argparse, io, json, os, time
import pandas as pd, requests

ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True); ap.add_argument("--genome", required=True); ap.add_argument("--cls", required=True); ap.add_argument("--celltypes", required=True)
ap.add_argument("--units", required=True); ap.add_argument("--min-score", type=float, default=100.0); ap.add_argument("--max-tfs", type=int, default=None)
ap.add_argument("--meta", default=f"{ROOT}/analysis_data/chipatlas_metadata/experimentList_hg38_mm10_TFs.tsv"); ap.add_argument("--out", default=None)
a = ap.parse_args()
OUT = a.out or f"{ROOT}/analysis_data/chipatlas_{a.tag}"; os.makedirs(OUT, exist_ok=True)
URL = "https://chip-atlas.dbcls.jp/data/{g}/target/{tf}.5.tsv"
em = pd.read_csv(a.meta, sep="\t", header=None, names=["id", "genome", "track", "antigen", "cls", "ctype"], dtype=str)
types = [t.strip() for t in a.celltypes.split(";") if t.strip()]
sel = em[(em.genome == a.genome) & (em.cls == a.cls) & (em.ctype.isin(types))]
missing = [t for t in types if t not in set(sel.ctype)]
print(f"{len(sel)} {a.genome} TF experiments in class '{a.cls}' x {len(types)} cell types; cell types with none: {missing}", flush=True)
by_tf = {}
for r in sel.itertuples(): by_tf.setdefault(r.antigen.upper(), set()).add(r.id)
antigen_name = sel.drop_duplicates("antigen").assign(u=lambda d: d.antigen.str.upper()).set_index("u").antigen.to_dict()
units = a.units.split(","); tfs, genes = set(), set()
for u in units:
    d = f"{ROOT}/analysis_data/{u}/data/stage1_tf_ge20"
    tfs |= set(pd.read_csv(f"{d}/tf_list.csv").TF.astype(str).str.upper()); genes |= set(pd.read_csv(f"{d}/candidates.tsv", sep="\t", header=None, usecols=[1])[1].astype(str).str.upper())
tfs = sorted(t for t in tfs if t in by_tf)
print(f"units {units}: {len(genes):,} measured target genes; TFs with >=1 selected experiment: {len(tfs)}", flush=True)
if a.max_tfs: tfs = tfs[:a.max_tfs]
cp = f"{OUT}/chipatlas_raw.json"; cache = json.load(open(cp)) if os.path.exists(cp) else {}
rows = []
for i, tf in enumerate(tfs):
    if tf not in cache:
        try:
            t = requests.get(URL.format(g=a.genome, tf=antigen_name[tf]), timeout=120); t.raise_for_status(); df = pd.read_csv(io.StringIO(t.text), sep="\t")
            cols = [c for c in df.columns[1:] if c.split("|")[0] in by_tf[tf]]
            if not cols: cache[tf] = {"status": "no_selected_experiments_in_target_file"}
            else:
                sc = df[cols].apply(pd.to_numeric, errors="coerce").max(axis=1); g_ = df.iloc[:, 0].astype(str).str.upper()
                cache[tf] = {"status": "ok", "n_experiments": len(cols), "bound_genes": sorted(set(g_[sc >= a.min_score]))}
        except Exception as e:
            cache[tf] = {"status": "failed", "error": str(e)}
        json.dump(cache, open(cp, "w")); time.sleep(0.3)
    e = cache[tf]
    if e["status"] == "ok":
        b = [g for g in e["bound_genes"] if g in genes and g != tf]; rows += [(tf, g) for g in b]
        print(f"[{i+1}/{len(tfs)}] {tf}: {e['n_experiments']} experiments, {len(b)} measured targets (score>={a.min_score:g})", flush=True)
    else: print(f"[{i+1}/{len(tfs)}] {tf}: {e['status']} {e.get('error','')[:80]}", flush=True)
C = pd.DataFrame(rows, columns=["TF", "gene"]).drop_duplicates()
C.to_csv(f"{OUT}/chip_candidates.tsv", sep="\t", header=False, index=False)
json.dump({"genome": a.genome, "class": a.cls, "cell_types": types, "n_experiments": int(len(sel)), "min_score": a.min_score, "units": units, "n_tfs_with_pairs": int(C.TF.nunique()), "n_pairs": int(len(C)),
           "tfs_without_selected_experiments": sorted(set(pd.read_csv(f"{ROOT}/analysis_data/{units[0]}/data/stage1_tf_ge20/tf_list.csv").TF.astype(str).str.upper()) - set(by_tf))[:50]}, open(f"{OUT}/selection.json", "w"), indent=1)
print(f"{OUT}/chip_candidates.tsv: {len(C):,} pairs, {C.TF.nunique()} TFs, {C.gene.nunique():,} targets", flush=True)
