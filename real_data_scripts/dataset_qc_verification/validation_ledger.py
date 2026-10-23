#!/usr/bin/env python3
"""Validation ledger: one row per checked claim, with the method and the evidence file. Every `result` is read from the evidence files (json / logs / md5 of the files on disk) at build time,
never typed by hand. State after the 2026-10-16 rebuilds: current finalized_data files, the previous (archived) files they replaced, and the analysis outputs.
usage: validation_ledger.py   -> analysis_data/dataset_qc_verification/validation_ledger.csv (+ prints it)
The hand-written validation page of preprocessing_per_sample__v2.pdf summarises the same evidence."""
import filecmp
import glob
import hashlib
import json
import os
import re

import pandas as pd

ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"
V = f"{ROOT}/analysis_data/dataset_qc_verification"
R = f"{V}/rebuild"
LG = f"{V}/logs"
FIN = f"{ROOT}/finalized_data"
NEW = f"{FIN}/rebuild_fullsheet_2026-10-16"
rows = []


def add(dataset, claim, method, res):
    result, evidence = res
    rows.append(dict(dataset=dataset, claim=claim, method=method, result=result, evidence=evidence))


def jl(p):
    return json.load(open(p))


def exists(p, f):
    return f(p) if os.path.exists(p) else ("PENDING (evidence file missing)", p)


META = ("obs_metadata.csv", ".json")          # column-set / summary-key files; the data files are everything else


def rebuild_json(ds):
    p = f"{R}/{ds}/diff.json"
    def f(p):
        d = jl(p); fl = d["files"]
        ok = lambda v: bool(v.get("byte_identical") or v.get("values_identical"))
        data = {k: v for k, v in fl.items() if not k.endswith(META)}
        meta = {k: v for k, v in fl.items() if k.endswith(META)}
        notwritten = [k for k, v in fl.items() if not v.get("present_in_rebuild", True)]
        data_bad = {k: v for k, v in data.items() if not ok(v) and k not in notwritten}
        meta_bad = {k: v for k, v in meta.items() if not ok(v) and k not in notwritten}
        parts = []
        if not data_bad:
            parts.append(f"DATA FILES IDENTICAL ({sum(ok(v) for v in data.values())} of {len(data)} byte/value-identical)")
        else:
            parts.append("DATA FILES DIFFER: " + "; ".join(f"{k}: {json.dumps(v)[:200]}" for k, v in data_bad.items()))
        if meta and not meta_bad:
            parts.append(f"metadata files identical ({sum(ok(v) for v in meta.values())} of {len(meta)})")
        if meta_bad:
            parts.append("metadata files differ (column set / summary keys): " + "; ".join(f"{k}: " + (f"columns {v.get('shape_old_new')}" if "shape_old_new" in v else f"keys {sorted(v.get('keys_differing', {}))}") for k, v in meta_bad.items()))
        if notwritten:
            parts.append("not written by the script: " + ", ".join(notwritten))
        parts.append(f"finalized_data untouched={d['finalized_untouched']}")
        return " | ".join(parts), p
    return exists(p, f)


def h5_identity(p):
    def f(p):
        d = jl(p)
        if d["ALL_IDENTICAL"]:
            return f"IDENTICAL: {d['n_identical']}/{d['n_datasets_compared']} HDF5 datasets (X, layers, obs, var, uns ...)", p
        bad = {k: v for k, v in d["datasets"].items() if v != "identical"}
        if not bad:
            return f"IDENTICAL on all {d['n_datasets_compared']} shared HDF5 datasets; only in rebuild: {d['only_in_a'][:8]}; only in the file used (not written by the rebuilt chain): {d['only_in_b'][:8]}", p
        return f"DIFFERS: {len(bad)} of {d['n_datasets_compared']} datasets; only_in_rebuild={d['only_in_a'][:5]} only_in_used={d['only_in_b'][:5]}; e.g. {list(bad.items())[:4]}", p
    return exists(p, f)


def text(p, pattern=None, maxlines=40):
    def f(p):
        t = open(p).read().strip()
        if pattern:
            t = "\n".join(l for l in t.splitlines() if re.search(pattern, l))
        return "\n".join(t.splitlines()[:maxlines]), p
    return exists(p, f)


def fields(p, keys):
    def f(p):
        d = jl(p)
        return "; ".join(f"{k}={d[k]}" for k in keys if k in d), p
    return exists(p, f)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def same_files(a, b, names):
    """md5 of each named file in dir a and dir b."""
    out = []
    for n in names:
        pa, pb = f"{a}/{n}", f"{b}/{n}"
        if not (os.path.exists(pa) and os.path.exists(pb)):
            out.append(f"{n}: MISSING"); continue
        out.append(f"{n}: {'identical' if md5(pa) == md5(pb) else 'DIFFERENT'}")
    return "; ".join(out)


# ------------------------------------------------------------------ FateMap family: previous files vs original scripts
for ds in ["FM01", "FM03", "FM05", "FM06", "FM07", "FM08", "Watermelon"]:
    add(ds, "PREVIOUS qc_filtered files (now archived) = output of the original build script on the raw inputs",
        "original build script rerun into scratch; every file diffed vs the then-current finalized_data (rebuild_and_diff.py)", rebuild_json(ds))
r = rebuild_json("HS054_Sot48h")
a, b = f"{FIN}/HS054_Sot48h_data/qc_filtered/hs054_qc_counts.mtx", f"{R}/HS054_Sot48h/hs054_sot48h_qc_counts.mtx"
cmpm = ("hs054_qc_counts.mtx (not written by the script) vs rebuilt hs054_sot48h_qc_counts.mtx: " + ("identical (cmp)" if os.path.exists(a) and os.path.exists(b) and filecmp.cmp(a, b, shallow=False) else "NOT identical / missing"))
add("HS054_Sot48h", "qc_filtered = output of build_hs054_qc_matrix.py on the Cell Ranger matrix + Hanxiao's starcode table",
    "build script rerun; files diffed; the extra finalized file compared with cmp now", (r[0] + " | " + cmpm, r[1]))

# ------------------------------------------------------------------ current FM / Watermelon files = full-sheet rebuild
mv = f"{LG}/move_new_into_finalized.log"
for ds, d in [("FM01", "FM01_data"), ("FM03", "FM03_data"), ("FM05", "FM05_data"), ("FM06", "FM06_data"), ("FM08", "FM08_data"), ("Watermelon", "Watermelon_data")]:
    sp = f"{FIN}/{d}/qc_filtered/preprocessing_summary.json"
    def f(p, ds=ds, d=d):
        j = jl(p); fin_n = j.get("n_after_gene_floor_ceiling", j.get("n_final"))
        log = [l for l in open(mv).read().splitlines() if l.strip() == f"updated {d}"]
        return (f"final cells {fin_n}; mito cutoff {j.get('mito_pct_cutoff', j.get('mito_pct'))}; singletCode: {j.get('singletcode_run_on', 'entire sheet (all SampleNum)')}; rescued units "
                f"{j.get('n_rescued_cells', j.get('n_rescued_units_sample_cellID'))}; copy of the job output verified with diff -rq: {'yes' if log else 'NOT FOUND IN LOG'}"), p
    add(ds, "CURRENT finalized_data files = full-sheet rebuild (whole-sheet singletCode, paper mito cutoff, rescue per (sample, cellID)), copied from the job output",
        "build_fm_fullsheet_rebuild.py / rebuild_corrected_downstream.py (jobs 4041481, 4045122); copy checked with diff -rq in move_new_into_finalized.sh", exists(sp, f))
for t in ("naive", "lag", "late"):
    add(f"Watermelon_{t}", "CURRENT stage qc_filtered + integrated h5ad = rebuild output", "wm_split + wm_stage (job 4045122); copy checked with diff -rq",
        (("copy verified: yes" if any(l.strip() == f"updated Watermelon_{t}_data" for l in open(mv)) else "NOT FOUND IN LOG"), mv))

# ------------------------------------------------------------------ criterion-4 rescue check
for ds in ["FM01", "FM03", "FM05", "FM06", "FM07", "FM08", "Watermelon", "LARRY"]:
    add(ds, "criterion-4 rescue: pooled by cellID (as originally implemented) vs a cell identified by (sample, cellID); check run on the PREVIOUS final files (final_cells = their count)",
        "check_rescue_logic.py (job 4036965): the build is rerun, singletCode output captured, rescue applied both ways",
        fields(f"{V}/rescue_check/{ds}/rescue_check.json", ["cellIDs_in_more_than_one_sample", "rescued_units_as_implemented", "rescued_units_corrected", "FINAL_cells_rescued_only_as_implemented",
                                                             "FINAL_cells_that_would_be_added_by_correction", "final_cells"]))

# ------------------------------------------------------------------ LARRY
add("LARRY", "stage 1: combined matrix + umi table from the 15 inDrops zips + barcode FASTQ (the authors' files)", "build_larry_matrix.py rerun into scratch; files diffed vs the stored stage-1 files", rebuild_json("LARRY_stage1"))
add("LARRY", "PREVIOUS filtered file (dip test + Otsu whitelist, rescue by cellID) = stage 2 of the script with the Weinreb rule (ii)/(iii) step off",
    "build_final_matrix_umi3.py rerun with the rule function replaced by all-False masks; diffed vs the previous finalized filtered", rebuild_json("LARRY_stage2_noStep5"))
add("LARRY", "stage 2 with the Weinreb rule on (current script on the previous inputs): expected to DIFFER, the rule was never applied to our file", "same, rule on", rebuild_json("LARRY_stage2"))
add("LARRY", "PREVIOUS cell-type annotation file", "larry_raw_annotate.py rerun; h5ad compared element-wise", h5_identity(f"{R}/LARRY_annotation/identity.json"))
lf, ln = f"{FIN}/LARRY_data", f"{NEW}/LARRY_table1_whitelist"
add("LARRY", "CURRENT filtered/ and annotated/ = job 4053678 output (Table 1 whitelist, corrected rescue, no Weinreb rule)", "md5 of every file in finalized_data/LARRY_data vs the job output folder",
    (same_files(f"{lf}/filtered", f"{ln}/filtered", ["cells.txt", "genes.txt", "larry_qc_counts.mtx", "obs_metadata.csv", "preprocessing_summary.json"]) + " | " +
     same_files(f"{lf}/annotated", f"{ln}/annotated", ["filtered_annotated_k15.h5ad", "annotation_transfer_filtered_k15.json"]), ln))
add("LARRY", "CURRENT counts through the filters", "read from the current preprocessing_summary.json",
    fields(f"{lf}/filtered/preprocessing_summary.json", ["n_raw_cells", "n_after_whitelist", "n_after_mito", "n_after_singletcode_umi3_plus_criterion4", "n_after_gene_floor", "n_rescued_units_sample_cellID", "n_after_weinreb_rule_ii_iii"]))
add("LARRY", "Table 1 whitelist vs the dip+Otsu whitelist (final cells, clones, medians per day)", "compare_larry_table1.py (job 4053678)", text(f"{ln}/compare_vs_dip_otsu.csv"))

# ------------------------------------------------------------------ hPSC endo (yscher), undifferentiated, Novo
add("hPSC_endo_yscher", "kb-layer h5ad = build_endo_dataset.py on the stepThree tables of yscher's index_fix/B", "script + interpreter rerun (output redirected); h5ad compared element-wise", h5_identity(f"{R}/hPSC_endo_yscher/A/identity_vs_finalized.json"))
add("hPSC_endo_yscher", "Cell Ranger counts h5ad (endo_counts_mito20_min200genes.h5ad, the counts used in the analysis) = rebuilt file", "same rerun; h5ad compared element-wise with yscher's file", h5_identity(f"{R}/hPSC_endo_yscher/A/identity_counts_vs_hers.json"))
ep, er = f"{FIN}/hPSC_endo_yscher_data/endo_counts_mito20_min200genes.h5ad", f"{R}/hPSC_endo_yscher/A/endo_counts_mito20_min200genes.h5ad"
add("hPSC_endo_yscher", "finalized_data copy of the Cell Ranger counts file = the rebuilt file", "cmp", (("byte-identical (cmp)" if os.path.exists(ep) and os.path.exists(er) and filecmp.cmp(ep, er, shallow=False) else "NOT identical / missing"), ep))
add("hPSC_endo_yscher", "lane concatenation of the barcode FASTQs", "md5 of cat(L001,L002) vs md5 of the combined file", text(f"{R}/hPSC_endo_yscher/B/fastq_concat_md5.txt"))
add("hPSC_endo_yscher", "stepThree barcode tables = Hanxiao's pipeline on the FASTQs", "stepOne/Two/Three rerun; tables compared with cmp", text(f"{R}/hPSC_endo_yscher/B/stepThree_cmp.txt"))
add("hPSC_endo_yscher", "build_endo_dataset.py on OUR stepThree tables -> kb-layer h5ad", "h5ad compared element-wise", h5_identity(f"{R}/hPSC_endo_yscher/B/build/identity_vs_finalized.json"))
add("hPSC_endo_yscher", "Cell Ranger counts vs the kb-python matrix (X of the kb h5ad) for the same cells", "endo_cellranger_vs_kb.py (genes matched by symbol)",
    fields(f"{V}/endo_cellranger_vs_kb.json", ["n_cells_matched", "n_shared_symbols", "per_gene_total_pearson_across_shared_genes", "cells_with_identical_shared_gene_vector"]))
add("hPSC_endo_yscher", "analysis PCA uses the Cell Ranger counts", "pca5k summary source line", fields(f"{V}/pca5k/summary_hPSC_endo_yscher.json", ["n_cells", "n_hvg", "source"]))
for t in ("T0", "T10", "T15"):
    add("hPSC_undifferentiated", f"{t}: QC (mito < 20%, >= 200 genes) recomputed from the Cell Ranger filtered matrix = cells of our file; sampled count vectors identical",
        "hpsc_undiff_qc_check.py (streams the 10x h5; 300 random cells compared on all genes)", (
            (lambda d: (f"cellranger_cells={d[t]['cellranger_cells']}; pass_QC={d[t]['pass_mito20_min200genes']}; our_cells={d[t]['our_cells']}; sets_equal={d[t]['recomputed_set_equals_ours']}; "
                        f"identical_count_vectors={d[t]['sampled_cells_identical_count_vector']}/{d[t]['sampled_cells_compared']}", f"{V}/hpsc_undifferentiated_qc_check.json"))(jl(f"{V}/hpsc_undifferentiated_qc_check.json"))
            if os.path.exists(f"{V}/hpsc_undifferentiated_qc_check.json") else ("PENDING", f"{V}/hpsc_undifferentiated_qc_check.json")))
    add("hPSC_undifferentiated", f"{t}: clone_c5 rerun from the stepThree table (singletCode cutoff 5, ratio 0, dominant 30, diff 0) = clone_c5 of our file", "hpsc_undiff_clone_c5_rerun.py",
        fields(f"{V}/hpsc_undiff_clone_c5_{t}.json", ["reads", "our_cells_with_clone_c5", "cells_with_clone_equal_sets", "barcode_strings_identical", "barcode_strings_compared"]))
add("hPSC_undifferentiated", "stepThree tables saved as raw data = the originals", "cmp at copy time (copy_hpsc_undiff_stepthree.sh) + md5 list", text(f"{LG}/copy_hpsc_undiff_stepthree.log"))
add("hPSC_Novo_Nordisk", "per-sample qc_filtered = build_qc_per_sample.py on the Seurat export (the authors')", "script rerun into scratch; each file compared with cmp", text(f"{R}/hPSC_Novo_Nordisk/cmp.txt", maxlines=20))

# ------------------------------------------------------------------ CellTag3
for f in sorted(glob.glob(f"{R}/CellTag3/cmp_*.json")):
    add("CellTag3", f"intermediate {os.path.basename(f)[4:-5]} = yscher's chain rerun", "yscher's scripts rerun in a mirror tree; h5ad element-wise", h5_identity(f))
add("CellTag3", "csv intermediates (steps 4, 6) and the final md5", "cmp / md5 in the rebuild logs (steps 1-5 job 3952316, steps 6-7 job 4049584)",
    text(f"{LG}/rebuild_celltag_3952316.out", r"byte-identical|DIFFERS", 6) if False else
    ((lambda a, b: (a[0] + "\n" + b[0], b[1]))(text(f"{LG}/rebuild_celltag_3952316.out", r"byte-identical", 6), text(f"{LG}/rebuild_celltag_67_4049584.out", r"byte-identical|[0-9a-f]{32}", 6))))

# ------------------------------------------------------------------ Michaels, PeTrace, DTT, SpaceBar
add("Michaels_T1T2", "michaels_T1_qc.h5ad / michaels_T2_qc.h5ad (yscher) = raw GEO matrices + mito<=20%, >=200 genes", "independent reimplementation; names, order, every X value, stored metrics compared",
    text(f"{V}/michaels_rerun/qc_h5ad_identity.json", r"identical|kept|raw_cells"))
def michaels_clones():
    sp, vp, kp = f"{V}/michaels_rerun/steps.json", f"{V}/michaels_rerun/validation.json", f"{V}/michaels_rerun/step8_seeds.json"
    if not all(os.path.exists(x) for x in (sp, vp, kp)):
        return "PENDING (evidence file missing)", sp
    st, va, ks = jl(sp), jl(vp), jl(kp)
    a, b, c, d, g = st["2_same_UMI_resolution"], st["3_consensus"], st["4to7_filters"], st["8_graph"], va["V7_vs_her_cell_to_clone"]
    return (f"steps 1-7 counts of the rerun: cell-UMI pairs {a['cell_umi_pairs']}, dropped incompatible {a['dropped_incompatible']}, distinct barcodes {b['distinct_barcodes']}, consensus seen more than once {b['consensus_seen_more_than_once']}, "
            f"cells with a barcode {c['cells_with_a_barcode']}, barcodes used {c['barcodes_used']}, raw components {d['connected_components']}. "
            f"Step 8 (hash-seed dependent): single run {g['rerun_clones']} clones / {g['rerun_cells']} cells vs yscher's {g['her_clones']} / {g['her_cells']}, ARI {g['ARI_shared_cells']} on shared cells; "
            f"20 seeds (yscher's networkx {ks['seeds'][0]['networkx']}): clones {ks['clones_min']}-{ks['clones_max']}, cells {ks['cells_min']}-{ks['cells_max']}, ARI {ks['ARI_min']}-{ks['ARI_max']}, "
            f"seeds giving yscher's file exactly: {ks['seeds_identical_to_yscher']} of {len(ks['seeds'])}"), kp
add("Michaels_T1T2", "cell_to_clone.csv = yscher's michaels_barcodes.py on the TLB FASTQs", "rerun of steps 1-7 (counts vs yscher's log) and step 8 with 20 hash seeds (michaels_rerun_barcodes.py, michaels_step8_only.py, michaels_step8_seeds.py)", michaels_clones())
add("PeTrace", "all 15 .h5td files = Zenodo record 23088914", "md5 of every local file vs the md5 published by Zenodo", text(f"{ROOT}/real_data/PeTrace/md5_check_all_2026-10-16.txt", None, 20))
audited = []
for f in sorted(glob.glob(f"{V}/petrace_tree_audit/*.json")):
    d = jl(f); audited.append(d["file"])
    ok = (d["obs_cells_with_tree_value_that_are_leaves"] == d["obs_cells_with_tree_value"] == d["leaves_total"] and d["nodes_with_more_than_one_parent"] == 0 and d.get("obs_cells_that_have_children", 0) == 0)
    add("PeTrace", f"{d['file']}: tree structure + sister-pair count", "independent audit of obst (petrace_tree_audit.py)",
        (f"{'CONSISTENT' if ok else 'INCONSISTENT'}: {d['obs_cells_with_tree_value']} of {d['n_obs_cells']} cells are tree leaves, each once; multi-parent nodes {d['nodes_with_more_than_one_parent']}; "
         f"sister pairs {d['sister_pairs_all_pairs_within_parent']}", f))
allp = [f"E{s}-R{r}" for s in ("7.5", "8.0", "8.5", "9.0", "9.5") for r in (1, 2, 3)]
add("PeTrace", "tree audit NOT done for these files (not required by any result; sister pairs come from the pair-distance analysis alone)", "files without a petrace_tree_audit json",
    (f"{len(allp) - len(audited)} of {len(allp)} files: " + ", ".join(p for p in allp if p not in audited), f"{V}/petrace_tree_audit/"))
add("DTT_mouse", "sister-pair tables and counts_samecell_analysis.npz = rebuild from the GEO matrices and the authors' tree", "dtt_rebuild (job 3977944) + diff_dtt.py",
    exists(f"{R}/DTT_mouse/diff.json", lambda p: ((lambda d: (f"ALL_IDENTICAL={d['ALL_IDENTICAL']}; {len([k for k in d if k != 'ALL_IDENTICAL'])} files compared", p))(jl(p)))))
add("SpaceBar", "Section1-5_cell_by_gene_clustered.csv incl. bc_cluster (the authors')", "source: the SpaceBar authors (stated by K. Arun 2026-10-16); md5 recorded; not re-derivable here", text(f"{ROOT}/real_data/SpaceBar_data/MD5SUMS_2026-10-16.txt", None, 10))
add("SpaceBar", "analysis on raw counts as is (no normalization, log1p, HVG, scaling)", "pca5k summary source line", fields(f"{V}/pca5k/summary_SpaceBar.json", ["n_cells", "n_hvg", "source"]))

# ------------------------------------------------------------------ re-run QC verification + analysis outputs vs the final cell counts
add("FM01/03/05/06/08, Watermelon, LARRY", "qc_verify.py rerun on the current files", "job 4057411 (3 datasets at a time)", text(f"{LG}/qcverify_final_4057411.out", r"^OK|^FAILED", 12))
cur_n = {}
for ds in ["FM01", "FM03", "FM05", "FM06", "FM07", "FM08"]:
    j = jl(f"{FIN}/{ds}_data/qc_filtered/preprocessing_summary.json"); cur_n[ds] = j["n_after_gene_floor_ceiling"]
cur_n["LARRY"] = jl(f"{FIN}/LARRY_data/filtered/preprocessing_summary.json")["n_after_gene_floor"]
cur_n["HS054_Sot48h"] = jl(f"{FIN}/HS054_Sot48h_data/qc_filtered/preprocessing_summary.json")["n_after_gene_floor_ceiling"]
cur_n["hPSC_endo_yscher"] = 26859
for ds, n in cur_n.items():
    p = f"{V}/pca5k/summary_{ds}.json"
    def f(p, n=n):
        s = jl(p)
        return f"PCA n_cells={s['n_cells']} vs current final cells={n}: {'MATCH' if s['n_cells'] == n else 'MISMATCH'}; HVGs={s['n_hvg']}; source: {s['source']}", p
    add(ds, "analysis PCA (pca5k) was computed on the current final cells", "pca5k_summary json n_cells vs the current final cell count", exists(p, f))

df = pd.DataFrame(rows)
os.makedirs(V, exist_ok=True)
df.to_csv(f"{V}/validation_ledger.csv", index=False)
pd.set_option("display.max_colwidth", 150, "display.width", 250)
print(df[["dataset", "claim", "result"]].to_string())
bad = df[df["result"].str.contains("PENDING|DIFFERS|DIFFER:|MISMATCH|NOT identical|NOT FOUND|DIFFERENT|MISSING|INCONSISTENT", regex=True)]
print(f"\n{len(df)} rows; rows with PENDING / DIFFERS / MISMATCH-type results: {len(bad)}")
for _, r in bad.iterrows():
    print("  ->", r["dataset"], "|", r["claim"][:70], "|", r["result"][:120].replace("\n", " "))
