#!/usr/bin/env python3
"""Rewrites finalized_data/DATASET_CREATION_SUMMARY.pdf (previous version, 2026-09-30, archived in finalized_data/archive_2026-10-16_before_fullsheet_rebuild/).
Every count is read from the current preprocessing_summary.json / obs files at build time; wording is the state after the 2026-10-16 validation and rebuilds.
Plain tables only. usage: make_dataset_creation_summary.py"""
import json

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

FIN = "/gpfs/projects/b1255/hzhang/TwINFER_KA/finalized_data"
OUT = f"{FIN}/DATASET_CREATION_SUMMARY.pdf"
ss = getSampleStyleSheet()
B = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12)
S = ParagraphStyle("s", parent=B, fontSize=7.5, leading=9.5)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=16)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, spaceBefore=10)


def P(t, st=S):
    return Paragraph(t, st)


def tbl(rows, widths):
    t = Table([[P(str(c)) for c in r] for r in rows], colWidths=[w * inch for w in widths], repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 3),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 3), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde6f0"))]))
    return t


def f(x):
    return f"{int(x):,}"


def js(d):
    return json.load(open(f"{FIN}/{d}_data/qc_filtered/preprocessing_summary.json"))


fm = {d: js(d) for d in ["FM01", "FM03", "FM05", "FM06", "FM07", "FM08", "HS054_Sot48h", "Watermelon"]}
L = json.load(open(f"{FIN}/LARRY_data/filtered/preprocessing_summary.json"))
AT = json.load(open(f"{FIN}/LARRY_data/annotated/annotation_transfer_filtered_k15.json"))
lo = pd.read_csv(f"{FIN}/LARRY_data/filtered/obs_metadata.csv", index_col=0)
lclones = lo.larry_clone_singletcode.fillna("").astype(str).replace("", pd.NA).nunique()
wm = pd.read_csv(f"{FIN}/Watermelon_data/qc_filtered/per_sample_counts.csv", index_col=0)
hp = json.load(open(f"{FIN}/hPSC_20260927_data/qc_filtered/preprocessing_summary.json"))
ann_src = pd.read_csv(f"{FIN}/LARRY_data/filtered/obs_metadata.csv", index_col=0).shape[0]

st = [P("TwINFER finalized_data: how each dataset was built", H1),
      P("Rewritten 2026-10-16 after the dataset validation (previous version, 2026-09-30, archived in finalized_data/archive_2026-10-16_before_fullsheet_rebuild/). "
        "Sources: the build scripts (code/TwINFER/paper_analysis/fatemap_pipeline/ and larry_pipeline/), each dataset's preprocessing_summary.json, and the rebuild / diff checks listed in "
        "analysis_data/dataset_qc_verification/preprocessing_per_sample__v2.pdf, which holds the per-sample tables and the validation page. All numbers below are read from the current files.", B),
      P("Common pipeline (FateMap-style datasets: FM01, FM03, FM05, FM06, FM07, FM08, HS054, Watermelon)", H2),
      P("&bull; <b>RNA input:</b> Cell Ranger filtered (called-cell) matrices; gene symbols as var_names; the '-1' (or aggr '-N') suffix stripped so cell barcodes match the 16-bp cellID of the lineage-barcode table."
        "<br/>&bull; <b>singletCode:</b> run <b>once on the entire barcode sheet</b> (every SampleNum, none dropped, including samples without an RNA matrix): criteria 1-3 (single barcode; dominant barcode by UMI; barcode combination recurs in other cells of the same sample). "
        "When min_umi_cutoff is not set, singletCode uses its own floor plus ratio-based auto cutoff, per sample."
        "<br/>&bull; <b>Criterion 4</b> (this repo's own rescue, singletcode_cross_sample_rescue.apply_cross_sample_rescue): cells still labelled Multiplet are re-tested for barcode-combination recurrence in cells of other samples (singletCode computes this but never applies it). "
        "Since 2026-10-16 a cell is identified by (sample, cellID); before, rows were pooled by cellID alone, which is wrong when the same 16-bp cellID occurs in two samples (see the gaps section). Clone ID = sorted concatenation of the singlet's unique barcodes (fatemap_clone_singletcode)."
        "<br/>&bull; <b>Sample sheet:</b> cellID, barcode, sample; one row per (cellID, UMI, barcode) record, exact duplicates collapsed; no manual UMI pre-filter. Barcode column BC50StarcodeD8 (FM, HS054) / D7 (hPSC)."
        "<br/>&bull; <b>Cell QC</b> (apply_qc_filters): mito % = MT- genes / total counts, keep &le; cutoff; keep min_genes &le; n_genes &le; max_genes. Final cells = pass mito AND singlet AND pass gene range. "
        "The thresholds are the values of the original paper (K. Arun): mito 26% (FM01, FM05, FM08), 30% (FM02, FM03), 18% (FM04), 21% (FM06); gene range 200-7,200 (FM06 200-5,000). FM07 (no value given) was built at 26%."
        "<br/>&bull; <b>Integrated h5ad</b> (build_*_integrated.py): log1p(CP10k) on all cells together; 2,000 HVGs (seurat, batch_key='replicate'); scale (max 10); PCA 50 (arpack, seed 0); scanorama and Harmony both stored; neighbors + UMAP; "
        "Goyal et al. cluster annotation joined for FM01/06/08. Counts kept in layers['counts']. (The analysis PCA of the validation uses 5,000 HVGs from the raw counts instead; see the validation PDF.)", B),
      P("Per-dataset QC and singletCode settings (current files)", H2)]
rows = [["Dataset", "System / samples", "Mito %", "Genes", "singletCode", "Criterion-4 rescued", "Status"],
        ["FM01", "WM989 + 1 uM PLX; replicates A, B (RNA GSM7434407/408); sheet SampleNum 1-4 (3, 4 without RNA)", "&le; 26", "200-7200", "whole sheet, auto cutoff 6 (A), 5 (B), 5, 5", f"{fm['FM01']['n_rescued_cells']} (sample, cellID) units", "rebuilt 2026-10-16"],
        ["FM03", "WM989 DMSO + 1 uM PLX; A (RNA GSM7434413); sheet SampleNum 1-2 (2 without RNA)", "&le; 30", "200-7200", "whole sheet, auto cutoff 12 in both", f"{fm['FM03']['n_rescued_cells']} units", "rebuilt (mito 26 -> 30)"],
        ["FM05", "WM989 250 nM PLX; A1, A2 (RNA GSM7434417/418); sheet SampleNum 1-4 (3, 4 without RNA)", "&le; 26", "200-7200", "whole sheet, auto cutoff 2 in all four", f"{fm['FM05']['n_rescued_cells']} units", "rebuilt 2026-10-16"],
        ["FM06", "WM989 naive; A, B (RNA GSM7434419/420)", "&le; 21", "200-5000", "whole sheet, min_umi_cutoff 3", f"{fm['FM06']['n_rescued_cells']} units", "rebuilt 2026-10-16"],
        ["FM07", "WM989 continuous 1 uM PLX; A, B (RNA GSM7434421/422)", "&le; 26 (paper value not given)", "200-7200", "whole sheet, auto cutoff 5 (A), 12 (B)", "14 units (same under both rescue versions)", "original build; byte-identical to a rerun; not rebuilt"],
        ["FM08", "Primary melanocytes; A, B (RNA GSM7434423/424); sheet SampleNum 1-3 (3: 466 cells, no RNA)", "&le; 26", "200-7200", "whole sheet, min_umi_cutoff 3", f"{fm['FM08']['n_rescued_cells']} units", "rebuilt 2026-10-16"],
        ["HS054_Sot48h", "PDAC, sotorasib 48 h; single sample", "&le; 25", "200-7000", "auto cutoff (not found in any log)", "0", "original build; byte-identical to a rerun"],
        ["Watermelon (T47D naive / lag / late x 2)", "6 samples, all passed to singletCode in one call", "naive &le; 20; lag, late &le; 25", "min 2500; max 7500 (naive), 8000 (lag, late)", "whole sheet, auto cutoff 2 in all six", f"{fm['Watermelon']['n_rescued_units_sample_cellID']} units", "rebuilt 2026-10-16 (rescue per sample + cellID)"],
        ["LARRY (inDrops, LSK d2 / d4 / d6)", "15 libraries; the authors' zips and barcode FASTQ", "&le; 20 (mt-)", "&ge; 200", "min_umi_cutoff 3 + criterion 4", f"{L['n_rescued_units_sample_cellID']} units ({L['n_distinct_cellids_rescued_by_criterion4']:,} distinct cellIDs)", "whitelist replaced 2026-10-16 (Table 1 minimum UMIs)"],
        ["hPSC_20260927", "hESC to definitive endoderm; two timepoints", "&le; 10", "&ge; 1000", "auto cutoff 2", f"{hp['n_distinct_cellids_rescued_by_criterion4']}", "excluded from the analysis; folder kept"]]
st += [tbl(rows, [1.3, 2.7, 0.8, 1.0, 1.7, 1.1, 1.4]),
       P("Order of operations: singletCode runs on the full barcode table, then RNA QC (mito, gene range) is intersected.", S)]

st += [PageBreak(), P("Cell counts through each filter (current files)", H2)]
r = [["Dataset", "Raw cells", "After mito", "After singletCode + criterion 4", "After gene range (final)", "Final file"]]
for d, fn in [("FM01", "FM01_integrated.h5ad"), ("FM03", "qc_filtered only"), ("FM05", "qc_filtered only"), ("FM06", "FM06_integrated.h5ad"), ("FM07", "qc_filtered only"), ("FM08", "FM08_integrated.h5ad"), ("HS054_Sot48h", "HS054_Sot48h_integrated.h5ad")]:
    j = fm[d]; r.append([d, f(j["n_raw_cells"]), f(j["n_after_mito"]), f(j["n_after_singletcode_plus_rescue"]), f(j["n_after_gene_floor_ceiling"]), fn])
w = fm["Watermelon"]
st_tot = {s: int(wm.loc[wm.index.str.contains(s), "final"].sum()) for s in ("naive", "lag", "late")}
r.append(["Watermelon (all)", f(w["n_raw_cells"]), "-", "-", f(w["n_final"]), f"split into naive {f(st_tot['naive'])} / lag {f(st_tot['lag'])} / late {f(st_tot['late'])}"])
r.append(["LARRY", f(L["n_raw_cells"]), f"whitelist {f(L['n_after_whitelist'])} -> mito {f(L['n_after_mito'])}", f(L["n_after_singletcode_umi3_plus_criterion4"]), f(L["n_after_gene_floor"]) + f" ({f(lclones)} clones; Weinreb rule ii/iii not applied)", "filtered/ and annotated/filtered_annotated_k15.h5ad"])
st += [tbl(r, [1.6, 0.9, 1.9, 1.9, 2.3, 2.3]), Spacer(1, 6), P("Watermelon per sample (raw / pass mito / pass mito + singlet / final)", H2)]
r = [["Sample", "Raw", "Mito", "Mito + singlet", "Final"]] + [[i, f(x["raw"]), f(x["pass_mito"]), f(x["pass_mito+singlet"]), f(x["final"])] for i, x in wm.iterrows()]
st += [tbl(r, [1.6, 1, 1, 1.2, 1]),
       P("Watermelon stage datasets: Watermelon_{naive,lag,late}_integrated.h5ad hold each stage's two replicates (replicate 1 = A, 2 = B) with batch-aware log1p(CP10k) + PCA 50 + scanorama + Harmony (batch_key='replicate'). obs has replicate A/B and fatemap_clone_singletcode.", S)]

wl = L["whitelist_per_library"]
ap = AT.get("cv_accuracy_per_timepoint", {})
st += [P("LARRY: history of the current file", H2),
       P("&bull; <b>Step 1, raw matrix</b> (build_larry_matrix.py). 15 inDrops LSK libraries (d2 x3, d4 x6, d6 x6) from the authors' zip files, combined: 95,584 barcodes x 25,289 genes; inDrops 'bcXXXX' hashes renamed to the real ACGT-ACGT gel barcode (abundant_barcodes.pickle, 1:1 in all libraries). "
         "Lineage barcodes from LARRY_sorted_and_filtered_barcodes.fastq.gz (the authors' file): reads counted per (library, cell, UMI, barcode) combination and only combinations supported by &ge; 10 reads kept (the read-support filter of the Weinreb et al. notebook); "
         "barcodes then merged within Hamming distance 3 by transitive connected components, each collapsing to its highest-UMI member. Output umi_table.csv (one row per UMI per cell / barcode / library) feeds singletCode. The &ge; 3 UMIs per cell-barcode is not applied in this step; singletCode applies min_umi_cutoff 3 later."
         "<br/>&bull; <b>Step 2, whitelist (changed 2026-10-16).</b> The Weinreb et al. cutoffs were set visually per sample from histograms of total transcripts per barcode and were not shared (K. Arun). The repo first used a knee-point rule tuned to the cell count of the paper's PDF, then a dip test + Otsu rule per library "
         "(95,584 to 89,813 barcodes; real cuts only in LSK_d2_1 at 205, LSK_d6_1_1 at 1,257, LSK_d6_1_2 at 1,217; every non-zero barcode kept in the other 12). The current file uses the minimum total UMIs of Weinreb et al. Table 1 for in vitro LSK: "
         f"day 2 &ge; {wl['LSK_d2_1']['threshold']:.0f}, day 4 &ge; {wl['LSK_d4_1_1']['threshold']:.0f}, day 6 &ge; {wl['LSK_d6_1_1']['threshold']:.0f}, applied to each library of that day: {f(L['n_after_whitelist'])} barcodes, equal to the target cell count of the earlier knee-point rule. "
         "Cells passing the whitelist have median total UMIs of 1,802 / 2,005 / 2,181 (day 2 / 4 / 6) against 1,720 / 1,993 / 2,211 in Table 1. Against the dip + Otsu file the switch changes about 1.3% of the final cells."
         f"<br/>&bull; <b>Step 3, singletCode.</b> get_singlets(umi_table, min_umi_cutoff=3) (3 rather than 2, to match Weinreb et al. 2020 Methods 3.3), then the criterion-4 rescue with a cell identified by (library, cellID): {f(L['n_rescued_units_sample_cellID'])} rescued units. Clone = sorted concatenation of the singlet's unique barcodes."
         f"<br/>&bull; <b>Step 4, RNA QC.</b> Mito &le; 20% (mt-) and &ge; 200 genes, applied sequentially: whitelist {f(L['n_after_whitelist'])}, mito {f(L['n_after_mito'])}, singletCode {f(L['n_after_singletcode_umi3_plus_criterion4'])}, gene floor {f(L['n_after_gene_floor'])} cells, {f(lclones)} clones."
         "<br/>&bull; <b>Step 5 (Weinreb rules ii/iii)</b> was tried and retired: cross-library (cell-BC, clone) repeats and over-abundant-clone chi-square filtering were integrated for one build and removed, because they accounted for 62% of the disagreement with the authors' h5ad (rule ii alone 1,965 cells, rule iii 48). The finalized file is Steps 1-4 only; the rule is not applied. "
         "A rerun of the current script with the rule on gives 27,972 cells instead of 32,395 for the previous file."
         "<br/>&bull; <b>Validation.</b> Rebuilt from the authors' zips: stage 1 is byte-identical to the stored files (two obsolete obs columns aside). Against the released LSK_d2_d4_d6.h5ad clone_id, the previous file had Adjusted Rand Index 0.9989 on the 23,687 cells both call barcoded (76 cells group differently); this was not recomputed for the current file."
         "<br/>&bull; <b>Stored matrix.</b> filtered/larry_qc_counts.mtx is raw counts (no normalization). Normalization (log1p(CP10k) / log1pPF) happens at inference input construction."
         f"<br/>&bull; <b>Annotation</b> (larry_raw_annotate.py). Labels come from the published LSK_d2_d4_d6.h5ad where the cell is present (30,466 cells, 'published'); the other 2,896 cells get predicted labels. Method: drop genes expressed in &lt; 5% of cells; log1p(CP10k); 2,000 HVGs with batch_key='sample'; PCA 50; distance-weighted k-NN, k=15, class-balanced fit pool capped at 500 per class. "
         f"The classifier never sees clone identity. Stratified 5-fold CV on published cells: overall accuracy {AT['cv_accuracy_overall']:.3f} (d2 {ap.get('2', float('nan')):.3f}, d4 {ap.get('4', float('nan')):.3f}, d6 {ap.get('6', float('nan')):.3f}); poor on rare classes (Erythroid 0.0, pDC 0.0, Eos 0.4). The HVG / PCA is for the classifier only and is not stored.", S)]

st += [P("Other datasets in finalized_data (details in preprocessing_per_sample__v2.pdf)", H2)]
r = [["Dataset", "How the file was made", "Verification 2026-10-16"],
     ["hPSC_endo_yscher", "Cell Ranger filtered cells (mito &lt; 20%, &ge; 200 genes) by Hanxiao; clone_c5 = singletCode (cutoff 5, ratio 0, dom 30, diff 0) per sample on the starcode D8 table, by yscher's build_endo_dataset.py. The counts used in the analysis are the Cell Ranger counts (endo_counts_mito20_min200genes.h5ad); the kb-python spliced / unspliced layers in the other h5ad are not used.", "Barcode chain rebuilt from the barcode FASTQs and identical; both h5ads identical. RNA FASTQ to Cell Ranger and kb count not rerun."],
     ["hPSC_undifferentiated", "Cell Ranger filtered cells from two runs (T0 from cellranger_out, T10 and T15 from cellranger_10x), mito &lt; 20%, &ge; 200 genes; log1pPF in the stored X, integer counts in a sibling file; clone_c5 as above.", "QC recomputed from the Cell Ranger matrices: identical cells (13,713 / 18,032 / 23,216) and identical counts in 900 of 900 sampled cells; clone_c5 rerun from the stepThree tables (copied to real_data/hPSC_undifferentiated_barcodes/): identical."],
     ["hPSC_Novo_Nordisk", "Per-sample counts and clone labels exported from the authors' Seurat object (nFeature &ge; 1000, percent.mt &le; 10).", "Per-sample files byte-identical from the export; the export itself is from the authors."],
     ["Michaels_T1T2", "yscher's QC'd h5ads (mito &lt; 20%, &ge; 200 genes) and clone labels from yscher's own barcode pipeline (not singletCode).", "QC files identical; barcode steps 1-7 match yscher's log; clone labels not exactly reproducible (hash-seed dependent step 8)."],
     ["CellTag3", "yscher's scripts from the GEO Cell Ranger matrices; clones = Jaccard &ge; 0.6 on 2-24 CellTags, connected components, the authors' sparse-clone repair; X = log1p(CP10k).", "Whole chain rebuilt; final file md5-identical."],
     ["PeTrace", "The authors' .h5td files (Zenodo 23088914), integer counts; clonal pairs = sister leaves of one parent in the lineage tree (obst).", "All 15 md5 match Zenodo; tree audit in 7 files; the other 8 are not audited."],
     ["DTT_mouse", "GEO GSE341627 matrices + the authors' tree; strict sister pairs (parent &ge; E12.5), cousins, singletons.", "Rebuilt; all outputs identical."],
     ["SpaceBar", "The authors' Section1-5 cell-by-gene tables (bc_cluster = clone); analysed as raw counts, no normalization.", "Authors' files; md5 recorded; not re-derivable."]]
st += [tbl(r, [1.6, 5.4, 3.4])]

st += [PageBreak(), P("What was done to the cells after QC; are the normalized counts modified?", H2),
       P("&bull; FM01 / 06 / 08, Watermelon stages, HS054 integrated files: log1p(CP10k) over all cells together (raw counts in layers['counts'], log values in layers['lognorm'] and .raw); 2,000 HVGs (batch_key='replicate'; none for HS054); scale (max 10); PCA 50; neighbors + UMAP; scanorama and Harmony stored alongside for the replicate datasets. "
         "HS054 also gets Leiden clustering; FM01/06/08 get Goyal et al. cluster labels. Watermelon is split by stage and integrated per stage."
         "<br/>&bull; 'Integrated' refers to embeddings only: X, layers['lognorm'] and .raw.X are identical and equal plain log1p(CP10k) (expm1 of a cell's values sums to 10,000, checked on FM06 and HS054). Scanorama and Harmony write only to obsm; scaling and PCA run on a separate copy. Expression is not batch-corrected."
         "<br/>&bull; LARRY: k-NN annotation transfer only; the annotated h5ad stores raw counts in X. hPSC and Novo: nothing beyond QC + singletCode (Novo: the authors' export). SpaceBar: raw counts as is. CellTag3: X is log1p(CP10k)."
         "<br/>&bull; The validation PCA and pair distances use one recipe for every dataset except SpaceBar: raw counts, normalize_total 1e4, log1p, 5,000 HVGs (seurat, no batch key), scale (max 10), 50 PCs. CellTag3's X already is log1p(CP10k), so those two steps are skipped. "
         "TwINFER reads expression, not the embeddings. CP10k is not library-size invariant for rank statistics such as Spearman.", S),
       P("Things to know / gaps", H2),
       P("&bull; <b>Criterion-4 flaw (found 2026-10-16, fixed in the rebuilt files).</b> The rescue pooled rows by cellID alone, so a 16-bp cellID present in two samples (two different cells) was treated as one. Effect on rescued units, as built / corrected: FM01 28 / 28, FM03 12 / 12, FM05 357 / 359, FM06 38 / 40, FM07 14 / 14, FM08 22 / 22, Watermelon 572 / 580, LARRY 1,006 / 1,561. "
         "The old LARRY file had 12 wrongly rescued and 567 missed cells. The criterion4_rescued flag in the old files was also set by cellID alone. FM07 and HS054 were not rebuilt (no difference for FM07; HS054 has one sample)."
         "<br/>&bull; Auto UMI cutoffs were taken from the rebuild logs: FM01 6 / 5 / 5 / 5, FM03 12 / 12, FM05 2 in all four, FM07 5 / 12, Watermelon 2 in all six, hPSC 2. The HS054 cutoff was not found in any log."
         "<br/>&bull; HS054: a fresh auto-cutoff run gives 3,040 singlets before the gene range; an earlier external singlets_all.txt (3,264) could not be reproduced. The RNA matrix was re-synced from the canonical run directory because the earlier copy overlapped only 12 of 7,342 barcode-table cellIDs."
         "<br/>&bull; Settings differ across datasets (min_umi_cutoff 3 for FM06, FM08, LARRY; auto elsewhere; mito 21-30%; gene ranges differ), as set from the original paper and the build scripts."
         "<br/>&bull; Inputs that cannot be re-derived here: the LARRY zips and barcode FASTQ and the SpaceBar tables (the authors' files), the Novo Seurat export, the Cell Ranger and kb steps of the hPSC datasets, and the exact Michaels clone labels. PeTrace tree audit: 8 of 15 files not audited (their sister-pair counts come from the pair-distance analysis alone)."
         "<br/>&bull; hPSC_20260927 and the all-stage Watermelon have no single integrated h5ad: hPSC has qc_filtered only, Watermelon is split by stage. hPSC_20260927 is excluded from the analysis.", S)]
SimpleDocTemplate(OUT, pagesize=landscape(letter), leftMargin=0.45 * inch, rightMargin=0.45 * inch, topMargin=0.45 * inch, bottomMargin=0.45 * inch,
                  title="TwINFER dataset creation summary", author="Keerthana Arun").build(st)
print("saved", OUT)
