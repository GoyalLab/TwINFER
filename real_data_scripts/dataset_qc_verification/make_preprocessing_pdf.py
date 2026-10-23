#!/usr/bin/env python3
"""PDF with the pre-processing of every sample: source, raw cells, each filter with its cutoff and the number of cells it removed
(recomputed from the raw matrices wherever raw data exist), final cells, and what is known / not known otherwise.

Inputs : <OUT>/<dataset>/verify.json (qc_verify.py), <OUT>/pca5k/summary_*.json, <OUT>/all_samples_list.csv,
         finalized_data/LARRY_data/filtered/preprocessing_summary.json, a small extra-metrics json (yscher/Novo observed ranges, see EXTRA).
Output : <OUT>/preprocessing_per_sample__v2.pdf
Plain tables only (no plots).
"""
import json
import os
import sys

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qc_common import FIN, OUT, ROOT

EXTRA = os.environ.get("EXTRA_METRICS_JSON", f"{OUT}/extra_metrics_yscher_novo.json")
PDF = f"{OUT}/preprocessing_per_sample__v2.pdf"   # renamed by K. Arun 2026-10-16: VS Code showed the old file name inverted (stored view state), a new name displays correctly

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11.5, spaceBefore=8, spaceAfter=3)
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=8.3, leading=10.4, spaceAfter=3)
SM = ParagraphStyle("SM", parent=B, fontSize=7.3, leading=9)
CELL = ParagraphStyle("CELL", parent=B, fontSize=7.1, leading=8.6, spaceAfter=0)
HEAD = ParagraphStyle("HEAD", parent=CELL, fontName="Helvetica-Bold")


def P(t, st=B):
    return Paragraph(t, st)


def table(rows, header, widths, zebra=True, font=7.1):
    data = [[Paragraph(str(h), HEAD) for h in header]] + [[Paragraph(str(c), CELL) for c in r] for r in rows]
    t = Table(data, colWidths=[w * inch for w in widths], repeatRows=1)
    st = [("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b8b8b8")), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6e6e6")),
          ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
          ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]
    t.setStyle(TableStyle(st))
    return t


def f(x):
    return f"{int(round(x)):,}" if x is not None and x == x else "-"


def vj(d):
    return json.load(open(f"{OUT}/{d}/verify.json"))


def cut(c):
    parts = [f"mito &le; {c['mito']:g}%"]
    g = f"genes &ge; {c['min_genes']:g}"
    if c["max_genes"] < 1e8:
        g += f", &le; {c['max_genes']:g}"
    parts.append(g)
    if c.get("max_total", float("inf")) < 1e12:
        parts.append(f"UMI &le; {c['max_total']:g}")
    return "; ".join(parts)


HDR = ["Sample (raw unit)", "Raw cells", "Pre-filter", "Removed: mito", "Removed: gene/UMI range", "Removed: not a singlet / no clone call", "Final cells",
       "Median genes raw", "Median UMI raw -> final", "Max mito% final", "Recomputed from raw?"]
W = [1.25, 0.6, 1.2, 0.85, 0.9, 0.9, 0.6, 0.55, 0.9, 0.5, 1.2]


def rows_from_verify(d):
    j = vj(d)
    rows = []
    for s, v in j["per_sample"].items():
        c = j["stated_cutoffs"][s]
        pre = f"{f(v['n_removed_by_pre_filter'])} removed" if v["n_removed_by_pre_filter"] else "-"
        chk = ("YES: %s; stored metrics match; 0 final cells break cutoffs" % v["final_rows_identical_to_raw_rows"].split(";")[0].replace(" sampled", " rows identical"))
        rows.append([f"<b>{s}</b><br/>{cut(c)}", f(v["n_raw"]), pre, f(v['n_removed_by_mito_after_pre']), f(v['n_removed_by_range_after_pre_mito']),
                     f(v['n_removed_as_non_singlet']), f"<b>{f(v['n_final'])}</b>", f(v["raw_n_genes_median"]),
                     f"{f(v['raw_total_counts_median'])} -> {f(v['final_total_counts_median'])}", f"{v['final_pct_mt_max']:.1f}", chk])
    return rows, j


def sums_ok(rows_j):
    pass


story = []
story += [P("Pre-processing of every sample", H1),
          P("Keerthana Arun - TwINFER real datasets - generated 2026-10-16. Every number marked 'recomputed' was recounted from the raw matrices in this session; "
            "numbers taken from build summaries are labelled as such. Cells removed are counted <b>sequentially</b> in the order shown "
            "(pre-filter, mito, gene/UMI range, then singlet/clone calling), so each row adds up: raw - (pre-filter + mito + range + non-singlet) = final. The three 'Removed' columns are counts of cells (not negative values).", B),
          P("<b>Status key.</b> <b>recomputed</b> = raw Cell Ranger matrix loaded, all metrics recomputed here, final cells matched to raw cells and 300 final rows compared "
            "value-by-value with the raw rows. <b>summary</b> = numbers from the build script / preprocessing_summary.json, not re-run. "
            "<b>final only</b> = no raw matrix is available locally, so only the final cells could be described.", B)]

# ---------------------------------------------------------------- validation by rebuild
LED = pd.read_csv(f"{OUT}/validation_ledger.csv")
vrows = [
 ["FM01, FM03, FM05, FM06, FM07, FM08, Watermelon", "Original build script rerun on the raw inputs into a scratch folder; every output file diffed against the then-current finalized_data (md5 taken before and after). Then the full-sheet / corrected-rescue rebuild (jobs 4041481, 4045122) was copied into finalized_data and checked with diff -rq.", "<b>Previous files (archived): byte-identical</b> to the rebuild, 5 files per FM dataset, 6 for Watermelon. <b>Current files</b> = the full-sheet rebuild (mito 26 / 30 / 26 / 21 / 26 for FM01 / 03 / 05 / 06 / 08, singletCode once on the whole sheet, rescue per (sample, cellID)), verified copies of the job output; FM07 and HS054 were not rebuilt."],
 ["HS054_Sot48h", "build_hs054_qc_matrix.py rerun; files diffed.", "<b>Byte-identical.</b> The extra finalized file hs054_qc_counts.mtx (the one the analysis loads) is not written by the script; cmp against the rebuilt hs054_sot48h_qc_counts.mtx = identical."],
 ["LARRY stage 1", "build_larry_matrix.py rerun on the 15 inDrops zips + barcode FASTQ.", "<b>Byte-identical</b> except two obsolete obs columns present only in the older file. The zips and the barcode file (LARRY_sorted_and_filtered_barcodes.fastq.gz) are the authors' files; they cannot be re-derived here."],
 ["LARRY stage 2", "build_final_matrix_umi3.py rerun on the stage-1 files, as the current script and with the Weinreb rule (ii)/(iii) step off.", "The <b>previous</b> file (dip test + Otsu whitelist, rescue by cellID alone, archived) equalled the output without the Weinreb rule step; the current script with the rule on gives 27,972 cells, so the rule was never applied to our file. The <b>current</b> file replaces the whitelist by the Table 1 thresholds and uses the corrected rescue (job 4053678); finalized_data/LARRY_data/{filtered,annotated} are verified copies (diff -rq) of that job's output."],
 ["LARRY annotation", "larry_raw_annotate.py rerun; h5ad compared element-wise (previous file).", "Previous annotation file: <b>identical</b>, 33 of 33 HDF5 datasets. Current annotation file: made by the same script on the new filtered files (job 4053678): 33,362 cells, published labels for 30,466 and k-NN transfer for 2,896; no independent second rerun."],
 ["hPSC_endo_yscher", "Barcode chain rerun in four links, starting from the barcode FASTQs: lane concatenation (md5), stepOne/Two/Three barcode tables (cmp), yscher's build_endo_dataset.py, final h5ad compared element-wise. The RNA FASTQ -> Cell Ranger step and the kb count step were not rerun (their outputs are used as given).", "<b>Identical at every link</b>: both barcode-FASTQ concatenations equal the two lanes; stepThree tables byte-identical (T0, T15); the kb-layer h5ad 31 of 31 HDF5 datasets identical, and the Cell Ranger counts h5ad (the file used) identical to yscher's (17 of 17)."],
 ["hPSC_Novo_Nordisk", "build_qc_per_sample.py rerun on the Seurat export; each file cmp'd.", "<b>Byte-identical</b>: 18 of 18 files (6 samples x counts.npz, genes.txt, obs_metadata.csv). The Seurat export itself comes from the authors; percent.mt recomputed from the counts differs slightly from the stored value (4 cells above 10% when recomputed; none by the stored value)."],
 ["Michaels T1 / T2 QC files", "Raw GEO matrices re-QC'd (mito &le; 20%, &ge; 200 genes) independently; compared with yscher's michaels_T1_qc.h5ad / michaels_T2_qc.h5ad.", "<b>Identical</b>: obs and var names and order, every X value, stored metrics (T1 8,584 raw -> 8,478; T2 5,747 cells)."],
 ["Michaels T1 / T2 clone labels", "yscher's michaels_barcodes.py rerun from the TLB FASTQs: steps 1-7 against yscher's log; step 8 with yscher's interpreter and 20 hash seeds.", "Steps 1-7: <b>every count equals yscher's log</b> (963,137 UMIs, 68,890 barcodes, 5,549 cells, 3,230 barcodes used, 1,829 raw components). Step 8 (min-cut / degree pruning) depends on PYTHONHASHSEED: 20 seeds give 1,359-1,372 clones, 4,062-4,125 cells, ARI 0.989-0.999 vs yscher's file; yscher's 1,371 clones / 4,118 cells is not reproduced by any of them. <b>The clone labels are therefore not exactly reproducible.</b>"],
 ["PeTrace", "md5 of all 15 .h5td vs the md5 published on Zenodo (record 23088914); independent audit of the tree field (obst) per file.", "<b>All 15 md5 match.</b> Tree audit done for E7.5 R1-R3, E8.0 R1-R3, E8.5-R1: every cell with a tree value is a leaf exactly once, no node has two parents. The remaining 8 files (E8.5 R2/R3, E9.0, E9.5) are not audited; their sister-pair counts come from the pair-distance analysis alone."],
 ["hPSC_undifferentiated", "QC recomputed from the Cell Ranger filtered matrices (mito &lt; 20%, &ge; 200 genes) and compared with the cells and counts of our file (hpsc_undiff_qc_check.py).", "<b>Identical</b>: 13,713 / 18,032 / 23,216 cells at T0 / T10 / T15, 0 cells differ; 300 random cells per timepoint have identical count vectors (900 of 900). yscher's reproduction notes record the two Cell Ranger runs (T0 from cellranger_out, T10 and T15 from cellranger_10x). The clone_c5 step was rerun as well (next line)."],
 ["hPSC_undifferentiated clone_c5", "singletCode on the three stepThree tables (copied to real_data/hPSC_undifferentiated_barcodes/, cmp-verified).", "<b>Identical</b>: same cells with a clone (5,981 / 7,845 / 10,617) and the same barcode string in all of them, T0 / T10 / T15. The stepThree tables themselves (Hanxiao's barcode pipeline on the barcode FASTQs) were not regenerated."],
 ["SpaceBar", "None possible.", "Section1-5_cell_by_gene_clustered.csv come from the SpaceBar authors (stated by K. Arun); md5 recorded in real_data/SpaceBar_data/MD5SUMS_2026-10-16.txt; cannot be re-derived here."],
 ["CellTag3", "yscher's unmodified scripts and interpreter rerun in a mirror of yscher's tree from the GEO Cell Ranger matrices (job 3952316, 6h07m, completed); every intermediate compared with yscher's file of the same name.",
  "<b>Steps 1-5 identical:</b> nosinglet counts h5ad 19/19 HDF5 datasets; QC'd h5ad (mito 20%, 200 genes) all 24 shared datasets identical (the file we hold has 8 extra entries: 7 X_pca_* embeddings in obsm and uns/normalization_names, written by yscher's separate build_celltag_embeddings.py as test embeddings; they are not part of the data and not used here); the two clone csv files (assignments, 2-24-tag pair overlaps) byte-identical; celltag_clones_published_nopad.h5ad and _counts.h5ad 16/16 identical. "
  "<b>Steps 6-7 identical too</b> (job 4049584, after installing igraph 0.9.11 in yscher's interpreter; the first attempts failed on missing / newer igraph, which rejects the authors' integer-named vertex column): clone_table_repaired.csv byte-identical; celltag_clones_repaired_nopad.h5ad and _counts.h5ad identical element by element; md5 of the rebuilt final file 6c39651f209805181a1d352eb5e16f49 = the file we use."],
 ["DTT mouse", "Rebuild of the sister-pair tables and counts_samecell_analysis.npz from the GEO matrices and the authors' tree (job 3977944).", "<b>Identical</b>: sister_pairs_strict.csv, cousin_cells.csv, all_cherries_age_edits.csv, counts_samecell_analysis_cells.csv and the seven per-GSM cell lists byte-identical; counts_samecell_analysis.npz (84,718 x 78,258, 72,959,919 non-zero values) and the seven per-GSM count files identical in values. Gene-column mapping verified earlier with markers."],
 ["Criterion-4 rescue check", "check_rescue_logic.py (job 4036965): the repo's rescue pools rows by cellID across samples; compared with the same rescue with a cell identified by (sample, cellID).", "Rescued units, as built / corrected: FM01 28 / 28, FM03 12 / 12, FM05 357 / 359, FM06 38 / 40, FM07 14 / 14, FM08 22 / 22, Watermelon 572 / 580, LARRY 1,006 / 1,561. The cellID-only version wrongly rescued 12 final LARRY cells and missed 567 (LARRY has 5,310 cellIDs in more than one sample); it never wrongly rescued a final cell in any other dataset. The current files use the corrected rescue (FM07 unchanged, same result)."]
]
story += [P("Validation by rebuild (2026-10-16): what was re-run and what it shows", H1),
          P("Each line is a rebuild of the original script from the stated inputs, compared with the files the analysis uses. Evidence files are listed in analysis_data/dataset_qc_verification/validation_ledger.csv. Lines marked not finished / not run contain no result.", B),
          table(vrows, ["Dataset", "What was re-run", "Result"], [1.6, 3.6, 5.0])]

# ---------------------------------------------------------------- common pipeline
story += [P("Common pipeline (FateMap-style datasets: FM01-08, HS054, Watermelon)", H2),
          P("1) RNA: Cell Ranger filtered (called-cell) matrices; gene symbols as var_names; barcode suffix (-1 / aggr -N) stripped to match the 16-bp cellID of the lineage table. "
            "2) Lineage barcodes: starcode-collapsed reads (BC50StarcodeD8; hPSC D7) to a sample sheet (cellID, barcode, sample), no manual UMI pre-filter. "
            "3) singletCode, run once on the entire barcode sheet (every SampleNum, none dropped): criteria 1-3 (single barcode; dominant barcode by UMI; barcode combination recurs in other cells of the sample), then this repo's criterion-4 rescue (barcode combination recurs in cells of other samples; a cell is identified by sample + cellID since 2026-10-16, before that by cellID alone, which does not change FM01, FM03, FM07, FM08 and moves FM05 / FM06 by 2 units); "
            "clone ID = sorted concatenation of the singlet's barcodes. 4) RNA QC: mito % = counts in MT- genes / total counts (13 MT- genes, human); keep mito &le; cutoff and min &le; genes detected &le; max. "
            "Final cells = pass mito AND pass gene range AND singlet. 5) Final count matrix = raw integer counts (no normalization stored). "
            "Cutoffs and singletCode UMI cutoffs differ by dataset (below); they are the values hard-coded in the build scripts. <b>Source of the numbers:</b> all thresholds are the values of the original paper (stated by K. Arun): mito &le; 26% (FM01, FM05, FM08), 30% (FM02, FM03), 18% (FM04), 21% (FM06); gene range 200-7,200 (FM06 200-5,000). No FM07 value was in the list given; its build uses 26%. Nothing in the repo records the paper citation, so the PDF cannot show where in the paper each number is. "
            "<b>singletCode auto UMI cutoff</b> (min_umi_cutoff not passed): per sample, cutoff = max(min_umi_cutoff default 2, round(3/4e5 x number of barcode rows in that sample of the sheet)); a barcode needs at least that many UMIs to count. It therefore scales with sequencing depth of the barcode library (2 for small sheets, 5-12 here).", B)]

# ---------------------------------------------------------------- FateMap
fm = [("FM01", "WM989 + 1 uM PLX; replicates A, B (merged = one sample). RNA GSM7434407/408. singletCode run once on the whole barcode sheet (SampleNum 1-4; 3 and 4 have no RNA matrix, none dropped); auto UMI cutoff per sample = 6 (A), 5 (B), 5, 5; 72 (sample, cellID) units rescued by criterion 4 (matched by barcode combination, cell = sample + cellID). mito &le; 26%. Rebuilt 2026-10-16: 10,475 final cells (the previous file, 10,455, used samples 1-2 only; archived)."),
      ("FM03", "WM989 DMSO + 1 uM PLX; one replicate (A). RNA GSM7434413. singletCode run once on the whole sheet (sample 2 has no RNA, not dropped), auto cutoff 12 in both samples; 12 units rescued. mito &le; 30% (paper value). Rebuilt 2026-10-16: 4,711 final cells (previous file, built at 26%: 4,705)."),
      ("FM05", "WM989 250 nM PLX; A1, A2. RNA GSM7434417/418. singletCode run once on the whole sheet (SampleNum 1-4, 3 and 4 without RNA), auto cutoff 2 in all four samples; 359 units rescued (the previous cellID-only rescue gave 357). mito &le; 26%. Rebuilt 2026-10-16: 13,291 final cells (previous 13,290)."),
      ("FM06", "WM989 naive; A, B (merged = one sample). RNA GSM7434419/420. singletCode min_umi_cutoff = 3 on the whole sheet (SampleNum 1 and 2, nothing to drop); 40 units rescued (previous cellID-only rescue: 38). mito &le; 21% (paper value). Rebuilt 2026-10-16: 12,865 final cells (previous 12,863)."),
      ("FM07", "WM989 continuous 1 uM PLX; A, B. RNA GSM7434421/422. singletCode run once on the whole sheet, auto cutoff 5 (A) and 12 (B) in that one run. Not rebuilt: the rescue check shows 14 rescued units identically under the cellID-only and the corrected rescue. mito &le; 26% as built (the paper value for FM07 was not in the list given)."),
      ("FM08", "Primary melanocytes; A, B (merged = one sample). RNA GSM7434423/424. singletCode min_umi_cutoff = 3 on the whole sheet, including SampleNum 3 (466 cells, no RNA); 24 units rescued. mito &le; 26%. Rebuilt 2026-10-16: 3,985 final cells (identical count to the previous file).")]
story += [PageBreak(), P("FateMap melanoma series (FM01, FM03, FM05, FM06, FM07, FM08)", H1)]
for d, note in fm:
    rows, j = rows_from_verify(d)
    story += [P(f"{d}", H2), P(note, SM), table(rows, HDR, W)]
    ih = [x for x in j["integrated_h5ad_checks"] if not x.get("missing")]
    if ih:
        x = ih[0]
        story += [P(f"Integrated h5ad ({os.path.basename(x['file'])}) - recomputed: counts layer equals the qc matrix: {x['counts_layer_equals_qc_mtx']}; X is plain log1p(CP10k) "
                    f"(expm1 row sums {x['X_expm1_rowsum_min_max_(expect_1e4)'][0]:.0f}-{x['X_expm1_rowsum_min_max_(expect_1e4)'][1]:.0f}, max |X - recomputed| = {x['X_equals_recomputed_log1p_CP10k_maxabs']:.1e}); "
                    f"{x['n_hvg_flag']} HVGs flagged; stored PCA {x['X_pca_shape']}; embeddings {', '.join(x['obsm'])}. Expression itself is not batch-corrected.", SM)]
    story.append(Spacer(1, 3))

# ---------------------------------------------------------------- HS054 + Watermelon
story += [PageBreak(), P("HS054 + sotorasib and Watermelon (T47D)", H1)]
rows, j = rows_from_verify("HS054_Sot48h")
story += [P("HS054_Sot48h (PDAC, sotorasib 48 h, single sample)", H2),
          P("RNA from /projects/b1042/GoyalLab/Hanxiao/twinfer/10x/HS054_Sot48h (re-synced into real_data/pancreatic_cancer: the earlier copy overlapped only 12 of 7,342 barcode-table cellIDs). "
            "singletCode auto cutoff (value not found in any log); 0 cellIDs rescued by criterion 4. A fresh singletCode run gives 3,040 singlets before the gene range; the earlier external singlets_all.txt (3,264) could not be reproduced.", SM),
          table(rows, HDR, W)]
ih = j["integrated_h5ad_checks"][0]
story += [P(f"Integrated h5ad - recomputed: counts layer = qc matrix: {ih['counts_layer_equals_qc_mtx']}; X = log1p(CP10k) (max |diff| {ih['X_equals_recomputed_log1p_CP10k_maxabs']:.1e}); {ih['n_hvg_flag']} HVGs; PCA {ih['X_pca_shape']}; no integration (single sample).", SM)]
rows, j = rows_from_verify("Watermelon")
story += [P("Watermelon (T47D breast cancer; naive, lag, late x 2 replicates)", H2),
          P("RNA: six cellranger-aggr samples (suffix -1..-6 stripped). Lineage table Watermelon_barcode_cellID_UMIcounts.csv is already a sample sheet. All six samples are passed to singletCode in ONE call with the auto UMI cutoff "
            "(logged cutoff 2 in every sample); 580 (sample, cellID) units rescued by criterion 4 (the previous cellID-only rescue gave 572; rebuilt 2026-10-16: 33,279 final cells, previous 33,271). Per-sample cutoffs as shown (min genes 2,500 in all; naive mito &le; 20 and max 7,500; lag/late mito &le; 25 and max 8,000). "
            "Stage files (naive / lag / late) hold the two replicates of that stage with batch-aware log1p(CP10k) and PCA 50 (summary).", SM),
          table(rows, HDR, W)]
for x in j["integrated_h5ad_checks"]:
    if x.get("missing"):
        continue
    story += [P(f"{os.path.basename(x['file'])} - recomputed: counts layer = qc matrix rows: {x['counts_layer_equals_qc_mtx']}; X = log1p(CP10k) (max |diff| {x['X_equals_recomputed_log1p_CP10k_maxabs']:.1e}); {x['n_hvg_flag']} HVGs.", SM)]

# ---------------------------------------------------------------- LARRY
ls = json.load(open(f"{FIN}/LARRY_data/filtered/preprocessing_summary.json"))
lv = vj("LARRY")["per_sample"]["LSK_d2_1"]
ob = pd.read_csv(f"{FIN}/LARRY_data/filtered/obs_metadata.csv", index_col=0)
fin = ob.groupby("library").size()
story += [PageBreak(), P("LARRY (inDrops, mouse LSK, d2 / d4 / d6)", H1),
          P("Pipeline (current file, rebuilt 2026-10-16): 15 inDrops libraries combined (95,584 barcodes x 25,289 genes), gel barcodes mapped to real ACGT-ACGT; lineage barcodes = (library, cell, UMI, barcode) combinations supported by &ge; 10 reads (N_READS), "
            "then barcodes merged within Hamming distance 3 (the &ge; 3 UMIs per cell-barcode is applied later, by singletCode). <b>Whitelist</b> = the Weinreb et al. Table 1 minimum total UMIs per barcode for in vitro LSK: day 2 &ge; 500, day 4 and day 6 &ge; 300, applied to every library of that day. "
            "<b>singletCode</b> min_umi_cutoff = 3 on the whole sheet, then criterion-4 rescue with a cell identified by (library, cellID) (1,547 rescued units). <b>RNA QC</b>: mito &le; 20% (mt-), &ge; 200 genes, applied to the survivors. Final matrix = raw counts. "
            "Totals: raw 95,584 -> whitelist 85,735 -> mito 83,735 -> singletCode 33,362 -> gene floor 33,362 cells, 5,499 clones. "
            "Raw inputs (zips, barcode FASTQ) are the authors' files. <i>This replaces the earlier file (dip test + Otsu whitelist, rescue by cellID alone: 95,584 -> 89,813 -> 87,492 -> 32,728 -> 32,395 cells, 5,387 clones), which is archived in finalized_data/archive_2026-10-16_before_fullsheet_rebuild/LARRY_data/.</i>", B)]
rows = []
for lib, w in ls["whitelist_per_library"].items():
    rows.append([lib, f"&ge; {w['threshold']:.0f} (Table 1)", f(w["n_pass"]), f(fin.get(lib, 0)), "rebuilt from the zips; matches the file used"])
story += [table(rows, ["Library", "Whitelist: min total UMIs per barcode", "Cell barcodes passing whitelist", "Final cells", "Status / note"], [1.6, 1.8, 1.5, 0.9, 3.6]),
          P("Analysis units: libraries are merged to one sample per day (day2 = LSK_d2_1..3, day4 = d4_*, day6 = d6_*), clones shared across libraries by barcode string. "
            "Annotation: labels from the published LSK_d2_d4_d6.h5ad where the cell is present (30,466 cells), others by distance-weighted k-NN (2,896 cells; cross-validated accuracy 0.868 overall, 0.915 / 0.886 / 0.848 for day 2 / 4 / 6); the classifier does not see clone identity.", SM)]

# ---------------------------------------------------------------- Michaels
story += [PageBreak(), P("Michaels et al. (human; GEO GSE248549)", H1)]
rows, j = rows_from_verify("Michaels_T1T2")
story += [P("Michaels T1 / T2 (clone labels from yscher's cell_to_clone.csv)", H2),
          P("Raw GSM7916614 (T1), GSM7916615 (T2). QC = yscher's (mito &le; 20%, &ge; 200 genes) - <b>rebuilt from the raw GEO matrices and compared with yscher's michaels_T1_qc.h5ad / michaels_T2_qc.h5ad: identical</b> (obs names and order, var names and order, every X value, n_genes / pct_mt / total_counts; T1 8,478 and T2 5,747 cells). 4,118 cells carry a clone label "
            "(T1 1,897; T2 2,221; 312 clone ids in both samples). <b>Clone labels</b> come from yscher's own barcode pipeline (helpers/michaels_barcodes.py), not singletCode: lineage barcodes are read from the TLB amplicon FASTQs (pattern AT-NNNN-CT-NNNN-AC-NNNN-TC-NNNN-GT), filtered by the paper's rules (UMI-level merge, consensus rescue, cell in the 10x matrix, barcode in &gt; 1 cell, &ge; 2 UMIs per cell-barcode pair, &ge; 50% of the cell's top barcode), cells sharing a barcode are linked in a graph and clones are the connected components after min-cut splitting and degree pruning. <b>Status: rerun 2026-10-16. Steps 1-7 reproduce yscher's log exactly; step 8 depends on the hash seed and is not exactly reproducible (see the validation page).</b>", SM), table(rows, HDR, W)]

# ---------------------------------------------------------------- hPSC (yscher, undifferentiated, Novo)
ex = json.load(open(EXTRA))
story += [PageBreak(), P("hPSC datasets (see the validation page for what was rebuilt)", H1),
          P("hPSC_20260927 (our own build) is not listed: removed from this analysis at your request.", SM)]
H2s = ["Sample", "Cells in final file", "Cells with clone label", "Median genes", "Median UMI", "Median mito %", "Observed min genes / max mito %", "Status"]
W2 = [1.2, 0.9, 0.9, 0.7, 0.7, 0.7, 1.4, 3.4]
lab = pd.read_csv(f"{OUT}/all_samples_list.csv")
cl = {(r.dataset, r["sample"]): r.n_cells_with_clone for _, r in lab.iterrows()} if False else {(r["dataset"], r["sample"]): r["n_cells_with_clone"] for _, r in lab.iterrows()}
rows = []
for s, v in ex["hPSC_endo_yscher"].items():
    rows.append([f"hPSC_endo_yscher {s}", f(v["n"]), f(cl.get(("hPSC_endo_yscher", s))), f(v["genes"]), f(v["umi"]), f"{v['mito']:.1f}", f"{f(v['gmin'])} / {v['mito_max']:.1f}",
                 "<b>Counts = Cell Ranger integer counts</b> (endo_counts_mito20_min200genes.h5ad, 26,859 cells x 38,606 genes; Cell Ranger run by Hanxiao, not rerun here); the per-cell numbers in this row are from that file. yscher's build (build_endo_dataset.py, README_PROVENANCE): Cell Ranger filtered cells with mito &lt; 20% and &ge; 200 genes. "
                 "clone_c5 = singletCode(cutoff 5, ratio 0, dom 30, diff 0) on BC50StarcodeD8, per sample, per yscher's scripts. The barcode chain (lane concatenation, stepOne-Three, yscher's build) was rebuilt from the barcode FASTQs and is identical (see validation page); the RNA FASTQ -> Cell Ranger step was not rerun. The kb-python spliced / unspliced layers (separate h5ad) are not used."])

for s, v in ex["hPSC_undifferentiated"].items():
    rows.append([f"hPSC_undifferentiated {s}", f(v["n"]), f(cl.get(("hPSC_undifferentiated", s))), f(v["genes"]), f(v["umi"]), f"{v['mito']:.1f}", f"{f(v['gmin'])} / {v['mito_max']:.1f}",
                 "<b>QC recomputed 2026-10-16</b> from the Cell Ranger filtered matrices (T0 cellranger_out/T0h; T10, T15 cellranger_10x): mito &lt; 20% and &ge; 200 genes give exactly these cells (13,713 / 18,032 / 23,216), and 900 of 900 sampled count vectors are identical. X in the final file is log1pPF; the integer counts used for the PCA are in the sibling twinfer2026_run1plus2_counts file (same cells). clone_c5 (singletCode, cutoff 5, ratio 0, dominant 30, diff 0, on the starcode table of Hanxiao's sidereaction_merged_2runs, copied to real_data/hPSC_undifferentiated_barcodes/) <b>rerun here: identical</b> in all three timepoints; the stepThree tables themselves were not regenerated."])
for s, v in ex["hPSC_Novo_Nordisk"].items():
    rows.append([f"hPSC_Novo_Nordisk {s}", f(v["n"]), f(cl.get(("hPSC_Novo_Nordisk", s))), f(v["genes"]), f(v["umi"]), f"{v['mito']:.1f}", f"{f(v['gmin'])} / {v['mito_max']:.1f}",
                 "final only (authors' Seurat object, Linneberg-Agerholm 2026). Only clone-labelled cells are in our filtered files (41,454 of 76,451). Authors' QC cutoffs are not stated in the object; the observed ranges are shown. Raw lineage tags are not in the object."])
story += [table(rows, H2s, W2)]

# ---------------------------------------------------------------- final-only others
story += [PageBreak(), P("Datasets with final data only (CellTag3, SpaceBar, DTT, PeTrace)", H1),
          P("For these no raw matrix is available locally, so filters could not be reconstructed. Shown: the final cells and metrics recomputed from the final matrix (min / median / max).", B)]
H3 = ["Sample", "Cells", "Genes detected (min / med / max)", "UMI (min / med / max)", "Mito % (min / med / max)", "What is known about its processing"]
W3 = [1.3, 0.7, 1.5, 1.6, 1.4, 3.7]


def mm(v):
    return "-" if v is None else f"{f(v[0])} / {f(v[1])} / {f(v[2])}"


def mm1(v):
    return "-" if v is None else f"{v[0]:.1f} / {v[1]:.1f} / {v[2]:.1f}"


rows = []
notes = {"CellTag3": "From yuval.scher's folder, built by yscher's scripts from the GEO matrices (celltag_clones_repaired_nopad.h5ad): X is normalized (values 0.05-8.6); the integer counts of the same cells are in celltag_clones_repaired_nopad_counts.h5ad (median total counts 25,053, max mito 20.0%). The per-cell metrics in this table are from the normalized file and are therefore not UMI counts. Clones: published and repaired labels; 2 timepoints (day 2.5 / 5).",
         "SpaceBar": "Spatial, targeted panel of 114 genes common to the 5 sections (median 28 genes per cell); cells with 0 counts are present; no mito genes in the panel. Clone = barcode cluster (bc_cluster); 191,373 of 314,153 cells have one.",
         "DTT_mouse": "counts_samecell_analysis.npz built from GSE341627 (our subset: 84,718 cells = strict sister pairs, cousins, singletons); 37 mito genes. Clone information = sister pairs, not clone ids.",
         "PeTrace": "Authors' .h5td per replicate (E7.5-E9.5, R1-R3), integer counts, 37 mito genes; Merged per day in the analysis; clonal pair = sister cells (leaves under the same parent node of the lineage tree). The files\' clone column is a whole tree and is NOT used as a clone. E9.5-R1/R2/R3 re-downloaded 2026-10-15 and verified by MD5."}
for d in ["CellTag3", "SpaceBar", "DTT_mouse"]:
    j = vj(d)
    for s, v in j["per_sample"].items():
        rows.append([f"{d} {s}", f(v["n_cells"]), mm(v.get("n_genes")), mm(v.get("total_counts")), mm1(v.get("pct_counts_mt")), notes[d] if s == list(j["per_sample"])[0] else "(same as above)"])
for st in ["E7.5", "E8.0", "E8.5", "E9.0", "E9.5"]:
    for r in (1, 2, 3):
        d = f"PeTrace_{st}-R{r}"
        if not os.path.exists(f"{OUT}/{d}/verify.json"):
            continue
        j = vj(d)
        for s, v in j["per_sample"].items():
            rows.append([f"PeTrace {s} (capture of {st}-R{r}.h5td)", f(v["n_cells"]), mm(v.get("n_genes")), mm(v.get("total_counts")), mm1(v.get("pct_counts_mt")),
                         notes["PeTrace"] if (st, r) == ("E7.5", 1) else "(same as above)"])
story += [table(rows, H3, W3)]

# ---------------------------------------------------------------- normalization + PCA
# ---------------------------------------------------------------- detailed methods for non-standard processing
story += [PageBreak(), P("Detailed description of the non-standard processing (what exactly was done, and why)", H1),
 P("These datasets were not processed by a standard function (Cell Ranger + singletCode). Each paragraph says what the files are, what we did to them, and the reason. 'Why' is given only where it is recorded in the code, handoff notes or the source papers; otherwise it says so.", B),
 P("LARRY (pilot, mouse LSK, inDrops)", H2),
 P("<b>Source.</b> 15 inDrops libraries (LSK_d2_1..3, d4_*, d6_*) from the LARRY resource; raw counts per library are zip files (e.g. LSK_d2_1.zip), plus lineage-barcode reads (LARRY_sorted_and_filtered_barcodes.fastq.gz); all provided by the authors. "
   "<b>Steps.</b> (1) Libraries concatenated (95,584 cells x 25,289 genes); inDrops gel barcodes mapped to real ACGT-ACGT cell ids. "
   "(2) Lineage barcode table (build_larry_matrix.py): reads are counted per (library, cell, UMI, barcode) combination and only combinations supported by &ge; 10 reads (N_READS) are kept - the read-support filter of the Weinreb et al. notebook (histogram of log10 reads per combination, cutoff N_READS; 'retaining X of Y (Sample, Cell-BC, UMI, GFP-BC) combinations'); barcodes within Hamming distance 3 are then merged (transitive components, each collapsing to its highest-UMI member). The &ge; 3 UMIs per cell-barcode is not applied here (that step is commented out in the script); it is applied afterwards by singletCode (min_umi_cutoff 3). "
   "(3) <b>Cell whitelist</b> (a cell-calling step is needed because the zip files are used as given and no called-cell list comes with them; Cell Ranger is not involved for inDrops): the minimum total UMIs per barcode of Weinreb et al. Table 1 for in vitro LSK - day 2 500, day 4 300, day 6 300 - applied to each library of that day. These thresholds were chosen by the authors from histograms of total transcripts per barcode. <b>Why this choice</b> (K. Arun, 2026-10-16): the earlier whitelist of this repo, dip test + Otsu per library (a principled automatic substitute for cutoffs the authors set visually and did not share), kept every non-zero barcode in 12 of 15 libraries and cut at 205 / 1,257 / 1,217 counts in LSK_d2_1 / LSK_d6_1_1 / LSK_d6_1_2; once Table 1's numbers were available they were used instead. <b>Check:</b> these thresholds give 85,735 whitelisted barcodes in total, exactly the cell count the repo's first (knee-point) whitelist had been tuned to, taken from the paper's own PDF; the median UMIs of barcodes at or above the thresholds are 1,802 / 2,005 / 2,181 for day 2 / 4 / 6 against 1,720 / 1,993 / 2,211 in Table 1. Compared with the dip + Otsu file, the switch changes 1.3% of the final cells (33,362 vs 32,941 in the same-rescue run; 32,421 in both) and moves day-2 clones of size &ge; 2 from 1,114 to 1,025 (twin pairs 3,491 to 2,872), day 4 from 1,968 to 1,957 and day 6 from 2,305 to 2,357. The only raw inputs are the 15 per-library zip files and the barcode file, all from the authors; no GEO inDrops file is used anywhere. "
   "(4) singletCode with min_umi_cutoff = 3 (default 2) - chosen to match the threshold in Weinreb et al. 2020 Methods 3.3 (build_final_matrix_umi3.py), then the repo's criterion-4 rescue by barcode combination across libraries with a cell identified by (library, cellID): 1,547 rescued units (1,405 distinct cellIDs). The earlier version pooled rows by cellID alone, which wrongly rescued 12 final cells and missed 567 (rescue check, job 4036965). "
   "(5) mito (mt-) &le; 20% and &ge; 200 genes on survivors. Clone id = sorted concatenation of the singlet's unique barcodes; the same string in different libraries is the same clone. The Weinreb et al. rule (ii)/(iii) is not applied. "
   "(6) Annotation: the published LSK_d2_d4_d6.h5ad labels for cells present in it (30,466), the remaining 2,896 cells by a distance-weighted k-NN on the expression embedding (file filtered_annotated_k15.h5ad); clone identity is not used by the classifier. "
   "<b>Pair-distance analysis.</b> Libraries of one day are merged to one sample per day (as stated by you); clonal pair = two cells with the same clone string in that day. "
   "<b>Verified here:</b> stage 1 (matrix + umi table from the zips and barcode FASTQ) was re-run and matches the stage-1 files exactly; stage 2 and the annotation were run for the current file in job 4053678 and the files in finalized_data are copies of that output (the previous file was also reproduced exactly, with the Weinreb rule step off). The 15 zips and the barcode FASTQ (LARRY_sorted_and_filtered_barcodes.fastq.gz) are the authors' files.", SM),
 P("DTT mouse (Shendure lab, GEO GSE341627)", H2),
 P("<b>Source.</b> 7 GSM sparse count matrices (78,258 unnamed columns) plus the authors' repository (lineage tree with 655,701 tips, LSD2-dated, CRISPR-tape edit data). "
   "<b>Gene names.</b> Column j = row j of the authors' df_gene (mm39) table; this is the same order as the first 78,258 rows of mouse.v37.geneID.txt. It was verified with markers (Alb 603x in hepatocytes; Pecam1/Cdh5/Ptprc ~0 in neural cells); GEO's df_gene.csv.gz is NOT the column list (it has 54,653 rows). "
   "<b>Clone definition.</b> There are no clone ids. A <b>sister pair</b> = a 'cherry' (internal tree node with exactly two leaf children) whose parent is dated &ge; E12.5, both cells in the 655,701-tip backbone: 318,598 cherries, 221,565 after the age filter, 32,146 'strict' pairs (16,294 same cell type, 15,852 different) - these numbers reproduce the authors' handout exactly. Cousins = leaf children of the uncle nodes. "
   "<b>Cell subset.</b> raw counts were extracted only for the 135,279 cells needed for the sister/cousin analyses; counts_samecell_analysis.npz = 84,718 cells (32,588 twin cells + 52,130 singletons), integer counts, no normalization stored. "
   "<b>Why.</b> The strict-sister definition and the &ge; E12.5 parent-age cut are what reproduces the handout numbers above; the biological reason for E12.5 is not recorded in our files. "
   "<b>Pair-distance analysis.</b> Clonal pairs = the 32,146 strict sister pairs (homotypic pairs per cell type); random pairs = random cell pairs from the same GSM file (sample), chosen because 17% of sister pairs share a file and other sample definitions gave 0.5-7%. Heterotypic sisters are ignored in per-cell-type plots. "
   "Dating of the tree (LSD2, root day 1.5, tips E13.5, literature cell-count minimum-age ladder) was not re-run; a '2-hour minimum branch' floor in the dating was my guess and is not confirmed.", SM),
 P("PeTrace (mouse embryo E7.5-E9.5; .h5td files from Zenodo)", H2),
 P("<b>Source.</b> One .h5td (HDF5 AnnData-like) file per day and replicate (E7.5, E8.0, E8.5, E9.0, E9.5 x R1-R3), 78,258 genes in every file, integer counts, 37 mito genes; E9.5-R1/R2/R3 were re-downloaded (MD5 verified) because the first copies were incomplete. "
   "<b>What the 'clone' column is.</b> It is a whole lineage tree, not a clone (E9.5-R1: 382k cells in 3 trees). Using it gives only a handful of 'clones' per day - this was my first, wrong, version and was replaced. "
   "<b>Clonal pair definition used.</b> The tree itself ('obst' field, JSON of parent -> children edges): leaf cells that are children of the same parent node are sister cells; all pairs among the leaves of one parent are clonal pairs. Pairs are taken per day with R1, R2, R3 pooled. "
   "<b>Why.</b> Sister cells are the closest relationship the tree gives and they are comparable to the DTT sister-pair definition. "
   "<b>Random pairs</b> = random pairs of cells from the same day (and, per cell type, the same cell_type label), not matched for capture/replicate - a known limitation. Heterotypic sister pairs ignored in per-cell-type plots. "
   "<b>Pre-processing.</b> The files hold the authors' already filtered and annotated cells (obs: capture, embryo, donor / host, cell type, cluster, lineage, germ layer, tree; obsm: X_scvi, X_umap and edit characters; no PCA is stored). We apply no QC and could not redo it, because no raw FASTQ / Cell Ranger output is local, so the authors' QC cannot be verified. The PCA is ours: computed from the integer counts by the common recipe on the day-merged matrix, for consistency with the other datasets; the authors' scVI latent and UMAPs are not used. Tree structure: the audit covers 7 of 15 files (see the validation page); the sister-pair analysis of the other 8 was done without it.", SM),
 P("CellTag3 (built by yuval.scher's scripts from the GEO matrices; clone caller = the authors' functions)", H2),
 P("<b>Source.</b> celltag_clones_repaired_nopad.h5ad (2.2 GB, md5 6c39651f209805181a1d352eb5e16f49), from yuval.scher's folder. <b>How it was made</b> (yscher's helpers/ scripts; all steps were rerun here from the GEO Cell Ranger matrices + all_samples.csv + the authors' master table and reproduce yscher's intermediates and the final file exactly): (1) build_nosinglet_dataset.py + finalize_nosinglet.py: combine the samples, no singlet filter; (2) celltag_apply_qc.py: mito &le; 20%, &ge; 200 genes; (3) build_corrected_clones.py: pairwise Jaccard overlap of CellTag sets on cells with 2-24 tags, clones = connected components of pairs with Jaccard &ge; 0.6 (authors' sf_utils_morris.call_clones); (4) build_celltag_schema_clones.py: the h5ad schema with these clone labels; (5) celltag_clones_repaired.py: the authors' sparse-clone repair (ident_sparse_clones / fix_sparse_clones: among the largest clones, those whose subgraph density is below 0.2 are chains and are split) and build_celltag_repaired_schema.py: writes the final file. <b>Steps (1)-(5) were all rebuilt and reproduce the file we use exactly</b> (final md5 6c39651f209805181a1d352eb5e16f49). In this file X is normalized (values 0.05-8.6, 31,055 genes) and the PCA so far used X as shipped (no normalize/log1p). <b>The same cells also exist as integer Cell Ranger counts</b> in celltag_clones_repaired_nopad_counts.h5ad (54,657 cells x 25,616 genes with a count; identical cell order and identical repaired_clone labels; recomputed from those counts: median total counts 25,053, min 512, max 192,421; min 239 genes per cell; 13 MT- genes; max mito 20.0%). "
   "<b>Clone definition.</b> obs column repaired_clone (the published_clone labels after the sparse-clone repair; is_padding flags padded rows). Samples = 'Time point' (day 2 and day 5; experimental day 2.5/5). Cell type = 'Cell type annotation'. Cells with NaN clone have no clone label. "
   "<b>Why.</b> The repaired labels were chosen over published_clone (the reason is documented in yscher's celltag_clones_repaired.py header: connected components chain A-B-C even when Jaccard(A,C) &lt; 0.6). Cross-timepoint clonal pairs (same clone at day 2 and day 5) are not counted in the per-sample plots.", SM),
 P("SpaceBar (spatial, targeted gene panel)", H2),
 P("<b>Source.</b> 5 tissue sections, Section{i}_cell_by_gene_clustered.csv (object_id, gene panel, barcode-call columns, bc_cluster). "
   "<b>What was done.</b> The gene panel was intersected over the 5 sections (114 common genes; median 28 detected per cell); the matrix is used as shipped (counts per segmented cell) and, because the method is smFISH-like, analysed as raw numbers: no normalization, log1p, HVG selection or scaling (K. Arun, 2026-10-16); the PCA runs on all 114 genes. "
   "<b>Clone definition.</b> bc_cluster = the authors' clustering of the barcode calls; cells with the same bc_cluster in a section are one clone (191,373 of 314,153 cells have one). No QC could be applied (no mito genes in the panel, cells with 0 counts present).", SM),
 P("Michaels et al. T1 / T2 clone labels", H2),
 P("Clone labels are from yuval.scher's cell_to_clone.csv (copied to finalized_data/Michaels_T1T2_data with a provenance README). They were written by yscher's helpers/michaels_barcodes.py (yscher's own graph-based pipeline, not singletCode); that script was re-run: steps 1-7 reproduce yscher's log exactly, step 8 is hash-seed dependent (20 seeds: 1,359-1,372 clones vs yscher's 1,371), so the labels are used as given and are not exactly reproducible. QC (mito &le; 20%, &ge; 200 genes): the two QC'd h5ad files were rebuilt from raw and are identical to yscher's.", SM)]

story += [PageBreak(), P("Normalization and PCA done for this analysis (all datasets, identical recipe)", H1),
          P("Raw counts -> normalize_total(1e4) -> log1p -> highly_variable_genes(n_top_genes = 5,000, flavor 'seurat', no batch key) -> scale(max_value = 10) on the 5,000-gene copy -> PCA 50 PCs (arpack, seed 0). "
            "Exceptions: CellTag3 (its X is already log1p(CP10k) - expm1(X) sums to 10,000 in every cell checked, and build_celltag_schema_clones.py normalizes to 1e4 then log1p - so normalize_total and log1p are skipped and the normalization is the same), SpaceBar (smFISH-like targeted panel of 114 genes: raw counts as is, PCA on all 114 genes - no normalization, log1p, HVG selection or scaling, by decision of K. Arun). Source of counts per dataset in the 'source' column. Elbow = the PC with the largest distance below the chord of the log variance-ratio curve (descriptive only; "
            "pair distances use all 50 PCs).", B)]
rows = []
for fpath in sorted(os.listdir(f"{OUT}/pca5k")):
    if fpath.startswith("summary_") and fpath.endswith(".json"):
        j = json.load(open(f"{OUT}/pca5k/{fpath}"))
        if j["dataset"] == "hPSC_20260927" or (j["dataset"].startswith("PeTrace_") and "-R" in j["dataset"]):
            continue                      # removed dataset / per-replicate PeTrace (listed merged per day)
        rows.append([j["dataset"], f(j["n_cells"]), f(j["n_genes_expressed"]), f(j["n_hvg"]), f"PC{j['elbow_pc']}", f"{100 * j['pc1_ratio']:.1f}", j["source"]])
story += [table(rows, ["Dataset", "Cells", "Genes expressed", "HVGs", "Elbow", "PC1 % var", "Source of counts"], [1.7, 0.8, 0.9, 0.6, 0.6, 0.7, 4.9]),
          P("PeTrace is listed merged per day (R1 + R2 + R3); per-replicate PCAs were also computed but are not listed.", SM)]

# ---------------------------------------------------------------- caveats
story += [P("Not verified in this document", H2),
          P("&bull; FM01-08, Watermelon and HS054: the build scripts (incl. singletCode and the rescue) were re-run and give byte-identical files; the criterion-4 rescue pools by cellID across samples (check pending). "
            "&bull; LARRY: stages 1 and 2 and the annotation were rebuilt and match (stage 2 without the Weinreb rule); the raw zips and barcode FASTQ are the authors' files and cannot be re-derived here. "
            "&bull; hPSC_endo_yscher (rebuilt from FASTQ, identical) and Novo (rebuilt from the Seurat export, identical) are validated; hPSC_undifferentiated (QC recomputed from the Cell Ranger matrices: identical cells and counts), CellTag3 (whole chain rebuilt from the GEO Cell Ranger matrices; final file md5-identical), SpaceBar (authors' files), DTT (sister-pair tables and counts rebuilt and identical, from the GEO matrices and the authors' tree) and PeTrace (md5 = Zenodo; tree audit partial) have no raw-to-final filter reconstruction here. "
            "&bull; Michaels T1/T2 clone labels: yscher's script re-run, step 8 not exactly reproducible. &bull; Inconsistent settings across datasets (FM06/FM08/LARRY min_umi_cutoff 3 vs auto elsewhere; mito 20-26%; gene ranges differ) are as hard-coded in the build scripts.", B)]

doc = SimpleDocTemplate(PDF, pagesize=landscape(letter), leftMargin=0.45 * inch, rightMargin=0.45 * inch, topMargin=0.45 * inch, bottomMargin=0.45 * inch,
                        title="Pre-processing of every sample", author="Keerthana Arun")
doc.build(story)
print("saved", PDF)
