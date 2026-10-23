#!/usr/bin/env python3
"""PDF with the pre-processing of every sample: source, raw cells, each filter with its cutoff and the number of cells it removed
(recomputed from the raw matrices wherever raw data exist), final cells, and what is known / not known otherwise.

Inputs : <OUT>/<dataset>/verify.json (qc_verify.py), <OUT>/pca5k/summary_*.json, <OUT>/all_samples_list.csv,
         finalized_data/LARRY_data/filtered/preprocessing_summary.json, a small extra-metrics json (yscher/Novo observed ranges, see EXTRA).
Output : <OUT>/preprocessing_per_sample.pdf
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
PDF = f"{OUT}/preprocessing_per_sample.pdf"

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

# ---------------------------------------------------------------- common pipeline
story += [P("Common pipeline (FateMap-style datasets: FM01-08, HS054, Watermelon)", H2),
          P("1) RNA: Cell Ranger filtered (called-cell) matrices; gene symbols as var_names; barcode suffix (-1 / aggr -N) stripped to match the 16-bp cellID of the lineage table. "
            "2) Lineage barcodes: starcode-collapsed reads (BC50StarcodeD8; hPSC D7) to a sample sheet (cellID, barcode, sample), no manual UMI pre-filter. "
            "3) singletCode criteria 1-3 (single barcode; dominant barcode by UMI; barcode combination recurs in other cells of the sample), then this repo's criterion-4 cross-sample rescue; "
            "clone ID = sorted concatenation of the singlet's barcodes. 4) RNA QC: mito % = counts in MT- genes / total counts (13 MT- genes, human); keep mito &le; cutoff and min &le; genes detected &le; max. "
            "Final cells = pass mito AND pass gene range AND singlet. 5) Final count matrix = raw integer counts (no normalization stored). "
            "Cutoffs and singletCode UMI cutoffs differ by dataset (below); they are the values hard-coded in the build scripts. <b>Source of the numbers:</b> mito &le; 26%, 200-7200 genes (FM01/03/05/07/08), mito &le; 21% and 200-5000 genes (FM06) are constants in build_fm01_qc_matrix.py / build_fm_variants_qc_matrix.py / the FM06-FM08 build (handoff 2026-09-22). No justification (paper citation or distribution-based rule) is recorded anywhere in the repo; I could only verify that the final cells obey them, not why they were chosen. "
            "<b>singletCode auto UMI cutoff</b> (min_umi_cutoff not passed): per sample, cutoff = max(min_umi_cutoff default 2, round(3/4e5 x number of barcode rows in that sample of the sheet)); a barcode needs at least that many UMIs to count. It therefore scales with sequencing depth of the barcode library (2 for small sheets, 5-12 here).", B)]

# ---------------------------------------------------------------- FateMap
fm = [("FM01", "WM989 + 1 uM PLX; replicates A, B (merged = one sample). RNA GSM7434407/408. singletCode: min_umi_cutoff auto = max(2, round(3/4e5 x barcode rows of the sample)) = 6 (A, 808,042 rows) and 5 (B, 694,802 rows) per the build log fm01_qc_7181595.out; 28 cellIDs rescued by criterion 4 (summary)."),
      ("FM03", "WM989 DMSO + 1 uM PLX; one replicate (A). RNA GSM7434413. singletCode auto cutoff = 12 (build log qc_3636305_0: 3,280,180 sheet rows, run on the whole sheet incl. the replicate without RNA)."),
      ("FM05", "WM989 250 nM PLX; A1, A2. RNA GSM7434417/418. singletCode auto cutoff."),
      ("FM06", "WM989 naive; A, B (merged = one sample). RNA GSM7434419/420. singletCode min_umi_cutoff = 3; 38 cellIDs rescued (summary)."),
      ("FM07", "WM989 continuous 1 uM PLX; A, B. RNA GSM7434421/422. singletCode auto cutoff = 5 (A) and 12 (B) (build log qc_3636305_1)."),
      ("FM08", "Primary melanocytes; A, B (merged = one sample). RNA GSM7434423/424. singletCode min_umi_cutoff = 3; SampleNum 3 (no RNA) dropped; 22 cellIDs rescued (summary).")]
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
            "(logged cutoff 2 in every sample); 572 cellIDs rescued by criterion 4 (summary). Per-sample cutoffs as shown (min genes 2,500 in all; naive mito &le; 20 and max 7,500; lag/late mito &le; 25 and max 8,000). "
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
          P("Pipeline (from the build summary): 15 inDrops libraries combined (95,584 cells x 25,289 genes), gel barcodes mapped to real ACGT-ACGT; lineage barcodes = (cellID, barcode, UMI) supported by &ge; 10 reads and &ge; 3 UMIs, "
            "barcodes merged within Hamming distance 3. <b>Whitelist</b> per library: Hartigan dip test on log10 total counts; if bimodal (p &lt; 0.05) cut at Otsu's threshold, otherwise keep every non-zero barcode. "
            "<b>singletCode</b> min_umi_cutoff = 3, then criterion-4 rescue (957 cellIDs). <b>RNA QC</b>: mito &le; 20% (mt-), &ge; 200 genes, applied to the survivors. Final matrix = raw counts. "
            "Totals (summary, not re-run): raw 95,584 -> whitelist 89,813 -> mito 87,492 -> singletCode 32,728 -> gene floor 32,395 cells, 5,387 clones. "
            "<b>Pilot recomputed from the raw zip for LSK_d2_1 only</b> (the other 14 libraries were not recomputed).", B)]
rows = []
for lib, w in ls["whitelist_per_library"].items():
    rows.append([lib, f"{w['threshold']:.0f} ({'bimodal' if w['bimodal'] else 'not bimodal'})", f(w["n_pass"]), f(fin.get(lib, 0)), "summary"])
rows.append([f"<b>LSK_d2_1 recomputed from zip</b>", f"> 205 total counts", f(lv["n_pass_pre: whitelist: total counts > 205.0 (stated Otsu threshold)"]) if "n_pass_pre: whitelist: total counts > 205.0 (stated Otsu threshold)" in lv else "8,148",
             f(lv["n_final"]), f"raw {f(lv['n_raw'])}; mito removed {f(lv['n_removed_by_mito_after_pre'])}; non-singlet {f(lv['n_removed_as_non_singlet'])}; all {f(lv['n_final'])} final cells found in raw: {lv['n_final_not_found_in_raw'] == 0}"])
story += [table(rows, ["Library", "Whitelist threshold (total counts)", "Cells passing whitelist", "Final cells", "Status / note"], [1.6, 1.8, 1.5, 0.9, 3.6]),
          P("Analysis units: libraries are merged to one sample per day (day2 = LSK_d2_1..3, day4 = d4_*, day6 = d6_*), clones shared across libraries by barcode string. "
            "Annotation: labels from the published LSK_d2_d4_d6.h5ad where the cell is present (29,351 cells), others by distance-weighted k-NN (3,044 cells); the classifier does not see clone identity.", SM)]

# ---------------------------------------------------------------- Michaels
story += [PageBreak(), P("Michaels et al. (human; GEO GSE248549)", H1)]
rows, j = rows_from_verify("Michaels_T1T2")
story += [P("Michaels T1 / T2 (clone labels from yscher's cell_to_clone.csv)", H2),
          P("Raw GSM7916614 (T1), GSM7916615 (T2). QC = yscher's (mito &le; 20%, &ge; 200 genes) - <b>rebuilt from the raw GEO matrices and compared with her michaels_T1_qc.h5ad / michaels_T2_qc.h5ad: identical</b> (obs names and order, var names and order, every X value, n_genes / pct_mt / total_counts; T1 8,478 and T2 5,747 cells). 4,118 cells carry a clone label "
            "(T1 1,897; T2 2,221; 312 clone ids in both samples). <b>Clone labels</b> come from yscher's own barcode pipeline (helpers/michaels_barcodes.py), not singletCode: lineage barcodes are read from the TLB amplicon FASTQs (pattern AT-NNNN-CT-NNNN-AC-NNNN-TC-NNNN-GT), filtered by the paper's rules (UMI-level merge, consensus rescue, cell in the 10x matrix, barcode in &gt; 1 cell, &ge; 2 UMIs per cell-barcode pair, &ge; 50% of the cell's top barcode), cells sharing a barcode are linked in a graph and clones are the connected components after min-cut splitting and degree pruning. <b>Status: described from reading her script; the re-run (job 3944546) is not finished, so these steps are NOT validated yet.</b>", SM), table(rows, HDR, W)]

# ---------------------------------------------------------------- hPSC (yscher, undifferentiated, Novo)
ex = json.load(open(EXTRA))
story += [PageBreak(), P("hPSC datasets (no raw matrices available locally for these builds)", H1),
          P("hPSC_20260927 (our own build) is not listed: removed from this analysis at your request.", SM)]
H2s = ["Sample", "Cells in final file", "Cells with clone label", "Median genes", "Median UMI", "Median mito %", "Observed min genes / max mito %", "Status"]
W2 = [1.2, 0.9, 0.9, 0.7, 0.7, 0.7, 1.4, 3.4]
lab = pd.read_csv(f"{OUT}/all_samples_list.csv")
cl = {(r.dataset, r["sample"]): r.n_cells_with_clone for _, r in lab.iterrows()} if False else {(r["dataset"], r["sample"]): r["n_cells_with_clone"] for _, r in lab.iterrows()}
rows = []
for s, v in ex["hPSC_endo_yscher"].items():
    rows.append([f"hPSC_endo_yscher {s}", f(v["n"]), f(cl.get(("hPSC_endo_yscher", s))), f(v["genes"]), f(v["umi"]), f"{v['mito']:.1f}", f"{f(v['gmin'])} / {v['mito_max']:.1f}",
                 "final only. yscher's build: mito &lt; 20%, &ge; 200 genes (README_PROVENANCE). clone_c5 = singletCode(cutoff 5, ratio 0, dom 30, diff 0) on BC50StarcodeD8 per her scripts; "
                 "<b>not independently confirmed for this file</b>. Layers: spliced / unspliced / mature / ambiguous (kb-python nac)."])
for s, v in ex["hPSC_undifferentiated"].items():
    rows.append([f"hPSC_undifferentiated {s}", f(v["n"]), f(cl.get(("hPSC_undifferentiated", s))), f(v["genes"]), f(v["umi"]), f"{v['mito']:.1f}", f"{f(v['gmin'])} / {v['mito_max']:.1f}",
                 "final only. File name states mito &lt; 20%, &ge; 200 genes, log1pPF normalization in X; raw counts taken from the sibling twinfer2026_run1plus2_counts file for PCA (same cells). Clone call (clone_c5) not documented here."])
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
notes = {"CellTag3": "From yuval.scher's folder: authors' processed object (celltag_clones_repaired_nopad.h5ad): X is normalized (values 0.05-8.6), not counts; total counts and mito % cannot be recomputed. Clones: published and repaired labels; 2 timepoints (day 2.5 / 5).",
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
            rows.append([f"PeTrace {st}-R{r}", f(v["n_cells"]), mm(v.get("n_genes")), mm(v.get("total_counts")), mm1(v.get("pct_counts_mt")),
                         notes["PeTrace"] if (st, r) == ("E7.5", 1) else "(same as above)"])
story += [table(rows, H3, W3)]

# ---------------------------------------------------------------- normalization + PCA
# ---------------------------------------------------------------- detailed methods for non-standard processing
story += [PageBreak(), P("Detailed description of the non-standard processing (what exactly was done, and why)", H1),
 P("These datasets were not processed by a standard function (Cell Ranger + singletCode). Each paragraph says what the files are, what we did to them, and the reason. 'Why' is given only where it is recorded in the code, handoff notes or the source papers; otherwise it says so.", B),
 P("LARRY (pilot, mouse LSK, inDrops)", H2),
 P("<b>Source.</b> 15 inDrops libraries (LSK_d2_1..3, d4_*, d6_*) from the LARRY resource; raw counts per library are zip files (e.g. LSK_d2_1.zip), plus lineage-barcode reads. "
   "<b>Steps.</b> (1) Libraries concatenated (95,584 cells x 25,289 genes); inDrops gel barcodes mapped to real ACGT-ACGT cell ids. "
   "(2) Lineage barcode table: a (cell, barcode, UMI) record is kept only with &ge; 10 reads and &ge; 3 UMIs; barcodes within Hamming distance 3 merged. "
   "(3) <b>Cell whitelist per library</b> (replaces Cell Ranger's called-cell list, which inDrops does not have): Hartigan dip test on log10 total counts; if bimodal (p &lt; 0.05) cells above Otsu's threshold on that distribution are kept, otherwise every non-zero barcode. "
   "(4) singletCode with min_umi_cutoff = 3 (default 2) - chosen to match the threshold in Weinreb et al. 2020 Methods 3.3 (build_final_matrix_umi3.py), then the repo's criterion-4 cross-sample rescue (957 cellIDs). "
   "(5) mito (mt-) &le; 20% and &ge; 200 genes on survivors. Clone id = sorted concatenation of the singlet's unique barcodes; the same string in different libraries is the same clone. "
   "(6) Annotation: the published LSK_d2_d4_d6.h5ad labels for cells present in it (29,351), the remaining 3,044 cells by a distance-weighted k-NN on the expression embedding (file filtered_annotated_k15.h5ad); clone identity is not used by the classifier. "
   "<b>Pair-distance analysis.</b> Libraries of one day are merged to one sample per day (as stated by you); clonal pair = two cells with the same clone string in that day. "
   "<b>Verified here:</b> only LSK_d2_1 was re-derived from the raw zip (cells pass the stated whitelist, mito and range filters; all final cells found in raw). The other 14 libraries and the singletCode result are taken from the build summary.", SM),
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
   "<b>Pre-processing.</b> None re-done: counts used as shipped, normalised and PCA'd by our common recipe on the day-merged matrix. No raw FASTQ/Cell Ranger output is local, so the authors' QC cannot be verified.", SM),
 P("CellTag3 (authors' object, taken from yuval.scher's folder)", H2),
 P("<b>Source.</b> celltag_clones_repaired_nopad.h5ad (2.2 GB). X is normalized (values 0.05-8.6), NOT counts, so total counts and mito % cannot be recomputed and the PCA used X as shipped (no normalize/log1p). "
   "<b>Clone definition.</b> obs column repaired_clone (the authors' published_clone after repair of padding artefacts; is_padding flags padded rows). Samples = 'Time point' (day 2 and day 5; experimental day 2.5/5). Cell type = 'Cell type annotation'. Cells with NaN clone have no clone label. "
   "<b>Why.</b> The repaired labels were chosen over published_clone (the repair step is not documented in our files). Cross-timepoint clonal pairs (same clone at day 2 and day 5) are not counted in the per-sample plots.", SM),
 P("SpaceBar (spatial, targeted gene panel)", H2),
 P("<b>Source.</b> 5 tissue sections, Section{i}_cell_by_gene_clustered.csv (object_id, gene panel, barcode-call columns, bc_cluster). "
   "<b>What was done.</b> The gene panel was intersected over the 5 sections (114 common genes; median 28 detected per cell); the matrix is used as shipped (counts per segmented cell), normalised by our recipe (113 HVGs because only 114 genes exist). "
   "<b>Clone definition.</b> bc_cluster = the authors' clustering of the barcode calls; cells with the same bc_cluster in a section are one clone (191,373 of 314,153 cells have one). No QC could be applied (no mito genes in the panel, cells with 0 counts present).", SM),
 P("Michaels et al. T1 / T2 clone labels", H2),
 P("Clone labels are from yuval.scher's cell_to_clone.csv (copied to finalized_data/Michaels_T1T2_data with a provenance README). They were written by her helpers/michaels_barcodes.py (her own graph-based pipeline, not singletCode); that script is read but not yet validated by a re-run, so the labels are used as given. QC (mito &le; 20%, &ge; 200 genes): the two QC'd h5ad files were rebuilt from raw and are identical to hers.", SM)]

story += [PageBreak(), P("Normalization and PCA done for this analysis (all datasets, identical recipe)", H1),
          P("Raw counts -> normalize_total(1e4) -> log1p -> highly_variable_genes(n_top_genes = 5,000, flavor 'seurat', no batch key) -> scale(max_value = 10) on the 5,000-gene copy -> PCA 50 PCs (arpack, seed 0). "
            "Exceptions: CellTag3 (matrix already normalized, used as shipped), SpaceBar (only 114 genes, 113 HVGs). Source of counts per dataset in the 'source' column. Elbow = the PC with the largest distance below the chord of the log variance-ratio curve (descriptive only; "
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
          P("&bull; singletCode UMI cutoffs and the number of criterion-4 rescued cells are taken from the build summary / logs; singletCode was not re-run. "
            "&bull; LARRY: only library LSK_d2_1 was recomputed from raw; the other 14 libraries and all LARRY totals are from the build summary. "
            "&bull; hPSC_endo_yscher, hPSC_undifferentiated, Novo Nordisk, CellTag3, SpaceBar, DTT and PeTrace: no raw matrix, so no filter could be reconstructed. "
            "&bull; Michaels T1/T2 clone labels: made by yscher's script (not re-run). &bull; Inconsistent settings across datasets (FM06/FM08/LARRY min_umi_cutoff 3 vs auto elsewhere; mito 20-26%; gene ranges differ) are as hard-coded in the build scripts.", B)]

doc = SimpleDocTemplate(PDF, pagesize=landscape(letter), leftMargin=0.45 * inch, rightMargin=0.45 * inch, topMargin=0.45 * inch, bottomMargin=0.45 * inch,
                        title="Pre-processing of every sample", author="Keerthana Arun")
doc.build(story)
print("saved", PDF)
