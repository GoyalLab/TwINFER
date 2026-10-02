from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
OUT="/gpfs/projects/b1255/hzhang/TwINFER_KA/finalized_data/DATASET_CREATION_SUMMARY.pdf"
ss=getSampleStyleSheet()
B=ParagraphStyle("b",parent=ss["BodyText"],fontSize=9,leading=12)
S=ParagraphStyle("s",parent=B,fontSize=7.5,leading=9.5)
H1=ParagraphStyle("h1",parent=ss["Heading1"],fontSize=16)
H2=ParagraphStyle("h2",parent=ss["Heading2"],fontSize=12,spaceBefore=10)
def P(t,st=S): return Paragraph(t,st)
def tbl(rows,widths,hdr=True):
    data=[[P(str(c)) for c in r] for r in rows]
    t=Table(data,colWidths=widths,repeatRows=1)
    st=[("GRID",(0,0),(-1,-1),0.4,colors.grey),("VALIGN",(0,0),(-1,-1),"TOP"),
        ("LEFTPADDING",(0,0),(-1,-1),3),("RIGHTPADDING",(0,0),(-1,-1),3)]
    if hdr: st.append(("BACKGROUND",(0,0),(-1,0),colors.HexColor("#dde6f0")))
    t.setStyle(TableStyle(st)); return t
W=10*inch
d=SimpleDocTemplate(OUT,pagesize=landscape(letter),leftMargin=.5*inch,rightMargin=.5*inch,topMargin=.5*inch,bottomMargin=.5*inch,title="TwINFER dataset creation summary")
E=[]
E.append(P("TwINFER finalized_data: how each dataset was built",H1))
E.append(P("Source: build scripts in <i>code/TwINFER/paper_analysis/fatemap_pipeline/</i> (and the LARRY summary JSON) plus each dataset's <i>preprocessing_summary.json</i>. Values are the ones hard-coded in the scripts / written to the summaries.",B))
E.append(P("Common pipeline (all FateMap-style datasets)",H2))
for t in [
"<b>RNA input:</b> Cell Ranger <i>filtered</i> (called-cell) matrices; gene symbols as var_names; '-1' (or aggr '-N') suffix stripped so cell barcodes match the 16-bp cellID in the lineage-barcode table. No whitelisting step (LARRY only).",
"<b>singletCode call (fatemap_qc_utils.run_singlet_calling):</b> <font face='Courier'>singletCode.check_sample_sheet</font> then <font face='Courier'>singletCode.get_singlets(sample_sheet, dataset_name, [min_umi_cutoff=N])</font>, i.e. criteria 1-3 (single barcode; dominant barcode by UMI; barcode combination recurs in other cells of the same sample). When <i>min_umi_cutoff</i> is None the kwarg is not passed and singletCode uses its own default floor plus ratio-based auto cutoff.",
"<b>Criterion 4 (this repo's own rescue, singletcode_cross_sample_rescue.apply_cross_sample_rescue):</b> cells still labelled Multiplet are re-tested for barcode-combination recurrence across samples (singletCode computes this but never applies it). Rescued cellIDs are flagged in obs as <i>criterion4_rescued</i>. Clone ID = sorted, concatenated unique barcodes of the singlet (<i>fatemap_clone_singletcode</i>).",
"<b>Sample sheet:</b> columns cellID, barcode, sample. One row per (cellID, UMI, barcode) record with exact duplicates collapsed; <b>no manual UMI-count pre-filter</b> (thresholding left to singletCode). Barcode column BC50StarcodeD8 (FM, HS054), BC50StarcodeD7 (hPSC, pooled).",
"<b>Cell QC (apply_qc_filters):</b> mito % = MT- genes (case-insensitive prefix) / total counts, keep &le; cutoff; keep min_genes &le; n_genes (genes with count &gt; 0) &le; max_genes. Final cells = pass mito AND singlet AND pass gene range.",
"<b>Integrated h5ad (build_*_integrated.py, fatemap_integration_utils):</b> log1p(CP10k) on all cells together; 2000 HVGs (seurat default flavor, batch_key='replicate'); scale (max 10); PCA 50 comps (arpack, seed 0); scanorama and Harmony (harmonypy on PCA) both stored; neighbors+UMAP per embedding; Goyal et al. cluster annotation joined for FM01/06/08. Counts kept in layers['counts'], log-norm in .raw/layers['lognorm']."]:
    E.append(P("&bull; "+t,B))
E.append(P("Per-dataset QC and singletCode settings",H2))
rows=[["Dataset","System / samples","RNA source","Barcode table","Mito %","Genes (min-max)","singletCode min_umi_cutoff","Crit.4 rescued cellIDs"],
["FM01","WM989 + 1 uM PLX; replicates A,B (sampleNum 1,2; 3,4 dropped, no RNA)","GSM7434407/408 FM01_A/B","stepFourStarcodeShavedReads.txt (no UMI column: each row = 1 UMI record, given unique id)","&le; 26","200-7200","auto (not set); <b>log: cutoff 6 (rep A), 5 (rep B)</b>","28"],
["FM06","WM989 naive; A,B (SampleNum 1,2)","GSM7434419/420","stepThreeStarcodeShavedReads.txt","&le; 21","200-5000","<b>3</b>","38"],
["FM08","Primary melanocytes; A,B (SampleNum 1,2; SampleNum 3 dropped: 466 cellIDs, overlap 1/0 with RNA)","GSM7434423/424","stepThreeStarcodeShavedReads.txt","&le; 26","200-7200","<b>3</b>","22"],
["HS054_Sot48h","PDAC, sotorasib 48 h; single sample","Hanxiao/twinfer/10x/HS054_Sot48h filtered_feature_bc_matrix","stepThree/HS054_Sot48h/stepThreeStarcodeShavedReads.txt (no SampleNum)","&le; 25","200-7000","auto (not set); cutoff not found in logs","0"],
["Watermelon (T47D naive/lag/late x 2)","6 samples, all passed to singletCode in ONE call as one experiment","watermelon_data/{sample}/ (aggr suffix -1..-6 stripped)","Watermelon_barcode_cellID_UMIcounts.csv (already a sample sheet: cellID, barcode, sample; used as is)","naive &le; 20; lag, late &le; 25","min 2500 all; max 7500 (naive), 8000 (lag, late)","auto (not set); <b>log: cutoff 2 in all 6 samples</b>","572"],
["hPSC_20260927","hESC to definitive endoderm; endo_T0, endo_T1 (two timepoints)","cellranger_out/endo_T{0,1} filtered_feature_bc_matrix.h5","stepThreed7/pooled/stepThreeStarcodeShavedReads_POOLED.txt (BC50StarcodeD7 collapsed jointly so shared clones have identical strings across timepoints)","&le; 10","min 1000 (no max)","auto (not set); <b>log: cutoff 2 in both samples</b> (logs/hpsc_qc_sf_7984268.out)","16"],
["LARRY (inDrops, LSK d2/d4/d6)","15 libraries; raw inDrops matrices","LARRY_data/raw/*.zip","LARRY_sorted_and_filtered_barcodes.fastq.gz","&le; 20 (mt-, case-insens.)","&ge; 200","<b>3</b> (log: 3 in all 15 libraries) + criterion 4","957"]]
E.append(tbl(rows,[1.0*inch,1.6*inch,1.3*inch,1.8*inch,.8*inch,.9*inch,1.4*inch,.7*inch]))
E.append(P("<b>Order of operations (same for every dataset):</b> singletCode runs on the full barcode table, then RNA QC (mito, gene range) is intersected. hPSC was originally built QC-first (sheet restricted to QC-passing cells); it was rerun singletCode-first (job 7984268) and the output was byte-identical (same 6,454 cells, same clone strings), so nothing downstream changed. The earlier QC-first output is kept in hPSC_20260927_data/qc_filtered_LEGACY_qcfirst/.",B))
E.append(PageBreak())
E.append(P("Cell counts through each filter",H2))
rows=[["Dataset","Raw cells","After mito","After singletCode + crit.4","After gene range (final)","Final file"],
["FM01","15,810","15,512","10,477","10,455","FM01_integrated.h5ad"],
["FM06","18,106","16,453","12,877","12,863","FM06_integrated.h5ad"],
["FM08","6,805","6,485","4,259","3,985","FM08_integrated.h5ad"],
["HS054_Sot48h","10,594","10,042","3,032","3,031","HS054_Sot48h_integrated.h5ad"],
["Watermelon (all)","36,664","-","-","33,271","split into naive 10,319 / lag 12,688 / late 10,264"],
["hPSC_20260927","endo_T0 21,281; endo_T1 5,726","20,896; 4,528 (QC alone)","5,601; 886 (singlet raw cells; before QC)","5,569 + 885 = 6,454","qc_filtered/ only (no integrated h5ad; log1p(CP10k) computed per gene set downstream)"],
["LARRY","95,584","whitelist 89,813 -> mito 87,492","32,728","32,395 (Weinreb rules ii/iii NOT applied)","filtered/ and annotated/filtered_annotated_k15.h5ad"]]
E.append(tbl(rows,[1.3*inch,1.4*inch,1.6*inch,1.6*inch,1.5*inch,2.6*inch]))
E.append(P("Watermelon per-sample (raw / pass mito / pass mito+singlet / final)",H2))
rows=[["Sample","Raw","Mito","Mito+singlet","Final"],["T47D-naive-1","5,748","5,717","5,192","5,124"],["T47D-naive-2","6,040","6,004","5,271","5,195"],["T47D-lag-1","6,989","6,912","6,758","6,674"],["T47D-lag-2","6,737","6,586","6,070","6,014"],["T47D-late-1","4,939","4,902","4,752","4,667"],["T47D-late-2","6,211","6,173","5,630","5,597"]]
E.append(tbl(rows,[1.6*inch,1*inch,1*inch,1.2*inch,1*inch]))
E.append(P("Watermelon stage datasets: Watermelon_{naive,lag,late}_integrated.h5ad are each stage's two replicates (replicate 1 = A, 2 = B), with batch-aware log1p(CP10k) + PCA(50) + scanorama + Harmony as above (batch_key='replicate'). obs has replicate A/B and fatemap_clone_singletcode.",B))
E.append(P("LARRY: full history of the finalized dataset",H2))
L=[
"<b>Step 1, raw matrix (build_larry_matrix.py).</b> 15 inDrops LSK libraries (d2 x3, d4 x6, d6 x6) combined, 95,584 cells x 25,289 genes; cell IDs renamed from inDrops 'bcXXXX' hashes to the real ACGT-ACGT gel barcode (abundant_barcodes.pickle, 1:1 in all libraries). Lineage barcodes from LARRY_sorted_and_filtered_barcodes.fastq.gz: keep (cellID, barcode, UMI) combinations supported by &ge; <b>10 reads</b> (n_reads=10), &ge; <b>3 UMIs</b> (n_umis=3), barcodes merged within <b>Hamming distance 3</b> by transitive connected components, each collapsing to its highest-UMI member (11,986 distinct barcodes before collapse, 7,893 clones after). Output umi_table.csv (one row per UMI per cellID/barcode/sample) feeds singletCode.",
"<b>Step 2, whitelist (dip-test + Otsu, per library).</b> Hartigan dip test on log10(total counts); if p &lt; 0.05 (bimodal) cut at Otsu's threshold, otherwise keep every nonzero-count barcode. No external target cell count. Only LSK_d2_1 (threshold 205), LSK_d6_1_1 (1257) and LSK_d6_1_2 (1217) were bimodal; other libraries kept at threshold 1-3. 95,584 to 89,813 cells. umi_table restricted to whitelisted cells.",
"<b>Step 3, singletCode.</b> <font face='Courier'>get_singlets(umi_table, dataset_name='LARRY_umi3_final', min_umi_cutoff=3)</font> (3 rather than the default 2, to match Weinreb et al. 2020 Methods 3.3), then cross-sample criterion-4 rescue (957 cellIDs rescued). Clone = sorted concatenation of the singlet's unique barcodes (larry_clone_singletcode).",
"<b>Step 4, RNA QC.</b> Mito &le; 20% (mt- prefix, case-insensitive) and &ge; 200 detected genes. Applied sequentially on the survivors: whitelist 89,813, mito 87,492, singletCode 32,728, gene floor 32,395 cells (5,387 distinct clones).",
"<b>Step 5 (Weinreb rules ii/iii) was tried and retired.</b> Cross-library (cell-BC, clone) repeats and over-abundant-clone chi-square (p &lt; 0.001) filtering were integrated for one build, then removed: they accounted for 62% of the disagreement with the authors' h5ad (rule ii alone 1,965 cells, rule iii 48). Finalized data is Steps 1-4 only; the code is kept commented out.",
"<b>Validation.</b> Against the released LSK_d2_d4_d6.h5ad clone_id: Adjusted Rand Index 0.9989 on the 23,687 cells both call barcoded (76 cells, 0.32%, group differently; listed in singletcode_vs_h5ad_diff_cells.csv). The h5ad has 1,221 barcoded cells not in our set.",
"<b>Stored matrix.</b> filtered/larry_qc_counts.mtx is raw counts (no normalization applied). Normalization (log1p(CP10k) / log1pPF) happens downstream at inference input construction.",
"<b>Annotation (larry_raw_annotate.py).</b> Labels come from the published LSK_d2_d4_d6.h5ad where the cell is present (29,351 cells, 'published'); the other 3,044 cells get predicted labels ('predicted'). Method: drop genes expressed in &lt; 5% of cells; log1p(CP10k); 2000 HVGs with batch_key='sample'; PCA 50; distance-weighted k-NN, <b>k=15</b>, class-balanced fit pool capped at 500 cells per class. The classifier never sees clone identity. Stratified 5-fold CV on published cells: overall accuracy 0.876 (d2 0.923, d4 0.892, d6 0.857); poor on rare classes (Erythroid 0.0, pDC 0.0, Eos 0.4). The HVG/PCA is for the classifier only and is not stored in the output, which keeps the full gene set."]
for t in L: E.append(P("&bull; "+t,B))
E.append(P("What was done to the cells after QC",H2))
for t in ["<b>FM01/06/08, Watermelon stages, HS054:</b> log1p(CP10k) over all cells together (raw counts in layers['counts'], log values in layers['lognorm'] and .raw); 2000 HVGs (batch_key='replicate'; none for HS054); scale (max 10); PCA 50 (arpack, seed 0); neighbors + UMAP. Scanorama and Harmony embeddings stored alongside for the replicate datasets (expression itself is not batch-corrected). HS054 also gets Leiden clustering. FM01/06/08 get Goyal et al. cluster labels joined. Watermelon is split by stage and integrated per stage across replicates (1=A, 2=B).",
"<b>hPSC:</b> nothing beyond QC + singletCode; log1p(CP10k) is computed per gene set at input build time.",
"<b>LARRY:</b> k-NN annotation transfer only (above). No other cell-level steps (ambient RNA removal, extra doublet tools, cell-cycle regression) appear in any of the scripts."]:
    E.append(P("&bull; "+t,B))
E.append(P("Are the normalized counts modified? Does 'integrated' change them?",H2))
for t in ["<b>No.</b> In the *_integrated.h5ad files, X, layers['lognorm'] and .raw.X are identical and equal plain log1p(CP10k): expm1 of a cell's values sums to exactly 10,000 (checked on FM06 and HS054). layers['counts'] holds the original integer counts.",
"<b>'Integrated' refers to embeddings only.</b> Scanorama (run on a copy of the HVG-restricted log-normalized matrix) and Harmony (run on the PCA) write only to obsm (X_scanorama, X_pca_harmony). Neither writes back to X or any layer, so expression is not batch-corrected and replicate effects are not removed from it.",
"<b>Scaling/PCA are side copies.</b> scale(max_value=10) and PCA run on a separate 2000-HVG copy; the stored matrix is never scaled or clipped. HVG selection is stored only as a flag in var. All genes and all QC-passing cells are retained.",
"<b>Only per-cell library-size rescaling (CP10k) plus log1p is applied.</b> HS054 is a single sample with no integration at all (PCA, UMAP, Leiden only). The LARRY annotated h5ad stores raw counts in X with no normalized layer and no embeddings; the classifier's normalization/HVG/PCA is not written out. hPSC has no h5ad; log1p(CP10k) is computed per gene set at input build time.",
"<b>TwINFER reads expression, not the embeddings</b>, so the integration outputs are for UMAPs and replicate-mixing checks only. (Note that CP10k is not library-size invariant for rank-based statistics such as Spearman.) The filename 'integrated' therefore means 'normalized + PCA + integrated embeddings', not 'batch-corrected expression'."]:
    E.append(P("&bull; "+t.replace("*_integrated","_integrated"),B))
E.append(P("Things to know / gaps",H2))
for t in ["<b>Auto UMI cutoffs</b> were recovered from the singletCode logs for FM01 (6 / 5) and Watermelon (2 in every sample) and from the summary JSON for hPSC (2). The HS054 cutoff was not found in any log. hPSC cutoff confirmed as 2 from the rerun log.",
"<b>HS054:</b> a fresh auto-cutoff run gives 3,040 singlets; the earlier external singlets_all.txt had 3,264 and could not be reproduced by any min_umi_cutoff in {None,1,2,3} x barcode column sweep. The fresh run is the definition used. The RNA matrix was re-synced from the canonical run directory because the earlier copy only overlapped 12/7,342 barcode-table cellIDs (canonical: 7,342/7,342).",
"<b>Inconsistent UMI settings across datasets:</b> FM06, FM08 and LARRY use min_umi_cutoff=3; FM01, HS054, Watermelon and hPSC use the auto cutoff.",

"<b>hPSC_20260927 and Watermelon (all-stage)</b> have no single integrated h5ad in finalized_data/: hPSC has qc_filtered only, Watermelon is split by stage."]:
    E.append(P("&bull; "+t,B))
d.build(E)
