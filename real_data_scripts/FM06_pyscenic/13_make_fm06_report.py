from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
OUT="/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/fm06/pyscenic/compare/FM06_SCENIC_GRNBoost2_vs_TwINFER_US.pdf"
IMG="/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/fm06/pyscenic/compare/FM06_venn_TFs_edges.png"
B=ParagraphStyle("b",fontName="Helvetica",fontSize=9.2,leading=12.2,spaceAfter=4); H1=ParagraphStyle("h1",fontName="Helvetica-Bold",fontSize=11.5,leading=14,spaceBefore=10,spaceAfter=4)
T=ParagraphStyle("t",fontName="Helvetica-Bold",fontSize=15,leading=18,spaceAfter=2,alignment=1); ST=ParagraphStyle("st",fontName="Helvetica",fontSize=9.5,leading=12,alignment=1,spaceAfter=8)
C=ParagraphStyle("c",fontName="Helvetica",fontSize=7.6,leading=9.2); CB=ParagraphStyle("cb",parent=C,fontName="Helvetica-Bold"); CAP=ParagraphStyle("cap",fontName="Helvetica-Oblique",fontSize=8.3,leading=10.4,spaceAfter=6)
BL=ParagraphStyle("bl",parent=B,leftIndent=12,bulletIndent=2)
def p(t,s=B): return Paragraph(t,s)
def bl(items): return [Paragraph(i,BL,bulletText="•") for i in items]
def tbl(rows,widths,cap=None,hdr=True,align_right_from=None):
    data=[[Paragraph(str(c),CB if (hdr and i==0) else C) for c in r] for i,r in enumerate(rows)]
    t=Table(data,colWidths=[w*inch for w in widths],repeatRows=1 if hdr else 0)
    st=[("VALIGN",(0,0),(-1,-1),"TOP"),("LINEABOVE",(0,0),(-1,0),0.9,colors.black),("LINEBELOW",(0,0),(-1,0),0.5,colors.black),("LINEBELOW",(0,-1),(-1,-1),0.9,colors.black),("TOPPADDING",(0,0),(-1,-1),2),("BOTTOMPADDING",(0,0),(-1,-1),2),("LEFTPADDING",(0,0),(-1,-1),3),("RIGHTPADDING",(0,0),(-1,-1),3)]
    t.setStyle(TableStyle(st)); out=[t]
    if cap: out.append(Paragraph(cap,CAP))
    out.append(Spacer(1,5)); return out
doc=SimpleDocTemplate(OUT,pagesize=letter,leftMargin=0.8*inch,rightMargin=0.8*inch,topMargin=0.75*inch,bottomMargin=0.75*inch,title="FM06: pySCENIC with GRNBoost2 vs TwINFER U and S",author="TwINFER")
def foot(c,d): c.saveState(); c.setFont("Helvetica",8); c.drawCentredString(letter[0]/2,0.45*inch,f"{d.page}"); c.restoreState()
s=[]
s+= [p("FM06 (melanoma): pySCENIC with GRNBoost2 vs TwINFER U and S",T),p("Regulon comparison, 13 October 2026",ST)]
s+= [p("Summary",H1)]+bl([
"<b>Question.</b> What happens to the SCENIC regulons when the GRNBoost2 TF–target weights are replaced by TwINFER's unshared correlation U, or by the same-cell correlation S? Everything downstream (motif pruning with cisTarget, AUCell) is identical.",
"<b>Data.</b> 12,863 cells (replicates A and B pooled as one sample), log1p(CP10k); 9,100 genes expressed in more than 5% of cells; 613 regulators. 3,167 clones have at least 2 cells, giving 18,022 within-sample twin pairs.",
"<b>Regulons.</b> GRNBoost2 gives 171 regulons, U gives 194 and S gives 211. 126 TFs have a regulon with all three methods and 262 have one with at least one (Figure 1).",
"<b>Similarity.</b> S is closer to GRNBoost2 than U is (edge overlap 0.38 against 0.32; paired over 126 shared TFs, p = 3 × 10<super>-6</super>). U and S resemble each other more than either resembles GRNBoost2 (edge overlap 0.45).",
"<b>Biology.</b> The top shared targets are recognisable: MITF → MLANA, PMEL, DCT, RAB27A; SOX10 → PLP1, MFSD12; ATF4 → ASNS, DDIT3, TRIB3, PSAT1, MTHFD2; E2F1 → PCNA, MCM2–7. GRNBoost2, U and S recover canonical targets about equally, except for FOSL1, whose regulon is large in GRNBoost2 (455 targets) and nearly absent in U (4) and S (20).",
"<b>Cell clusters.</b> Regulon activity (AUCell) recovers the annotated clusters equally well with all three (accuracy 0.88–0.89).",
"<b>Caution.</b> A hub-gene program (MKI67, CENPF, ATAD2 and others) appears in the regulons of many TFs and tracks cell-cycle and library-complexity signals. The overlap with CollecTRI is about 1.5% for every method, so it cannot separate them."])
s+= [p("Methods",H1)]+bl([
"<b>Normalisation.</b> Raw counts layer → counts per 10,000 → log1p. Genes with non-zero counts in more than 5% of cells were kept (9,100 of 32,738). Regulators were the 613 TFs of the FM06 TF list (analysis_data/fm06/gene_modules/all_tfs.txt), all inside the gene set.",
"<b>GRNBoost2 arm.</b> pySCENIC <i>grn</i> (arboreto GRNBoost2, seed 666, 32 workers): 725,723 TF → target links.",
"<b>TwINFER arm.</b> S, C and U = S − C were computed for every TF × gene pair with clone-weighted ranks over within-sample twin pairs (the TwinScore procedure, sections 2, 4 and 5). For each TF the top targets by |U| (or |S|) were kept, using the same number of edges per TF as the GRNBoost2 adjacency. Edge counts per TF therefore come from GRNBoost2; the identity and ranking of targets come from U or S.",
"<b>Common steps.</b> pySCENIC <i>ctx</i> with the hg38 cisTarget rankings (10 kb up/down and 500 bp up / 100 bp down of the TSS, v10 clustered motifs) and the motif-to-TF annotation file, default thresholds (NES ≥ 3, rank threshold 5000); activating (+) regulons only. AUCell with the default top 5% of ranked genes.",
"<b>Self-edges.</b> pySCENIC adds each TF to its own regulon; 62–67% of regulons contained their TF. These self-edges are removed from every number in this report."])
s+= [p("1. Regulon sets",H1)]
s+= tbl([["","GRNBoost2","TwINFER U","TwINFER S"],["TFs with a regulon","171","194","211"],["Regulon edges (TF → target)","24,331","26,996","35,869"],["Median targets per regulon","38","47","40"],["Largest regulon (targets)","2,364 (MITF)","2,042","2,601"],["Edges where the target is a TF","7.2%","7.5%","7.1%"],["Edges also in CollecTRI","1.52%","1.53%","1.48%"]],[2.4,1.4,1.4,1.4],"Table 1. Regulons after motif pruning, self-edges removed.")
s+= [KeepTogether([Image(IMG,width=6.9*inch,height=6.9*inch*1300/2750),p("Figure 1. Overlap of the three regulon sets: TFs that have a regulon (left) and TF → target edges inside regulons (right). GRNBoost2 is blue, U is orange, S is green; the number outside each circle is the set total.",CAP)])]
s+= [p("2. How similar are the arms?",H1)]
s+= tbl([["Comparison","TFs in both","TF-set overlap (Jaccard)","Edge overlap (Jaccard)","Median target overlap per shared TF (IQR)"],
["U vs GRNBoost2","133 of 171 (78% found by U)","0.573","0.324 (12,561 shared edges)","0.149 (0.08–0.30)"],
["S vs GRNBoost2","142 of 171 (83% found by S)","0.592","0.384 (16,690 shared edges)","0.179 (0.09–0.33)"],
["U vs S","165","0.69","0.448","0.312 (0.15–0.48)"]],[1.1,1.45,1.15,1.6,1.6],"Table 2. Overlap of regulons between methods. TF-set overlap is the Jaccard index of the sets of TFs having a regulon; edge and target overlap are Jaccard indices.")
s+= [p("Over the 126 TFs that have a regulon in all three methods, the median per-TF target overlap with GRNBoost2 is 0.159 for U and 0.191 for S; S is higher for 84 TFs and U for 41 (Wilcoxon signed-rank p = 2.9 × 10<super>-6</super>).")]
s+= [p("3. Edge categories",H1)]
s+= tbl([["Edge set","Edges","In GRNBoost2 regulon","In raw GRNBoost2 adjacency","In CollecTRI"],["Common to U and S","19,438","57.4%","67.8%","1.72%"],["S only","16,431","33.7%","53.6%","1.19%"],["U only","7,558","18.7%","48.2%","1.06%"],["GRNBoost2 only (not U or S)","6,228","100%","100%","1.04%"]],[1.9,0.9,1.4,1.7,1.0],"Table 3. Each edge is assigned to one category by which regulons contain it. The raw GRNBoost2 adjacency holds about 13% of all TF – gene pairs.")
s+= tbl([["Edge set","Edges","Mean target expression","Housekeeping-like targets","Target is a TF","Median TF–target correlation","Positive correlation"],
["Common to U and S","19,438","0.44","2.6%","7.6%","0.054","100%"],["U only","7,558","0.49","3.1%","7.2%","0.045","100%"],["S only","16,431","0.52","4.0%","6.5%","0.049","100%"],["GRNBoost2 only","6,228","0.43","3.3%","7.6%","0.046","99.9%"],
["U ∩ GRNBoost2","12,561","0.42","3.0%","7.3%","0.058","100%"],["U, not in GRNBoost2","14,435","0.48","2.6%","7.7%","0.046","100%"],["S ∩ GRNBoost2","16,690","0.43","3.2%","7.0%","0.057","100%"],["S, not in GRNBoost2","19,179","0.52","3.3%","7.2%","0.048","100%"]],[1.5,0.7,1.0,1.0,0.8,1.1,0.8],"Table 4. Properties of the target genes in each edge set (log1p(CP10k) expression and within-cell Spearman correlation, 6,000-cell sample). Housekeeping-like = ribosomal, mitochondrial, translation, proteasome and histone genes. Edge sets differ little.")
s+= [p("4. Which TFs form regulons",H1)]
s+= tbl([["Region","TFs"],["All three methods","126"],["GRNBoost2 and S, not U","16"],["GRNBoost2 and U, not S","7"],["U and S, not GRNBoost2","39"],["GRNBoost2 only","22"],["U only","22"],["S only","30"],["Union","262"]],[3.0,1.0],"Table 5. Regions of the TF Venn diagram (Figure 1, left).")
s+= [p("TFs with a regulon in all three methods and the largest regulons: MITF, BHLHE41, ETV4, ETV5, TFDP1, HES6, USF2, ZNF704, YY1, E2F1, TFAP2A, STAT1, ETS1, E2F8, TEAD1, FOSL1, SOX10, KLF6, ZNF148, E2F7, ATF4, TFAP2C, EGR1, SOX6, SOX5, CEBPB, MYBL2, MYC, MYBL1, SP3.",B)]
s+= [p("5. Shared and differing targets of key TFs",H1)]
rows=[["TF","Targets G / U / S; common / U-only / S-only","Top common targets","Top U-only targets","Top S-only targets"],
["MITF","2,364 / 2,042 / 2,601; 1,868 / 174 / 733","MLANA, PMEL, FRMD4B, PIK3R3, DCT, LGALS3, RAB27A, ASAH1, CAPN3","SLC27A3, IFI16, MTOR, TTYH2, CDCA8","CHCHD6, CDK2, CHCHD10, PSMA7, PYCARD"],
["SOX10","447 / 472 / 510; 347 / 125 / 163","PLP1, MITF, DSTYK, ZNF106, MFSD12, PSAP, MYO10","MYH10, PIK3R3, GAS7, GPM6B, MLANA, PMEL","H2AFZ, CDK2, GNPTAB, NBL1, ATM"],
["TFAP2A","575 / 592 / 703; 543 / 49 / 160","MYH10, ZEB2, IRS2, AKAP12, MXI1, CADPS, PIK3R3","MCAM, FAT1, RAB6A, ARID5B","PLP1, SAT1, PMP22, MFSD12, RAB38"],
["TFAP2C","296 / 61 / 388; 42 / 19 / 346","FN1, CCND1, NGFR, BASP1, AXL, PRNP, TWIST1","IER3, DDIT4, CCNB2, MARCKS, MMP1","LGALS1, CAV1, THBS1, SERPINE2, VIM"],
["JUN","251 / 127 / 245; 118 / 9 / 127","ACTB, BASP1, TPM4, THBS1, F2R, PPP1R15A, CRIM1","FOSL2, DUSP6, ACTG1","CAV1, FN1, ANXA2, TNFRSF12A, VIM"],
["E2F1","607 / 767 / 869; 728 / 39 / 141","PCNA, ATAD2, CLSPN, MCM4, HELLS, CCNE2, MCM3, DUT","RAD1, NBL1, PKM, YBX1","FOSL1, FADS1, LMNB1, NCL, NEAT1"],
["MYBL2","289 / 357 / 395; 297 / 60 / 98","RRM2, ATAD2, DUT, KIAA0101, CLSPN, PCNA, TYMS, CDK1","CDC45, RRM1, AURKB, RNF8","HIST1H1D, DHFR, EXO1, MCM6, DNA2"],
["FOXM1","225 / 414 / 215; 171 / 243 / 44","HMGB2, CDK1, SMC4, KPNA2, CKS2, PRC1, PLK1, BUB1B","UBE2C, CENPF, KIF14, CENPE, KIF20B, CEP55","MKI67, AURKB, KIF2C, CDCA2, WEE1"],
["ATF4","381 / 420 / 448; 352 / 68 / 96","ASNS, DDIT3, MTHFD2, RCAN1, DDIT4, EIF5, NUPR1, TRIB3, PSAT1","PTP4A1, STK4, GLS, RAD23B","IER3, CCND1, PLIN2, SLC7A1, MAT2A"],
["DDIT3","232 / 285 / 319; 189 / 96 / 130","EIF1, ATF4, GADD45A, EIF5, TRIB3, HERPUD1, DDIT4","CKS1B, ASPM, TPX2, GTSE1, BUB1","SLC7A11, AURKA, NUPR1, PSAT1, GCLM"],
["HES6","729 / 701 / 1,148; 620 / 81 / 528","PMEL, MLANA, DCT, ASAH1, PIK3R3, SLC45A2, TRPM1, BIRC7","FASTKD2, MRPL34, CHCHD10, SDHB","FRMD4B, GDF15, TXN, PRDX1, APOE"],
["TEAD1","484 / 603 / 560; 473 / 130 / 87","DST, HNRNPU, SPTBN1, MYH9, TRIO, MYO10","RPS20, EIF4G2, CSDE1, APC","BPTF, ACTR3, ATM, FN1, FAT1"]]
s+= tbl(rows,[0.55,1.3,2.0,1.35,1.4],"Table 6. Top-ranked targets per TF (ranked by the TwINFER score: the mean of |U| and |S| for common edges, |U| for U-only, |S| for S-only). The full table of 165 TFs is in U_vs_S_targets_per_common_TF.csv.")
s+= [p("6. Benchmarks against known biology",H1)]
s+= tbl([["Check","GRNBoost2","U","S"],["MITF regulon contains: 19 pigmentation / melanocyte genes","16","14","16"],["ATF4 regulon contains: 17 integrated-stress-response genes","17","16","17"],["E2F1 regulon contains: 22 S-phase / replication genes","21","21","21"],["SOX10 regulon contains: 11 glial and melanocyte genes","6","7","7"],["First 50 pairs of the TwinScore melanoma top-100 (Table 1) found in regulons","26","17","22"]],[4.0,0.9,0.7,0.7],"Table 7. Recovery of canonical targets. The gene lists are my own selections from known biology, not curated databases. 15 of the 50 TwinScore pairs are found by all three methods. The U and S shortfall on the TwinScore pairs is largely FOSL1: 455 targets in GRNBoost2, 4 in U and 20 in S. SNAI2 has no regulon in any method.")
s+= [p("7. Regulon activity and the annotated clusters",H1)]
s+= tbl([["Method","Regulons","Accuracy","Balanced accuracy","Macro F1"],["GRNBoost2","171","0.881","0.796","0.815"],["U","194","0.885","0.809","0.828"],["S","211","0.888","0.821","0.837"]],[1.6,1.0,1.0,1.4,1.0],"Table 8. Five-fold cross-validated logistic regression of the four annotated clusters (Neural Crest Marker 5,831 cells, Melanocytic Marker 4,737, Canonical Resistant Cells 1,962, Unknown 333) from AUCell scores. Always predicting the largest cluster gives accuracy 0.453.")
s+= tbl([["Cluster","GRNBoost2","U","S"],["Canonical Resistant Cells","HMGA1, IRF1, CEBPB, JUNB, E2F7, HMGA2","IRF1, JUNB, HMGA1, HMGA2, CEBPB, ETV1","CEBPB, IRF1, HMGA2, JUNB, NR2F2, ETV1"],["Melanocytic Marker","ETV4, BHLHE41, USF2, MITF, HES6, ETV5","BHLHE41, USF2, ESRRA, HES6, GATA4, ETV4","HEY1, USF2, BHLHE41, GATA4, ETV4, RXRG"],["Neural Crest Marker","SOX4, SOX5, CEBPD, FOSB, POU3F2, STAT2","SOX4, JUND, DLX5, POU3F2, STAT2, SOX5","DLX5, SOX4, SREBF2, POU3F2, KLF9, TCF7L1"],["Unknown","GATA4, MLX, POU2F1, ZNF398, DDIT3, UBP1","ZNF639, ONECUT2, FOXK2, ETV5, CREBZF, CREB3","ZXDC, ONECUT2, ZBTB25, FOXP1, ZNF678, PKNOX1"]],[1.4,1.8,1.8,1.8],"Table 9. Regulons most enriched in each annotated cluster (largest difference in standardised AUCell, first six).")
s+= [p("8. TFs found by only one of U and S, and their known roles",H1),p("Roles are from general knowledge of the literature and were not looked up for this report; zinc-finger proteins with little annotation are grouped. Targets are counts in that method's regulon; +G means the TF also has a GRNBoost2 regulon.",B)]
s+= tbl([["TF (targets)","Known role"],
["ZBTB10 (75, +G), ZBTB11 (71, +G), ZNF267 (65)","BTB-zinc-finger or zinc-finger repressors with little functional annotation"],
["PLAGL2 (31)","zinc-finger oncogene (Wnt pathway)"],["RREB1 (23)","Ras-responsive element binding factor"],["HBP1 (19)","HMG-box repressor; cell cycle and Wnt, tumour suppressor"],["JDP2 (17)","AP-1 repressor"],["PHF20 (16, +G)","component of the NSL histone acetyltransferase complex"],["SIX1 (10)","developmental homeobox factor"],["SATB2 (10)","chromatin organiser"],["NFIL3 (10)","circadian and immune bZIP repressor"],["ARNTL / BMAL1 (8)","circadian clock"],["REL (3)","NF-κB subunit"]],[2.4,4.5],"Table 10a. 29 TFs with a regulon in U but not S (largest first; the table lists the 11 groups with annotated roles; the others are mostly zinc fingers).")
s+= tbl([["TF (targets)","Known role"],
["MXI1 (269, +G)","MAD family antagonist of MYC"],["NR2F2 (178, +G)","COUP-TFII orphan nuclear receptor; mesenchymal and neural-crest development"],["RUNX3 (104, +G), RUNX1 (29)","RUNX transcription factors"],["TWIST1 (95)","bHLH regulator of EMT; linked to the invasive melanoma state"],["FOXP1 (41, +G), PBX3 (30, +G), FOXJ3 (28, +G)","developmental transcription factors"],["THAP11 (28, +G)","growth and cell-cycle regulation"],["ZNF207 (26)","BuGZ, spindle-checkpoint factor"],["NFATC3 (25, +G)","calcineurin–NFAT calcium signalling"],["MXD4 (23, +G), MNT (13)","MAD family MYC antagonists"],["NR1D2 (21)","REV-ERBβ, circadian nuclear receptor"],["RELB (19)","non-canonical NF-κB subunit"],["ARID3A (15), TERF2 (15), MEF2D (12), DBP (11), NCOA2 (8)","chromatin and DNA-binding factor, telomere protein TRF2, muscle/neural MEF2 factor, circadian bZIP, nuclear receptor coactivator"]],[2.4,4.5],"Table 10b. 46 TFs with a regulon in S but not U (largest first).")
s+= [p("GRNBoost2 only (22 TFs), largest first: MBNL2 (57, an RNA-splicing regulator), KLF2 (49), TCFL5 (40), NFAT5 (38, osmotic stress), SKI (33, TGF-β/SMAD co-repressor), ZNF7, MAZ, ZBTB1, MXD3 and MEF2A.",B)]
s+= [p("9. The hub-gene program",H1)]
s+= [p("The 25 genes that occur in the U or S regulons of the most TFs include MKI67 (55 TFs), SPTBN1 (55), CLTC, HNRNPU, RAD21, TMPO, KPNB1, ANKRD11, MYH10, APP and CENPF. I scored every cell on these genes (6,000-cell sample) and correlated the score with other properties.",B)]
s+= tbl([["Hub score correlated with","Spearman r"],["Mitotic spindle genes (Hallmark)","+0.73"],["G2-M checkpoint genes","+0.66"],["E2F target genes","+0.61"],["Genes detected per cell","+0.55"],["Total counts per cell","+0.43"],["Apoptosis / unfolded protein response / hypoxia (Hallmark)","+0.44 / +0.40 / +0.34"],["p53 pathway","+0.27"],["Immediate-early genes / heat-shock genes","+0.11 / +0.24"],["Mitochondrial fraction","+0.14"],["Melanocytic markers (MITF, SOX10, MLANA, PMEL, DCT, TYR)","+0.03"]],[4.2,1.6],"Table 11. The hub program follows the cell-cycle program and library complexity most strongly. Stress sets correlate moderately but overlap with cell-cycle genes, so I cannot separate the two. By annotated cluster the mean hub score is near zero in the three main clusters and −0.67 in Unknown.")
s+= [p("Examples of the effect: in STAT1, the S-only targets are enriched for the G2-M checkpoint set; in DDIT3 the U-only targets include CKS1B, ASPM, TPX2 and GTSE1. These edges reflect cell-cycle co-expression, not necessarily regulation by those TFs.",B)]
s+= [p("10. Limits",H1)]+bl([
"There is no ground truth for the regulons. Overlap with CollecTRI is about 1.5% for all methods and does not distinguish them. The canonical-gene lists in Table 7 are my own.",
"U and S rest on 18,022 twin pairs; regulon calls depend on one set of default thresholds (correlation sign cut-off 0.03, NES ≥ 3), which I did not tune.",
"Per-TF edge counts for U and S are inherited from GRNBoost2 by design (equal edge budget per TF).",
"The cell-cycle and library-complexity program passes into many regulons. I did not regress it out for FM06; a complexity-regression test for the hPSC data is planned."])
s+= [p("Files",H1),p("Everything is in analysis_data/fm06/pyscenic/. Regulons: regulons_FM06.csv, regulons_U_FM06.csv, regulons_S_FM06.csv. AUCell: auc_FM06.csv, auc_U_FM06.csv, auc_S_FM06.csv. In compare/: all_edges_FM06_GRNBoost2_U_S.csv (49,655 edges with scores, ranks and flags), U_vs_S_targets_per_common_TF.csv (165 TFs), target_set_enrichment_FM06.csv, regulons_noself.pkl and this report. Scripts: code/TwINFER/real_data_scripts/FM06_pyscenic/.",B)]
doc.build(s,onFirstPage=foot,onLaterPages=foot); print("wrote",OUT)
