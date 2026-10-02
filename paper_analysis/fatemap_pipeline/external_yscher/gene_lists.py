# [copied 2026-09-30 from /gpfs/projects/b1255/yscher/Transcriptomic Distance/exports/_probe_tmp/hive_fm06/work/fm06-builder-contrarian/scripts/gene_lists.py (author/owner: yscher/laj2116), md5 28b011d23df7aee3fb32fc8dbf7018b8; unmodified below this header]
"""Gene lists used by the contrarian diagnosis. PROVENANCE: typed from memory by the agent (no file on disk holds them; checked
2026-09-24 with find/grep over the project and gzu's tree). S_GENES and G2M_GENES = Tirosh et al. 2016 Science (Seurat cc.genes,
regev_lab_cell_cycle_genes.txt; 43 + 54 symbols, hg19 symbols). IEG_CORE = a core subset of the dissociation-induced genes of
van den Brink et al. 2017 Nat Methods (immediate-early and heat-shock genes); the full 140-gene table is NOT on disk, so this is a
subset, labelled as such wherever it is used."""
S_GENES = ["MCM5","PCNA","TYMS","FEN1","MCM2","MCM4","RRM1","UNG","GINS2","MCM6","CDCA7","DTL","PRIM1","UHRF1","MLF1IP","HELLS","RFC2",
           "RPA2","NASP","RAD51AP1","GMNN","WDR76","SLBP","CCNE2","UBR7","POLD3","MSH2","ATAD2","RAD51","RRM2","CDC45","CDC6","EXO1",
           "TIPIN","DSCC1","BLM","CASP8AP2","USP1","CLSPN","POLA1","CHAF1B","BRIP1","E2F8"]
G2M_GENES = ["HMGB2","CDK1","NUSAP1","UBE2C","BIRC5","TPX2","TOP2A","NDC80","CKS2","NUF2","CKS1B","MKI67","TMPO","CENPF","TACC3","FAM64A",
             "SMC4","CCNB2","CKAP2L","CKAP2","AURKB","BUB1","KIF11","ANP32E","TUBB4B","GTSE1","KIF20B","HJURP","CDCA3","HN1","CDC20","TTK",
             "CDC25C","KIF2C","RANGAP1","NCAPD2","DLGAP5","CDCA2","CDCA8","ECT2","KIF23","HMMR","AURKA","PSRC1","ANLN","LBR","CKAP5","CENPE",
             "CTCF","NEK2","G2E3","GAS2L3","CBX5","CENPA"]
IEG_CORE = ["FOS","FOSB","JUN","JUNB","JUND","EGR1","EGR2","EGR3","ATF3","IER2","IER3","IER5","NR4A1","NR4A2","DUSP1","DUSP6","ZFP36","KLF2",
            "KLF4","KLF6","HSPA1A","HSPA1B","HSPA8","HSPB1","HSP90AA1","DNAJB1","DNAJA1","SOCS3","BTG2","PPP1R15A","CEBPB","CEBPD","GADD45B",
            "NFKBIA","MCL1","RHOB","SGK1","CYR61","CTGF","ERRFI1"]
assert len(S_GENES) == 43 and len(G2M_GENES) == 54 and len(set(S_GENES)) == 43 and len(set(G2M_GENES)) == 54
