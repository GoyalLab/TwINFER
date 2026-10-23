#======================================================================================================================================
# Input Files: outputs from stepOne and one or more cellranger filtered matrices (barcodes.tsv.gz).
# Per-sample matrix lookup comes from staggers.txt (optional 4th column) or fall back to the default.
#
# Usage (CLI):
#   Rscript stepTwo.R <staggers_path> <default_filtered_bc_matrix_dir> <stepOne_base_dir> <output_base_dir> <sample1> [sample2 ...]
#======================================================================================================================================

args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 5) {
  staggersPath <- args[1]
  defaultMatrixDir <- args[2]
  input2Base <- args[3]
  outputBase <- args[4]
  samples <- args[5:length(args)]
} else {
  stop("Usage: Rscript stepTwo.R <staggers_path> <default_filtered_bc_matrix_dir> <stepOne_base_dir> <output_base_dir> <sample1> [sample2 ...]")
}

library(tidyverse, quietly = TRUE)
library(stringdist, quietly = TRUE)
library(gridExtra, quietly = TRUE)

parse_staggers_matrix_lookup <- function(path, default_matrix_dir) {
  lookup <- character(0)
  for (line in readLines(path)) {
    line <- trimws(line)
    if (nchar(line) == 0 || startsWith(line, "#")) next
    parts <- strsplit(line, "\t", fixed = TRUE)[[1]]
    sample_name <- trimws(parts[1])
    matrix_dir <- if (length(parts) >= 4 && nchar(trimws(parts[4])) > 0) {
      trimws(parts[4])
    } else {
      default_matrix_dir
    }
    lookup[sample_name] <- matrix_dir
  }
  lookup
}

load_filtered_barcodes <- local({
  cache <- list()
  function(matrix_dir) {
    if (is.null(matrix_dir) || is.na(matrix_dir) || nchar(matrix_dir) == 0) {
      stop("No filtered_bc_matrix_dir available (default empty and staggers entry blank).")
    }
    if (is.null(cache[[matrix_dir]])) {
      barcodes_file <- file.path(matrix_dir, "barcodes.tsv.gz")
      if (!file.exists(barcodes_file)) {
        stop("barcodes.tsv.gz not found at: ", barcodes_file)
      }
      df <- as_tibble(read.table(barcodes_file, stringsAsFactors = FALSE)) %>%
        dplyr::rename(cellID = V1)
      df <- as_tibble(substring(df$cellID, 1, nchar(df[1, 1]) - 2)) %>%
        dplyr::rename(cellID = value)
      cache[[matrix_dir]] <<- df
    }
    cache[[matrix_dir]]
  }
})

matrix_lookup <- parse_staggers_matrix_lookup(staggersPath, defaultMatrixDir)

allBarcodes <- character(0)
join_summary <- list()  

for (i in samples) {
  matrix_dir <- if (i %in% names(matrix_lookup)) matrix_lookup[[i]] else defaultMatrixDir
  cat(i, "→ matrix:", matrix_dir, "\n")
  data1file <- load_filtered_barcodes(matrix_dir)

  input2Directory <- file.path(input2Base, i)
  data2file = as_tibble(read.table(file.path(input2Directory,"uniqueShavedReads.txt"), stringsAsFactors=F)) %>% dplyr::rename(cellID = V1,UMI = V2, BC = V3) %>%
    mutate(BC50 = substring(BC,1,50),
           BC40 = substring(BC,1,40),
           BC30 = substring(BC,1,30))
  cat(i, "loaded\n")
  cellIDUMIBarcodes = inner_join(data1file, data2file, by = "cellID")
  Barcodes = unique(cellIDUMIBarcodes$BC50)
  cellIDs = unique(cellIDUMIBarcodes$cellID)
  
  ## --- JOIN SUMMARY: how many cells have both CellRanger AND barcode data --------------------
  n_cellranger <- nrow(data1file)                          # real cells from CellRanger
  n_bc_reads   <- length(unique(data2file$cellID))         # distinct cells appearing in barcode reads
  n_joined     <- length(cellIDs)                          # cells in BOTH (survived the join)
  pct_recovered <- round(100 * n_joined / n_cellranger, 2)
  join_summary[[i]] <- data.frame(
    sample                   = i,
    cellranger_cells         = n_cellranger,
    cells_with_barcode_reads = n_bc_reads,
    cells_in_both            = n_joined,
    pct_cellranger_recovered = pct_recovered
  )
  cat(sprintf("  JOIN: %d CellRanger cells | %d with barcode reads | %d in BOTH | %.1f%% recovered\n",
              n_cellranger, n_bc_reads, n_joined, pct_recovered))

  allBarcodes <- c(allBarcodes, Barcodes)

  set.seed(2059)
  subsample1 = sample(Barcodes,5000)
  subsample2 = sample(Barcodes,5000)
  subsample3 = sample(Barcodes,5000)
  cat("Starting distance calculation...\n")
  BarcodesLv1 = as.integer(stringdistmatrix(subsample1, method = "lv"), useBytes = TRUE, nthread = 8)
  BarcodesLv2 = as.integer(stringdistmatrix(subsample2, method = "lv"), useBytes = TRUE, nthread = 8)
  BarcodesLv3 = as.integer(stringdistmatrix(subsample3, method = "lv"), useBytes = TRUE, nthread = 8)
  lBarcodesLv = length(BarcodesLv1)
  
  cat("Comparing samplings...\n")
  BarcodesLv = tibble(
    lvdist = c(BarcodesLv1, BarcodesLv2, BarcodesLv3),
    subsamNum = c(rep("subsamping1", lBarcodesLv), rep("subsamping2", lBarcodesLv), rep("subsamping3", lBarcodesLv)))
  
  BarcodesLvHist <- BarcodesLv %>% group_by(subsamNum, lvdist) %>% summarise(length(lvdist)) %>%
    group_by(subsamNum) %>% mutate(totalNum = sum(`length(lvdist)`), fracLvDist = `length(lvdist)`/totalNum)
  
  BarcodesLvHistPlot <- ggplot(BarcodesLvHist, aes(lvdist, fracLvDist)) +
    geom_bar(width = 0.5, stat = 'identity') +
    facet_wrap(facets = vars(subsamNum)) +
    coord_cartesian(xlim = c(0, 50)) +
    theme_classic()

  ymin <- 1e-7   # floor just below the smallest fraction you have

  BarcodesLvHistPlotLog <- ggplot(BarcodesLvHist %>% filter(fracLvDist > 0),
                                  aes(xmin = lvdist - 0.4, xmax = lvdist + 0.4,
                                      ymin = ymin, ymax = fracLvDist)) +
    geom_rect() +
    facet_wrap(facets = vars(subsamNum)) +
    scale_y_log10(expand = expansion(mult = c(0, 0.05))) +
    coord_cartesian(xlim = c(0, 50)) +
    theme_classic() +
    ylab("Fraction (log10)") + xlab("LV Distance")

  cellIDsLv = tibble(lvdist = as.integer(stringdistmatrix(cellIDs, method = "lv")))
  cellIDsHist <- cellIDsLv  %>% group_by(lvdist)%>% summarise(length(lvdist)) %>% mutate(totalNum = sum(`length(lvdist)`), fracLvDist = `length(lvdist)`/totalNum)
  
  cellIDsHistPlot <- ggplot(cellIDsHist, aes(lvdist, fracLvDist)) +
    geom_bar(width = 0.5, stat = 'identity') +
    theme_classic()
  
  #writing files
  outputDirectory <- file.path(outputBase, i)
  if (!dir.exists(outputDirectory)) {
    dir.create(outputDirectory, recursive = TRUE)
  }
  cat("Saving...\n")
  ggsave(BarcodesLvHistPlot,file=file.path(outputDirectory,'stepTwoBarcodesLvBeforeStarcode_50.pdf'))
  ggsave(BarcodesLvHistPlotLog,file = file.path(outputDirectory, 'stepTwoBarcodesLvBeforeStarcode_50_log.pdf'))
  ggsave(cellIDsHistPlot,file=file.path(outputDirectory,'stepTwoCellIdsLvBeforeStarcode.pdf'))
  write.table(cellIDUMIBarcodes, file= file.path(outputDirectory,'stepTwoCellIDUMIBarcodes.txt'),row.names=F,col.names=T,quote=F,sep="\t")
  write.table(cellIDUMIBarcodes[,4], file= file.path(outputDirectory,'stepTwoBarcodes50.txt'),row.names=F,col.names=F,quote=F,sep="\t")
  write.table(cellIDUMIBarcodes[,5], file= file.path(outputDirectory,'stepTwoBarcodes40.txt'),row.names=F,col.names=F,quote=F,sep="\t")
  write.table(cellIDUMIBarcodes[,6], file= file.path(outputDirectory,'stepTwoBarcodes30.txt'),row.names=F,col.names=F,quote=F,sep="\t")
  cat("Done\n\n")
}

## JOIN SUMMARY: write per-sample recovery table
join_summary_df <- do.call(rbind, join_summary)
write.table(join_summary_df,
            file = file.path(outputBase, "stepTwo_join_summary.txt"),
            row.names = FALSE, quote = FALSE, sep = "\t")
cat("\n=== JOIN SUMMARY (cells with both CellRanger + barcode data) ===\n")
print(join_summary_df)
cat("Written to:", file.path(outputBase, "stepTwo_join_summary.txt"), "\n\n")

## MERGED barcode LV analysis across ALL samples
cat("=== MERGED barcode LV (all samples pooled) ===\n")
mergedBarcodes <- unique(allBarcodes)
cat("Merged unique BC50 barcodes:", length(mergedBarcodes), "\n")
 
nsub <- min(5000, length(mergedBarcodes))
set.seed(2059)
m1 <- sample(mergedBarcodes, nsub)
m2 <- sample(mergedBarcodes, nsub)
m3 <- sample(mergedBarcodes, nsub)
 
cat("Starting merged distance calculation...\n")
MergedLv1 <- as.integer(stringdistmatrix(m1, method = "lv"), useBytes = TRUE, nthread = 8)
MergedLv2 <- as.integer(stringdistmatrix(m2, method = "lv"), useBytes = TRUE, nthread = 8)
MergedLv3 <- as.integer(stringdistmatrix(m3, method = "lv"), useBytes = TRUE, nthread = 8)
lMergedLv <- length(MergedLv1)
 
MergedLv <- tibble(
  lvdist = c(MergedLv1, MergedLv2, MergedLv3),
  subsamNum = c(rep("subsamping1", lMergedLv), rep("subsamping2", lMergedLv), rep("subsamping3", lMergedLv)))
 
MergedLvHist <- MergedLv %>% group_by(subsamNum, lvdist) %>% summarise(length(lvdist)) %>%
  group_by(subsamNum) %>% mutate(totalNum = sum(`length(lvdist)`), fracLvDist = `length(lvdist)`/totalNum)
 
MergedLvHistPlot <- ggplot(MergedLvHist, aes(lvdist, fracLvDist)) +
  geom_bar(width = 0.5, stat = 'identity') +
  facet_wrap(facets = vars(subsamNum)) +
  coord_cartesian(xlim = c(0, 50)) +
  theme_classic() +
  ggtitle("Merged (all samples) barcode LV before starcode")
 
ymin <- 1e-7   # floor just below the smallest fraction (adjust to your data)

MergedLvHistPlotLog <- ggplot(MergedLvHist %>% filter(fracLvDist > 0),
                              aes(xmin = lvdist - 0.4, xmax = lvdist + 0.4,
                                    ymin = ymin, ymax = fracLvDist)) +
  geom_rect() +
  facet_wrap(facets = vars(subsamNum)) +
  scale_y_log10(expand = expansion(mult = c(0, 0.05))) +
  coord_cartesian(xlim = c(0, 50)) +
  theme_classic() +
  ggtitle("Merged (all samples) barcode LV before starcode (log)") +
  ylab("Fraction (log10)") + xlab("LV Distance")

mergedDir <- file.path(outputBase, "merged")
if (!dir.exists(mergedDir)) {
  dir.create(mergedDir, recursive = TRUE)
}
ggsave(MergedLvHistPlot, file = file.path(mergedDir, 'stepTwoBarcodesLvBeforeStarcode_50_MERGED.pdf'))
ggsave(MergedLvHistPlotLog, file = file.path(mergedDir, 'stepTwoBarcodesLvBeforeStarcode_50_MERGED_log.pdf'))

cat("Merged mean LV before starcode:", mean(c(MergedLv1, MergedLv2, MergedLv3)), "\n")
cat("Merged LV analysis done.\n")


