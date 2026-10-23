# Per-cell reference values from the original .qs for a seeded random 10% of cells (compared by validate_cells_vs_R.py)
suppressPackageStartupMessages({library(qs); library(Seurat); library(SeuratObject); library(Matrix); library(data.table)})
f   <- "/gpfs/projects/b1255/hzhang/TwINFER_KA/real_data/Novo_nordisk_hPSC_endo/seurat_lineage_tracing_linneberg_agerholm_2026.qs"
out <- "/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/cell_checks_R.csv"
obj <- qread(f, nthreads = 2)
cells <- colnames(obj); set.seed(1); idx <- sort(sample(length(cells), ceiling(0.1 * length(cells))))
# locale-free gene weight: polynomial hash of the codepoints, mod 1e9+7 (exact in double precision)
gw <- function(g) vapply(g, function(s) { h <- 0; for (c in utf8ToInt(s)) h <- (h * 131 + c) %% 1000000007; h }, 0, USE.NAMES = FALSE)
stats <- function(m, pre) {
  m <- m[, idx, drop = FALSE]; w <- gw(rownames(m))
  d <- data.table(Matrix::colSums(m), diff(m@p), as.numeric(Matrix::crossprod(m, w)))
  setnames(d, paste0(pre, c("_sum", "_nnz", "_w"))); d
}
res <- data.table(cell = cells[idx], row = idx)
res <- cbind(res, stats(LayerData(obj[["RNA"]], layer = "counts"), "rna"),
                  stats(slot(obj[["SCT"]], "counts"), "sctc"),
                  stats(slot(obj[["SCT"]], "data"), "sctd"))
sd <- slot(obj[["SCT"]], "scale.data")[, idx, drop = FALSE]
res[, scale_sum := colSums(sd)][, scale_sumsq := colSums(sd^2)]
fwrite(res, out); cat("wrote", nrow(res), "cells to", out, "\n")
