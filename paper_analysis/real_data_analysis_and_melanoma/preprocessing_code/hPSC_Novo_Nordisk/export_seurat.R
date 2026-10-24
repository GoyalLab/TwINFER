# Export every assay/layer/reduction/graph/meta.data of the Novo Nordisk hPSC Seurat object
# to neutral files (MatrixMarket + CSV). Step 1 of 2; build_h5ad.py assembles the h5ad.
suppressPackageStartupMessages({library(qs); library(Seurat); library(SeuratObject); library(Matrix); library(data.table)})
# the authors' Seurat object and the output folder are given by the launcher (run_rebuild_novo.sbatch): NOVO_QS = .../seurat_lineage_tracing_linneberg_agerholm_2026.qs, NOVO_EXPORT_DIR
f   <- Sys.getenv("NOVO_QS");         if (f == "")   stop("set NOVO_QS to the authors' Seurat .qs file")
out <- Sys.getenv("NOVO_EXPORT_DIR"); if (out == "") stop("set NOVO_EXPORT_DIR to the output folder")
dir.create(out, recursive = TRUE, showWarnings = FALSE)

obj <- qread(f, nthreads = 4)
cells <- colnames(obj)
stopifnot(!anyDuplicated(cells))
chk <- list(n_cells = length(cells))   # checksums for the python-side validation

wr_csc <- function(m, name) {   # raw little-endian CSC: i (int32), p (int32/int64 via double if nnz>2^31), x (float64), dims
  if (!inherits(m, "dgCMatrix")) m <- as(as(m, "generalMatrix"), "dgCMatrix")
  writeBin(m@i, file.path(out, paste0(name, "_i.bin")), size = 4)
  writeBin(as.double(m@p), file.path(out, paste0(name, "_p.bin")), size = 8)
  writeBin(m@x, file.path(out, paste0(name, "_x.bin")), size = 8)
  writeLines(as.character(dim(m)), file.path(out, paste0(name, "_dim.txt")))
}
wr_mtx <- function(m, name) {          # genes x cells dgCMatrix, as stored in Seurat
  m <- as(m, "CsparseMatrix")
  # writeMM(m, file.path(out, paste0(name, ".mtx")))   # text MatrixMarket: ~3 min / 6.6 GB per layer, replaced by raw binary CSC
  wr_csc(m, name)
  fwrite(data.table(feature = rownames(m)), file.path(out, paste0(name, "_features.csv")))
  chk[[paste0(name, "_dim")]]   <<- dim(m)
  chk[[paste0(name, "_sum")]]   <<- sum(m)
  chk[[paste0(name, "_nnz")]]   <<- length(m@x)
  cat(name, dim(m), "sum", sum(m), "nnz", length(m@x), "\n")
}
fwrite(data.table(cell = cells), file.path(out, "cells.csv"))

## RNA assay (Assay5): raw counts -> X
rna <- obj[["RNA"]]
stopifnot(identical(Layers(rna), "counts"))
m <- LayerData(rna, layer = "counts"); stopifnot(identical(colnames(m), cells)); wr_mtx(m, "RNA_counts")

## SCT assay (v3-style SCTAssay): counts, data, scale.data
sct <- obj[["SCT"]]
m <- slot(sct, "counts"); stopifnot(identical(colnames(m), cells)); wr_mtx(m, "SCT_counts")
m <- slot(sct, "data");   stopifnot(identical(colnames(m), cells)); wr_mtx(m, "SCT_data")
sd <- slot(sct, "scale.data")                       # 3000 x cells, dense
stopifnot(identical(colnames(sd), cells))
fwrite(data.table(feature = rownames(sd), sd), file.path(out, "SCT_scale_data.csv"))
chk$SCT_scale_data_dim <- dim(sd); chk$SCT_scale_data_sum <- sum(sd)
fwrite(data.table(feature = VariableFeatures(obj, assay = "SCT")), file.path(out, "SCT_variable_features.csv"))
chk$sct_features_subset_of_rna <- all(rownames(sct) %in% rownames(obj[["RNA"]]))
cat("SCT features subset of RNA:", chk$sct_features_subset_of_rna, "\n")
# feature-level metadata, if any
for (a in c("RNA", "SCT")) {
  mf <- tryCatch(obj[[a]][[]], error = function(e) NULL)
  if (!is.null(mf) && ncol(mf) > 0) { fwrite(data.table(feature = rownames(mf), mf), file.path(out, paste0(a, "_feature_meta.csv"))); cat("feature meta", a, dim(mf), "\n") }
}

## reductions (embeddings + PCA loadings/stdev)
for (r in Reductions(obj)) {
  e <- Embeddings(obj[[r]]); stopifnot(identical(rownames(e), cells))
  fwrite(data.table(cell = rownames(e), e), file.path(out, paste0("red_", r, ".csv")))
  l <- Loadings(obj[[r]])
  if (length(l) > 0 && nrow(l) > 0) fwrite(data.table(feature = rownames(l), l), file.path(out, paste0("red_", r, "_loadings.csv")))
  cat("reduction", r, dim(e), "assay", DefaultAssay(obj[[r]]), "\n")
}

## graphs (cells x cells)
# for (g in Graphs(obj)) { G <- as(obj[[g]], "CsparseMatrix"); writeMM(G, file.path(out, paste0("graph_", g, ".mtx"))); cat("graph", g, dim(G), "\n") }
for (g in Graphs(obj)) { G <- as(obj[[g]], "CsparseMatrix"); wr_csc(G, paste0("graph_", g)); fwrite(data.table(cell = rownames(G)), file.path(out, paste0("graph_", g, "_cells.csv"))); cat("graph", g, dim(G), "\n") }

## meta.data (all columns, raw, factors kept as strings + level order saved)
md <- obj@meta.data; stopifnot(identical(rownames(md), cells))
fwrite(data.table(cell = rownames(md), md), file.path(out, "obs.csv"), na = "NA")
lv <- lapply(md[sapply(md, is.factor)], levels)
writeLines(jsonlite::toJSON(lv, auto_unbox = FALSE), file.path(out, "obs_factor_levels.json"))
writeLines(vapply(md, function(x) class(x)[1], ""), file.path(out, "obs_classes.txt"))
# clone sizes as seen by R (cells per clone.id, NA excluded)
cs <- table(md$clone.id); chk$n_clones <- length(cs); chk$n_cells_with_clone <- sum(!is.na(md$clone.id))
chk$max_clone_size <- max(cs); chk$clone_size_sum <- sum(cs)
fwrite(data.table(clone.id = names(cs), size = as.integer(cs)), file.path(out, "clone_sizes_R.csv"))
cat("clone.id: n_clones", chk$n_clones, "cells with clone.id", chk$n_cells_with_clone, "of", length(cells), "\n")
cat("cells with cell.barcode.ct:", sum(!is.na(md$cell.barcode.ct)), "\n")
print(table(StageID = md$StageID, has_clone = !is.na(md$clone.id)))
print(table(SampleID = md$SampleID, has_clone = !is.na(md$clone.id)))
print(summary(as.integer(cs)))

## per-cell sums for sample-level validation
chk$RNA_counts_colsums_first5 <- head(Matrix::colSums(LayerData(rna, layer = "counts")), 5)
writeLines(jsonlite::toJSON(chk, auto_unbox = TRUE, digits = NA), file.path(out, "R_checksums.json"))
cat("done\n")
