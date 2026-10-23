suppressPackageStartupMessages({library(qs); library(Seurat); library(SeuratObject); library(Matrix)})
f <- "/gpfs/projects/b1255/hzhang/TwINFER_KA/real_data/Novo_nordisk_hPSC_endo/seurat_lineage_tracing_linneberg_agerholm_2026.qs"
t0 <- Sys.time()
obj <- qread(f, nthreads = 4)
cat("read in", format(Sys.time() - t0), "\n")
cat("class:", class(obj), "\n")
cat("Seurat version in object:", as.character(obj@version), "\n")
print(obj)
cat("\nDefaultAssay:", DefaultAssay(obj), "\n")
for (a in Assays(obj)) {
  A <- obj[[a]]
  cat("\n== assay", a, "class", class(A), "\n")
  if (inherits(A, "Assay5")) {
    for (l in Layers(A)) { m <- LayerData(A, layer = l); cat("  layer", l, class(m), dim(m), "\n") }
  } else {
    for (s in c("counts", "data", "scale.data")) { m <- slot(A, s); cat("  slot", s, class(m), dim(m), "\n") }
  }
}
cat("\n== reductions:", Reductions(obj), "\n")
for (r in Reductions(obj)) cat("  ", r, dim(Embeddings(obj[[r]])), "\n")
cat("== graphs:", Graphs(obj), "\n")
cat("== images:", Images(obj), "\n")
cat("== misc names:", names(obj@misc), "\n")
cat("== tools names:", names(obj@tools), "\n")
md <- obj@meta.data
cat("\n== meta.data", dim(md), "\n")
str(md, list.len = 200, vec.len = 3)
cat("\n== duplicated cell names:", anyDuplicated(colnames(obj)), "\n")
