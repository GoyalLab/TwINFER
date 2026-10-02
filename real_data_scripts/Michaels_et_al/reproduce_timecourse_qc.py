from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import scipy.io
import scipy.sparse
import gzip
from pathlib import Path

DATA_DIR = Path(f'{TWINFER_PROJECT_ROOT}/real_data/Michaels_et_al')
OUT_DIR = Path(f'{TWINFER_PROJECT_ROOT}/real_data/Michaels_et_al')

pools = {
    "PoolA": "GSM7916611_PoolA",
    "PoolC": "GSM7916612_PoolC",
    "PoolD": "GSM7916613_PoolD",
}

def load_pool(prefix):
    mtx_path = DATA_DIR / f"{prefix}_matrix.mtx.gz"
    barcodes_path = DATA_DIR / f"{prefix}_barcodes.tsv.gz"
    features_path = DATA_DIR / f"{prefix}_features.tsv.gz"
    hto_path = DATA_DIR / f"{prefix}_HTOclusters.csv.gz"

    mat = scipy.io.mmread(mtx_path).T.tocsr()  # cells x genes
    barcodes = pd.read_csv(barcodes_path, header=None, sep="\t")[0].values
    features = pd.read_csv(features_path, header=None, sep="\t")
    features.columns = ["gene_id", "gene_symbol", "feature_type"][: features.shape[1]]

    adata = ad.AnnData(X=mat)
    adata.obs_names = barcodes
    adata.var_names = features["gene_id"].values
    adata.var["gene_symbol"] = features["gene_symbol"].values
    adata.var_names_make_unique()

    hto = pd.read_csv(hto_path)
    hto = hto.set_index("Barcode")
    adata.obs["HTO_cluster"] = hto["Cluster"].reindex(adata.obs_names).values

    return adata


adatas = {}
for pool_name, prefix in pools.items():
    a = load_pool(prefix)
    a.obs["pool"] = pool_name
    adatas[pool_name] = a
    print(f"{pool_name}: loaded {a.n_obs} barcodes x {a.n_vars} genes")

# Check gene_id sets are identical across pools before concatenating
gene_sets = [set(a.var_names) for a in adatas.values()]
common_genes = set.intersection(*gene_sets)
union_genes = set.union(*gene_sets)
print(f"\nGene ID overlap: intersection={len(common_genes)}, union={len(union_genes)}")

# Concatenate on the outer join of genes (anndata default is inner join; use outer to match union)
combined = ad.concat(list(adatas.values()), join="outer", label="batch", keys=list(pools.keys()), index_unique="-", merge="first")
print(f"\nConcatenated raw: {combined.n_obs} cells x {combined.n_vars} genes")

# Step 1: use demultiplexed HTO cluster assignments to remove empty droplets / doublets / negatives
singlet_mask = ~combined.obs["HTO_cluster"].isin(["Doublet", "Negative"]) & combined.obs["HTO_cluster"].notna()
print(f"\nSinglets after HTO demux filter: {singlet_mask.sum()} / {combined.n_obs}")
adata = combined[singlet_mask].copy()

# Step 2: per-cell QC filters as described in the paper
adata.var["mt"] = adata.var["gene_symbol"].astype(str).str.upper().str.startswith("MT-")
print(f"\nNumber of MT genes detected: {adata.var['mt'].sum()}")

sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

n_before = adata.n_obs
mask_genes = adata.obs["n_genes_by_counts"] >= 1000  # "fewer than 1000 genes" filtered OUT
mask_mito = adata.obs["pct_counts_mt"] <= 20         # "more than 20% mitochondrial reads" filtered OUT
mask_reads = adata.obs["total_counts"] <= 85000       # "more than 85000 reads" filtered OUT

print(f"\nFiltered by <1000 genes: {(~mask_genes).sum()} cells removed")
print(f"Filtered by >20% mito: {(~mask_mito).sum()} cells removed")
print(f"Filtered by >85000 counts: {(~mask_reads).sum()} cells removed")

qc_mask = mask_genes & mask_mito & mask_reads
adata = adata[qc_mask].copy()
print(f"\nCells after per-cell QC filters: {adata.n_obs} (paper reports 7899)")

# Step 3: gene filtering (standard scanpy tutorial default, not explicitly stated in text but
# implied by "following the scanpy preprocessing tutorial")
n_genes_before_filter = adata.n_vars
sc.pp.filter_genes(adata, min_cells=3)
print(f"\nGenes after filter_genes(min_cells=3): {adata.n_vars} (from {n_genes_before_filter}); paper reports 21888")

adata.write(OUT_DIR / "timecourse_after_cell_gene_qc.h5ad")

# Step 4: normalization + log transform
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Step 5: HVGs
sc.pp.highly_variable_genes(adata)
print(f"\nHighly variable genes flagged: {adata.var['highly_variable'].sum()}")

# Step 6: cell cycle scoring (standard Regev/Tirosh S and G2M gene lists, translated from mouse/human symbols)
s_genes = ['MCM5','PCNA','TYMS','FEN1','MCM2','MCM4','RRM1','UNG','GINS2','MCM6','CDCA7','DTL',
           'PRIM1','UHRF1','MLF1IP','HELLS','RFC2','RPA2','NASP','RAD51AP1','GMNN','WDR76','SLBP',
           'CCNE2','UBR7','POLD3','MSH2','ATAD2','RAD51','RRM2','CDC45','CDC6','EXO1','TIPIN',
           'DSCC1','BLM','CASP8AP2','USP1','CLSPN','POLA1','CHAF1B','BRIP1','E2F8']
g2m_genes = ['HMGB2','CDK1','NUSAP1','UBE2C','BIRC5','TPX2','TOP2A','NDC80','CKS2','NUF2','CKS1B',
             'MKI67','TMPO','CENPF','TACC3','FAM64A','SMC4','CCNB2','CKAP2L','CKAP2','AURKB','BUB1',
             'KIF11','ANP32E','TUBB4B','GTSE1','KIF20B','HJURP','CDCA3','HN1','CDC20','TTK','CDC25C',
             'KIF2C','RANGAP1','NCAPD2','DLGAP5','CDCA2','CDCA8','ECT2','KIF23','HMMR','AURKA','PSRC1',
             'ANLN','LBR','CKAP5','CENPE','CTCF','NEK2','G2E3','GAS2L3','CBX5','CENPA']

var_symbols = adata.var["gene_symbol"].astype(str)
adata.var_names_symbol_backup = adata.var_names.copy()
sym_to_idx = pd.Series(adata.var_names.values, index=var_symbols.values)

s_genes_present = [g for g in s_genes if g in sym_to_idx.index]
g2m_genes_present = [g for g in g2m_genes if g in sym_to_idx.index]
print(f"\nCell cycle gene sets found: S={len(s_genes_present)}/{len(s_genes)}, G2M={len(g2m_genes_present)}/{len(g2m_genes)}")

# Temporarily switch var_names to symbols for score_genes_cell_cycle convenience
adata.var["ensembl_id"] = adata.var_names
adata.var_names = var_symbols.values
adata.var_names_make_unique()

sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes_present, g2m_genes=g2m_genes_present)
print(adata.obs["phase"].value_counts())

# restore ensembl var_names
adata.var_names = adata.var["ensembl_id"].values

# Step 7: regress out cell cycle scores
sc.pp.regress_out(adata, ["S_score", "G2M_score"])

# Step 8: PCA, neighbors, UMAP
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)

print(f"\nFINAL processed dataset: {adata.n_obs} cells x {adata.n_vars} genes")
print("Paper reports: 7899 cells, 21888 genes")

adata.write(OUT_DIR / "timecourse_fully_processed.h5ad")
print(f"\nSaved:\n - {OUT_DIR / 'timecourse_after_cell_gene_qc.h5ad'} (post cell+gene QC, pre-normalization)\n - {OUT_DIR / 'timecourse_fully_processed.h5ad'} (fully processed: norm/log/HVG/CC-regressed/PCA/UMAP)")
