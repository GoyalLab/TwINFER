from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import scanpy as sc
import pandas as pd
import numpy as np

DATA_DIR = f'{TWINFER_PROJECT_ROOT}/real_data/Michaels_et_al'
adata = sc.read_h5ad(f"{DATA_DIR}/timecourse_fully_processed.h5ad")
print(f"Loaded: {adata.n_obs} cells x {adata.n_vars} genes")
print(adata.obs[["pool", "HTO_cluster"]].value_counts().sort_index())

# switch var_names to symbols for marker lookup
sym = adata.var["gene_symbol"].astype(str)
adata.var["ensembl_id"] = adata.var_names
adata.var_names = sym.values
adata.var_names_make_unique()

marker_sets = {
    "arterial_endo_day0-1": ["GJA4", "HEY1", "DLL4", "MECOM"],
    "hsc_day2_peak": ["RAB27B", "MYB", "STAT5A", "SPINK2"],
    "t_lineage_day7plus": ["CD7", "CD3E", "IL7R", "PTCRA"],
}

for name, genes in marker_sets.items():
    present = [g for g in genes if g in adata.var_names]
    missing = [g for g in genes if g not in adata.var_names]
    print(f"\n{name}: present={present} missing={missing}")
    sc.tl.score_genes(adata, present, score_name=name)

celltype_markers = {
    "Endothelial": ["MDK", "LAPTM4B", "ECSCR", "IFITM3", "PLVAP"],
    "MPP_HSC": ["ANGPT1", "CYTL1", "C1QTNF4", "IGFBP7", "RPS14"],
    "Erythroid": ["HBG1", "HBG2", "RHAG", "BLVRB", "GYPB", "KLF1"],
    "Mast": ["CD63", "IL1RL1", "KIT", "SLC18A2"],
    "Myeloid": ["HLA-DRA", "CD74", "HLA-DPA1", "HLA-DRB1", "SPI1"],
    "ProT": ["CD3D", "CD3E", "IL32", "LTB", "GATA3"],
}

for name, genes in celltype_markers.items():
    present = [g for g in genes if g in adata.var_names]
    missing = [g for g in genes if g not in adata.var_names]
    print(f"\n{name}: present={present} missing={missing}")
    sc.tl.score_genes(adata, present, score_name=f"ct_{name}")

score_cols = [f"ct_{k}" for k in celltype_markers] + list(marker_sets.keys())

summary = adata.obs.groupby(["pool", "HTO_cluster"], observed=True)[score_cols].mean()
summary["n_cells"] = adata.obs.groupby(["pool", "HTO_cluster"], observed=True).size()
summary = summary.sort_values(["pool", "HTO_cluster"])

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
print("\n\n=== Mean marker/celltype scores per pool x hashtag ===")
print(summary.round(3).to_string())

summary.to_csv(f"{DATA_DIR}/hashtag_timepoint_test_summary.csv")
print(f"\nSaved: {DATA_DIR}/hashtag_timepoint_test_summary.csv")

# Merged across pools: score by hashtag alone, ignoring which of A/C/D it came from
merged = adata.obs.groupby("HTO_cluster", observed=True)[score_cols].mean()
merged["n_cells"] = adata.obs.groupby("HTO_cluster", observed=True).size()
merged = merged.sort_index()

print("\n\n=== Mean marker/celltype scores per hashtag, MERGED across PoolA+C+D ===")
print(merged.round(3).to_string())

merged.to_csv(f"{DATA_DIR}/hashtag_timepoint_test_merged.csv")
print(f"\nSaved: {DATA_DIR}/hashtag_timepoint_test_merged.csv")

# Also weight by pool composition to see which pools contribute to each merged hashtag
contrib = adata.obs.groupby(["HTO_cluster", "pool"], observed=True).size().unstack(fill_value=0)
print("\n\n=== Cell count contribution per pool, by hashtag ===")
print(contrib.to_string())
