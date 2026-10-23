"""Consistency of clone.id vs cell.barcode.ct in the exported Seurat meta.data (obs.csv).
Run after export_seurat.R:  ~/.conda/envs/twinfer-code/bin/python -I check_clone_consistency.py
"""
import pandas as pd

EXP = "/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/export"
obs = pd.read_csv(f"{EXP}/obs.csv", index_col=0, low_memory=False)
n = len(obs)
cid, cbc = obs["clone.id"], obs["cell.barcode.ct"]

print(f"cells {n}; clone.id non-NA {cid.notna().sum()}; cell.barcode.ct non-NA {cbc.notna().sum()}")
print("\n1) NA pattern (rows = clone.id present, cols = cell.barcode.ct present)")
print(pd.crosstab(cid.notna().rename("clone.id"), cbc.notna().rename("cell.barcode.ct")))

both = obs[cid.notna() & cbc.notna()]
print(f"\n2) cell.barcode.ct uniqueness: {cbc.dropna().duplicated().sum()} duplicated values "
      f"among {cbc.notna().sum()} non-NA")
g = both.groupby("cell.barcode.ct")["clone.id"].nunique()
print(f"   barcodes mapping to >1 clone.id: {(g > 1).sum()}")

print("\n3) cell.barcode.ct vs Seurat cell name")
sample_ct = cbc.dropna().str.rsplit("_", n=1).str[0]
bc_ct = cbc.dropna().str.rsplit("_", n=1).str[1]
print("   cell-name examples:", list(obs.index[:3]), "| ct examples:", list(cbc.dropna().iloc[:3]))
bc_name = obs.loc[bc_ct.index].index.to_series().str.extract(r"([ACGT]{16}-\d+)")[0]
print(f"   16-mer barcode in cell name equals one in cell.barcode.ct: {(bc_name == bc_ct).mean():.4f}")
print("   ct sample prefix vs SampleID:")
print(pd.crosstab(sample_ct, obs.loc[sample_ct.index, "SampleID"]))

print("\n4) clone.id vs sample / stage (a clone should live within one lineage-tracing sample)")
per = both.groupby("clone.id")["SampleID"].nunique()
print(f"   clones spanning >1 SampleID: {(per > 1).sum()} of {len(per)}")
print(pd.crosstab(obs["StageID"], cid.notna().rename("has_clone")))

sizes = cid.value_counts()
print("\n5) clone size (cells per clone.id)")
print(sizes.describe())
print(f"   singletons {(sizes == 1).sum()}, size>=2 {(sizes >= 2).sum()}")
