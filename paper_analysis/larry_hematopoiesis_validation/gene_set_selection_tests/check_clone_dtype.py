from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import anndata as ad
import pandas as pd
import numpy as np

# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] a = ad.read_h5ad(f'{TWINFER_PROJECT_ROOT}/code/TwINFER/real_data/LSK_d2_d4_d6.h5ad', backed='r')
a = ad.read_h5ad(f'{TWINFER_PROJECT_ROOT}/clean_data/real_data/LSK_d2_d4_d6.h5ad', backed='r')
obs = a.obs
col = obs['clone_id']
print("dtype:", col.dtype)
print("is categorical:", pd.api.types.is_categorical_dtype(col))
if pd.api.types.is_categorical_dtype(col):
    print("n categories:", len(col.cat.categories))
    print("sample categories:", col.cat.categories[:10].tolist())
print("value_counts head:")
print(col.value_counts(dropna=False).head(10))
print("n unique raw:", col.nunique(dropna=False))
print("n empty-string:", (col.astype(str) == '').sum())
print("n literal 'nan' string:", (col.astype(str).str.lower() == 'nan').sum())
print("n literal '-1':", (col.astype(str) == '-1').sum())
