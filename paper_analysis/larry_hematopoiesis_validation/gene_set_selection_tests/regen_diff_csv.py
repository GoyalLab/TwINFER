from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import anndata as ad
import pandas as pd
import numpy as np

# [2026-10-01 commented out: /scratch/gzu5140 is purged scratch; see REPOINT_LOG.tsv] PROC = '/scratch/gzu5140/ka_twinfer/larry_dataset/processed'
from twinfer.utils.paths import get_larry_dataset_dir  # [2026-10-01 added]
PROC = str(get_larry_dataset_dir() / "processed")
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] H5AD = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/real_data/LSK_d2_d4_d6.h5ad'
H5AD = f'{TWINFER_PROJECT_ROOT}/clean_data/real_data/LSK_d2_d4_d6.h5ad'

obs_qc = pd.read_csv(f'{PROC}/qc_filtered/obs_metadata.csv', index_col=0)
a = ad.read_h5ad(H5AD, backed='r')
obs_h5 = a.obs
h5_key = obs_h5['Library'].astype(str).to_numpy() + ':' + obs_h5.index.astype(str).to_numpy()
h5_df = pd.DataFrame({'key': h5_key, 'clone_id': obs_h5['clone_id'].to_numpy()}).drop_duplicates('key').set_index('key')

merged = obs_qc.join(h5_df, how='inner')
merged = merged[merged['clone_id'] != -1]

sib_ours = merged.groupby('larry_clone_singletcode', group_keys=False).apply(lambda g: frozenset(g.index), include_groups=False)
sib_h5 = merged.groupby('clone_id', group_keys=False).apply(lambda g: frozenset(g.index), include_groups=False)
ours_map = {k: s for s in sib_ours for k in s}
h5_map = {k: s for s in sib_h5 for k in s}

rows = []
for k in merged.index:
    if ours_map[k] != h5_map[k]:
        rows.append({
            "key": k,
            "our_clone": merged.loc[k, "larry_clone_singletcode"],
            "h5ad_clone_id": merged.loc[k, "clone_id"],
            "our_sibling_group_size": len(ours_map[k]),
            "h5ad_sibling_group_size": len(h5_map[k]),
        })
diff_df = pd.DataFrame(rows)
diff_df.to_csv(f"{PROC}/singletcode_vs_h5ad_diff_cells.csv", index=False)
print(f"wrote {len(diff_df)} rows")
