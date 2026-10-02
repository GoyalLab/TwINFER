from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import anndata as ad
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score

# [2026-10-01 commented out: /scratch/gzu5140 is purged scratch; see REPOINT_LOG.tsv] PROC = '/scratch/gzu5140/ka_twinfer/larry_dataset/processed'
from twinfer.utils.paths import get_larry_dataset_dir  # [2026-10-01 added]
PROC = str(get_larry_dataset_dir() / "processed")
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] H5AD = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/real_data/LSK_d2_d4_d6.h5ad'
H5AD = f'{TWINFER_PROJECT_ROOT}/clean_data/real_data/LSK_d2_d4_d6.h5ad'

obs_qc = pd.read_csv(f'{PROC}/qc_filtered/obs_metadata.csv', index_col=0)
print("qc_filtered/ cells:", len(obs_qc))
print("distinct clones:", obs_qc['larry_clone_singletcode'].nunique())

a = ad.read_h5ad(H5AD, backed='r')
obs_h5 = a.obs
h5_key = obs_h5['Library'].astype(str).to_numpy() + ':' + obs_h5.index.astype(str).to_numpy()
h5_clone = obs_h5['clone_id'].to_numpy()
h5_df = pd.DataFrame({'key': h5_key, 'clone_id': h5_clone}).drop_duplicates('key').set_index('key')

merged = obs_qc.join(h5_df, how='inner')
merged = merged[merged['clone_id'] != -1]
print("overlap cells both call clone-barcoded:", len(merged))

ari = adjusted_rand_score(merged['larry_clone_singletcode'], merged['clone_id'])
print("ARI:", ari)

# disagreement count via sibling-set method
sib_ours = merged.groupby('larry_clone_singletcode').apply(lambda g: frozenset(g.index))
sib_h5 = merged.groupby('clone_id').apply(lambda g: frozenset(g.index))
ours_map = {}
for s in sib_ours:
    for k in s:
        ours_map[k] = s
h5_map = {}
for s in sib_h5:
    for k in s:
        h5_map[k] = s
disagree = sum(1 for k in merged.index if ours_map[k] != h5_map[k])
print(f"disagree: {disagree} / {len(merged)} = {100*disagree/len(merged):.2f}%")

# gap: h5ad clone cells not in our qc_filtered at all
final_cells = set(obs_qc.index)
has_clone = obs_h5['clone_id'].to_numpy() != -1
h5_clone_keys = set(pd.Series(h5_key)[has_clone])
gap = h5_clone_keys - final_cells
print("h5ad clone_id cells NOT in new qc_filtered/:", len(gap))
