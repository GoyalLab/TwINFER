from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import anndata as ad
import pandas as pd

# [2026-10-01 commented out: /scratch/gzu5140 is purged scratch; see REPOINT_LOG.tsv] PROC = '/scratch/gzu5140/ka_twinfer/larry_dataset/processed'
from twinfer.utils.paths import get_larry_dataset_dir  # [2026-10-01 added]
PROC = str(get_larry_dataset_dir() / "processed")

# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] a = ad.read_h5ad(f'{TWINFER_PROJECT_ROOT}/code/TwINFER/real_data/LSK_d2_d4_d6.h5ad', backed='r')
a = ad.read_h5ad(f'{TWINFER_PROJECT_ROOT}/clean_data/real_data/LSK_d2_d4_d6.h5ad', backed='r')
obs = a.obs
has_clone = obs['clone_id'].notna().to_numpy()
h5_keys_all = (obs['Library'].astype(str).to_numpy() + ':' + obs.index.astype(str).to_numpy())
h5_keys_clone = set(h5_keys_all[has_clone])

cells_qc = set(open(f'{PROC}/qc_filtered/cells.txt').read().split())

print("h5ad total cells:", obs.shape[0])
print("h5ad cells WITH clone_id:", has_clone.sum())
print("our final (qc_filtered/) cells:", len(cells_qc))

overlap_clone = h5_keys_clone & cells_qc
gap = h5_keys_clone - cells_qc

print("h5ad clone_id cells that ARE in our final set:", len(overlap_clone))
print("h5ad clone_id cells NOT in our final set:", len(gap))
