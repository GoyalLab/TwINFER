from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_a2a3302f_2026-09-28_why_twinfer_fails; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import pandas as pd
import numpy as np
from scipy.stats import rankdata

df = pd.read_csv(f'{TWINFER_PROJECT_ROOT}/simulation_data/real_data/df_rows_0_0_0_0_0_12072026_005933_ncells_6000_mCAD_1_0_c733c5af.csv')
genes = [f"gene_{i+1}" for i in range(5)]
for t in [0, 1]:
    dt = df[df.time_step == t]
    A = dt[dt.replicate == 1].sort_values("clone_id")
    B = dt[dt.replicate == 2].sort_values("clone_id")
    print(f"--- t={t} ---")
    for g in genes:
        for suffix in ["mRNA", "protein"]:
            c = f"{g}_{suffix}"
            r = np.corrcoef(rankdata(A[c]), rankdata(B[c]))[0, 1]
            mean_a = A[c].mean()
            print(f"  {g} {suffix:8s} corr={r:.3f}  mean_count={mean_a:.1f}")
