from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_a2a3302f_2026-09-28_why_twinfer_fails; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import numpy as np, pandas as pd, warnings
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
warnings.filterwarnings("ignore")
R = f'{TWINFER_PROJECT_ROOT}'
OLD = f"{R}/simulation_data/twinfer_format"
genes = [l.strip() for l in open(f"{OLD}/HSC/gene_order.txt")]
M = np.loadtxt(f"{OLD}/HSC/interaction_matrix.txt", delimiter=",")
f = f"{R}/simulation_data/real_data/df_rows_0_0_0_0_0_0_0_0_0_0_0_12072026_020848_ncells_6000_HSC_balanced_0_0_6bff4482.csv"
cols = ["cell_id","time_step","clone_id"]+[f"gene_{i+1}_protein" for i in range(len(genes))]
df = pd.read_csv(f, usecols=cols)
last = df[df.time_step==df.time_step.max()]
cs = last.clone_id.value_counts(); pairs = cs[cs==2].index
sub = last[last.clone_id.isin(pairs)]
a = sub.groupby("clone_id").nth(0).sort_values("clone_id"); b = sub.groupby("clone_id").nth(1).sort_values("clone_id")
keep_idx = [i for i,g in enumerate(genes) if a[f"gene_{i+1}_protein"].std()>1e-9 and b[f"gene_{i+1}_protein"].std()>1e-9]
print("dropped (zero-variance):", [genes[i] for i in range(len(genes)) if i not in keep_idx])
gc = [f"gene_{i+1}_protein" for i in keep_idx]
Mk = M[np.ix_(keep_idx, keep_idx)]
n = len(keep_idx)
Ra = np.column_stack([rankdata(a[c]) for c in gc]); Rb = np.column_stack([rankdata(b[c]) for c in gc])
pool = np.vstack([Ra, Rb]); S = np.corrcoef(pool.T)
C_raw = np.corrcoef(np.hstack([Ra, Rb]).T)[:n, n:]; C = (C_raw + C_raw.T) / 2.0
iu = np.triu_indices(n, 1)
corr_S_C = np.corrcoef(S[iu], C[iu])[0, 1]
und = ((Mk != 0) | (Mk.T != 0))[iu].astype(int)
auprc_S = average_precision_score(und, np.abs(S)[iu]) / und.mean()
auprc_zreg = average_precision_score(und, np.abs((S-C))[iu]) / und.mean()
twin_diag = np.mean([C[i,i] for i in range(n)])
print(f"HSC (Fli1/Scl dropped, n={n} genes): twin(C diag) mean={twin_diag:.3f}  corr(S,C)={corr_S_C:.3f}  "
      f"AUPRC-x |S|={auprc_S:.2f}  AUPRC-x |S-C|={auprc_zreg:.2f} (change {auprc_zreg-auprc_S:+.2f})")
