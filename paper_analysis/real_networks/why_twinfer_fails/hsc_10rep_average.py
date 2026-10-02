from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_a2a3302f_2026-09-28_why_twinfer_fails; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import glob
import numpy as np
import pandas as pd
from scipy.stats import rankdata

R = f'{TWINFER_PROJECT_ROOT}/simulation_data/real_data'
files = sorted(f for f in glob.glob(f"{R}/df_rows_*HSC_balanced*.csv") if "before_division" not in f)[:10]
print(f"using {len(files)} replicates")
for f in files:
    print(" ", f.split("/")[-1])

genes = [l.strip() for l in open(f'{TWINFER_PROJECT_ROOT}/simulation_data/twinfer_format/HSC/gene_order.txt')]
gi = {g: i + 1 for i, g in enumerate(genes)}
reg, tgt = "Gata1", "Fog1"
cr, ct = f"gene_{gi[reg]}_mRNA", f"gene_{gi[tgt]}_mRNA"

all_S, all_C = [], []
for f in files:
    df = pd.read_csv(f, usecols=["time_step", "clone_id", "replicate", cr, ct])
    steps = sorted(df.time_step.unique())
    Ss, Cs = [], []
    for t in steps:
        dt = df[df.time_step == t]
        A = dt[dt.replicate == 1].sort_values("clone_id")
        B = dt[dt.replicate == 2].sort_values("clone_id")
        ra_r, ra_t = rankdata(A[cr]), rankdata(A[ct])
        rb_r, rb_t = rankdata(B[cr]), rankdata(B[ct])
        S = np.corrcoef(np.concatenate([ra_r, rb_r]), np.concatenate([ra_t, rb_t]))[0, 1]
        C1 = np.corrcoef(ra_r, rb_t)[0, 1]
        C2 = np.corrcoef(rb_r, ra_t)[0, 1]
        C = (C1 + C2) / 2
        Ss.append(S); Cs.append(C)
    all_S.append(Ss); all_C.append(Cs)

all_S = np.array(all_S)  # (n_reps, n_t)
all_C = np.array(all_C)
mean_S = all_S.mean(axis=0)
mean_C = all_C.mean(axis=0)
se_S = all_S.std(axis=0, ddof=1) / np.sqrt(len(files))
se_C = all_C.std(axis=0, ddof=1) / np.sqrt(len(files))
gap = np.abs(mean_S) - np.abs(mean_C)
se_gap = np.sqrt(se_S**2 + se_C**2)

print(f"\n{reg}->{tgt}, averaged over {len(files)} replicates (mRNA)")
print(f"{'t':>4s} {'mean|S|':>9s} {'mean|C|':>9s} {'gap':>8s} {'se_gap':>8s} {'gap/se':>8s}")
for t in [0, 1, 5, 10, 20, 30, 40, 48]:
    idx = t
    print(f"{t:4d} {abs(mean_S[idx]):9.4f} {abs(mean_C[idx]):9.4f} {gap[idx]:+8.4f} {se_gap[idx]:8.4f} {gap[idx]/se_gap[idx] if se_gap[idx]>0 else float('nan'):8.2f}")
