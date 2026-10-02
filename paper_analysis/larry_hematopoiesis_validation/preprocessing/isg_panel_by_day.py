"""Direct check: does the day-2-only interferon-stimulated-gene (ISG) signature actually resolve
by days 4/6, or is it present throughout and just diluted out by cell-count weighting in the
pooled all-cells run? Tracks % cells detecting + mean CP10k expression of a small ISG panel across
days 2/4/6 directly -- cheaper and more precise than re-diffing whole gene-set membership per day.

Run directly: python3 isg_panel_by_day.py
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import os

import numpy as np
import pandas as pd
import scipy.io as sio

SOURCE = f'{TWINFER_PROJECT_ROOT}/finalized_data/LARRY_data/filtered'
ISG_GENES = ["Isg15", "Irf1", "Irf7", "Rsad2", "Oasl1", "Cxcl10", "Gbp2", "Ly6e"]


def log(m):
    print(m, flush=True)


X = sio.mmread(os.path.join(SOURCE, "larry_qc_counts.mtx")).tocsr()
genes = pd.Index(open(os.path.join(SOURCE, "genes.txt")).read().split())
obs = pd.read_csv(os.path.join(SOURCE, "obs_metadata.csv"), index_col=0)
assert X.shape == (len(obs), len(genes))

day = obs["library"].astype(str).str.extract(r"_d(\d+)_", expand=False)
assert day.notna().all()
day = day.astype(int).to_numpy()
days = sorted(set(day))
log(f"days present: {days}  |  cells per day: " +
    ", ".join(f"{d}={int((day == d).sum()):,}" for d in days))

lib_size = np.asarray(X.sum(axis=1)).ravel()  # per-cell total counts, for CP10k normalization

gidx = {g: genes.get_loc(g) for g in ISG_GENES if g in genes}
missing = [g for g in ISG_GENES if g not in genes]
if missing:
    log(f"WARNING: not in genes.txt (dropped): {missing}")

panel = X[:, list(gidx.values())].toarray().astype(np.float64)  # (n_cells, len(gidx))
cp10k = panel / lib_size[:, None] * 1e4

log(f"\n{'gene':<10}" + "".join(f"{'day'+str(d)+' %det':>12}{'day'+str(d)+' CP10k':>14}" for d in days))
for j, g in enumerate(gidx):
    row = f"{g:<10}"
    for d in days:
        m = day == d
        pct_det = float((panel[m, j] > 0).mean()) * 100
        mean_cp10k = float(cp10k[m, j].mean())
        row += f"{pct_det:>11.1f}%{mean_cp10k:>14.2f}"
    log(row)

log("\npanel-level summary (mean across the 8 genes, per day):")
pct_det_by_day = {d: float((panel[day == d] > 0).mean()) * 100 for d in days}
mean_cp10k_by_day = {d: float(cp10k[day == d].mean()) for d in days}
for d in days:
    log(f"  day {d}: mean %detected = {pct_det_by_day[d]:.1f}%   mean CP10k = {mean_cp10k_by_day[d]:.2f}")
