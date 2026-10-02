#!/usr/bin/env python
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_a2a3302f_2026-09-28_why_twinfer_fails; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Gillespie-native test of handoff section 3.1: is a sister cell exactly as informative
as the (impossible) same-cell measurement?

No trunk trajectory needed: right after division the two twins are essentially identical
(they've had ~0 time to diverge), so twin_A's own value at t1 (first post-division step)
IS a valid proxy for "the cell itself" at the division moment. Then:
  same-cell(t1->t2)  := corr(twin_A[t1], twin_A[t2])   (same lineage, continuing)
  sister(t1->t2)     := corr(twin_A[t1], twin_B[t2])   (sibling, diverged at division)
compared across a range of t2 (protein columns, per project convention: no raw mRNA).
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import numpy as np
import pandas as pd

ROOT = f'{TWINFER_PROJECT_ROOT}'
F = f"{ROOT}/simulation_data/real_data/df_rows_0_0_0_0_0_12072026_005933_ncells_6000_mCAD_1_0_c733c5af.csv"
GENES = ["Fgf8", "Emx2", "Pax6", "Coup", "Sp8"]

df = pd.read_csv(F)
cols = [f"gene_{i+1}_protein" for i in range(5)]
steps = sorted(df.time_step.unique())
t1 = steps[0]
print(f"time_steps: {steps[0]}..{steps[-1]} (n={len(steps)})")

A = df[(df.replicate == 1) & (df.time_step == t1)].sort_values("clone_id").set_index("clone_id")[cols]
B0 = df[(df.replicate == 2) & (df.time_step == t1)].sort_values("clone_id").set_index("clone_id")[cols]
common0 = A.index.intersection(B0.index)
print(f"n twin pairs at t1: {len(common0)}")
# sanity: twins should be ~identical at t1 (just post-division)
diffs = (A.loc[common0].to_numpy() - B0.loc[common0].to_numpy())
print("twin_A vs twin_B at t1, mean abs relative diff per gene:",
      np.round(np.mean(np.abs(diffs) / (np.abs(A.loc[common0].to_numpy()) + 1e-9), axis=0), 4))

rows = []
for t2 in steps[1:]:
    Bt2 = df[(df.replicate == 2) & (df.time_step == t2)].sort_values("clone_id").set_index("clone_id")[cols]
    At2 = df[(df.replicate == 1) & (df.time_step == t2)].sort_values("clone_id").set_index("clone_id")[cols]
    common = common0.intersection(Bt2.index).intersection(At2.index)
    a1 = A.loc[common]
    a2 = At2.loc[common]
    b2 = Bt2.loc[common]
    for gi, g in enumerate(GENES):
        c = cols[gi]
        same_cell = np.corrcoef(a1[c], a2[c])[0, 1]
        sister = np.corrcoef(a1[c], b2[c])[0, 1]
        rows.append(dict(t2=t2, gene=g, same_cell=same_cell, sister=sister, n=len(common)))

out = pd.DataFrame(rows)
out.to_csv(f"{ROOT}/analysis_data/boolode_sims_real_networks/diagnostics/gillespie_same_cell_vs_sister_mCAD.csv", index=False)
print(out.pivot(index="t2", columns="gene", values="same_cell").round(3))
print()
print(out.pivot(index="t2", columns="gene", values="sister").round(3))
print()
gap = out.copy()
gap["gap"] = gap.same_cell - gap.sister
print("max |same_cell - sister| gap across all t2/genes:", gap.gap.abs().max())
print("mean |gap|:", gap.gap.abs().mean())
