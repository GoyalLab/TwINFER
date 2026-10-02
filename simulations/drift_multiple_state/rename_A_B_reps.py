"""
Renumber the regenerated A_B (no-reg) 2-state drift sims to a clean
rep_1 .. rep_20 set. All A_B sims use identical params (rows [[0,0]]); the
'rep' field in the filename is just a label, and the regen races left some
duplicate rep numbers. Each CSV is an independent stochastic realization, so
we treat the (oldest) 20 of them as replicates 1..20.

Renames df_rows_0_0_<old>_... -> df_rows_0_0_<i>_... (keeps timestamp + hash),
and the matching simulation_before_division_ companion. Any surplus files go
to drift_simulation/extra_A_B/ (kept, not deleted).
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob
import os
import re

D = f'{TWINFER_PROJECT_ROOT}/simulation_data/drift_simulation'
N = 20
PAT = re.compile(
    r"df_rows_0_0_(\d+)_(\d+_\d+)_ncells_6000_A_B_no_reg_2_states_([0-9a-f]+)\.csv")

files = sorted(glob.glob(os.path.join(D, "df_rows_0_0_*_A_B_no_reg_2_states_*.csv")),
               key=os.path.getmtime)
keep, extra = files[:N], files[N:]

if extra:
    os.makedirs(os.path.join(D, "extra_A_B"), exist_ok=True)

plan = []
for i, f in enumerate(keep, 1):
    b = os.path.basename(f)
    m = PAT.match(b)
    ts, h = m.group(2), m.group(3)
    new_b = f"df_rows_0_0_{i}_{ts}_ncells_6000_A_B_no_reg_2_states_{h}.csv"
    plan.append((b, new_b))

print("RENAME PLAN:")
for old, new in plan:
    print(f"  {old}\n   -> {new}")
print(f"\nEXTRA (moved to extra_A_B/): {[os.path.basename(x) for x in extra]}")

# execute
for old_b, new_b in plan:
    if old_b == new_b:
        continue
    os.rename(os.path.join(D, old_b), os.path.join(D, new_b))
    comp = os.path.join(D, f"simulation_before_division_{old_b}")
    if os.path.exists(comp):
        os.rename(comp, os.path.join(D, f"simulation_before_division_{new_b}"))

for f in extra:
    b = os.path.basename(f)
    os.rename(f, os.path.join(D, "extra_A_B", b))
    comp = os.path.join(D, f"simulation_before_division_{b}")
    if os.path.exists(comp):
        os.rename(comp, os.path.join(D, "extra_A_B", f"simulation_before_division_{b}"))

print("\nDONE. A_B reps now:",
      sorted(int(PAT.match(os.path.basename(x)).group(1))
             for x in glob.glob(os.path.join(D, "df_rows_0_0_*_A_B_no_reg_2_states_*.csv"))))
