#!/usr/bin/env python
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_a2a3302f_2026-09-28_why_twinfer_fails; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Fill the missing EMT ALL_PAIRS TwINFER result: infer_real_network_allpairs.py's build_new_tasks()
only looks in the `simulate/latest` symlink, which for EMT points to `multistate_2000steps` (a
DIFFERENT simulation, tuned with kadd_scale=2 to induce multistability -- not a renamed copy of the
same data). The canonical, non-multistate EMT_rep_N files this whole session has used live in
`simulate/20260825_224653/` instead, which the symlink never reaches. This script reuses
infer_real_network_allpairs.py's exact settings (make_base_config, build_simulation_record, T1=1,
T2=20, OUTPUT_DIR) but points directly at the canonical directory, so the output lands in the
normal place and is picked up by everything downstream (todo4v2 scoring, full_metrics_table.py)
with no other changes needed.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob
import os
import re
import sys

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/package")
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/work_in_progress/benchmark")
from benchmarks.network_benchmarks.infer.infer_real_network_allpairs import (
    NETWORKS, T1, T2, OUTPUT_DIR, make_base_config, build_simulation_record,
)

net = "EMT"
cfg = NETWORKS[net]
base_config = make_base_config(net, cfg)
sim_dir = f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/EMT/simulate/20260825_224653'
token = cfg["filename_token"]
file_re = re.compile(rf"^df_rows_.*_ncells_\d+_{re.escape(token)}_rep_(\d+)_([0-9a-fA-F]{{8}})\.csv$")

tasks = []
for f in sorted(glob.glob(f"{sim_dir}/df_*_ncells_*.csv")):
    name = os.path.basename(f)
    if name.startswith("simulation_before_division"):
        continue
    m = file_re.match(name)
    if not m:
        continue
    rep, file_hash = m.groups()
    rep_id = f"{rep}_{file_hash}"
    tasks.append((f, net, rep_id, base_config))

print(f"Found {len(tasks)} EMT replicate files in the canonical (non-multistate) directory:")
for path, _, rep_id, _ in tasks:
    print(f"  rep_id={rep_id}  {os.path.basename(path)}")

for path, sim_type, rep_id, bc in tasks:
    print(f"\n=== running {sim_type} rep {rep_id} ===", flush=True)
    result = build_simulation_record(path, sim_type, rep_id, bc, T1, T2, str(OUTPUT_DIR), n_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    print(result, flush=True)
