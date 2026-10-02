from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_a2a3302f_2026-09-28_why_twinfer_fails; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import sys, time
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/package")
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/work_in_progress/benchmark")
print("importing...", flush=True)
t_import = time.time()
from benchmarks.network_benchmarks.infer.infer_real_network_allpairs import NETWORKS, T1, T2, make_base_config, build_simulation_record
print(f"import done in {time.time()-t_import:.1f}s", flush=True)
net = "EMT"
cfg = NETWORKS[net]
bc = make_base_config(net, cfg)
path = f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/EMT/simulate/20260825_224653/df_rows_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_25082026_234904_ncells_6000_EMT_rep_0_839e1156.csv'
t0 = time.time()
print("starting inference...", flush=True)
r = build_simulation_record(path, net, "0_839e1156_TEST", bc, T1, T2,
                             f'{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/real_networks/why_twinfer_fails',
                             n_cores=16)
print(r, "elapsed", time.time() - t0, flush=True)
