from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_807d072d_2026-09-20_real_network_sim; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import sys
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/network_analysis")
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/package")
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/work_in_progress/benchmark")
import numpy as np
from simulations.network_analysis.predict_multistability import MeanField, load_params
from benchmarks.network_benchmarks.simulate.real_network_multistate_sim import NETWORKS as SIM_SPEC

def read_input_matrix(path):
    M = np.loadtxt(path, dtype=int, delimiter=",")
    if M.ndim == 0:
        M = M.reshape((1,1))
    return M

for net in ["VSC", "GSD", "HSC"]:
    spec = SIM_SPEC[net]
    M = read_input_matrix(f"{TWINFER_PROJECT_ROOT}/input_data/real_world_networks/{net}.txt")
    P = load_params()
    mf = MeanField(M, P, kadd_scale=spec["kadd_scale"], n_hill=spec["n_hill"])
    fps = [f for f in mf.fixed_points(n_starts=250) if f["stable"]]
    prot = np.array([f["p"] for f in fps])  # (k, n_genes)
    print(f"\n=== {net}  k={len(fps)} states,  {prot.shape[1]} genes ===")
    header = "gene".ljust(10) + "".join(f"state{s}".rjust(14) for s in range(len(fps))) + "   spread(max/min)"
    print(header)
    for gi in range(prot.shape[1]):
        vals = prot[:, gi]
        spread = (vals.max()+1) / (vals.min()+1)
        print(f"gene_{gi+1}".ljust(10) + "".join(f"{v:14.1f}" for v in vals) + f"   {spread:8.2f}x")
