from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import sys, pandas as pd, numpy as np
from scipy.stats import spearmanr
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0,"/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/work_in_progress/benchmark")
from benchmarks.network_benchmarks.to_beeline import boolode_real_convert as c
R=f'{TWINFER_PROJECT_ROOT}'; RW=f"{R}/input_data/real_world_networks"; OLD=f"{R}/simulation_data/twinfer_format"

def check(net, genes, matrix_path, targets):
    d = pd.read_csv(f"{c.SIM}/{net}/replicate_0/twin_final_states.csv")
    tA = d[d.source=="twin_A"].sort_values(["step","pair_id"])
    steps = sorted(tA.step.unique()); t_div, t_final = steps[0], steps[-1]
    M = np.loadtxt(matrix_path, delimiter=",")
    gi = {g:i for i,g in enumerate(genes)}
    def s(g,t): return tA[tA.step==t].sort_values("pair_id")[g].values
    for tgt in targets:
        print(f"\n=== {net}: target={tgt} (t_div={t_div}, t_final={t_final}) ===")
        rows=[]
        for reg in genes:
            is_edge = M[gi[reg], gi[tgt]] != 0
            r0 = spearmanr(s(reg,t_div), s(tgt,t_div))[0]
            rF = spearmanr(s(reg,t_div), s(tgt,t_final))[0]
            rows.append((reg, is_edge, r0, rF))
        for reg, is_edge, r0, rF in sorted(rows, key=lambda x: -abs(x[3])):
            flag = "EDGE" if is_edge else "    "
            print(f"  {flag} {reg:10s}->{tgt:10s}  corr(t_div,t_div)={r0:6.2f}  corr(t_div,t_final)={rF:6.2f}")

HSC_GENES = ["Gata2","Gata1","Gfi1","Fog1","Eklf","Fli1","Scl","Cebpa","Pu1","cJun","EgrNab"]
check("HSC", HSC_GENES, f"{OLD}/HSC/interaction_matrix.txt", ["Pu1","Gata2"])

PLURI_GENES = c.NETS["Pluripotent_real"]
check("Pluripotent_real", PLURI_GENES, f"{RW}/Pluripotent.txt", ["ZNF398"])
