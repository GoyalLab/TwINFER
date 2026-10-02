"""todo4v2 (old/new/nogate) scoring of the BoolODE real-network all-pairs inference JSONs, plus a side-by-side
with the TwinScore_supplement (gated bootstrap) metrics computed on the same replicates.
Output: todo4v2_boolode_real_networks_results.csv (this dir)."""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob, json, os, re, numpy as np, pandas as pd
from benchmarks.network_benchmarks.score import todo4v2_sim_scoring as t
R = f'{TWINFER_PROJECT_ROOT}'; RW = f"{R}/input_data/real_world_networks"; OLD = f"{R}/simulation_data/twinfer_format"
TOPO = {"B_cell_activation": f"{RW}/B_cell.txt", "EMT_real": f"{RW}/EMT.txt", "Pluripotent_real": f"{RW}/Pluripotent.txt",
        **{n: f"{OLD}/{n}/interaction_matrix.txt" for n in ("GSD", "HSC", "mCAD", "VSC")}}
INF = f"{R}/analysis_data/boolode_sims_real_networks/twinfer_inference_allpairs"
TS = f"{R}/analysis_data/boolode_sims_real_networks/twinscore_supp_gated_bootstrap"
rows = []
for p in sorted(glob.glob(f"{INF}/*_all_results.json")):
    net, rep = re.match(r"(.+?)_rep_(\d+)_", os.path.basename(p)).groups()
    r = t.score_one_json(p, lambda d: TOPO[d["sim_type"]])
    if not r: continue
    r.update(net=net, rep=int(rep))
    m = f"{TS}/{net}_rep_{rep}_metrics.json"
    if os.path.exists(m):
        mm = json.load(open(m))["metrics"]
        for k in ("TwinScore", "PAIR", "D(PIDC)", "abs_C"): r[f"ts_{k}_auprc_x"] = mm[k]["auprc_x"]
    rows.append(r)
# [2026-09-30 replaced os.path.dirname(os.path.abspath(__file__)): the CSV used to be written next to the script in the (original) work_in_progress/benchmark dir; that location is kept]
# [2026-09-30 commented out: result files now in clean_data/, see REPOINT_LOG.tsv] df = pd.DataFrame(rows); df.to_csv(os.path.join(f"{TWINFER_PROJECT_ROOT}/code/TwINFER/work_in_progress/benchmark", "todo4v2_boolode_real_networks_results.csv"), index=False)
df = pd.DataFrame(rows); df.to_csv(os.path.join(f"{TWINFER_PROJECT_ROOT}/clean_data/benchmarks/network_benchmarks/score_results", "todo4v2_boolode_real_networks_results.csv"), index=False)
print(len(df), "replicates scored")
