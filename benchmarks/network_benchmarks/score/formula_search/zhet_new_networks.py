import sys
import pandas as pd
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, '.')
from benchmarks.network_benchmarks.score.formula_search.zhet_everywhere import run_real_data, ROOT

EXTRA_SIM_DIRS = {
    "EMT": f"{ROOT}/analysis_data/paper_analysis/EMT/simulate/20260825_224653",
    "B_cell_activation": f"{ROOT}/analysis_data/paper_analysis/B_cell_activation/simulate/20260812_153622",
    "Pluripotent": f"{ROOT}/analysis_data/paper_analysis/Pluripotent/simulate/20260825_224511",
}

rows = []
for tok, topo in [("EMT", "EMT.txt"), ("B_cell_activation", "B_cell.txt"), ("Pluripotent", "Pluripotent.txt")]:
    rows += run_real_data(tok, topo, sim_dir_override=EXTRA_SIM_DIRS[tok])

df = pd.DataFrame(rows)
df.to_csv("zhet_new_networks_results.csv", index=False)
summ = df.groupby(["dataset", "T1"]).agg(mean_auprc=("auprc", "mean"),
                                          mode_sign=("sign", lambda x: int((x > 0).mean() > 0.5) * 2 - 1),
                                          n=("auprc", "size"))
print(summ.round(3).to_string())
