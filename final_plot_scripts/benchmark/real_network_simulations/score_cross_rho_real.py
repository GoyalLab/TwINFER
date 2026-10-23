"""Directional cross-correlation ranking (rho_cross_xy for x->y, rho_cross_yx for y->x, ranked separately)
on the real-network inference, scored against the canonical ground-truth matrices, and compared with
TODO4v2 (directed unsigned) from summary_metrics_table_real_networks_t1_1.csv. Same scoring as
score_cross_rho_directional.py. Date: 2026-10-06.
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/gpfs/projects/b1255/hzhang/TwINFER_KA")
sys.path.insert(0, str(ROOT / "code" / "TwINFER" / "package"))
sys.path.insert(0, str(ROOT / "code" / "TwINFER"))
from benchmarks.network_benchmarks.score import score_mixed_network_sweep as M
from score_cross_rho_directional import directional_ranked, score_one

RW = ROOT / "input_data" / "real_world_networks"
INF = ROOT / "analysis_data" / "paper_analysis" / "real_networks" / "twinfer_inference_allpairs"
TABLE = ROOT / "analysis_data" / "benchmarks" / "network_benchmarks" / "score_results" / "summary_metrics_table_real_networks_t1_1.csv"
OUT = ROOT / "analysis_data" / "synthetic_network_simulations"
TOPO = {"B_cell_activation": "B_cell.txt"}


def gt_df_for(net, n):
    M_topo = np.loadtxt(RW / TOPO.get(net, f"{net}.txt"), delimiter=",")
    names = [f"gene_{i + 1}" for i in range(n)]
    recs = [(names[i], names[j], "+" if M_topo[i, j] > 0 else "-")
            for i in range(n) for j in range(n) if i != j and M_topo[i, j] != 0]
    return pd.DataFrame(recs, columns=["Gene1", "Gene2", "Type"])


rows = []
for f in sorted(glob.glob(f"{INF}/*_all_results.json")):
    d = json.load(open(f))
    ds = d.get("dataset_id") or d.get("sim_type")
    net = re.sub(r"_rep_?\d+$", "", ds)
    if net.startswith("Circadian") or net == "Pluripotent6000":
        continue  # Circadian excluded; Pluripotent6000 is a separate run
    if net.startswith("Circadian"):
        continue  # excluded from the real-network analysis
    if net.startswith("Circadian"):
        continue  # excluded from the real-network analysis
    gt = gt_df_for(net, d["n_genes"])
    ranked = directional_ranked(d)
    if len(ranked) < 3:
        continue
    auprc, auprc_x, f1 = score_one(ranked, gt)
    rows.append(dict(network=net, raw_id=ds, auprc=auprc, auprc_x=auprc_x, f1_topk=f1))
df = pd.DataFrame(rows)
df.to_csv(OUT / "cross_rho_directional_real_replicates.csv", index=False)
mine = df.groupby("network")[["auprc", "auprc_x", "f1_topk"]].mean()

t = pd.read_csv(TABLE)
t = t[(t.variant == "directed_unsigned") & (t.method == "TODO4v2")].set_index("dataset_id")
cmp = mine.join(t[["auprc", "f1_topk"]].rename(columns={"auprc": "todo_auprc", "f1_topk": "todo_f1"}), how="left")
cmp.to_csv(OUT / "cross_rho_directional_real.csv")
print(cmp.round(3).to_string())
print("replicates", len(df), df.groupby("network").size().to_dict())
