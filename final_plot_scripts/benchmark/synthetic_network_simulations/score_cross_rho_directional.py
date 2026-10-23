"""Directional cross-correlation ranking for the synthetic sweeps, compared with TODO4v2 (directed unsigned).
Each pair gives two directed candidates, ranked separately: gene_1 -> gene_2 scored by rho_cross_xy, and
gene_2 -> gene_1 scored by rho_cross_yx. compute_auprc ranks by |score| (as the other scorers do).
AUPRC× = AUPRC / base rate, base rate = n_true_directed / (n*(n-1)) from each dataset's ground truth.

Date:   2026-10-06
Output: analysis_data/synthetic_network_simulations/cross_rho_directional_<family>.csv
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/gpfs/projects/b1255/hzhang/TwINFER_KA")
sys.path.insert(0, str(ROOT / "code" / "TwINFER" / "package"))
sys.path.insert(0, str(ROOT / "code" / "TwINFER"))
from benchmarks.network_benchmarks.analysis_plots import summary_table_todo4v2 as S
from benchmarks.network_benchmarks.score import score_mixed_network_sweep as M

OUT = ROOT / "analysis_data" / "synthetic_network_simulations"
OUT.mkdir(parents=True, exist_ok=True)
BEELINE = ROOT / "code" / "Beeline" / "inputs"


def families():
    A = ROOT / "analysis_data" / "network_sweep_final"
    B = ROOT / "analysis_data" / "mixed_network_sweep"
    e13 = BEELINE / "e13_pos100"
    nsf = BEELINE / "network_sweep_final"
    mns = BEELINE / "mixed_network_sweep"
    return {
        "network_sweep": dict(
            json_dirs=[(f"{A}/e13_pos100/twinfer_inference_allpairs", lambda ds: e13 / ds / "GroundTruthNetwork.csv"),
                       (f"{A}/twinfer_inference_allpairs", lambda ds: nsf / ds / "GroundTruthNetwork.csv")]),
        "mixed_network_sweep": dict(
            json_dirs=[(f"{B}/twinfer_inference_allpairs", lambda ds: mns / ds / "GroundTruthNetwork.csv")]),
    }


def directional_ranked(d):
    tsi = d["twin_score_inputs"]
    dd = pd.DataFrame(tsi["data"], columns=tsi["columns"])
    dd = dd[dd.gene_1 != dd.gene_2]
    fwd = pd.DataFrame({"Gene1": dd.gene_1, "Gene2": dd.gene_2, "EdgeWeight": dd.rho_cross_xy})
    rev = pd.DataFrame({"Gene1": dd.gene_2, "Gene2": dd.gene_1, "EdgeWeight": dd.rho_cross_yx})
    ranked = pd.concat([fwd, rev], ignore_index=True).dropna()
    return ranked


def score_one(ranked, gt_df):
    possible_edges, true_edges = M.build_edge_universe(gt_df)
    n = len({*gt_df.Gene1, *gt_df.Gene2})
    base = len(true_edges) / (n * (n - 1))
    auprc = M.compute_auprc(ranked, gt_df)
    predicted = M.dedupe_predictions(ranked)
    sel, _ = M.top_k_tie_aware_selection(predicted, len(true_edges))
    _, _, f1 = M.precision_recall_f1(sel, true_edges)
    return auprc, auprc / base, f1


def run(name, cfg):
    rows = []
    for json_dir, gt_resolver in cfg["json_dirs"]:
        for f in sorted(glob.glob(f"{json_dir}/*_all_results.json")):
            d = json.load(open(f))
            ds = d.get("dataset_id") or d.get("sim_type")
            gt_path = gt_resolver(ds)
            if not gt_path.exists():
                continue
            gt_df = pd.read_csv(gt_path)
            ranked = directional_ranked(d)
            if len(ranked) < 3:
                continue
            auprc, auprc_x, f1 = score_one(ranked, gt_df)
            rows.append(dict(group=S.strip_rep(ds), raw_id=ds, auprc=auprc, auprc_x=auprc_x, f1_topk=f1))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"cross_rho_directional_{name}_replicates.csv", index=False)
    summ = df.groupby("group")[["auprc", "auprc_x", "f1_topk"]].mean()
    summ.to_csv(OUT / f"cross_rho_directional_{name}.csv")
    print(name, "replicates", len(df))
    print("overall  AUPRC %.3f  AUPRCx %.3f  F1 %.3f" % (df.auprc.mean(), df.auprc_x.mean(), df.f1_topk.mean()))
    return df


if __name__ == "__main__":
    for name, cfg in families().items():
        run(name, cfg)
