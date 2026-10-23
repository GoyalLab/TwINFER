"""Remake the real-network summary table with HSC re-scored against the CANONICAL network (no Beeline ground truth).
Base = analysis_data/benchmarks/network_benchmarks/score_results/summary_metrics_table_real_networks_t1_1.csv
(rows for the other networks are unchanged, as checked against the canonical radar data). HSC rows (all 8 methods,
all 3 variants) are replaced with canonical scores; ALL rows are recomputed as the mean over datasets.
Output: clean_data/.../summary_metrics_table_real_networks_t1_1_20261006.csv (original left untouched).
Date: 2026-10-06
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/gpfs/projects/b1255/hzhang/TwINFER_KA")
sys.path.insert(0, str(ROOT / "code" / "TwINFER" / "package"))
sys.path.insert(0, str(ROOT / "code" / "TwINFER"))
from benchmarks.network_benchmarks.score import score_mixed_network_sweep as M
from benchmarks.network_benchmarks.analysis_plots import summary_table_todo4v2 as S
from rescore_hsc_canonical import canonical_gt  # HSC-only helper; generic version below

BASE = ROOT / "analysis_data" / "benchmarks" / "network_benchmarks" / "score_results" / "summary_metrics_table_real_networks_t1_1.csv"
OUT = Path("/home/gzu5140/TwINFER_KA/clean_data/benchmarks/network_benchmarks/score_results/summary_metrics_table_real_networks_t1_1_20261006.csv")
OUTPUTS = ROOT / "benchmarking_analysis" / "twinfer_real" / "outputs"
TW = ROOT / "analysis_data" / "paper_analysis" / "real_networks" / "twinfer_inference_allpairs"
RESCORE = ["HSC", "GSD", "VSC", "mCAD", "EMT", "Pluripotent", "B_cell_activation"]  # all re-scored on canonical matrices
BEELINE_T1 = ROOT / "analysis_data" / "paper_analysis" / "real_networks" / "beeline_inference"  # t1=1 outputs
CANON_FILE = {"B_cell_activation": "B_cell"}  # canonical matrix name where it differs from the network key
VARIANT_SUFFIX = {"directed_unsigned": "", "signed_directed": "_signed", "undirected": "_undirected"}


def competitor_metrics(ranked, gt):
    out = {}
    _, true_e = M.build_edge_universe(gt)
    true_s = M.build_signed_edge_universe(gt)
    _, true_u = M.build_undirected_edge_universe(gt)
    pred = M.dedupe_predictions(ranked)
    pred_s = M.add_predicted_sign(pred)
    pred_u = M.dedupe_predictions_undirected(ranked)
    out["auprc"] = M.compute_auprc(ranked, gt)
    out["auprc_signed"] = M.compute_auprc_signed(ranked, gt)
    out["auprc_undirected"] = M.compute_auprc_undirected(ranked, gt)
    sel, _ = M.top_k_tie_aware_selection(pred, len(true_e))
    p, _, f = M.precision_recall_f1(sel, true_e)
    out.update(precision_topk=p, f1_topk=f)
    sel, _ = M.top_k_tie_aware_selection_signed(pred_s, len(true_s))
    p, _, f = M.precision_recall_f1(sel, true_s)
    out.update(precision_topk_signed=p, f1_topk_signed=f)
    sel, _ = M.top_k_tie_aware_selection_undirected(pred_u, len(true_u))
    p, _, f = M.precision_recall_f1(sel, true_u)
    out.update(precision_topk_undirected=p, f1_topk_undirected=f)
    return out


def canonical_gt_for(net):
    Mx = np.loadtxt(ROOT / "input_data" / "real_world_networks" / f"{CANON_FILE.get(net, net)}.txt", delimiter=",")
    n = Mx.shape[0]
    recs = [(f"gene_{i + 1}", f"gene_{j + 1}", "+" if Mx[i, j] > 0 else "-")
            for i in range(n) for j in range(n) if i != j and Mx[i, j] != 0]
    return pd.DataFrame(recs, columns=["Gene1", "Gene2", "Type"])


def net_rows(net):
    gt = canonical_gt_for(net)
    per = {}  # method -> list of per-replicate dicts
    for f in sorted(glob.glob(f"{TW}/{net}_rep_*_all_results.json")):
        row = S.todo4v2_row_for_dataset(f, gt)
        if row:
            per.setdefault("TODO4v2", []).append(row)
    roots = [OUTPUTS / net] if (OUTPUTS / net).exists() else [BEELINE_T1 / net]
    for rk in sorted(glob.glob(f"{roots[0]}/*/*/rankedEdges.csv")):
        alg = Path(rk).parent.name
        r = pd.read_csv(rk, sep="\t")
        r = r.rename(columns={r.columns[0]: "Gene1", r.columns[1]: "Gene2", r.columns[2]: "EdgeWeight"}).dropna()
        if len(r):
            per.setdefault(alg, []).append(competitor_metrics(r, gt))
    rows = []
    for method, items in per.items():
        df = pd.DataFrame(items)
        for variant, suf in VARIANT_SUFFIX.items():
            rows.append(dict(variant=variant, dataset_id=net, method=method,
                             f1_topk=df[f"f1_topk{suf}"].mean(), precision_topk=df[f"precision_topk{suf}"].mean(),
                             auprc=df[f"auprc{suf}"].mean()))
    return pd.DataFrame(rows)


def main():
    base = pd.read_csv(BASE)
    base = base[~base.dataset_id.isin(RESCORE + ["ALL"])]
    new = pd.concat([net_rows(net) for net in RESCORE], ignore_index=True)
    print("replicates per network/method:")
    table = pd.concat([base, new], ignore_index=True)
    all_rows = (table.groupby(["variant", "method"])[["f1_topk", "precision_topk", "auprc"]].mean().reset_index()
                .assign(dataset_id="ALL"))
    table = pd.concat([table, all_rows[table.columns]], ignore_index=True)
    table = table.round(4)
    table.to_csv(OUT, index=False)
    print("wrote", OUT, len(table), "rows")
    print(new[new.variant == "directed_unsigned"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
