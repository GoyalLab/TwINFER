"""Re-score the HSC real network against the CANONICAL ground truth (input_data/real_world_networks/HSC.txt,
23 off-diagonal directed edges), for all eight methods, and compare with the Beeline HSC ground truth used in
summary_metrics_table_real_networks_t1_1.csv.

Gene names: gene_1..gene_11 follow network_labels.txt (#HSC): cJun, SCL, Eklf, Gata1, PU.1, CEBPa, Gata2, Gfi1,
EgrNab, Fli1, Fog. TwINFER JSONs and competitor rankedEdges already use gene_i names. No Beeline ground truth is used.

Date: 2026-10-06
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
from benchmarks.network_benchmarks.score import score_mixed_network_sweep as M
from benchmarks.network_benchmarks.analysis_plots import summary_table_todo4v2 as S

HSC_ORDER = ["cJun", "SCL", "Eklf", "Gata1", "PU.1", "CEBPa", "Gata2", "Gfi1", "EgrNab", "Fli1", "Fog"]
BEELINE_TO_CANON = {"cJun": "gene_1", "SCL": "gene_2", "Eklf": "gene_3", "Gata1": "gene_4", "Pu1": "gene_5",
                    "PU.1": "gene_5", "Cebpa": "gene_6", "CEBPa": "gene_6", "Gata2": "gene_7", "Gfi1": "gene_8",
                    "EgrNab": "gene_9", "Fli1": "gene_10", "Fog1": "gene_11", "Fog": "gene_11"}
CANON_FILE = ROOT / "input_data" / "real_world_networks" / "HSC.txt"
OUT = ROOT / "analysis_data" / "synthetic_network_simulations"
OUT.mkdir(parents=True, exist_ok=True)


def canonical_gt():
    Mx = np.loadtxt(CANON_FILE, delimiter=",")
    n = Mx.shape[0]
    recs = [(f"gene_{i + 1}", f"gene_{j + 1}", "+" if Mx[i, j] > 0 else "-")
            for i in range(n) for j in range(n) if i != j and Mx[i, j] != 0]
    return pd.DataFrame(recs, columns=["Gene1", "Gene2", "Type"])


def score_ranked(ranked, gt):
    _, true_edges = M.build_edge_universe(gt)
    n = len({*gt.Gene1, *gt.Gene2})
    base = len(true_edges) / (n * (n - 1))
    auprc = M.compute_auprc(ranked, gt)
    predicted = M.dedupe_predictions(ranked)
    sel, _ = M.top_k_tie_aware_selection(predicted, len(true_edges))
    _, _, f1 = M.precision_recall_f1(sel, true_edges)
    return auprc, f1, auprc / base


def main():
    gt = canonical_gt()
    rows = []
    # TwINFER (TODO4v2 nogate, directed unsigned), canonical ground truth
    for f in sorted(glob.glob(str(ROOT / "analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs/HSC_rep_*_all_results.json"))):
        row = S.todo4v2_row_for_dataset(f, gt)
        if row:
            rows.append(dict(method="TODO4v2", run=Path(f).name, auprc=row["auprc"], f1_topk=row["f1_topk"]))
    # competitors from the Beeline HSC outputs (already gene_i names, canonical order)
    for rk in sorted(glob.glob(str(ROOT / "benchmarking_analysis/twinfer_real/outputs/HSC/*/*/rankedEdges.csv"))):
        alg = Path(rk).parent.name
        run = Path(rk).parent.parent.name
        r = pd.read_csv(rk, sep="\t")
        r = r.rename(columns={r.columns[0]: "Gene1", r.columns[1]: "Gene2", r.columns[2]: "EdgeWeight"})
        if len(r) == 0:
            continue
        auprc, f1, _ = score_ranked(r, gt)
        rows.append(dict(method=alg, run=run, auprc=auprc, f1_topk=f1))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "hsc_canonical_replicates.csv", index=False)
    summ = df.groupby("method")[["auprc", "f1_topk"]].agg(["mean", "count"]).round(3)
    print(summ.to_string())


if __name__ == "__main__":
    main()
