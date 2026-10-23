"""AUPRC-x table for the synthetic sweeps (network_sweep_final, mixed_network_sweep), same metric as the
real-network radar: AUPRC-x = raw AUPRC / base_rate, base_rate = n_true_directed / (n*(n-1)) per dataset,
n = number of genes in that dataset's GroundTruthNetwork.csv. TwINFER = TODO4v2 (nogate), directed unsigned.
Competitors = BEELINE rows (scheme == twin_paired) from the existing competitor CSVs, same as the TODO4v2 summary.

Date:   2026-10-04
Output: analysis_data/synthetic_network_simulations/auprc_x_<family>.csv (one row per method x dataset group)
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/gpfs/projects/b1255/hzhang/TwINFER_KA")
sys.path.insert(0, str(ROOT / "code" / "TwINFER" / "package"))
sys.path.insert(0, str(ROOT / "code" / "TwINFER"))
from benchmarks.network_benchmarks.analysis_plots import summary_table_todo4v2 as S

BEELINE = ROOT / "code" / "Beeline" / "inputs"
OUT = ROOT / "analysis_data" / "synthetic_network_simulations"
OUT.mkdir(parents=True, exist_ok=True)


def base_rate(gt_csv):
    gt = pd.read_csv(gt_csv)
    genes = set(gt.Gene1) | set(gt.Gene2)
    n = len(genes)
    return len(gt) / (n * (n - 1))


def families():
    e13 = BEELINE / "e13_pos100"
    nsf = BEELINE / "network_sweep_final"
    mns = BEELINE / "mixed_network_sweep"
    A = ROOT / "analysis_data" / "network_sweep_final"
    B = ROOT / "analysis_data" / "mixed_network_sweep"
    return {
        "network_sweep": dict(
            json_dirs=[(f"{A}/e13_pos100/twinfer_inference_allpairs", lambda ds: e13 / ds / "GroundTruthNetwork.csv"),
                       (f"{A}/twinfer_inference_allpairs", lambda ds: nsf / ds / "GroundTruthNetwork.csv")],
            beeline=[f"{A}/e13_pos100/beeline_analysis_output.csv", f"{A}/beeline_gmm_analysis_output.csv"],
            gt_of=lambda ds: (e13 if (e13 / ds).exists() else nsf) / ds / "GroundTruthNetwork.csv"),
        "mixed_network_sweep": dict(
            json_dirs=[(f"{B}/twinfer_inference_allpairs", lambda ds: mns / ds / "GroundTruthNetwork.csv")],
            beeline=[f"{B}/beeline_analysis_output.csv"],
            gt_of=lambda ds: mns / ds / "GroundTruthNetwork.csv"),
    }


def run(name, cfg):
    rows = []
    import glob, json
    for json_dir, gt_resolver in cfg["json_dirs"]:
        for f in sorted(glob.glob(f"{json_dir}/*_all_results.json")):
            d = json.load(open(f))
            ds = d.get("dataset_id") or d.get("sim_type")
            gt_path = gt_resolver(ds)
            if not gt_path.exists():
                continue
            gt_df = pd.read_csv(gt_path)
            row = S.todo4v2_row_for_dataset(f, gt_df)
            if row is None:
                continue
            br = base_rate(gt_path)
            rows.append(dict(method="TODO4v2", dataset_id=S.strip_rep(ds), raw_id=ds, auprc=row["auprc"], base_rate=br,
                             auprc_x=row["auprc"] / br))
    for csv in cfg["beeline"]:
        b = pd.read_csv(csv)
        if "scheme" in b.columns:
            b = b[b.scheme == "twin_paired"]
        for _, r in b.iterrows():
            gt_path = cfg["gt_of"](r["dataset_id"])
            if not gt_path.exists():
                continue
            br = base_rate(gt_path)
            rows.append(dict(method=r["algorithm"], dataset_id=S.strip_rep(r["dataset_id"]), raw_id=r["dataset_id"], auprc=r["auprc"],
                             base_rate=br, auprc_x=r["auprc"] / br))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"auprc_x_{name}_replicates.csv", index=False)
    summary = (df.groupby(["method", "dataset_id"])["auprc_x"].mean().reset_index())
    summary.to_csv(OUT / f"auprc_x_{name}.csv", index=False)
    overall = df.groupby("method")["auprc_x"].mean().sort_values(ascending=False)
    print(name, "rows", len(df), "groups", df.dataset_id.nunique())
    print(overall.round(3).to_string())
    return df


if __name__ == "__main__":
    for name, cfg in families().items():
        run(name, cfg)
