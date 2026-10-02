"""Per-dataset (per individual sim replicate) win/loss count: TODO4v2(nogate) auprc_x vs the
BEST competitor's auprc_x on that SAME replicate (not the pooled per-benchmark means used
elsewhere), at t1=10, grouped by network/sim type within each of the 4 benchmark families.

Join keys per family (t1=10 JSON -> BEELINE run_id):
  network_sweep_e13, network_sweep_final_broader, mixed_network_sweep:
      JSON dataset_id + JSON "label" (sim replicate index, string) -> BEELINE
      (dataset_id, run_id=f"simrep{label}_twin_paired")
  real_networks: JSON has no dataset_id/label; its "analysis_key" (e.g.
      "B_cell_activation_rep_0_08a5aeab") ends in the same random hash BEELINE's run_id carries
      (e.g. "simrep0_08a5aeab_twin_paired") -- joined by that hash substring instead.
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/larry_hematopoiesis_validation")
from benchmarks.network_benchmarks.score import todo4v2_sim_scoring as T

from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: project root for the clean_data/ repointing, see REPOINT_LOG.tsv]
ROOT = T.ROOT
HERE = T.HERE
TOPO_MAP = {"GSD": "GSD.txt", "HSC": "HSC.txt", "VSC": "VSC.txt", "mCAD": "mCAD.txt",
            "B_cell_activation": "B_cell.txt", "Circadian_cycle": "circadian.txt",
            "EMT": "EMT.txt", "Pluripotent": "Pluripotent.txt"}
HASH_RE = re.compile(r"([0-9a-f]{8})$")


def strip_rep(dataset_id):
    """Topology/sim-type grouping key: strip a trailing _repN."""
    return re.sub(r"_rep\d+$", "", dataset_id)


def per_replicate_rows(json_dir, resolver, has_dataset_id=True):
    files = sorted(glob.glob(f"{json_dir}/*_all_results.json"))
    rows = []
    for f in files:
        d = json.load(open(f))
        r = T.score_one_json(f, resolver)
        if r is None:
            continue
        r["json_file"] = os.path.basename(f)
        if has_dataset_id:
            r["sim_rep_label"] = d.get("label")
        else:
            r["analysis_key"] = d.get("analysis_key")
            r["sim_type"] = d.get("sim_type")
        rows.append(r)
    return rows


def best_competitor_per_run(beeline_csv):
    b = pd.read_csv(beeline_csv)
    return b.groupby(["dataset_id", "run_id"])["auprc"].max().reset_index()


def join_simple(rows, beeline_csv):
    best = best_competitor_per_run(beeline_csv)
    best_map = {(r.dataset_id, r.run_id): r.auprc for r in best.itertuples()}
    out = []
    for r in rows:
        run_id = f"simrep{r['sim_rep_label']}_twin_paired"
        comp_auprc = best_map.get((r["dataset_id"], run_id))
        if comp_auprc is None:
            continue
        rand = r["n_true"] / r["n_pairs"]
        comp_auprc_x = comp_auprc / rand if rand > 0 else np.nan
        out.append(dict(group=strip_rep(r["dataset_id"]), dataset_id=r["dataset_id"],
                         run_id=run_id, todo4v2_auprc_x=r["todo4v2_nogate_auprc_x"],
                         best_comp_auprc_x=comp_auprc_x))
    return pd.DataFrame(out)


def join_real_networks(rows, beeline_csv):
    b = pd.read_csv(beeline_csv)
    best = b.groupby(["dataset_id", "run_id"])["auprc"].max().reset_index()
    out = []
    for r in rows:
        m = HASH_RE.search(r["analysis_key"] or "")
        if not m:
            continue
        h = m.group(1)
        cand = best[(best.dataset_id == r["sim_type"]) & (best.run_id.str.contains(h))]
        if cand.empty:
            continue
        comp_auprc = cand.auprc.iloc[0]
        rand = r["n_true"] / r["n_pairs"]
        comp_auprc_x = comp_auprc / rand if rand > 0 else np.nan
        out.append(dict(group=r["sim_type"], dataset_id=r["sim_type"], run_id=cand.run_id.iloc[0],
                         todo4v2_auprc_x=r["todo4v2_nogate_auprc_x"], best_comp_auprc_x=comp_auprc_x))
    return pd.DataFrame(out)


def summarize(df, name):
    df = df.dropna(subset=["todo4v2_auprc_x", "best_comp_auprc_x"])
    df["win"] = df.todo4v2_auprc_x > df.best_comp_auprc_x
    print(f"\n=== {name}: {df.win.sum()}/{len(df)} replicates win overall ===")
    g = df.groupby("group")["win"].agg(["sum", "count"])
    g["pct"] = (100 * g["sum"] / g["count"]).round(1)
    print(g.sort_values("group").to_string())
    return df


def main():
    all_dfs = []

    rows = per_replicate_rows(f"{ROOT}/analysis_data/network_sweep_final/e13_pos100/twinfer_inference_allpairs_t1_10",
                               lambda d: d.get("ground_truth_matrix"))
    df = join_simple(rows, f"{ROOT}/analysis_data/network_sweep_final/e13_pos100/beeline_analysis_output_t1_10.csv")
    all_dfs.append(summarize(df, "network_sweep_e13"))

    rows = per_replicate_rows(f"{ROOT}/analysis_data/network_sweep_final/twinfer_inference_allpairs_t1_10",
                               lambda d: d.get("ground_truth_matrix"))
    df = join_simple(rows, f"{ROOT}/analysis_data/network_sweep_final/beeline_gmm_analysis_output_t1_10.csv")
    all_dfs.append(summarize(df, "network_sweep_final_broader"))

    rows = per_replicate_rows(f"{ROOT}/analysis_data/mixed_network_sweep/twinfer_inference_allpairs_t1_10",
                               lambda d: f"{ROOT}/input_data/mixed_network_sweep/{d.get('dataset_id')}.txt")
    df = join_simple(rows, f"{ROOT}/analysis_data/mixed_network_sweep/beeline_analysis_output_t1_10.csv")
    all_dfs.append(summarize(df, "mixed_network_sweep"))

    real_beeline_csv = f"{ROOT}/analysis_data/paper_analysis/real_networks/beeline_analysis_output_t1_10.csv"
    if os.path.exists(real_beeline_csv):
        rows = per_replicate_rows(f"{ROOT}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs_t1_10",
                                   lambda d: f"{ROOT}/input_data/real_world_networks/{TOPO_MAP.get(d.get('sim_type'), '')}",
                                   has_dataset_id=False)
        df = join_real_networks(rows, real_beeline_csv)
        all_dfs.append(summarize(df, "real_networks"))
    else:
        print(f"\n=== real_networks: SKIPPED -- {real_beeline_csv} not built yet ===")

    combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    # [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] combined.to_csv(f"{HERE}/per_dataset_wins_t1_10.csv", index=False)
    combined.to_csv(f"{TWINFER_PROJECT_ROOT}/analysis_data/benchmarks/network_benchmarks/score_results/per_dataset_wins_t1_10.csv", index=False)


if __name__ == "__main__":
    main()
