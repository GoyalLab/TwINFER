"""Part 5 follow-up (handoff/2026-09-18_todo4v2_signed_metrics_and_t1_10_pipeline.md, open item 3):
tests the "real-data-noise vs clean-sim" hypothesis for why cross-corr-only dominates TODO4v2 on
every simulated benchmark but loses to it on LARRY (Part 5's finding: cross-corr wins only 3/18
LARRY panel/twin_def combos, mean auprc_x 1.104x vs TODO4v2's 1.557x -- opposite of every sim).

Unsupervised signal-strength proxy: frac(|z_dagger| > 2.576) per dataset/panel, where z_dagger is
the same z_signed(rho_cross_xy, DEFAULT_SD["cross"]) used inside todo4v2_score/todo4v2_sim_scoring.
2.576 = norm.ppf(1 - 0.01/2), the two-sided 1% significance threshold. If a single bivariate
statistic (rho_cross_xy) is inherently noisier per-pair in real LARRY data than in a clean
Gillespie simulation, LARRY's frac should be systematically LOWER than every sim benchmark's --
that's the leading hypothesis for why TODO4v2's extra (partially-independent) terms boost SNR on
LARRY but only dilute cross-corr's near-ceiling signal on sims.

Uses the EXISTING t1=1 sim benchmark data (this diagnostic is about real-vs-sim noise in general,
not about the t1=10 rerun) and the already-computed LARRY yscher panels (9 gene sets x 2 twin_defs,
same source apply_todo4v2_allpairs_with_competitors.py reads).
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/larry_hematopoiesis_validation")
from analytic_zscores import DEFAULT_SD, z_signed

ROOT = "/home/gzu5140/TwINFER_KA"
HERE = "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/work_in_progress/benchmark"
LARRY_R = "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/larry_hematopoiesis_validation/resources"

Z_THR = 2.576  # two-sided 1% significance

GENE_SETS = ["variability_high", "variability_mid", "variability_low",
             "detection_high", "detection_mid", "detection_low",
             "correlation_high", "correlation_mid", "correlation_low"]
TWIN_DEFS = {
    "unfiltered": "analytic_infer_yscher_nrand200",
    "annotfilter": "analytic_infer_yscher_annotfilter_nrand200",
}


def frac_strong_zdagger(rho_cross_xy):
    z = z_signed(np.asarray(rho_cross_xy, float), DEFAULT_SD["cross"])
    z = z[np.isfinite(z)]
    if len(z) == 0:
        return np.nan, 0
    return float((np.abs(z) > Z_THR).mean()), len(z)


def sim_json_frac(json_path):
    d = json.load(open(json_path))
    tsi = d.get("twin_score_inputs")
    if tsi is None:
        return None
    dd = pd.DataFrame(tsi["data"], columns=tsi["columns"])
    dd = dd[dd.gene_1 != dd.gene_2]
    dd = dd[dd.rho_cross_xy.notna()]
    if len(dd) < 3:
        return None
    frac, n = frac_strong_zdagger(dd.rho_cross_xy.to_numpy())
    return dict(dataset_id=d.get("dataset_id", os.path.basename(json_path)), frac_strong=frac, n_pairs=n)


def collect_sim(name, json_dir):
    files = sorted(glob.glob(f"{json_dir}/*_all_results.json"))
    rows = [r for r in (sim_json_frac(f) for f in files) if r]
    for r in rows:
        r["benchmark"] = name
    print(f"{name}: {len(rows)}/{len(files)} usable replicates")
    return rows


def collect_larry():
    rows = []
    for twin_def, analytic_dir in TWIN_DEFS.items():
        for gs in GENE_SETS:
            dpath = f"{LARRY_R}/{analytic_dir}/{gs}/twin_score_inputs.csv"
            if not os.path.exists(dpath):
                continue
            dd = pd.read_csv(dpath)
            dd = dd[dd.gene_1 != dd.gene_2]
            dd = dd[dd.rho_cross_xy.notna()]
            if len(dd) < 3:
                continue
            frac, n = frac_strong_zdagger(dd.rho_cross_xy.to_numpy())
            rows.append(dict(dataset_id=f"{twin_def}/{gs}", benchmark="LARRY",
                              twin_def=twin_def, gene_set=gs, frac_strong=frac, n_pairs=n))
    print(f"LARRY: {len(rows)}/18 usable panels")
    return rows


def main():
    all_rows = []
    all_rows += collect_sim("network_sweep_final",
                             f"{ROOT}/analysis_data/network_sweep_final/twinfer_inference_allpairs")
    all_rows += collect_sim("network_sweep_e13",
                             f"{ROOT}/analysis_data/network_sweep_final/e13_pos100/twinfer_inference_allpairs")
    all_rows += collect_sim("mixed_network_sweep",
                             f"{ROOT}/analysis_data/mixed_network_sweep/twinfer_inference_allpairs")
    all_rows += collect_sim("real_networks",
                             f"{ROOT}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs")
    all_rows += collect_larry()

    df = pd.DataFrame(all_rows)
    out_csv = f"{HERE}/zdagger_signal_strength_larry_vs_sims.csv"
    df.to_csv(out_csv, index=False)

    print(f"\nwrote {out_csv} ({len(df)} rows)\n")
    print("=== mean frac(|z_dagger|>2.576) per benchmark ===")
    summary = df.groupby("benchmark")["frac_strong"].agg(["mean", "median", "std", "count"])
    summary = summary.reindex(["LARRY", "network_sweep_final", "network_sweep_e13",
                                "mixed_network_sweep", "real_networks"])
    print(summary.to_string())

    larry_vals = df.loc[df.benchmark == "LARRY", "frac_strong"].dropna()
    sim_vals = df.loc[df.benchmark != "LARRY", "frac_strong"].dropna()
    if len(larry_vals) > 0 and len(sim_vals) > 0:
        u, p = mannwhitneyu(larry_vals, sim_vals, alternative="less")
        print(f"\nMann-Whitney U (LARRY < pooled sims): U={u:.1f}, p={p:.4g}")
        print(f"LARRY mean={larry_vals.mean():.4f}  pooled sims mean={sim_vals.mean():.4f}  "
              f"ratio={larry_vals.mean()/sim_vals.mean() if sim_vals.mean() else float('nan'):.3f}")

    print("\n=== LARRY panel detail ===")
    print(df[df.benchmark == "LARRY"][["twin_def", "gene_set", "frac_strong", "n_pairs"]].to_string(index=False))


if __name__ == "__main__":
    main()
