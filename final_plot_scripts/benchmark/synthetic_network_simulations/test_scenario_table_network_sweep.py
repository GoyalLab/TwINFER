"""Test the four-scenario decision table on the network_sweep TwINFER inference (main five families, e13 excluded).
Per ordered pair (gene_1 -> gene_2) of each replicate:
  z_rho      = z_abs_rho_t1                                (correlation significance; |z| > 1.96)
  z_hat      = heterogeneity z_het for the pair            (twin-difference significance)
  z_rhos     = direction z_scores for the pair             (shared-part significance)
Scenario: 1 nothing; 2 regulation (z_rho sig, shared not); 3 regulation + heterogeneity (shared sig, twin-diff sig);
4 heterogeneity (shared sig, twin-diff not sig).  Note: the table uses z_rho only in scenarios 1–2 and z_hat only in 3–4.
Truth: directed edge present in the sweep ground-truth matrix (input_data/network_sweep_final/<dataset>.txt).
Date: 2026-10-06
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/gpfs/projects/b1255/hzhang/TwINFER_KA")
INF = ROOT / "analysis_data" / "network_sweep_final" / "twinfer_inference_allpairs"
GT = ROOT / "input_data" / "network_sweep_final"
OUT = ROOT / "analysis_data" / "synthetic_network_simulations"
THR = 1.96


def sig(z):
    return (z is not None) and np.isfinite(z) and abs(z) > THR


def scenario(z_rho, z_rhos, z_hat):
    if sig(z_rhos):
        return 3 if sig(z_hat) else 4
    return 2 if sig(z_rho) else 1


rows = []
for f in sorted(glob.glob(str(INF / "*_all_results.json"))):
    d = json.load(open(f))
    ds = d.get("dataset_id") or d.get("sim_type")
    if "e13" in ds or "e17" in ds:
        continue
    stem = re.sub(r"_rep\d+$", "", ds)
    rep = int(re.search(r"_rep(\d+)$", ds).group(1))
    gt_path = GT / f"{ds}.txt"
    if not gt_path.exists():
        continue
    M = np.loadtxt(gt_path, delimiter=",")
    tsi = d["twin_score_inputs"]
    dd = pd.DataFrame(tsi["data"], columns=tsi["columns"])
    zs = d["direction"]["z_scores"]
    het = d["heterogeneity"]["z_het"]
    for _, r in dd.iterrows():
        for a, b in [(r.gene_1, r.gene_2), (r.gene_2, r.gene_1)]:
            if a == b:
                continue
            i, j = int(a.split("_")[1]) - 1, int(b.split("_")[1]) - 1
            z_rho = r.z_abs_rho_t1
            z_rhos = zs.get(f"{a}__{b}")
            z_hat = het.get(f"{a}__{b}")
            truth = M[i, j] != 0
            rows.append(dict(family=stem, rep=rep, src=a, dst=b, z_rho=z_rho, z_rhos=z_rhos, z_hat=z_hat,
                             scenario=scenario(z_rho, z_rhos, z_hat), truth=truth))
df = pd.DataFrame(rows)
df.to_csv(OUT / "scenario_test_network_sweep_pairs.csv", index=False)
print("pairs", len(df), "datasets", df.family.nunique())
print("\nscenario counts by truth (pairs):")
print(pd.crosstab(df.scenario, df.truth, margins=True).to_string())
print("\nfraction of true edges in each scenario:")
t = df[df.truth]
print((t.scenario.value_counts(normalize=True).sort_index()).round(3).to_string())
print("\nfraction of non-edges in each scenario:")
n = df[~df.truth]
print((n.scenario.value_counts(normalize=True).sort_index()).round(3).to_string())
