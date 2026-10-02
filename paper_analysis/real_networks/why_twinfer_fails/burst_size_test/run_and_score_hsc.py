#!/usr/bin/env python
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_a2a3302f_2026-09-28_why_twinfer_fails; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Simulate HSC at baseline (k_prod_mRNA=2) vs boosted (k_prod_mRNA=10) burst size, then score both:
mRNA dispersion (var/mean, per gene), plain existence AUPRC-x (|S|), and the actual TwINFER
bootstrap-calibrated zreg for HSC's single-input edge Gata1->Fog1 at t=48 -- same statistic used in
2.7e/2.7f of this session's investigation, to see whether a bigger transcriptional burst size raises
mRNA's regulation signature.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/package")
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline")
from numba import set_num_threads
from twinfer.simulation.gillespie_simulations import process_param_set
from paper_analysis.fatemap_pipeline import twinscore_supp_helpers as supp
from paper_analysis.fatemap_pipeline import apply_twinscore_supplement_fatemap_gated_bootstrap as gb

R = f'{TWINFER_PROJECT_ROOT}'
MATRIX = f"{R}/simulation_data/twinfer_format/HSC/interaction_matrix.txt"
GENES = [l.strip() for l in open(f"{R}/simulation_data/twinfer_format/HSC/gene_order.txt")]
PARAM_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_params_hsc.csv")
OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f"{OUT}/logs", exist_ok=True)

cores = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 8))
set_num_threads(cores)

M = np.loadtxt(MATRIX, delimiter=",")
n_genes = len(GENES)
gi = {g: i + 1 for i, g in enumerate(GENES)}

paths = {}
for tag, row in [("hsc_burst2_baseline", 0), ("hsc_burst10_boosted", 1)]:
    out_file_glob = f"{OUT}/df_{tag}_*ncells_2000_HSC_{tag}_*.csv"
    import glob
    existing = [f for f in glob.glob(out_file_glob) if "before_division" not in f]
    if existing:
        print(f"=== {tag}: already exists, skipping sim -> {existing[0]}", flush=True)
        paths[tag] = existing[0]
        continue
    cfg = {
        "n_cells": 2000,
        "simulation_time_before_division": 300,
        "twin_simulation_time_after_division": 48,
        "twin_measurement_resolution": 1,
        "path_to_connectivity_matrix": MATRIX,
        "param_csv": PARAM_CSV,
        "rows_to_use": [[row] * n_genes],
        "output_folder": OUT,
        "log_file": f"{OUT}/logs/HSC_{tag}.log",
        "type": f"HSC_{tag}",
        "combinatorial_interaction_type": "additive",
    }
    print(f"=== running {tag} (param row {row}) ===", flush=True)
    path = process_param_set(cfg["rows_to_use"][0], tag, cfg)
    print(f"=== {tag} DONE -> {path} ===", flush=True)
    paths[tag] = path


def score(tag, path):
    print(f"\n{'='*20} SCORING {tag} {'='*20}")
    df = pd.read_csv(path)
    mrna_cols = [f"gene_{i+1}_mRNA" for i in range(n_genes)]
    last = df[df.time_step == df.time_step.max()].drop_duplicates("cell_id")

    print("-- mRNA dispersion (var/mean) per gene --")
    disps = {}
    for i, g in enumerate(GENES):
        c = f"gene_{i+1}_mRNA"
        v = last[c].to_numpy(dtype=float)
        m = v.mean()
        disps[g] = v.var() / m if m > 0 else np.nan
        print(f"  {g:8s} mean={m:8.3f}  dispersion={disps[g]:.3f}")
    print(f"  median dispersion: {np.nanmedian(list(disps.values())):.3f}")

    print("-- plain existence AUPRC-x |S| (rank-based, mRNA, last timepoint) --")
    vals = {f"gene_{i+1}": last[f"gene_{i+1}_mRNA"].to_numpy(dtype=float) for i in range(n_genes)}
    n = n_genes
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                S[i, j] = 1.0
            else:
                S[i, j] = spearmanr(vals[f"gene_{i+1}"], vals[f"gene_{j+1}"])[0]
    iu = np.triu_indices(n, 1)
    und = ((M != 0) | (M.T != 0))[iu].astype(int)
    auprc = average_precision_score(und, np.nan_to_num(np.abs(S)[iu])) / und.mean()
    print(f"  AUPRC-x |S| = {auprc:.2f}")

    print("-- TwINFER bootstrap-calibrated zreg for Gata1->Fog1 at t=48 (the actual production statistic) --")
    reg, tgt = "Gata1", "Fog1"
    t_raw = df[df.time_step == df.time_step.max()].reset_index(drop=True)
    rc, tc = f"gene_{gi[reg]}_mRNA", f"gene_{gi[tgt]}_mRNA"
    tmp = t_raw.rename(columns={rc: f"{reg}_mRNA", tc: f"{tgt}_mRNA"})
    sub_genes = [reg, tgt]
    Sm = supp.same_cell_matrix(tmp, sub_genes)
    Cm, meff = supp.sister_matrix_from_frame(tmp, sub_genes)
    sd_reg, nperm = gb.bootstrap_null_sd_offdiag(tmp, sub_genes, seed=42, kind="reg",
                                                  extra=pd.DataFrame(1.0, index=sub_genes, columns=sub_genes))
    s_val, c_val = Sm.loc[reg, tgt], Cm.loc[reg, tgt]
    zreg = (s_val - c_val) / sd_reg if sd_reg and np.isfinite(sd_reg) and sd_reg > 0 else np.nan
    print(f"  S={s_val:+.4f}  C={c_val:+.4f}  S-C={s_val-c_val:+.4f}  sd_reg={sd_reg:.5f} (n_perm={nperm})  zreg={zreg:+.2f}")

    return dict(tag=tag, median_dispersion=np.nanmedian(list(disps.values())), auprc_S=auprc,
                gata1_fog1_S=s_val, gata1_fog1_C=c_val, gata1_fog1_zreg=zreg)


results = [score(tag, path) for tag, path in paths.items()]
print(f"\n\n{'='*20} SUMMARY: baseline (k_prod_mRNA=2) vs boosted (k_prod_mRNA=10) {'='*20}")
for r in results:
    print(r)
