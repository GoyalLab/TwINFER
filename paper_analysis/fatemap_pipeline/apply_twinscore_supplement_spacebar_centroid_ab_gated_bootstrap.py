#!/usr/bin/env python3
"""TwinScore_supplement FINAL (gated + bootstrapped phi) on SpaceBar's centroid A/B split, 2026-09-23.

Supersedes apply_twinscore_supplement_spacebar_centroid_ab.py (built on the older ungated
persistence_no_gate/clipped-phi variant, now SUPERSEDED). Reuses run_gene_set from
apply_twinscore_supplement_fatemap_gated_bootstrap.py UNCHANGED (same formula, same bootstrap noise
floors, same phi h-gate), swapping only the two dataset-specific pieces:
  * load_raw_ab   -> SpaceBar centroid split (run_infer_spacebar_centroid_ab.build_ab_split_with_full_panel):
                     A = central (time_step 0), B = peripheral (time_step 1); all clones n>=3, oversized
                     clones subsampled to 30 per side
  * compute_Wz    -> in-memory split-half Wz from the raw 114-gene count matrix (SpaceBar's per-section
                     CSVs ARE the QC'd raw counts; no qc_dir/mtx wrapper needed)
z_dagger: gb.run_gene_set reads z_dagger_spacebar_{gs}_absplit_allpairs.json; the centroid inference run
wrote z_dagger_spacebar_{gs}_centroid_ab_allpairs.json, so a symlink under the expected name is created.
Outputs: analysis_data/spacebar/data/twinscore_supp_gated_bootstrap/ (own folder + README + run_params).
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
import pandas as pd

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_analysis.fatemap_pipeline import apply_twinscore_supplement_fatemap_gated_bootstrap as gb
from paper_analysis.fatemap_pipeline.run_infer_spacebar_centroid_ab import build_ab_split_with_full_panel, GENE_SETS

supp = gb.supp  # twinscore_supp_helpers
log = gb.log
SEED = gb.SEED
DATA = f'{TWINFER_PROJECT_ROOT}/analysis_data/spacebar/data'
_STATE = {}


def load_raw_ab_spacebar(dataset, gene_set_name):
    df, X_full, genes_full = build_ab_split_with_full_panel(gene_set_name)
    _STATE["X_full"], _STATE["genes_full"] = X_full, genes_full
    return df, list(GENE_SETS[gene_set_name]), None, genes_full, None


def compute_Wz_spacebar(raw, genes, qc_dir, label, genes_full, gidx, seed=SEED, n_boot=20):
    """In-memory analog of helpers.compute_Wz (same body; X_full is aligned 1:1 with `raw`)."""
    X_full, genes_full = _STATE["X_full"], _STATE["genes_full"]
    t0 = time.time()
    clone_ids = raw["clone_id"].to_numpy()
    n = len(genes)
    W_draws, meff_twin_list = [], []
    for b in range(n_boot):
        Wb, meff_b = supp._one_split_half_W(X_full, clone_ids, genes, genes_full, seed=seed + b)
        W_draws.append(Wb)
        meff_twin_list.append(meff_b)
    W_draws = np.stack(W_draws, axis=0)
    log(f"    Wz: {n_boot} bootstrap split-half draws done ({time.time()-t0:.0f}s), {X_full.shape[0]:,} cells")
    W_mean = W_draws.mean(axis=0)
    W_var_boot = W_draws.var(axis=0, ddof=1)
    iu = np.triu_indices(n, k=1)
    denom = np.sum(W_mean[iu] ** 2)
    a_star = float(np.clip(W_var_boot[iu].sum() / denom, 0.0, 1.0)) if denom > 0 else 1.0

    def shrink_invert(Wm):
        Wshrunk = (1 - a_star) * Wm + a_star * np.eye(n)
        P = -np.linalg.pinv(Wshrunk + 1e-8 * np.eye(n))
        d = np.sqrt(np.clip(np.outer(np.diag(P), np.diag(P)), 1e-12, None))
        PC = P / d
        np.fill_diagonal(PC, 0.0)
        return PC

    PC_mean = shrink_invert(W_mean)
    PC_draws = np.stack([shrink_invert(W_draws[b]) for b in range(n_boot)], axis=0)
    se = PC_draws.std(axis=0, ddof=1)
    iu_full = ~np.eye(n, dtype=bool)
    se_floor = max(1e-9, 0.05 * float(np.median(np.abs(PC_mean[iu_full]))))
    se_safe = np.where(se > se_floor, se, se_floor)
    Wz_clipped = np.clip(PC_mean / se_safe, -20.0, 20.0)
    Wz = pd.DataFrame(Wz_clipped, index=genes, columns=genes)
    v = supp.signal_share(Wz.to_numpy()[~np.eye(n, dtype=bool)])
    log(f"    Wz: a*={a_star:.3f}  v={v:.3f}")
    return Wz, v, a_star


gb.load_raw_ab = load_raw_ab_spacebar
gb.compute_Wz = compute_Wz_spacebar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene-set", choices=[k for k in GENE_SETS if k.startswith("correlation_")],
                    default="correlation_high")
    ap.add_argument("--n-shuffles", type=int, default=2000)
    ap.add_argument("--n-cores", type=int, default=8)
    ap.add_argument("--no-wz", action="store_true")
    args = ap.parse_args()
    gs = args.gene_set

    link = f"{DATA}/z_dagger_spacebar_{gs}_absplit_allpairs.json"
    target = f"z_dagger_spacebar_{gs}_centroid_ab_allpairs.json"
    if not os.path.lexists(link):
        os.symlink(target, link)

    ct = pd.read_csv(supp.COLLECTRI_PATH, sep="\t")
    ct = ct[ct.source_genesymbol != ct.target_genesymbol]
    CE = set(zip(ct.source_genesymbol, ct.target_genesymbol))

    out_dir = f"{DATA}/twinscore_supp_gated_bootstrap"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "README.md"), "w") as fh:
        fh.write(gb.GATED_README)
    suffix = "_absplit_gated_bootstrap"

    pair_df, gene_df, diag = gb.run_gene_set("SPACEBAR", gs, CE, True, n_shuffles=args.n_shuffles,
                                              n_cores=args.n_cores, compute_wz=not args.no_wz)
    pair_out = os.path.join(out_dir, f"twinscore_supplement_spacebar_{gs}{suffix}_pair_terms.csv")
    gene_out = os.path.join(out_dir, f"twinscore_supplement_spacebar_{gs}{suffix}_gene_terms.csv")
    pair_df.to_csv(pair_out, index=False)
    gene_df.to_csv(gene_out, index=False)
    log(f"wrote {pair_out}")
    log(f"wrote {gene_out}")
    log(f"diagnostics: {diag}")
    params = dict(dataset="SPACEBAR", split="centroid A/B (A=central, B=peripheral)", gene_set=gs,
                  n_shuffles=args.n_shuffles, compute_wz=not args.no_wz, phi_mode="gated_bootstrap",
                  gate_mult=2.0, phi_clip=None, script=os.path.basename(__file__),
                  z_dagger_input=f"{DATA}/{target}",
                  written=datetime.datetime.now().isoformat(timespec="seconds"), diagnostics=diag)
    with open(os.path.join(out_dir, f"run_params_spacebar_{gs}{suffix}.json"), "w") as fh:
        json.dump(params, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
