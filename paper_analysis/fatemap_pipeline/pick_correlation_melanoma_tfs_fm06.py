#!/usr/bin/env python3
"""FM06: a correlation_high-style panel that uses ALL the melanoma ground-truth source TFs (MITF, SOX10, JUN, JUND, FOSL1, FOSL2, TEAD1-4).

Same recipe as pick_correlation_clean_fm06.py, with the truth swapped from CollecTRI to the melanoma ChIP-Atlas network
(melanoma_tf_network_fm06.tsv, evidence chip_only|both, self-loops removed):
  * genes eligible if detected in >= 5% of the cells of EVERY replicate (pk.MIN_DETECTION_FRAC); expression = FM06_integrated.h5ad .raw (log1p CP10k)
  * targets lose yscher's cell-cycle / growth-flagged and curated-machinery genes and must be in her universe (as in correlation_high_clean); the TFs themselves are only
    required to be eligible by detection (they are the sources of the truth, so they are kept even if flagged) -- flagged TFs are reported
  * |Spearman rho| over all cells for every (TF, ChIP target) pair; each eligible TF keeps its --n-targets strongest targets; panel = TFs + targets
Writes key `correlation_high_melanoma_tfs` into gene_sets_fm06.json (backup gene_sets_fm06.json.bak_20260926 first) and gene_sets_detail_fm06_melanoma_tfs.json.
usage: pick_correlation_melanoma_tfs_fm06.py [--n-targets 8] [--key correlation_high_melanoma_tfs] [--random-seed S --detail-out PATH] [--dry-run]
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import argparse
import json
import os
import shutil
import sys

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import rankdata

PIPE = os.path.dirname(os.path.abspath(__file__))
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, PIPE)
from paper_analysis.fatemap_pipeline import pick_gene_sets_fatemap as pk

BASE = f'{TWINFER_PROJECT_ROOT}'
DATA = f"{BASE}/analysis_data/fm06/data"
# [2026-10-01 commented out: yscher data now copied to clean_data/external_yscher (user yes); see REPOINT_LOG.tsv] Y = "/gpfs/projects/b1255/yscher/Transcriptomic Distance/exports/_probe_tmp"
Y = f"{TWINFER_PROJECT_ROOT}/clean_data/external_yscher/Transcriptomic_Distance/exports/_probe_tmp"
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, f"{Y}/tf_wide")
from paper_analysis.fatemap_pipeline.external_yscher.machinery import is_machinery  # noqa: E402

TFS = ["MITF", "SOX10", "JUN", "JUND", "FOSL1", "FOSL2", "TEAD1", "TEAD2", "TEAD3", "TEAD4"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-targets", type=int, default=8)
    ap.add_argument("--key", default="correlation_high_melanoma_tfs")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--random-seed", type=int, default=None, help="pick each TF's targets at random (this seed) among its eligible ChIP targets instead of the strongest |rho|")
    ap.add_argument("--detail-out", default=f"{DATA}/gene_sets_detail_fm06_melanoma_tfs.json")
    a = ap.parse_args()
    rng = np.random.default_rng(a.random_seed) if a.random_seed is not None else None

    FL = pd.read_csv(f"{Y}/tf_wide/fm06/gene_flags.csv")
    in_universe = set(FL.gene)
    cyc = set(FL[(FL.cycle) | (FL.growth)].gene)
    mach = {g for g in in_universe if is_machinery(g)}
    ok_target = lambda g: g in in_universe and g not in cyc and g not in mach

    gt = pd.read_csv(f"{DATA}/melanoma_tf_network_fm06.tsv", sep="\t")
    gt = gt[(gt.TF != gt.target) & gt.evidence.isin(["chip_only", "both"])]

    A_full = ad.read_h5ad(f"{BASE}/finalized_data/FM06_data/FM06_integrated.h5ad")
    X, genes = A_full.raw.X, A_full.raw.var_names
    replicate = A_full.obs["replicate"].to_numpy()
    reps = sorted(set(replicate))
    frac = np.stack([np.asarray((X[replicate == r] > 0).sum(axis=0)).ravel() / (replicate == r).sum() for r in reps]).min(axis=0)
    eligible = set(genes[frac >= pk.MIN_DETECTION_FRAC])
    print("TF eligibility (min detection over replicates >= 0.05):")
    tf_ok = []
    for t in TFS:
        f = float(frac[list(genes).index(t)]) if t in set(genes) else float("nan")
        flagged = t in cyc or t in mach
        print(f"  {t}: min-replicate detection {f:.3f}{'  (cycle/machinery flagged)' if flagged else ''}{'' if t in eligible else '  -> NOT ELIGIBLE'}")
        if t in eligible:
            tf_ok.append(t)

    pairs = sorted({(r.TF, r.target) for r in gt.itertuples(index=False) if r.TF in tf_ok and r.target in eligible and ok_target(r.target)})
    need = sorted({g for p in pairs for g in p})
    idx = {g: k for k, g in enumerate(need)}
    cols = [list(genes).index(g) for g in need]
    M = X[:, cols]
    M = np.asarray(M.todense()) if hasattr(M, "todense") else np.asarray(M)
    R = np.apply_along_axis(rankdata, 0, M).astype(np.float32)
    R -= R.mean(axis=0)
    S = np.sqrt((R ** 2).sum(axis=0))
    S[S < 1e-12] = 1.0
    C = np.abs((R.T @ R) / np.outer(S, S))
    absrho = {p: float(C[idx[p[0]], idx[p[1]]]) for p in pairs}

    detail, panel = {}, set()
    for t in tf_ok:
        cand = sorted([p for p in pairs if p[0] == t], key=lambda p: absrho[p], reverse=True)
        tg = cand[: a.n_targets] if rng is None else [cand[i] for i in sorted(rng.choice(len(cand), size=min(a.n_targets, len(cand)), replace=False))]
        detail[t] = [[p[1], round(absrho[p], 4)] for p in tg]
        panel.add(t)
        panel.update(p[1] for p in tg)
        print(f"{t}: {sum(1 for p in pairs if p[0] == t)} eligible ChIP targets; kept {[f'{p[1]}({absrho[p]:.2f})' for p in tg]}")
    panel = sorted(panel)
    sources_in = [t for t in TFS if t in panel]
    n_true = sum(1 for x in panel for y in panel if (x, y) in {(r.TF, r.target) for r in gt.itertuples(index=False)})
    print(f"\npanel {len(panel)} genes; source TFs in panel {len(sources_in)}: {sources_in}; ChIP edges among panel genes (any source): {n_true}")
    if a.dry_run:
        print("dry run: nothing written")
        return
    gp = f"{DATA}/gene_sets_fm06.json"
    bak = f"{gp}.bak_20260926"
    if not os.path.exists(bak):
        shutil.copy(gp, bak)
    gs = json.load(open(gp))
    gs[a.key] = panel
    json.dump(gs, open(gp, "w"), indent=2)
    json.dump(detail, open(a.detail_out, "w"), indent=2)
    print(f"wrote key {a.key} to {gp} (backup {bak})")


if __name__ == "__main__":
    main()
