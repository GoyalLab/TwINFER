#!/usr/bin/env python3
"""Split the combined Watermelon QC'd counts matrix into per-stage qc_filtered folders that look like FM06/FM08's
(finalized_data/Watermelon_{stage}_data/qc_filtered/{label}_qc_counts.mtx, genes.txt, cells.txt, obs_metadata.csv with
replicate A/B), which the TwinScore_supplement Wz term / A-B loaders read."""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import os

import pandas as pd
import scipy.io as sio

QC_DIR = f'{TWINFER_PROJECT_ROOT}/finalized_data/Watermelon_data/qc_filtered'
X = sio.mmread(f"{QC_DIR}/watermelon_qc_counts.mtx").tocsr()
genes = open(f"{QC_DIR}/genes.txt").read().split()
cells = pd.Index(open(f"{QC_DIR}/cells.txt").read().split())
obs = pd.read_csv(f"{QC_DIR}/obs_metadata.csv", index_col=0, dtype={"replicate": str})
assert X.shape == (len(cells), len(genes)) == (len(obs), len(genes)) and (obs.index == cells).all()
obs["replicate_orig"] = obs["replicate"]
obs["replicate"] = obs["replicate"].map({"1": "A", "2": "B"})
for stage in ("naive", "lag", "late"):
    sel = (obs["stage"] == stage).to_numpy()
    out = f"{TWINFER_PROJECT_ROOT}/finalized_data/Watermelon_{stage}_data/qc_filtered"
    os.makedirs(out, exist_ok=True)
    sio.mmwrite(f"{out}/watermelon_{stage}_qc_counts.mtx", X[sel])
    open(f"{out}/genes.txt", "w").write("\n".join(genes) + "\n")
    open(f"{out}/cells.txt", "w").write("\n".join(cells[sel]) + "\n")
    obs[sel].to_csv(f"{out}/obs_metadata.csv")
    print(f"[{stage}] wrote {sel.sum():,} cells to {out}", flush=True)
