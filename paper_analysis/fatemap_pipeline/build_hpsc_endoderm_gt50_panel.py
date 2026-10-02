#!/usr/bin/env python3
"""54-gene panel for hPSC_20260927 chosen to contain exactly 50 ground-truth ChIP-Atlas+
Perturb-seq edges (hpsc_endoderm_candidate_pairs_final.csv), restricted to genes detected
in >=5% of QC-filtered cells (same detection floor as the target gene universe). TFs:
NANOG, POU5F1 (both expanded to top-20 targets earlier in the pipeline), SOX2, FOXH1,
TCF3, KDM2A (4/4/1/1 targets respectively among the detection-passing subset). Unlike the
top100 panel, genes here are NOT selected by expression magnitude -- they ARE the ground
truth, so AUPRC-x against this 50-edge set is directly meaningful (unlike top100's n=3).
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-10-01 added: replaces hardcoded project-root paths]
import json

import numpy as np
import pandas as pd
import scipy.io as sio

# QC_DIR = "/gpfs/projects/b1255/hzhang/TwINFER_KA/finalized_data/hPSC_20260927_data/qc_filtered"   # [2026-10-01 replaced by TWINFER_PROJECT_ROOT]
QC_DIR = f'{TWINFER_PROJECT_ROOT}/finalized_data/hPSC_20260927_data/qc_filtered'
# DATA_DIR = "/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/hPSC_20260927/data"   # [2026-10-01 replaced by TWINFER_PROJECT_ROOT]
DATA_DIR = f'{TWINFER_PROJECT_ROOT}/analysis_data/hPSC_20260927/data'
CHOSEN_TFS = ["NANOG", "POU5F1", "KDM2A", "TCF3", "SOX2", "FOXH1"]
MIN_DETECT_FRAC = 0.05
MAX_CLONE_SIZE = 50


def main():
    X = sio.mmread(f"{QC_DIR}/hPSC_20260927_qc_counts.mtx").tocsr()
    genes = np.array(open(f"{QC_DIR}/genes.txt").read().split())
    cells = np.array(open(f"{QC_DIR}/cells.txt").read().split())
    obs = pd.read_csv(f"{QC_DIR}/obs_metadata.csv", index_col=0)
    n_cells = X.shape[1]

    det_frac = np.asarray((X > 0).sum(axis=1)).ravel() / n_cells
    det_series = pd.Series(det_frac, index=genes).groupby(level=0).max()

    gt = pd.read_csv(f"{DATA_DIR}/hpsc_endoderm_candidate_pairs_final.csv")
    gt["tf_det"] = gt["TF"].map(det_series)
    gt["target_det"] = gt["target"].map(det_series)
    gt_ok = gt[(gt["tf_det"] >= MIN_DETECT_FRAC) & (gt["target_det"] >= MIN_DETECT_FRAC)].copy()

    gt50 = gt_ok[gt_ok["TF"].isin(CHOSEN_TFS)][["TF", "target"]].reset_index(drop=True)
    print(f"ground-truth edges selected: {len(gt50)}", flush=True)
    print(gt50.groupby("TF").size().to_string(), flush=True)
    gt50.to_csv(f"{DATA_DIR}/hpsc_endoderm_gt50_ground_truth_edges.csv", index=False)

    targets = sorted(gt50["target"].unique())
    panel = sorted(set(CHOSEN_TFS) | set(targets))
    assert all(det_series.get(g, 0) >= MIN_DETECT_FRAC for g in panel), "panel gene below detection floor"
    print(f"panel: {len(panel)} genes ({len(CHOSEN_TFS)} TFs + {len(targets)} targets)", flush=True)
    json.dump({"tfs": CHOSEN_TFS, "targets": targets, "panel": panel, "n_ground_truth_edges": len(gt50)},
              open(f"{DATA_DIR}/hpsc_endoderm_gt50_panel.json", "w"))

    gidx = {g: i for i, g in enumerate(genes)}
    col = [gidx[g] for g in panel]
    cell_total = np.asarray(X.sum(axis=0)).ravel().astype(float)
    cell_total[cell_total == 0] = 1.0
    expr = np.log1p(X[col, :].toarray().astype(float) / cell_total[None, :] * 1e4).T  # cells x genes

    df = pd.DataFrame(expr, columns=[f"{g}_mRNA" for g in panel], index=cells)
    df.insert(0, "time_step", obs["time_step"].to_numpy())
    df.insert(0, "cell_id", obs["cell_barcode"].to_numpy())
    df.insert(0, "clone_id", obs["fatemap_clone_singletcode"].astype(str).to_numpy())
    df = df.reset_index(drop=True)

    counts = df["clone_id"].value_counts()
    oversized = counts[counts > MAX_CLONE_SIZE]
    if len(oversized):
        print(f"dropping {len(oversized)} oversized clone(s) (>{MAX_CLONE_SIZE} cells)", flush=True)
    keep_clones = counts[(counts >= 2) & (counts <= MAX_CLONE_SIZE)].index
    n_before = len(df)
    df = df[df["clone_id"].isin(keep_clones)].reset_index(drop=True)
    print(f"clone-size filter: {n_before:,} -> {len(df):,} cells ({len(keep_clones):,} clones)", flush=True)
    print(df["time_step"].value_counts().to_dict(), flush=True)

    out_csv = f"{DATA_DIR}/twinfer_input_hpsc_20260927_gt50.csv"
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv} ({len(df):,} cells x {len(panel)} genes)", flush=True)


if __name__ == "__main__":
    main()
