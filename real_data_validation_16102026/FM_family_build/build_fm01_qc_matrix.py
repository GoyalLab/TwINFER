#!/usr/bin/env python3
"""FM01 (WM989 + 1uM PLX) QC pipeline: singletCode doublet removal (criteria 1-4, no manual UMI
pre-filter, no min_umi_cutoff set) + human MT- mito filter (<=26%) + gene-count floor/ceiling
(200-7200). Mirrors larry_pipeline/filtering/build_final_matrix_umi3.py, minus
the whitelisting step (FM01's 10x matrices are already Cell Ranger filtered).
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
import os

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

from paper_analysis.fatemap_pipeline.fatemap_qc_utils import (
    RAW_DIR,
    apply_qc_filters,
    build_sample_sheet,
    load_10x_replicate,
    run_singlet_calling,
)

STARCODE_PATH = (
    f"{RAW_DIR}/Processed_FM01_FM03_FM04_FM05_FM06_FM08_FM09_FM10/FM01/"
    "stepFourStarcodeShavedReads.txt"
)
REPLICATES = {"A": "GSM7434407_FM01_A_1uMPLX", "B": "GSM7434408_FM01_B_1uMPLX"}
# sampleNum 1/2 = RNA replicates A/B (barcode-overlap check: 8028/8371 and 7200/7439 cells match); sampleNum 3/4
# have no RNA matrix (GEO has only FM01_A/B) and are dropped by build_sample_sheet.
SAMPLE_NUM_TO_REPLICATE = {1: "A", 2: "B"}
OUT_DIR = f'{TWINFER_PROJECT_ROOT}/finalized_data/FM01_data/qc_filtered'
DATASET_NAME = "FM01"
MITO_PCT_CUTOFF = 26.0
MIN_GENES = 200
MAX_GENES = 7200
MIN_UMI_CUTOFF = None  # not set: singletCode auto-chooses its UMI cutoff (package default floor + ratio method)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[FM01] loading 10x replicates", flush=True)
    adatas = [load_10x_replicate(prefix, rep) for rep, prefix in REPLICATES.items()]
    var_names = adatas[0].var_names
    for a in adatas[1:]:
        assert (a.var_names == var_names).all(), "gene order differs between replicates"
    X = sp.vstack([a.X for a in adatas]).tocsr()
    cell_ids = np.concatenate([a.obs_names.to_numpy() for a in adatas])
    replicate = np.concatenate([a.obs["replicate"].to_numpy() for a in adatas])
    n_raw = X.shape[0]
    print(f"[FM01] raw cells across replicates: {n_raw}", flush=True)

    print("[FM01] building singletCode sample sheet", flush=True)
    sample_sheet = build_sample_sheet(STARCODE_PATH, SAMPLE_NUM_TO_REPLICATE)
    print(f"[FM01] sample sheet rows: {len(sample_sheet)}", flush=True)

    print("[FM01] running singletCode + cross-sample rescue", flush=True)
    singlets, singlet_clone, rescued_cellids = run_singlet_calling(
        sample_sheet, dataset_name=DATASET_NAME, min_umi_cutoff=MIN_UMI_CUTOFF
    )
    singlet_keys = set(zip(singlets["sample"], singlets["cellID"]))
    is_singlet = np.array(
        [(rep, cid) in singlet_keys for rep, cid in zip(replicate, cell_ids)]
    )
    n_singlet = int(is_singlet.sum())
    print(
        f"[FM01] singletCode: {len(singlets)} singlet rows, "
        f"{len(set(rescued_cellids))} distinct cellIDs rescued by criterion 4, "
        f"{n_singlet} raw cells matched to a singlet cellID",
        flush=True,
    )

    print("[FM01] applying mito + gene-count filters", flush=True)
    qc = apply_qc_filters(X, var_names, MITO_PCT_CUTOFF, MIN_GENES, MAX_GENES)

    after_mito = qc["pass_mito"]
    after_singlet = after_mito & is_singlet
    after_genes = after_singlet & qc["pass_genes"]

    n1, n2, n3 = int(after_mito.sum()), int(after_singlet.sum()), int(after_genes.sum())
    print(
        f"[FM01] after raw={n_raw}  after mito={n1}  after singletCode(+rescue)={n2}  "
        f"after gene floor/ceiling={n3}",
        flush=True,
    )

    keep = after_genes
    keep_idx = np.flatnonzero(keep)
    X_qc = X[keep_idx]
    kept_cells = pd.Series([f"{r}:{c}" for r, c in zip(replicate[keep], cell_ids[keep])])

    obs_qc = pd.DataFrame(
        {
            "replicate": replicate[keep],
            "cell_barcode": cell_ids[keep],
            "pct_counts_mt": qc["pct_counts_mt"][keep],
            "n_genes": qc["n_genes"][keep],
            "total_counts": qc["total_counts"][keep],
        },
        index=kept_cells,
    )
    clone_lookup = {f"{s}:{c}": bc for (s, c), bc in singlet_clone.items()}
    obs_qc["fatemap_clone_singletcode"] = kept_cells.map(clone_lookup).fillna("").to_numpy()
    rescued_set = set(rescued_cellids)
    obs_qc["criterion4_rescued"] = pd.Series(cell_ids[keep]).isin(rescued_set).to_numpy()

    assert X_qc.shape[0] == len(obs_qc) == len(kept_cells)
    assert obs_qc.index.is_unique

    sio.mmwrite(f"{OUT_DIR}/fm01_qc_counts.mtx", X_qc)
    open(f"{OUT_DIR}/genes.txt", "w").write("\n".join(var_names) + "\n")
    open(f"{OUT_DIR}/cells.txt", "w").write("\n".join(kept_cells) + "\n")
    obs_qc.to_csv(f"{OUT_DIR}/obs_metadata.csv")

    summary = {
        "dataset": DATASET_NAME,
        "n_raw_cells": n_raw,
        "n_after_mito": n1,
        "n_after_singletcode_plus_rescue": n2,
        "n_after_gene_floor_ceiling": n3,
        "mito_pct_cutoff": MITO_PCT_CUTOFF,
        "min_genes": MIN_GENES,
        "max_genes": MAX_GENES,
        "singletcode_min_umi_cutoff": "auto (singletCode default)",
        "n_distinct_cellids_rescued_by_criterion4": len(rescued_set),
        "sample_num_to_replicate": SAMPLE_NUM_TO_REPLICATE,
    }
    open(f"{OUT_DIR}/preprocessing_summary.json", "w").write(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[FM01] wrote {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
