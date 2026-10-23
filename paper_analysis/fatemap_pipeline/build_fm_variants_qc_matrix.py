#!/usr/bin/env python3
"""QC pipeline for the GSE233766 'more data' variants FM03 (DMSO/1uM PLX), FM05 (A1, A2; 250 nM PLX) and FM07
(continuous 1uM PLX). Same recipe as build_fm01_qc_matrix.py: singletCode (criteria 1-4, no manual UMI pre-filter, no
min_umi_cutoff set) + human MT- mito filter + gene-count floor/ceiling.

Difference from FM01/FM06: singletCode is run on the ENTIRE barcode sheet (every SampleNum in the table, including the
ones with no RNA matrix: FM03 sample 2, FM05 samples 3 and 4); nothing is subset before calling. Only afterwards are the
RNA cells matched to their singlet calls.

usage: python build_fm_variants_qc_matrix.py FM03|FM05|FM07
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent
import json
import os
import sys

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

SUB = "GSE233766_RAW_more_data_extracted"
STARCODE = f"{RAW_DIR}/Processed_FM01_FM03_FM04_FM05_FM06_FM08_FM09_FM10/%s/stepThreeStarcodeShavedReads.txt"
CFG = {
    # replicate label -> 10x prefix; sample_num -> label for the WHOLE sheet (labels without an RNA matrix are still passed to singletCode)
    "FM03": dict(reps={"A": "GSM7434413_FM03_A_DMSO_1uMPLX"},
                 sheet={1: "A", 2: "S2_noRNA"}),
    "FM05": dict(reps={"A1": "GSM7434417_FM05_A1_250nMPLX", "A2": "GSM7434418_FM05_A2_250nMPLX"},
                 sheet={1: "A1", 2: "A2", 3: "S3_noRNA", 4: "S4_noRNA"}),
    "FM07": dict(reps={"A": "GSM7434421_FM07_A_Continuous_1uMPLX", "B": "GSM7434422_FM07_B_Continuous_1uMPLX"},
                 sheet={1: "A", 2: "B"}),
}
# sample number -> RNA replicate verified by barcode overlap (92-99.5% of RNA barcodes in the matching sample, <=0.3% elsewhere)
MITO_PCT_CUTOFF = 26.0   # same as FM01 (FM06 used 21); MIN/MAX genes as FM01
MIN_GENES = 200
MAX_GENES = 7200
MIN_UMI_CUTOFF = None    # singletCode auto cutoff, as FM01


def main(ds):
    c = CFG[ds]
    out = f"{TWINFER_PROJECT_ROOT}/finalized_data/{ds}_data/qc_filtered"
    os.makedirs(out, exist_ok=True)
    print(f"[{ds}] loading 10x replicates", flush=True)
    adatas = [load_10x_replicate(f"{SUB}/{p}", rep) for rep, p in c["reps"].items()]
    var_names = adatas[0].var_names
    for a in adatas[1:]:
        assert (a.var_names == var_names).all()
    X = sp.vstack([a.X for a in adatas]).tocsr()
    cell_ids = np.concatenate([a.obs_names.to_numpy() for a in adatas])
    replicate = np.concatenate([a.obs["replicate"].to_numpy() for a in adatas])
    n_raw = X.shape[0]
    print(f"[{ds}] raw cells {n_raw}", flush=True)

    sheet = build_sample_sheet(STARCODE % ds, c["sheet"])
    print(f"[{ds}] sample sheet rows {len(sheet)}; per sample:\n{sheet['sample'].value_counts()}", flush=True)
    singlets, singlet_clone, rescued = run_singlet_calling(sheet, dataset_name=ds, min_umi_cutoff=MIN_UMI_CUTOFF)
    keys = set(zip(singlets["sample"], singlets["cellID"]))
    is_singlet = np.array([(r, i) in keys for r, i in zip(replicate, cell_ids)])
    print(f"[{ds}] singlet rows {len(singlets)}, rescued cellIDs {len(set(rescued))}, RNA cells matched {int(is_singlet.sum())}", flush=True)

    qc = apply_qc_filters(X, var_names, MITO_PCT_CUTOFF, MIN_GENES, MAX_GENES)
    keep = qc["pass_mito"] & is_singlet & qc["pass_genes"]
    n1, n2, n3 = int(qc["pass_mito"].sum()), int((qc["pass_mito"] & is_singlet).sum()), int(keep.sum())
    print(f"[{ds}] raw={n_raw} mito={n1} singlet={n2} genes={n3}", flush=True)

    ki = np.flatnonzero(keep)
    kept = pd.Series([f"{r}:{x}" for r, x in zip(replicate[keep], cell_ids[keep])])
    obs = pd.DataFrame({"replicate": replicate[keep], "cell_barcode": cell_ids[keep],
                        "pct_counts_mt": qc["pct_counts_mt"][keep], "n_genes": qc["n_genes"][keep],
                        "total_counts": qc["total_counts"][keep]}, index=kept)
    lookup = {f"{s}:{x}": b for (s, x), b in singlet_clone.items()}
    obs["fatemap_clone_singletcode"] = kept.map(lookup).fillna("").to_numpy()
    obs["criterion4_rescued"] = pd.Series(cell_ids[keep]).isin(set(rescued)).to_numpy()
    assert obs.index.is_unique

    sio.mmwrite(f"{out}/{ds.lower()}_qc_counts.mtx", X[ki])
    open(f"{out}/genes.txt", "w").write("\n".join(var_names) + "\n")
    open(f"{out}/cells.txt", "w").write("\n".join(kept) + "\n")
    obs.to_csv(f"{out}/obs_metadata.csv")
    summ = dict(dataset=ds, n_raw_cells=n_raw, n_after_mito=n1, n_after_singletcode_plus_rescue=n2,
                n_after_gene_floor_ceiling=n3, mito_pct_cutoff=MITO_PCT_CUTOFF, min_genes=MIN_GENES, max_genes=MAX_GENES,
                singletcode_min_umi_cutoff="auto", sheet_sample_num_to_label={str(k): v for k, v in c["sheet"].items()},
                singletcode_run_on="entire sheet (all SampleNum, none dropped)",
                n_clones=int(obs.loc[obs.fatemap_clone_singletcode != "", "fatemap_clone_singletcode"].nunique()))
    open(f"{out}/preprocessing_summary.json", "w").write(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
