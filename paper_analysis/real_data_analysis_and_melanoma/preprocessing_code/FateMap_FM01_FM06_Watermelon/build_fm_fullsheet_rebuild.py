#!/usr/bin/env python3
"""Rebuild of FM01, FM03, FM06, FM08 qc_filtered with (a) paper mito cutoffs and (b) singletCode on the ENTIRE barcode sheet
(every SampleNum, none dropped, one get_singlets call), as requested 2026-10-16. The original build scripts are untouched.

  mito %  : FM01 26, FM03 30, FM06 21, FM08 26 (paper values, given by K. Arun)
  genes   : FM01/FM03/FM08 200-7200, FM06 200-5000 (as in the original scripts)
  UMI     : FM01/FM03 singletCode auto cutoff; FM06/FM08 min_umi_cutoff 3 (as in the original scripts)
  rescue  : criterion 4 matches by barcode combination ACROSS samples, a cell identified by (sample, cellID) - the cell id is made
            sample-unique before the repo's apply_cross_sample_rescue and restored afterwards (the original pools rows by cellID alone)
usage: build_fm_fullsheet_rebuild.py FM01|FM03|FM05|FM06|FM08 <out_root>      writes <out_root>/<DS>_data/qc_filtered/
"""
import importlib
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

import singletCode
from fatemap_qc_utils import RAW_DIR, apply_qc_filters, build_sample_sheet, load_10x_replicate
from singletcode_cross_sample_rescue import apply_cross_sample_rescue

SH = "Processed_FM01_FM03_FM04_FM05_FM06_FM08_FM09_FM10"
CFG = {  # sheet: SampleNum -> label for the WHOLE sheet; labels without RNA are still given to singletCode
    "FM01": dict(mod="build_fm01_qc_matrix", mito=26.0, gmin=200, gmax=7200, umi=None, sheet={1: "A", 2: "B", 3: "S3_noRNA", 4: "S4_noRNA"}),
    "FM03": dict(mod="build_fm_variants_qc_matrix", mito=30.0, gmin=200, gmax=7200, umi=None, sheet={1: "A", 2: "S2_noRNA"}),
    "FM05": dict(mod="build_fm_variants_qc_matrix", mito=26.0, gmin=200, gmax=7200, umi=None, sheet={1: "A1", 2: "A2", 3: "S3_noRNA", 4: "S4_noRNA"}),   # unchanged settings; only the rescue differs
    "FM06": dict(mod="build_fm06_qc_matrix", mito=21.0, gmin=200, gmax=5000, umi=3, sheet={1: "A", 2: "B"}),
    "FM08": dict(mod="build_fm08_qc_matrix", mito=26.0, gmin=200, gmax=7200, umi=3, sheet={1: "A", 2: "B", 3: "S3_noRNA"}),
}


def main(ds, out_root):
    c = CFG[ds]
    m = importlib.import_module(f"{c['mod']}")
    if ds in ("FM03", "FM05"):
        reps = {k: f"{m.SUB}/{v}" for k, v in m.CFG[ds]["reps"].items()}
        starcode = m.STARCODE % ds
    else:
        reps, starcode = m.REPLICATES, m.STARCODE_PATH
    out = f"{out_root}/{ds}_data/qc_filtered"
    assert "/finalized_data/FM" not in out.replace(out_root, ""), out
    os.makedirs(out, exist_ok=True)
    adatas = [load_10x_replicate(p, r) for r, p in reps.items()]
    var_names = adatas[0].var_names
    for a in adatas[1:]:
        assert (a.var_names == var_names).all()
    X = sp.vstack([a.X for a in adatas]).tocsr()
    cell_ids = np.concatenate([a.obs_names.to_numpy() for a in adatas])
    replicate = np.concatenate([a.obs["replicate"].to_numpy() for a in adatas])
    n_raw = X.shape[0]
    sheet = build_sample_sheet(starcode, c["sheet"])
    print(f"[{ds}] raw cells {n_raw}; sheet rows {len(sheet)}\n{sheet['sample'].value_counts()}", flush=True)
    singletCode.check_sample_sheet(sheet)
    kw = {} if c["umi"] is None else {"min_umi_cutoff": c["umi"]}
    good, _ = singletCode.get_singlets(sheet, dataset_name=ds, **kw)       # ONE call on the whole sheet
    good = good.copy()
    good["cellID"] = good["sample"].astype(str) + "\t" + good["cellID"].astype(str)
    good, rescued_u = apply_cross_sample_rescue(good)
    good["cellID"] = good["cellID"].str.split("\t", n=1).str[1]
    rescued = {tuple(x.split("\t", 1)) for x in rescued_u}
    sing = good[good["label"] == "Singlet"]
    clone = sing.groupby(["sample", "cellID"])["barcode"].apply(lambda s: "".join(sorted(set(s))))
    keys = set(clone.index)
    is_singlet = np.array([(r, i) in keys for r, i in zip(replicate, cell_ids)])
    qc = apply_qc_filters(X, var_names, c["mito"], c["gmin"], c["gmax"])
    keep = qc["pass_mito"] & is_singlet & qc["pass_genes"]
    n1, n2, n3 = int(qc["pass_mito"].sum()), int((qc["pass_mito"] & is_singlet).sum()), int(keep.sum())
    print(f"[{ds}] raw={n_raw} mito={n1} singlet={n2} genes={n3}", flush=True)
    ki = np.flatnonzero(keep)
    kept = pd.Series([f"{r}:{x}" for r, x in zip(replicate[keep], cell_ids[keep])])
    obs = pd.DataFrame({"replicate": replicate[keep], "cell_barcode": cell_ids[keep], "pct_counts_mt": qc["pct_counts_mt"][keep],
                        "n_genes": qc["n_genes"][keep], "total_counts": qc["total_counts"][keep]}, index=kept)
    obs["fatemap_clone_singletcode"] = kept.map({f"{s}:{x}": b for (s, x), b in clone.items()}).fillna("").to_numpy()
    obs["criterion4_rescued"] = [(r, i) in rescued for r, i in zip(replicate[keep], cell_ids[keep])]
    assert obs.index.is_unique
    sio.mmwrite(f"{out}/{ds.lower()}_qc_counts.mtx", X[ki])
    open(f"{out}/genes.txt", "w").write("\n".join(var_names) + "\n")
    open(f"{out}/cells.txt", "w").write("\n".join(kept) + "\n")
    obs.to_csv(f"{out}/obs_metadata.csv")
    summ = dict(dataset=ds, n_raw_cells=n_raw, n_after_mito=n1, n_after_singletcode_plus_rescue=n2, n_after_gene_floor_ceiling=n3,
                mito_pct_cutoff=c["mito"], min_genes=c["gmin"], max_genes=c["gmax"], singletcode_min_umi_cutoff=c["umi"] or "auto",
                sheet_sample_num_to_label={str(k): v for k, v in c["sheet"].items()},
                singletcode_run_on="entire sheet, one call (all SampleNum, none dropped)",
                criterion4="barcode combination across samples; cell = (sample, cellID)", n_rescued_cells=len(rescued),
                n_clones=int(obs.loc[obs.fatemap_clone_singletcode != "", "fatemap_clone_singletcode"].nunique()))
    open(f"{out}/preprocessing_summary.json", "w").write(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
