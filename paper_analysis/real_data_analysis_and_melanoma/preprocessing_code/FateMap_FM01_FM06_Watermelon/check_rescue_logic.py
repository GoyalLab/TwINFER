#!/usr/bin/env python3
"""Does the repo's criterion-4 'cross-sample rescue' change any FINAL cell's singlet call because it pools rows by cellID across samples?

apply_cross_sample_rescue(good_data) builds each multiplet's barcode combination from ALL rows with that cellID, in every sample, and then
relabels every row with a rescued cellID (every sample) as Singlet. A cellID in two samples is two different cells, so the pooled
combination can mix two cells' lineage barcodes. Here the build script is rerun (output redirected to scratch), the singletCode output
(good_data, labels after criteria 1-3) is captured, and the rescue is applied twice:
  as implemented (key = cellID)  and  corrected (key = sample + cellID, i.e. the same function on a table whose cellID is made sample-unique).
Reported per dataset: rescued (sample, cellID) units under each, the differences, and how many FINAL cells (finalized obs) differ.
usage: check_rescue_logic.py <DATASET> ...   (FM01 FM03 FM05 FM06 FM07 FM08 Watermelon LARRY)
"""
import contextlib
import importlib
import io
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "LARRY")]      # this folder, fig5_paths.py, build_final_matrix_umi3.py (LARRY)
from fig5_paths import FINAL as FIN, REBUILD                                            # noqa: E402
OUTR = os.path.join(REBUILD, "rescue_check")
from singletcode_cross_sample_rescue import apply_cross_sample_rescue as ORIG           # noqa: E402

MOD = {"FM01": "build_fm01_qc_matrix", "FM06": "build_fm06_qc_matrix", "FM08": "build_fm08_qc_matrix",
       "FM03": "build_fm_variants_qc_matrix", "FM05": "build_fm_variants_qc_matrix", "FM07": "build_fm_variants_qc_matrix",
       "Watermelon": "build_watermelon_qc_matrix", "LARRY": None}


def final_keys(ds):
    if ds == "LARRY":
        o = pd.read_csv(f"{FIN}/LARRY_data/filtered/obs_metadata.csv", index_col=0)
        return {(lib, cid.split(":", 1)[1]) for lib, cid in zip(o["library"], o.index)}, o
    o = pd.read_csv(f"{FIN}/{ds}_data/qc_filtered/obs_metadata.csv", index_col=0)
    samp = o["sample"] if ds == "Watermelon" else o["replicate"]
    return set(zip(samp.astype(str), o["cell_barcode"].astype(str))), o


def run(ds):
    out = f"{OUTR}/{ds}"; os.makedirs(out, exist_ok=True)
    cap = {}

    def capture(good_data):
        cap["gd"] = good_data.copy()
        return ORIG(good_data)

    log = io.StringIO()
    if ds == "LARRY":
        m = importlib.import_module("build_final_matrix_umi3")
        m.PROCESSED = f"{FIN}/LARRY_data/intermediate_data_and_qc_details"; m.OUT_DIR = f"{out}/build"
        m.apply_weinreb_rule_ii_iii = lambda obs_qc, kept: (np.zeros(len(obs_qc), bool), np.zeros(len(obs_qc), bool))
        m.apply_cross_sample_rescue = capture
        with contextlib.redirect_stdout(log): m.main()
    else:
        u = importlib.import_module("fatemap_qc_utils")
        u.apply_cross_sample_rescue = capture
        m = importlib.import_module(f"{MOD[ds]}")
        with contextlib.redirect_stdout(log):
            if MOD[ds] == "build_fm_variants_qc_matrix":
                m.TWINFER_PROJECT_ROOT = f"{out}/_root"; m.main(ds)
            else:
                m.OUT_DIR = f"{out}/build"; m.main()
    gd = cap["gd"]
    assert not any(str(p).startswith(FIN) for p in [getattr(m, "OUT_DIR", out)])
    pre_mult = set(map(tuple, gd.loc[gd["label"] == "Multiplet", ["sample", "cellID"]].astype(str).drop_duplicates().to_numpy()))
    # as implemented
    g1, resc1 = ORIG(gd)
    resc1 = set(resc1)
    units_impl = {(s, c) for s, c in pre_mult if c in resc1}                   # (sample, cellID) units whose label flips to Singlet
    # corrected: same function, cellID made sample-unique
    gd2 = gd.copy(); gd2["cellID"] = gd2["sample"].astype(str) + "\t" + gd2["cellID"].astype(str)
    _, resc2 = ORIG(gd2)
    units_corr = {tuple(x.split("\t", 1)) for x in resc2}
    fk, o = final_keys(ds)
    cid_samples = gd.groupby("cellID")["sample"].nunique()
    r = dict(dataset=ds, samples=int(gd["sample"].nunique()), cellIDs=int(gd["cellID"].nunique()),
             cellIDs_in_more_than_one_sample=int((cid_samples > 1).sum()),
             multiplet_units_before_rescue=len(pre_mult),
             rescued_units_as_implemented=len(units_impl), rescued_units_corrected=len(units_corr),
             rescued_only_as_implemented=len(units_impl - units_corr), rescued_only_corrected=len(units_corr - units_impl),
             FINAL_cells_rescued_only_as_implemented=len((units_impl - units_corr) & fk),
             FINAL_cells_that_would_be_added_by_correction=len(units_corr - units_impl),
             final_cells=len(fk),
             examples_wrong=sorted(units_impl - units_corr)[:10])
    json.dump(r, open(f"{out}/rescue_check.json", "w"), indent=1, default=str)
    pd.DataFrame(sorted((units_impl - units_corr) & fk), columns=["sample", "cellID"]).to_csv(f"{out}/final_cells_rescued_only_as_implemented.csv", index=False)
    print(json.dumps(r, default=str), flush=True)


if __name__ == "__main__":
    for d in sys.argv[1:]:
        run(d)
