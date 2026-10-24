#!/usr/bin/env python3
"""Corrected-rescue / full-sheet rebuild of the downstream files, written to <REBUILD>/FateMap (NEW, see fig5_paths.py). final_data is never overwritten.
The criterion-4 rescue is applied with a cell identified by (sample, cellID) and matched by barcode combination across samples
(check_rescue_logic.py: the original pools rows by cellID alone).
usage: rebuild_corrected_downstream.py wm_qc | wm_split | larry | larry_table1 | fm_integrated <FM01|FM06|FM08> | wm_stage <naive|lag|late>
"""
import contextlib
import importlib
import io
import os
import runpy
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "LARRY")]      # this folder, fig5_paths.py, build_final_matrix_umi3.py (LARRY)
from fig5_paths import FINAL as FIN, REBUILD                                            # noqa: E402
NEW = os.path.join(REBUILD, "FateMap")           # rebuilt files are written here; final_data is never overwritten
SCR = os.path.join(REBUILD, "scratch")
os.makedirs(SCR, exist_ok=True)
from singletcode_cross_sample_rescue import apply_cross_sample_rescue as ORIG           # noqa: E402


RESC_UNITS = set()          # (sample, cellID) units rescued; the builders only receive plain cellIDs, so their criterion4_rescued flag is rewritten afterwards


def corrected_rescue(good_data):
    """same function, cell = (sample, cellID): returns (good_data with original cellIDs, rescued as plain cellIDs of rescued (sample, cellID) units)"""
    g = good_data.copy()
    g["cellID"] = g["sample"].astype(str) + "\t" + g["cellID"].astype(str)
    g, resc = ORIG(g)
    g["cellID"] = g["cellID"].str.split("\t", n=1).str[1]
    RESC_UNITS.update(tuple(x.split("\t", 1)) for x in resc)
    return g, [x.split("\t", 1)[1] for x in resc]


def main(a):
    st = a[0]
    if st == "wm_qc":
        u = importlib.import_module("fatemap_qc_utils")
        u.apply_cross_sample_rescue = corrected_rescue
        m = importlib.import_module("build_watermelon_qc_matrix")
        m.OUT_DIR = f"{NEW}/Watermelon_data/qc_filtered"
        assert m.OUT_DIR.startswith(NEW)
        m.main()
        import pandas as pd
        o = pd.read_csv(f"{m.OUT_DIR}/obs_metadata.csv", index_col=0)
        o["criterion4_rescued"] = [(str(s), str(c)) in RESC_UNITS for s, c in zip(o["sample"], o["cell_barcode"])]
        o.to_csv(f"{m.OUT_DIR}/obs_metadata.csv")
        import json
        sj = f"{m.OUT_DIR}/preprocessing_summary.json"; j = json.load(open(sj))
        j["n_rescued_units_sample_cellID"] = len(RESC_UNITS); j["criterion4"] = "barcode combination across samples; cell = (sample, cellID); criterion4_rescued flag per (sample, cellID)"
        json.dump(j, open(sj, "w"), indent=2)
        print(f"criterion4_rescued flag per (sample, cellID): {int(o['criterion4_rescued'].sum())} cells, {len(RESC_UNITS)} units", flush=True)
    elif st in ("larry", "larry_table1"):
        m = importlib.import_module("build_final_matrix_umi3")
        m.PROCESSED = f"{FIN}/LARRY_data/intermediate_data_and_qc_details"
        m.OUT_DIR = f"{NEW}/LARRY_data/filtered" if st == "larry" else f"{NEW}/LARRY_table1_whitelist/filtered"
        if st == "larry_table1":
            # whitelist = Weinreb et al. Table 1 minimum UMIs for in vitro LSK (day2 500, day4 300, day6 300), per library by its day;
            # build_final_matrix_umi3.main() calls diptest_otsu_threshold once per library in sorted-library order
            import pandas as pd
            ob = pd.read_csv(f"{m.PROCESSED}/obs_metadata.csv", index_col=0)
            libs = sorted(ob["library"].unique()); nlib = ob["library"].value_counts().to_dict(); TB = {"2": 500, "4": 300, "6": 300}
            it = iter(libs)

            def table1_whitelist(counts):
                lib = next(it)
                assert len(counts) == nlib[lib], (lib, len(counts), nlib[lib])
                thr = TB[lib.split("_d")[1][0]]
                c = counts[counts > 0]
                print(f"[table1] {lib}: min UMIs {thr}", flush=True)
                return c >= thr, float(thr), False, -1.0
            m.diptest_otsu_threshold = table1_whitelist
        assert m.OUT_DIR.startswith(NEW)
        m.apply_weinreb_rule_ii_iii = lambda obs_qc, kept: (np.zeros(len(obs_qc), bool), np.zeros(len(obs_qc), bool))   # as the finalized file
        m.apply_cross_sample_rescue = corrected_rescue
        m.main()
        import json
        import pandas as pd
        o = pd.read_csv(f"{m.OUT_DIR}/obs_metadata.csv", index_col=0)
        o["criterion4_rescued"] = [(str(l), i.split(":", 1)[1]) in RESC_UNITS for l, i in zip(o["library"], o.index)]
        o.to_csv(f"{m.OUT_DIR}/obs_metadata.csv")
        sj = f"{m.OUT_DIR}/preprocessing_summary.json"; j = json.load(open(sj))
        j["n_rescued_units_sample_cellID"] = len(RESC_UNITS); j["criterion4"] = "barcode combination across samples; cell = (sample, cellID); criterion4_rescued flag per (sample, cellID)"
        json.dump(j, open(sj, "w"), indent=2)
        print(f"criterion4_rescued flag per (sample, cellID): {int(o['criterion4_rescued'].sum())} cells, {len(RESC_UNITS)} units", flush=True)
    elif st == "fm_integrated":
        ds = a[1]
        m = importlib.import_module(f"build_{ds.lower()}_integrated")
        m.QC_DIR = f"{NEW}/{ds}_data/qc_filtered"
        m.OUT_H5AD = f"{NEW}/{ds}_data/{ds}_integrated.h5ad"
        m.PLOT_DIR = f"{SCR}/{ds}_plots"; m.DATA_DIR = f"{SCR}/{ds}_data"
        assert m.OUT_H5AD.startswith(NEW) and os.path.isdir(m.QC_DIR)
        m.main()
    elif st == "wm_split":                               # prep_watermelon_stage_qc.py is a module-level script: run it as __main__
        runpy.run_path(os.path.join(HERE, "prep_watermelon_stage_qc.py"), run_name="__main__")
    elif st == "wm_stage":
        stage = a[1]
        m = importlib.import_module("build_watermelon_stage_integrated")
        m.QC_DIR = f"{NEW}/Watermelon_data/qc_filtered"               # out_dir of the stage script is <NEW>/Watermelon_<stage>_data
        m.PLOT_DIR = f"{SCR}/wm_plots"; m.DATA_DIR = f"{SCR}/wm_data"
        assert os.path.isdir(m.QC_DIR)
        sys.argv = ["x", stage]
        m.main()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
