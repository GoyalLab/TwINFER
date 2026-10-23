#!/usr/bin/env python3
"""Corrected-rescue / full-sheet rebuild of the downstream files, written to finalized_data/rebuild_fullsheet_2026-10-16 (NEW). Originals untouched.
The criterion-4 rescue is applied with a cell identified by (sample, cellID) and matched by barcode combination across samples
(check_rescue_logic.py: the original pools rows by cellID alone).
usage: rebuild_corrected_downstream.py wm_qc | wm_split | larry | fm_integrated <FM01|FM06|FM08> | wm_stage <naive|lag|late>
"""
import contextlib
import importlib
import io
import os
import sys

import numpy as np

ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"
CODE = f"{ROOT}/code/TwINFER"
sys.path[:0] = [CODE, f"{CODE}/package"]
FIN = f"{ROOT}/finalized_data"
NEW = f"{FIN}/rebuild_fullsheet_2026-10-16"
SCR = f"{ROOT}/analysis_data/dataset_qc_verification/fullsheet_scratch"
os.makedirs(SCR, exist_ok=True)
from paper_analysis.larry_hematopoiesis_validation.preprocessing.singletcode_cross_sample_rescue import apply_cross_sample_rescue as ORIG


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
        u = importlib.import_module("paper_analysis.fatemap_pipeline.fatemap_qc_utils")
        u.apply_cross_sample_rescue = corrected_rescue
        m = importlib.import_module("paper_analysis.fatemap_pipeline.build_watermelon_qc_matrix")
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
    elif st == "larry":
        m = importlib.import_module("paper_analysis.larry_pipeline.filtering.build_final_matrix_umi3")
        m.PROCESSED = f"{FIN}/LARRY_data/intermediate_data_and_qc_details"
        m.OUT_DIR = f"{NEW}/LARRY_data/filtered"
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
        m = importlib.import_module(f"paper_analysis.fatemap_pipeline.build_{ds.lower()}_integrated")
        m.QC_DIR = f"{NEW}/{ds}_data/qc_filtered"
        m.OUT_H5AD = f"{NEW}/{ds}_data/{ds}_integrated.h5ad"
        m.PLOT_DIR = f"{SCR}/{ds}_plots"; m.DATA_DIR = f"{SCR}/{ds}_data"
        assert m.OUT_H5AD.startswith(NEW) and os.path.isdir(m.QC_DIR)
        m.main()
    elif st == "wm_split":                               # prep_watermelon_stage_qc.py is a module-level script: run its source with the root replaced
        tree = f"{SCR}/wm_tree"; os.makedirs(tree, exist_ok=True)
        if not os.path.islink(f"{tree}/finalized_data"):
            os.symlink(NEW, f"{tree}/finalized_data")
        src = open(f"{CODE}/paper_analysis/fatemap_pipeline/prep_watermelon_stage_qc.py").read().splitlines()
        i = [k for k, l in enumerate(src) if l.startswith("TWINFER_PROJECT_ROOT = ")]
        assert len(i) == 1
        src[i[0]] = f'TWINFER_PROJECT_ROOT = "{tree}"'
        exec(compile("\n".join(src), "prep_watermelon_stage_qc.py(redirected)", "exec"), {"__name__": "__main__"})
    elif st == "wm_stage":
        stage = a[1]
        m = importlib.import_module("paper_analysis.fatemap_pipeline.build_watermelon_stage_integrated")
        tree = f"{SCR}/wm_tree"; os.makedirs(tree, exist_ok=True)
        if not os.path.islink(f"{tree}/finalized_data"):
            os.symlink(NEW, f"{tree}/finalized_data")
        m.TWINFER_PROJECT_ROOT = tree                      # out_dir = <tree>/finalized_data/Watermelon_<stage>_data -> NEW
        m.QC_DIR = f"{NEW}/Watermelon_data/qc_filtered"
        m.PLOT_DIR = f"{SCR}/wm_plots"; m.DATA_DIR = f"{SCR}/wm_data"
        assert os.path.isdir(m.QC_DIR)
        sys.argv = ["x", stage]
        m.main()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
