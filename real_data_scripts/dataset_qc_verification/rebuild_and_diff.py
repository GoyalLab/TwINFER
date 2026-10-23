#!/usr/bin/env python3
"""Rerun the ORIGINAL build script of a dataset with its output redirected to a scratch directory, then compare every output file
with the one in finalized_data element by element. finalized_data is never written: the module's output path is replaced before
main() runs and an assertion checks it; md5 sums of the finalized files are recorded before and after the run.

usage: rebuild_and_diff.py <DATASET> [<DATASET> ...]
   DATASET in FM01 FM06 FM08 FM03 FM05 FM07 HS054_Sot48h Watermelon Watermelon_stages LARRY
Writes analysis_data/dataset_qc_verification/rebuild/<DATASET>/  (rebuilt files + diff.json)
"""
import contextlib
import glob
import hashlib
import importlib
import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.io as sio

ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"
CODE = f"{ROOT}/code/TwINFER"
sys.path[:0] = [CODE, f"{CODE}/package"]
FIN = f"{ROOT}/finalized_data"
OUTR = f"{ROOT}/analysis_data/dataset_qc_verification/rebuild"

# dataset -> (module, finalized qc dir, how to redirect)
SPEC = {
    "FM01": ("paper_analysis.fatemap_pipeline.build_fm01_qc_matrix", f"{FIN}/FM01_data/qc_filtered", "OUT_DIR"),
    "FM06": ("paper_analysis.fatemap_pipeline.build_fm06_qc_matrix", f"{FIN}/FM06_data/qc_filtered", "OUT_DIR"),
    "FM08": ("paper_analysis.fatemap_pipeline.build_fm08_qc_matrix", f"{FIN}/FM08_data/qc_filtered", "OUT_DIR"),
    "FM03": ("paper_analysis.fatemap_pipeline.build_fm_variants_qc_matrix", f"{FIN}/FM03_data/qc_filtered", "ROOT"),
    "FM05": ("paper_analysis.fatemap_pipeline.build_fm_variants_qc_matrix", f"{FIN}/FM05_data/qc_filtered", "ROOT"),
    "FM07": ("paper_analysis.fatemap_pipeline.build_fm_variants_qc_matrix", f"{FIN}/FM07_data/qc_filtered", "ROOT"),
    "HS054_Sot48h": ("paper_analysis.fatemap_pipeline.build_hs054_qc_matrix", f"{FIN}/HS054_Sot48h_data/qc_filtered", "OUT_DIR"),
    "Watermelon": ("paper_analysis.fatemap_pipeline.build_watermelon_qc_matrix", f"{FIN}/Watermelon_data/qc_filtered", "OUT_DIR"),
    # LARRY stage 1: per-library inDrops zips + barcode fastq (finalized_data/LARRY_data/raw) -> combined matrix + umi table
    "LARRY_stage1": ("paper_analysis.larry_pipeline.build_matrix.build_larry_matrix", f"{FIN}/LARRY_data/intermediate_data_and_qc_details", "LARRY1"),
    # LARRY stage 2: the finalized stage-1 outputs -> whitelist, mito, singletCode(+rescue), gene floor, Weinreb rules (current script)
    "LARRY_stage2": ("paper_analysis.larry_pipeline.filtering.build_final_matrix_umi3", f"{FIN}/LARRY_data/filtered", "LARRY2"),
    # same, with the Weinreb rule (ii)/(iii) step switched off (rule function returns all-False masks); tests whether finalized = pre-Step-5 output
    "LARRY_stage2_noStep5": ("paper_analysis.larry_pipeline.filtering.build_final_matrix_umi3", f"{FIN}/LARRY_data/filtered", "LARRY2_NOSTEP5"),
}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def snapshot(d):
    return {os.path.basename(p): md5(p) for p in sorted(glob.glob(f"{d}/*")) if os.path.isfile(p)}


def compare_dir(new, old):
    """element-wise comparison of every file present in `old` (the finalized dir)."""
    res = {}
    for p_old in sorted(glob.glob(f"{old}/*")):
        if not os.path.isfile(p_old):
            continue
        name = os.path.basename(p_old)
        p_new = f"{new}/{name}"
        r = {"present_in_rebuild": os.path.exists(p_new)}
        if not r["present_in_rebuild"]:
            res[name] = r
            continue
        r["byte_identical"] = md5(p_new) == md5(p_old)
        if not r["byte_identical"]:
            if name.endswith(".mtx"):
                A = sio.mmread(p_old).tocsr(); B = sio.mmread(p_new).tocsr()
                A.sort_indices(); B.sort_indices()
                r["shape_old_new"] = [A.shape, B.shape]
                if A.shape == B.shape:
                    D = (A - B)
                    r["n_entries_differing"] = int((abs(D) > 0).sum())
                    r["max_abs_diff"] = float(abs(D).max()) if D.nnz else 0.0
                    r["values_identical"] = r["n_entries_differing"] == 0
            elif name.endswith(".txt"):
                a = open(p_old).read().splitlines(); b = open(p_new).read().splitlines()
                r["n_lines_old_new"] = [len(a), len(b)]
                r["same_lines_same_order"] = a == b
                r["same_set"] = set(a) == set(b)
                r["only_old_first5"] = sorted(set(a) - set(b))[:5]; r["only_new_first5"] = sorted(set(b) - set(a))[:5]
            elif name.endswith(".csv"):
                A = pd.read_csv(p_old, index_col=0); B = pd.read_csv(p_new, index_col=0)
                r["shape_old_new"] = [A.shape, B.shape]
                r["same_index_order"] = list(A.index) == list(B.index)
                r["same_columns"] = list(A.columns) == list(B.columns)
                if r["same_index_order"] and r["same_columns"]:
                    cols = {}
                    for c in A.columns:
                        a, b = A[c], B[c]
                        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                            d = np.abs(a.to_numpy(float) - b.to_numpy(float))
                            cols[c] = dict(n_diff=int((d > 0).sum()), max_abs=float(np.nanmax(d)) if len(d) else 0.0)
                        else:
                            cols[c] = dict(n_diff=int((a.astype(str) != b.astype(str)).sum()))
                    r["columns"] = cols
            elif name.endswith(".json"):
                a = json.load(open(p_old)); b = json.load(open(p_new))
                r["keys_differing"] = {k: [a.get(k), b.get(k)] for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}
        res[name] = r
    return res


def run(ds):
    out = f"{OUTR}/{ds}"
    os.makedirs(out, exist_ok=True)
    mod_name, fin_dir, how = SPEC[ds]
    before = snapshot(fin_dir)
    m = importlib.import_module(mod_name)
    t0 = time.time()
    log = io.StringIO()
    if how == "OUT_DIR":
        m.OUT_DIR = out
        assert m.OUT_DIR == out and not m.OUT_DIR.startswith(FIN)
        with contextlib.redirect_stdout(log):
            m.main()
        new_dir = out
    elif how == "LARRY1":
        from pathlib import Path
        m.DATASET_DIR = Path(f"{FIN}/LARRY_data/raw"); m.FASTQ_PATH = m.DATASET_DIR / "LARRY_sorted_and_filtered_barcodes.fastq.gz"
        m.OUT_DIR = Path(out)
        assert not str(m.OUT_DIR).startswith(FIN)
        with contextlib.redirect_stdout(log):
            m.main()
        new_dir = out
    elif how == "LARRY2_NOSTEP5":
        m.PROCESSED = f"{FIN}/LARRY_data/intermediate_data_and_qc_details"; m.OUT_DIR = out
        m.apply_weinreb_rule_ii_iii = lambda obs_qc, kept_cells: (np.zeros(len(obs_qc), bool), np.zeros(len(obs_qc), bool))
        with contextlib.redirect_stdout(log):
            m.main()
        new_dir = out
    elif how == "LARRY2":
        m.PROCESSED = f"{FIN}/LARRY_data/intermediate_data_and_qc_details"; m.OUT_DIR = out
        assert not m.OUT_DIR.startswith(FIN)
        with contextlib.redirect_stdout(log):
            m.main()
        new_dir = out
    else:  # variants: output path = f"{TWINFER_PROJECT_ROOT}/finalized_data/{ds}_data/qc_filtered"
        fake = f"{out}/_root"
        m.TWINFER_PROJECT_ROOT = fake
        new_dir = f"{fake}/finalized_data/{ds}_data/qc_filtered"
        with contextlib.redirect_stdout(log):
            m.main(ds)
    after = snapshot(fin_dir)
    open(f"{out}/build_stdout.log", "w").write(log.getvalue())
    res = dict(dataset=ds, module=mod_name, finalized_dir=fin_dir, rebuilt_dir=new_dir, runtime_s=round(time.time() - t0),
               finalized_untouched=before == after, files=compare_dir(new_dir, fin_dir))
    res["ALL_IDENTICAL"] = all(v.get("byte_identical") or v.get("values_identical") or
                               (v.get("same_lines_same_order") is True) for v in res["files"].values())
    json.dump(res, open(f"{out}/diff.json", "w"), indent=1, default=str)
    print(ds, "ALL_IDENTICAL =", res["ALL_IDENTICAL"], {k: ("byte" if v.get("byte_identical") else v) for k, v in res["files"].items()}, flush=True)
    return res


if __name__ == "__main__":
    for ds in sys.argv[1:]:
        run(ds)
