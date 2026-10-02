#!/usr/bin/env python3
"""Final gated + bootstrap TwinScore_supplement with the D term taken from the Julia PIDC (NetworkInference.jl) instead of the Python PIDC.
Identical to apply_twinscore_supplement_fatemap_gated_bootstrap.py in everything else; only compute_D is replaced by a reader of
<BASE>/analysis_data/fm06/data/pidc_julia/D_<gene_set>/outFile.txt (Julia output on the replicate-B cells, made by prep_julia_D_fm06.py + run_pidc_julia.jl).
Julia PIDC gives one score per unordered pair, so D is symmetric here. Same arguments as the final script; give --out-dir (recommended:
<BASE>/analysis_data/fm06/data/twinscore_supp_gated_bootstrap_juliaD)."""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import argparse
import os
import sys

import pandas as pd

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
# TWINFER_GB selects the scorer module: default = final serial script; apply_twinscore_supplement_fatemap_gated_bootstrap_parallel = parallel bootstraps
gb = importlib.import_module(os.environ.get("TWINFER_GB", "apply_twinscore_supplement_fatemap_gated_bootstrap"))

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--gene-set", default="correlation_high")
_ap.add_argument("--d-file", default=None, help="Julia PIDC output to use as D (default pidc_julia/D_<gene_set>/outFile.txt = symmetric package PIDC; "
                  "use .../outFile_directed.txt for the directed variant, see run_pidc_julia_directed.jl)")
_known, _rest = _ap.parse_known_args()
_GS = _known.gene_set
_DFILE = _known.d_file or f"{TWINFER_PROJECT_ROOT}/analysis_data/fm06/data/pidc_julia/D_{_GS}/outFile.txt"
sys.argv = [sys.argv[0]] + _rest + ["--gene-set", _GS]   # --d-file is not known to the final script's parser


def julia_D(t2_raw, genes):
    jl = pd.read_csv(_DFILE, sep="\t", header=None, names=["g1", "g2", "w"])
    D = pd.DataFrame(0.0, index=genes, columns=genes)
    directed = "directed" in os.path.basename(_DFILE)
    for a, b, v in zip(jl.g1, jl.g2, jl.w):
        if a in D.index and b in D.columns:
            D.loc[a, b] = v          # source -> target for the directed file; the same weight is in both rows for the package file
            if not directed:
                D.loc[b, a] = v
    gb.log(f"    D term: Julia PIDC from {_DFILE} ({len(jl)} rows)")
    return D


gb.compute_D = julia_D

if __name__ == "__main__":
    gb.main()
