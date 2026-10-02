#!/usr/bin/env python3
"""Expression table for the Julia-PIDC D term: exactly the cells/genes the scorer's compute_D sees, i.e. replicate-B cells (t2_raw) of the
A/B-spanning clones for the gene set, log1p CP10k. Tab-separated, genes x cells -> <BASE>/analysis_data/fm06/data/pidc_julia/D_<gs>/ExpressionData.csv
usage: prep_julia_D_fm06.py <gene_set>"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import os, sys
import pandas as pd
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline")
from paper_analysis.fatemap_pipeline import apply_twinscore_supplement_fatemap_gated_bootstrap as gb
gs = sys.argv[1]
full, genes, *_ = gb.load_raw_ab("FM06", gs)
t2 = full[full.time_step == 1].reset_index(drop=True)
E = pd.DataFrame(t2[[f"{g}_mRNA" for g in genes]].to_numpy(float).T, index=genes, columns=[f"c{i}" for i in range(len(t2))])
const = E.index[E.nunique(axis=1) <= 1]
if len(const):
    print("dropping zero-variance genes:", list(const)); E = E.drop(index=const)
out = f"{TWINFER_PROJECT_ROOT}/analysis_data/fm06/data/pidc_julia/D_{gs}"
os.makedirs(out, exist_ok=True)
E.to_csv(f"{out}/ExpressionData.csv", sep="\t", header=True, index=True)
print(f"wrote {out}/ExpressionData.csv : {E.shape[0]} genes x {E.shape[1]} replicate-B cells")
