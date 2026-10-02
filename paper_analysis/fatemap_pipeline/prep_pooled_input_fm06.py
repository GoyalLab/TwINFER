#!/usr/bin/env python3
"""Writes the pooled TwINFER input table twinfer_input_fm06_<gene_set>.csv (all clone-barcoded cells, clone size 2-50, log1p(CP10k) of the panel genes)
that run_competitors_fatemap.py reads. Same builder as run_infer_fatemap.py (build_twinfer_input); that script also runs pooled inference, which is not
needed here.   usage: prep_pooled_input_fm06.py <gene_set> [--out PATH]"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import argparse
import json
import os
import sys

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_analysis.fatemap_pipeline.run_infer_fatemap import build_twinfer_input  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("gene_set")
ap.add_argument("--out", default=None)
a = ap.parse_args()
base = f'{TWINFER_PROJECT_ROOT}/analysis_data/fm06/data'
genes = json.load(open(f"{base}/gene_sets_fm06.json"))[a.gene_set]
df = build_twinfer_input("FM06", genes)
out = a.out or f"{base}/twinfer_input_fm06_{a.gene_set}.csv"
df.to_csv(out, index=False)
print(f"{a.gene_set}: {len(genes)} genes, {len(df):,} cells, {df['clone_id'].nunique():,} clones -> {out}", flush=True)
