#!/usr/bin/env python3
"""Runs pick_gene_sets_fatemap.py's CollecTRI panel-picking recipe on HS054_Sot48h.
Single sample (no replicate split), so the "per-replicate minimum detection floor" and
"batch_key='replicate' HVG" steps in pick_gene_sets() reduce to plain overall detection
fraction / ungrouped HVG selection -- achieved here by aliasing obs['replicate'] to the
single sample label, rather than duplicating pick_gene_sets()'s ~150 lines.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import os

import anndata as ad

from paper_analysis.fatemap_pipeline.pick_gene_sets_fatemap import pick_gene_sets

H5AD_PATH = f'{TWINFER_PROJECT_ROOT}/finalized_data/HS054_Sot48h_data/HS054_Sot48h_integrated.h5ad'
OUT_DIR = f'{TWINFER_PROJECT_ROOT}/analysis_data/hs054/data'
LABEL = "HS054_Sot48h"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    A = ad.read_h5ad(H5AD_PATH)
    A.obs["replicate"] = A.obs["sample"]  # single sample -> single "replicate" category
    tmp_path = H5AD_PATH.replace(".h5ad", "_replicate_alias_tmp.h5ad")
    A.write_h5ad(tmp_path)
    try:
        pick_gene_sets(
            tmp_path,
            os.path.join(OUT_DIR, "gene_sets_hs054.json"),
            os.path.join(OUT_DIR, "gene_sets_detail_hs054.json"),
            LABEL,
        )
    finally:
        os.remove(tmp_path)


if __name__ == "__main__":
    main()
