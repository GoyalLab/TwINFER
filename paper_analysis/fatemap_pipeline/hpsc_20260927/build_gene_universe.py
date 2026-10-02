#!/usr/bin/env python3
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Detected-gene universe for hPSC_20260927: genes detected (>0 counts) in >=5% of the
6,454 QC+singlet-filtered cells (endo_T0+endo_T1 combined), per the PDF's "Genes and
regulators" rule, minus MALAT1/NEAT1/XIST/FIRRE + mitochondrial genes (PDF's candidate
-pair exclusions) and minus the GO/Tirosh/histone exclusion set (build_gene_exclusion_list.py).
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json

import numpy as np
import pandas as pd
import scipy.io as sio

QC_DIR = f'{TWINFER_PROJECT_ROOT}/finalized_data/hPSC_20260927_data/qc_filtered'
EXCLUSION_JSON = f'{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/fatemap_pipeline/hpsc_20260927/gene_exclusion_list.json'
OUT_DIR = f'{TWINFER_PROJECT_ROOT}/analysis_data/hPSC_20260927/data'

PDF_ALWAYS_EXCLUDE = {"MALAT1", "NEAT1", "XIST", "FIRRE"}


def main():
    X = sio.mmread(f"{QC_DIR}/hPSC_20260927_qc_counts.mtx").tocsr()
    genes = np.array(open(f"{QC_DIR}/genes.txt").read().split())
    n_cells = X.shape[1]
    print(f"loaded {X.shape[0]} genes x {n_cells} cells")

    frac_detected = np.asarray((X > 0).sum(axis=1)).ravel() / n_cells
    detected = genes[frac_detected >= 0.05]
    print(f"genes detected in >=5% of cells: {len(detected)}")

    mito = {g for g in detected if g.upper().startswith("MT-")}
    exclusion = json.load(open(EXCLUSION_JSON))
    go_tirosh_histone = set(exclusion["_union_all"])
    always_excl = PDF_ALWAYS_EXCLUDE | mito

    excluded_total = go_tirosh_histone | always_excl
    targets = sorted(set(detected) - excluded_total)
    print(f"MALAT1/NEAT1/XIST/FIRRE present & excluded: {len(always_excl & set(detected) - mito)}")
    print(f"mito genes present & excluded: {len(mito)}")
    print(f"GO/Tirosh/histone genes present & excluded: {len(go_tirosh_histone & set(detected))}")
    print(f"final target gene universe: {len(targets)} genes")

    json.dump(targets, open(f"{OUT_DIR}/hpsc_endoderm_target_gene_universe.json", "w"))
    json.dump(sorted(detected.tolist()), open(f"{OUT_DIR}/hpsc_endoderm_detected_genes_5pct.json", "w"))
    print(f"wrote {OUT_DIR}/hpsc_endoderm_target_gene_universe.json")


if __name__ == "__main__":
    main()
