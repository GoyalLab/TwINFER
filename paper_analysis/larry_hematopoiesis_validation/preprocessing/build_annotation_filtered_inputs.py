"""Build twin-input CSVs where a cell's clone_id is combined with its cell-type annotation
(`{clone_id}__{Cell type annotation}`), so that assign_twin_id / _build_cross_time_twins in
twinfer.inference.correlation_functions -- which group purely on the `clone_id` column -- can no
longer pair two cells from the same LARRY clone unless they also carry the same annotation. This
implements TODO 2.1 ("remove twins that are not similar -- also are cells of the same cell type")
without touching the twinfer package: it only relabels clone_id before the existing pipelines
(run_analytic_twinfer.py, run_infer_correlation_high.py) read it.

Annotation source: finalized_data/LARRY_data/annotated/filtered_annotated_k15.h5ad, built by
larry_raw_annotate.py (published Weinreb et al. labels where available, KNN-transferred
elsewhere -- see obs['annotation_source']; overall CV accuracy in
annotation_transfer_filtered_k15.json). That file's AnnData was built from the SAME
obs_metadata.csv used by build_twinfer_inputs_per_geneset.py, over the SAME row order (no cells
dropped, only genes) -- confirmed positionally, so cell_id is joined by row position, not a
barcode transform.

Writes one relabeled CSV per (input dir, gene set) into a sibling `{input_dir}_annotfilter/`
directory -- e.g. resources/twinfer_input_cp10k/variability_high.csv ->
resources/twinfer_input_cp10k_annotfilter/variability_high.csv -- so nothing already built is
overwritten and existing scripts pick the new twin definition up via INPUT_DIR=..._annotfilter.

Run: python3 build_annotation_filtered_inputs.py
Override which input dir(s) / gene-sets file (e.g. for yscher's panels, normalized-only):
  INPUT_DIRS=twinfer_input_yscher_cp10k GENE_SETS_JSON=resources/gene_sets_yscher.json \
      python3 build_annotation_filtered_inputs.py
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
import os

import anndata as ad
import pandas as pd

HERE = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation'
from twinfer.utils.paths import get_data_root as _res_gdr  # [2026-10-01 LARRY resources/ moved out of the code tree to analysis_data/paper_analysis/larry_hematopoiesis_validation/resources (user)]
RES_HERE = f"{_res_gdr()}/paper_analysis/larry_hematopoiesis_validation"
SOURCE_OBS = f'{TWINFER_PROJECT_ROOT}/finalized_data/LARRY_data/filtered/obs_metadata.csv'
ANNOTATED = f'{TWINFER_PROJECT_ROOT}/finalized_data/LARRY_data/annotated/filtered_annotated_k15.h5ad'
INPUT_DIRS = os.environ.get("INPUT_DIRS", "twinfer_input_cp10k,twinfer_input").split(",")
GENE_SETS_JSON = os.environ.get("GENE_SETS_JSON", "resources/gene_sets.json")
GENE_SETS = list(json.load(open(os.path.join(HERE, GENE_SETS_JSON))).keys())


def log(m):
    print(m, flush=True)


obs = pd.read_csv(SOURCE_OBS, index_col=0)
A = ad.read_h5ad(ANNOTATED, backed="r")
assert A.n_obs == len(obs), f"row count mismatch: annotated={A.n_obs} obs_metadata={len(obs)}"
ann = pd.Series(A.obs["Cell type annotation"].astype(str).to_numpy(), index=obs.index, name="annotation")
src = pd.Series(A.obs["annotation_source"].astype(str).to_numpy(), index=obs.index, name="annotation_source")
log(f"loaded {len(ann):,} cell annotations from {os.path.basename(ANNOTATED)} "
    f"({(src == 'published').sum():,} published, {(src == 'predicted').sum():,} predicted)")
log("label counts:\n" + ann.value_counts().to_string())

for input_dir in INPUT_DIRS:
    # [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] src_dir = os.path.join(HERE, "resources", input_dir)
    src_dir = os.path.join(RES_HERE, "resources", input_dir)
    # [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] out_dir = os.path.join(HERE, "resources", f"{input_dir}_annotfilter")
    out_dir = os.path.join(RES_HERE, "resources", f"{input_dir}_annotfilter")
    os.makedirs(out_dir, exist_ok=True)
    for gs in GENE_SETS:
        df = pd.read_csv(os.path.join(src_dir, f"{gs}.csv"))
        missing = set(df["cell_id"]) - set(ann.index)
        if missing:
            raise ValueError(f"[{input_dir}/{gs}] {len(missing)} cell_id(s) not in annotation map, "
                              f"e.g. {sorted(missing)[:3]}")
        cell_ann = ann.loc[df["cell_id"]].to_numpy()
        n_clones_before = df["clone_id"].nunique()
        df["clone_id"] = df["clone_id"].astype(str) + "__" + cell_ann
        n_clones_after = df["clone_id"].nunique()
        out_path = os.path.join(out_dir, f"{gs}.csv")
        df.to_csv(out_path, index=False)
        log(f"[{input_dir}/{gs}] {len(df):,} rows, clones {n_clones_before:,} -> {n_clones_after:,} "
            f"(split by annotation) -> wrote {out_path}")

log("\ndone")
