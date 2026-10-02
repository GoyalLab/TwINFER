"""Build one TwINFER canonical-input CSV per gene set in resources/gene_sets.json, following the
same approach convert_10x_to_twinfer_input.ipynb already uses for this dataset (its cell efddd986:
PRE-annotation matrix -- full 25,289 genes, since the annotated matrix is missing several TFs --
larry_clone_singletcode as clone_id, day parsed from library as time_step) -- just looped over all
9 gene sets instead of the notebook's single 12-gene TF panel.

Canonical shape per file: clone_id, cell_id, time_step, {gene}_mRNA, {gene}_mRNA, ...
(is_simulation_data=False, so no `replicate` column -- matches
twinfer/inference/infer.py's _validate_canonical_input contract, reproduced here verbatim.)

By default keeps all cells (days 2/4/6). Set DAYS to restrict to a subset of timepoints, e.g.
DAYS=2,4 to drop day 6 -- written to its own day-suffixed subdirectory so the all-cells CSVs
(if already built) aren't overwritten.

Run directly: python3 build_twinfer_inputs_per_geneset.py
Restrict to certain days: DAYS=2,4 python3 build_twinfer_inputs_per_geneset.py
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
import os

import numpy as np
import pandas as pd
import scipy.io as sio

HERE = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation'
from twinfer.utils.paths import get_data_root as _res_gdr  # [2026-10-01 LARRY resources/ moved out of the code tree to analysis_data/paper_analysis/larry_hematopoiesis_validation/resources (user)]
RES_HERE = f"{_res_gdr()}/paper_analysis/larry_hematopoiesis_validation"
SOURCE = f'{TWINFER_PROJECT_ROOT}/finalized_data/LARRY_data/filtered'
# GENE_SETS_JSON overrides which gene-sets file to build from (default: resources/gene_sets.json).
# e.g. GENE_SETS_JSON=resources/gene_sets_yscher.json for yscher's own panel definitions.
GENE_SETS_PATH = os.path.join(HERE, os.environ.get("GENE_SETS_JSON", "resources/gene_sets.json"))

DAYS = [int(d) for d in os.environ["DAYS"].split(",")] if "DAYS" in os.environ else None
# NORMALIZE=cp10k  ->  gene columns are log1p(count / cell_total * 1e4) instead of raw counts.
# Spearman (and hence every TwINFER correlation) is NOT invariant to per-cell library-size
# normalization -- it removes the sequencing-depth confound that otherwise swamps the weak
# co-expression signal (raw vs cp10k: P@R 0.6x -> 2.5x on the corr benchmark). Written to a
# _cp10k-suffixed dir so the raw-count CSVs are not overwritten.
NORMALIZE = os.environ.get("NORMALIZE", "").lower()
# OUT_TAG distinguishes an alternate gene-sets source (e.g. "yscher") in the output dir name so it
# never collides with the default resources/twinfer_input{...}/ built from gene_sets.json.
OUT_TAG = os.environ.get("OUT_TAG", "")
_suffix = (f"_{OUT_TAG}" if OUT_TAG else "") + ("" if DAYS is None else f"_day{'+'.join(map(str, DAYS))}") + (f"_{NORMALIZE}" if NORMALIZE else "")
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] OUT_DIR = os.path.join(HERE, "resources", f"twinfer_input{_suffix}")
OUT_DIR = os.path.join(RES_HERE, "resources", f"twinfer_input{_suffix}")
os.makedirs(OUT_DIR, exist_ok=True)


def log(m):
    print(m, flush=True)


def validate_canonical_input(df, is_simulation_data=False):
    """Mirrors _validate_canonical_input in twinfer/inference/infer.py exactly (also reproduced
    in convert_10x_to_twinfer_input.ipynb's cell e7163937)."""
    required = ["clone_id", "cell_id", "time_step"]
    if is_simulation_data:
        required.append("replicate")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing required column(s): {missing}")
    gene_cols = [c for c in df.columns if c.endswith("_mRNA")]
    if not gene_cols:
        raise ValueError("must contain at least one '*_mRNA' column")
    forbidden = [c for c in ("twin_id", "pair_id", "n_twins", "n_pairs") if c in df.columns]
    if forbidden:
        raise ValueError(f"remove twin-expansion column(s) before saving: {forbidden}")
    check_cols = required + gene_cols
    if df[check_cols].isna().any().any():
        bad = [c for c in check_cols if df[c].isna().any()]
        raise ValueError(f"missing values in: {bad}")
    duplicated = df.duplicated(subset=["cell_id", "time_step"], keep=False)
    if duplicated.any():
        examples = df.loc[duplicated, ["cell_id", "time_step"]].drop_duplicates().head(5).to_dict("records")
        raise ValueError(f"duplicate (cell_id, time_step) rows, e.g.: {examples}")
    log(f"    OK: {len(df):,} rows, {len(gene_cols)} gene column(s), no duplicates, no missing values")


# ---------------------------------------------------------------- load once, reuse for all 9 sets
X = sio.mmread(os.path.join(SOURCE, "larry_qc_counts.mtx")).tocsr()
genes = pd.Index(open(os.path.join(SOURCE, "genes.txt")).read().split())
obs = pd.read_csv(os.path.join(SOURCE, "obs_metadata.csv"), index_col=0)
assert X.shape == (len(obs), len(genes))
log(f"loaded {X.shape[0]:,} cells x {X.shape[1]:,} genes")

day = obs["library"].astype(str).str.extract(r"_d(\d+)_", expand=False).astype(int)
gene_index = {g: i for i, g in enumerate(genes)}

if DAYS is not None:
    keep = day.isin(DAYS).to_numpy()
    X = X[keep]
    obs = obs[keep].copy()
    day = day[keep]
    log(f"restricted to day(s) {DAYS}: {X.shape[0]:,} / {len(keep):,} cells kept")

cell_total = np.asarray(X.sum(axis=1)).ravel().astype(float)  # per-cell total over the full transcriptome
if NORMALIZE == "cp10k":
    log("NORMALIZE=cp10k: gene columns = log1p(count / cell_total * 1e4)")
elif NORMALIZE:
    raise ValueError(f"unknown NORMALIZE={NORMALIZE!r} (supported: cp10k)")

gene_sets = json.load(open(GENE_SETS_PATH))
log(f"{len(gene_sets)} gene sets: " + ", ".join(f"{k}({len(v)})" for k, v in gene_sets.items()))

for name, gene_list in gene_sets.items():
    missing = [g for g in gene_list if g not in gene_index]
    if missing:
        raise ValueError(f"[{name}] gene(s) missing from genes.txt: {missing}")
    col_idx = [gene_index[g] for g in gene_list]

    mat = X[:, col_idx].toarray().astype(float)
    if NORMALIZE == "cp10k":
        mat = np.log1p(mat / cell_total[:, None] * 1e4)
    df = pd.DataFrame(mat, columns=[f"{g}_mRNA" for g in gene_list], index=obs.index)
    df.insert(0, "time_step", day.to_numpy())
    df.insert(0, "cell_id", obs.index)
    df.insert(0, "clone_id", obs["larry_clone_singletcode"].to_numpy())
    df = df.reset_index(drop=True)

    log(f"[{name}] {len(gene_list)} genes")
    validate_canonical_input(df, is_simulation_data=False)

    out_path = os.path.join(OUT_DIR, f"{name}.csv")
    df.to_csv(out_path, index=False)
    log(f"    wrote {out_path}")

log(f"\ndone -- {len(gene_sets)} CSVs written to {OUT_DIR}")
