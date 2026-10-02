#!/usr/bin/env python3
"""TwinScore_supplement, FINAL gated + bootstrapped version, for HS054_Sot48h.

Reuses apply_twinscore_supplement_fatemap_gated_bootstrap.py's run_gene_set() verbatim
(dataset-agnostic once given a loader) -- only load_raw_pooled's file-path convention
differs here: HS054_Sot48h's finalized/analysis-data folders don't follow the
{label}={dataset.lower()} naming FM06/FM08/Watermelon use (qc_filtered mtx is
hs054_sot48h_qc_counts.mtx under finalized_data/HS054_Sot48h_data/, but gene
sets/z_dagger/outputs live under analysis_data/hs054/ -- see run_infer_hs054.py). This
script monkeypatches twinscore_supp_helpers.load_raw with the correct paths before
calling the shared run_gene_set(), single-timepoint (t1=t2, no --absplit option --
HS054 has no second sample/replicate to split against).
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.io as sio

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_analysis.fatemap_pipeline import twinscore_supp_helpers as supp
from paper_analysis.fatemap_pipeline import apply_twinscore_supplement_fatemap_gated_bootstrap as gb

log = supp.log
MAX_CLONE_SIZE = supp.MAX_CLONE_SIZE

QC_DIR = f'{TWINFER_PROJECT_ROOT}/finalized_data/HS054_Sot48h_data/qc_filtered'
QC_MTX = f"{QC_DIR}/hs054_sot48h_qc_counts.mtx"
GENE_SETS_JSON = f'{TWINFER_PROJECT_ROOT}/analysis_data/hs054/data/gene_sets_hs054.json'
OUT_DIR = f'{TWINFER_PROJECT_ROOT}/analysis_data/hs054/data/twinscore_supp_gated_bootstrap'
Z_DAGGER_PATH_TMPL = f'{TWINFER_PROJECT_ROOT}/analysis_data/hs054/data/z_dagger_hs054_{{gs}}.json'


def load_raw_hs054(dataset, gene_set_name):
    gene_sets = json.load(open(GENE_SETS_JSON))
    genes = sorted(gene_sets[gene_set_name])

    X = sio.mmread(QC_MTX).tocsr()
    genes_full = np.array(open(f"{QC_DIR}/genes.txt").read().split())
    obs = pd.read_csv(f"{QC_DIR}/obs_metadata.csv", index_col=0)
    assert X.shape[0] == len(obs)
    gidx = {g: i for i, g in enumerate(genes_full)}
    col = [gidx[g] for g in genes]
    cell_total = np.asarray(X.sum(axis=1)).ravel().astype(float)
    mat = np.log1p(X[:, col].toarray().astype(float) / np.where(cell_total == 0, 1, cell_total)[:, None] * 1e4)

    df = pd.DataFrame(mat, columns=[f"{g}_mRNA" for g in genes])
    df.insert(0, "time_step", 0)
    df.insert(0, "cell_id", obs.index.to_numpy())
    clone = obs["fatemap_clone_singletcode"].astype(str)
    df.insert(0, "clone_id", clone.to_numpy())

    has_clone = clone.str.len() > 0
    df = df[has_clone.to_numpy()].reset_index(drop=True)
    counts = df["clone_id"].value_counts()
    keep = counts[(counts >= 2) & (counts <= MAX_CLONE_SIZE)].index
    dropped = int((~df["clone_id"].isin(keep)).sum())
    df = df[df["clone_id"].isin(keep)].reset_index(drop=True)
    log(f"[HS054_Sot48h/{gene_set_name}] {len(df):,} cells, {df['clone_id'].nunique():,} clones "
        f"(dropped {dropped:,} cells: no clone, singleton, or >MAX_CLONE_SIZE={MAX_CLONE_SIZE})")
    return df, genes, QC_DIR, genes_full, gidx


def main():
    import argparse
    import datetime
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene-set", default="correlation_high")
    ap.add_argument("--n-shuffles", type=int, default=500)
    ap.add_argument("--n-cores", type=int, default=8)
    ap.add_argument("--no-wz", action="store_true")
    args = ap.parse_args()

    # monkeypatch: run_gene_set() calls load_raw_pooled() -> supp.load_raw(); swap in the
    # HS054-correct path logic without touching the shared FM06/FM08/Watermelon helper.
    gb.supp.load_raw = load_raw_hs054

    ct = pd.read_csv(supp.COLLECTRI_PATH, sep="\t")
    ct = ct[ct.source_genesymbol != ct.target_genesymbol]
    CE = set(zip(ct.source_genesymbol, ct.target_genesymbol))

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "README.md"), "w") as fh:
        fh.write(gb.GATED_README)

    # run_gene_set's z_dagger loader expects z_dagger_{label}_{gs}{suffix}.json with
    # label=dataset.lower() -- "HS054" -> "hs054", suffix="" (no _allpairs, matches
    # run_infer_hs054.py's plain filename), so pass dataset="HS054" for that path to resolve.
    pair_df, gene_df, diag = gb.run_gene_set(
        "HS054", args.gene_set, CE, absplit=False, n_shuffles=args.n_shuffles,
        n_cores=args.n_cores, compute_wz=not args.no_wz,
    )
    pair_out = os.path.join(OUT_DIR, f"twinscore_supplement_hs054_{args.gene_set}_gated_bootstrap_pair_terms.csv")
    gene_out = os.path.join(OUT_DIR, f"twinscore_supplement_hs054_{args.gene_set}_gated_bootstrap_gene_terms.csv")
    pair_df.to_csv(pair_out, index=False)
    gene_df.to_csv(gene_out, index=False)
    log(f"wrote {pair_out}")
    log(f"wrote {gene_out}")
    log(f"diagnostics: {diag}")
    params = dict(dataset="HS054_Sot48h", gene_set=args.gene_set, absplit=False, n_shuffles=args.n_shuffles,
                  compute_wz=not args.no_wz, phi_mode="gated_bootstrap", gate_mult=2.0, phi_clip=None,
                  script=os.path.basename(__file__),
                  z_dagger_input=Z_DAGGER_PATH_TMPL.format(gs=args.gene_set),
                  written=datetime.datetime.now().isoformat(timespec="seconds"), diagnostics=diag)
    with open(os.path.join(OUT_DIR, f"run_params_hs054_{args.gene_set}_gated_bootstrap.json"), "w") as fh:
        json.dump(params, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
