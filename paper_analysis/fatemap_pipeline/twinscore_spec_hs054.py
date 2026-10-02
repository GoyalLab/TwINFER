#!/usr/bin/env python3
"""HS054_Sot48h: the most recent TwinScore-recent spec (2026-09-25, yscher's fanout_delta/spec_v3.py,
as ported to this repo in twinscore_spec_fm06.py), merged-sample mode only (HS054 has a single
sample/timepoint -- no A/B split to run the "ab" mode against). Reuses
twinscore_spec_fm06.py's Sample/steps/panel_expression/clone_filter verbatim (all dataset-agnostic);
only load_raw's file-path convention differs (see apply_twinscore_supplement_hs054_gated_bootstrap.py
for the same HS054-path adaptation). No covariate regression (matches
melanoma_gt_analysis/score_all_panels_newgt.py's simpler usage: panel_expression(..., uni=None)),
since HS054 has no yscher gene_flags.csv/gene_lists.py resources of its own.

Score = R (steps I-IV; see twinscore_spec_fm06.py's module docstring for the full formula),
reported gated (stage1 & called) and ungated, for the analytic/clone/bootstrap null variants.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.metrics import auc, precision_recall_curve

PIPE = os.path.dirname(os.path.abspath(__file__))
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, PIPE)
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/package")
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/larry_hematopoiesis_validation")
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/yscher/Transcriptomic Distance/exports/_probe_tmp")
from paper_analysis.fatemap_pipeline import twinscore_supp_helpers as supp
from paper_analysis.fatemap_pipeline.twinscore_spec_fm06 import Sample, clone_filter, panel_expression, steps  # noqa: E402

log = supp.log
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] COLLECTRI = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation/resources/collectri_human.tsv'
COLLECTRI = f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/larry_hematopoiesis_validation/resources/collectri_human.tsv'
QC_DIR = f'{TWINFER_PROJECT_ROOT}/finalized_data/HS054_Sot48h_data/qc_filtered'
QC_MTX = f"{QC_DIR}/hs054_sot48h_qc_counts.mtx"
GENE_SETS_JSON = f'{TWINFER_PROJECT_ROOT}/analysis_data/hs054/data/gene_sets_hs054.json'
NET_DIR = f'{TWINFER_PROJECT_ROOT}/analysis_data/hs054/data/networks'
OUT_DIR = f'{TWINFER_PROJECT_ROOT}/analysis_data/hs054/data/twinscore_spec'


def load_raw_hs054(gs):
    genes = sorted(json.load(open(GENE_SETS_JSON))[gs])
    X = sio.mmread(QC_MTX).tocsr()
    genes_full = np.array(open(f"{QC_DIR}/genes.txt").read().split())
    obs = pd.read_csv(f"{QC_DIR}/obs_metadata.csv", index_col=0)
    gidx = {g: i for i, g in enumerate(genes_full)}
    tot = np.asarray(X.sum(axis=1)).ravel().astype(float)
    bc = obs["fatemap_clone_singletcode"].astype(str).to_numpy()
    keep = np.array([len(b) > 0 for b in bc])
    df = pd.DataFrame(dict(cell_id=obs.index.to_numpy(), barcode=bc))[keep].reset_index(drop=True)
    return df, X[keep], tot[keep], genes, np.array([gidx[g] for g in genes]), genes_full


def full_report(sc, y):
    sc = np.asarray(sc, float)
    sc = np.nan_to_num(sc, nan=np.nanmin(sc) - 1.0 if np.isfinite(sc).any() else 0.0)
    prec, rec, _ = precision_recall_curve(y, sc)
    a = auc(rec, prec)
    rand = float(y.mean())
    k = int(y.sum())
    tp = int((y[np.argsort(-sc, kind="stable")[:k]] == 1).sum())
    return dict(auprc_x=a / rand if rand > 0 else float("nan"), topk_precision=tp / k if k else float("nan"), k=k, tp=tp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gene_set")
    ap.add_argument("--n-null", type=int, default=100)
    ap.add_argument("--nulls", default="analytic,clone,bootstrap")
    ap.add_argument("--out-dir", default=OUT_DIR)
    a = ap.parse_args()
    gs = a.gene_set
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(supp.SEED)

    df, Xc, tot, genes, pidx, genes_full = load_raw_hs054(gs)
    G = len(genes)
    ct = pd.read_csv(COLLECTRI, sep="\t")
    ct = ct[ct.source_genesymbol != ct.target_genesymbol]
    edges, sources = set(zip(ct.source_genesymbol, ct.target_genesymbol)), set(ct.source_genesymbol)

    clone = df.barcode.to_numpy()
    m = clone_filter(pd.DataFrame(dict(c=clone)), "c")
    rows_ = np.flatnonzero(m)
    V = panel_expression(Xc[rows_], tot[rows_], pidx, None, genes_full)  # uni=None: no covariate regression
    smp = Sample(V, clone[m], df.cell_id.to_numpy()[m], rng)

    kinds = a.nulls.split(",")
    boot = None
    if "bootstrap" in kinds:
        boot = smp.bootstrap_sds(a.n_null)
        log(f"    bootstrap sds from {a.n_null} draws")

    rows = [(x, y) for x in range(G) for y in range(G) if x != y]
    T = pd.DataFrame(dict(gene_1=[genes[x] for x, _ in rows], gene_2=[genes[y] for _, y in rows]))
    T["collectri_edge"] = [int((a_, b_) in edges) for a_, b_ in zip(T.gene_1, T.gene_2)]
    xi, yi = np.array([x for x, _ in rows]), np.array([y for _, y in rows])

    y = T["collectri_edge"].to_numpy()
    results = {}
    for kind in kinds:
        sd = smp.null_sds(kind, boot)
        r = steps(smp, sd)
        R_, called, stage1 = r["R"], r["called"], r["stage1"]
        ok = stage1 & called
        T[f"TwinScore-recent [{kind}]: R (called)"] = np.where(ok, R_, -np.inf)[xi, yi]
        T[f"TwinScore-recent [{kind}]: R ungated"] = R_[xi, yi]
        for k_ in ("z_rho", "h", "lam", "zreg", "zstar"):
            T[f"{kind}_{k_}"] = r[k_][xi, yi]
        log(f"    [{kind}] g {r['g']:.3f}; stage I {int(stage1[~np.eye(G, dtype=bool)].sum())}, "
            f"called {int(called[~np.eye(G, dtype=bool)].sum())}, both {int(ok[~np.eye(G, dtype=bool)].sum())} of {G * (G - 1)}")
        if y.sum() >= 3:
            m_gated = full_report(T[f"TwinScore-recent [{kind}]: R (called)"].to_numpy(), y)
            m_ungated = full_report(T[f"TwinScore-recent [{kind}]: R ungated"].to_numpy(), y)
            results[kind] = dict(gated_auprc_x=m_gated["auprc_x"], gated_topk=m_gated["topk_precision"],
                                  ungated_auprc_x=m_ungated["auprc_x"], ungated_topk=m_ungated["topk_precision"])
            log(f"    [{kind}] vs CollecTRI (n_true={int(y.sum())}/{len(y)}): "
                f"gated auprc_x={m_gated['auprc_x']:.3f}x  ungated auprc_x={m_ungated['auprc_x']:.3f}x")

    out_csv = f"{a.out_dir}/twinscore_spec_merged_{'_'.join(kinds)}_hs054_{gs}_pair_terms.csv"
    T.to_csv(out_csv, index=False)
    log(f"wrote {out_csv}")
    if results:
        summary_csv = f"{a.out_dir}/twinscore_spec_merged_hs054_{gs}_summary.csv"
        pd.DataFrame(results).T.to_csv(summary_csv)
        log(f"wrote {summary_csv}")


if __name__ == "__main__":
    main()
