#!/usr/bin/env python3
"""Candidate-pair generation for hPSC_20260927, replicating yscher's real code EXACTLY
(/gpfs/projects/b1255/yscher/Transcriptomic Distance/exports/_probe_tmp/fanout_delta/
chip_candidates.py, confirmed via subagent read -- her own "hpsc" AND "endo" dataset keys
both map to CLASS="Pluripotent stem cell", genome=hg38), not the substring-keyword
approximation in v1 (build_hpsc_endoderm_chipatlas_candidates.py, THR=250, kept for
reference/comparison but superseded by this script):

  THR = 100 (not 250)
  keep_srx = {SRX : experimentList.tab columns (genome==hg38) & (antigen-class=="TFs and
              others") & (cell-type-class=="Pluripotent stem cell")}
  per TF: fetch the +/-5kb-of-TSS per-target-gene TSV (chip-atlas.dbcls.jp/data/hg38/target/
          {tf}.5.tsv -- same endpoint her pre-built resources/reference/chipatlas5/ files
          come from, per the subagent's read of her chip_candidates.py), keep only columns
          whose SRX id is in keep_srx (exact ID match, not header-text keyword matching),
          take the max score across those columns per target gene, candidate edge if
          max score >= THR.

DET=0.05 (detection floor, applied earlier in the pipeline already via the target gene
universe -- see build_gene_universe.py) and TF-catalogue / MALAT1,NEAT1,XIST,FIRRE / mito
exclusion (also already applied upstream) match her convention too.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import argparse
import io
import json
import os
import time

import pandas as pd
import requests

BASE = f'{TWINFER_PROJECT_ROOT}'
OUT_DIR = f"{BASE}/analysis_data/hPSC_20260927/data"
CACHE_DIR = f"{OUT_DIR}/chipatlas_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

GENOME = "hg38"
DIST = "5"
THR = 100.0  # yscher's chip_candidates.py: THR=float(os.environ.get("CHIP_THR","100"))
URL = "https://chip-atlas.dbcls.jp/data/{genome}/target/{tf}.{dist}.tsv"
LAMBERT_URL = "https://humantfs.ccbr.utoronto.ca/download/v_1.01/DatabaseExtract_v_1.01.csv"
KEEP_SRX_PATH = f"{CACHE_DIR}/keep_srx_pluripotent.txt"


def fetch_lambert_tfs():
    r = requests.get(LAMBERT_URL, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    tfs = df.loc[df["Is TF?"] == "Yes", "HGNC symbol"].dropna().astype(str)
    return sorted(set(tfs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets-json", required=True)
    ap.add_argument("--limit-tfs", type=int, default=None)
    args = ap.parse_args()

    targets = set(json.load(open(args.targets_json)))
    keep_srx = set(open(KEEP_SRX_PATH).read().split())
    print(f"[hPSC_20260927 v2] {len(targets)} candidate target genes, "
          f"{len(keep_srx)} keep_srx (hg38, TFs and others, Pluripotent stem cell), THR={THR}", flush=True)

    tfs = fetch_lambert_tfs()
    if args.limit_tfs:
        tfs = tfs[: args.limit_tfs]
    print(f"[hPSC_20260927 v2] {len(tfs)} Lambert-catalogue TFs to query", flush=True)

    cache_path = f"{CACHE_DIR}/chipatlas_v2_pluripotent_raw.json"
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    edges = []
    rows = []
    for i, tf in enumerate(tfs):
        if tf not in cache:
            url = URL.format(genome=GENOME, tf=tf, dist=DIST)
            try:
                r = requests.get(url, timeout=90)
                r.raise_for_status()
                df = pd.read_csv(io.StringIO(r.text), sep="\t")
            except Exception as e:
                cache[tf] = {"status": "failed", "error": str(e)}
                print(f"[{i+1}/{len(tfs)}] {tf}: FAILED ({e})", flush=True)
                json.dump(cache, open(cache_path, "w"))
                time.sleep(0.3)
                continue
            if df.empty or df.shape[1] < 2:
                cache[tf] = {"status": "empty"}
                json.dump(cache, open(cache_path, "w"))
                time.sleep(0.3)
                continue
            gene_col = df.columns[0]
            # column header format: "SRX####|CellType" -- match the SRX id (before "|") exactly
            col_srx = [str(c).split("|")[0] for c in df.columns[1:]]
            cols = [c for c, srx in zip(df.columns[1:], col_srx) if srx in keep_srx]
            if not cols:
                cache[tf] = {"status": "no_matching_experiments", "n_cols_total": int(df.shape[1] - 1)}
                print(f"[{i+1}/{len(tfs)}] {tf}: no Pluripotent-stem-cell-class experiments "
                      f"(of {df.shape[1]-1} total cols)", flush=True)
                json.dump(cache, open(cache_path, "w"))
                time.sleep(0.3)
                continue
            scores = df[cols].apply(pd.to_numeric, errors="coerce").max(axis=1)
            bound_genes = sorted(set(df.loc[scores >= THR, gene_col].astype(str)))
            cache[tf] = {
                "status": "ok", "n_experiments": len(cols), "experiment_cols": list(cols),
                "n_bound_genes_total": len(bound_genes), "bound_genes": bound_genes,
            }
            print(f"[{i+1}/{len(tfs)}] {tf}: {len(cols)} matching experiment(s), "
                  f"{len(bound_genes)} bound genes (score>={THR:.0f})", flush=True)
            json.dump(cache, open(cache_path, "w"))
            time.sleep(0.35)

        entry = cache[tf]
        bound_in_targets = sorted(set(entry.get("bound_genes", [])) & targets)
        for t in bound_in_targets:
            edges.append((tf, t))
        rows.append(dict(
            TF=tf, status=entry.get("status", "?"),
            n_experiments=entry.get("n_experiments", 0),
            n_bound_genes_total=entry.get("n_bound_genes_total", 0),
            n_bound_in_targets=len(bound_in_targets),
        ))

    summary = pd.DataFrame(rows)
    summary.to_csv(f"{OUT_DIR}/hpsc_endoderm_tf_chipatlas_coverage_v2.csv", index=False)

    edges_df = pd.DataFrame(edges, columns=["TF", "target"])
    edges_df.to_csv(f"{OUT_DIR}/hpsc_endoderm_chipatlas_candidate_pairs_v2.csv", index=False)

    print("\n=== SUMMARY (v2, THR=100, exact SRX-ID cell-class match) ===", flush=True)
    print(f"TFs checked: {len(summary)}", flush=True)
    ok = summary[summary.status == "ok"]
    print(f"TFs with >=1 Pluripotent-stem-cell-class ChIP-Atlas experiment: {len(ok)}/{len(summary)}", flush=True)
    print(f"Total candidate (TF, target) edges: {len(edges_df)}", flush=True)
    print(f"Distinct TFs with >=1 candidate edge: {edges_df.TF.nunique() if len(edges_df) else 0}", flush=True)
    print(f"Distinct target genes with >=1 candidate edge: {edges_df.target.nunique() if len(edges_df) else 0}", flush=True)
    print(f"\nwrote {OUT_DIR}/hpsc_endoderm_chipatlas_candidate_pairs_v2.csv", flush=True)


if __name__ == "__main__":
    main()
