#!/usr/bin/env python3
"""Build the PDF's "Candidate pairs" set for hPSC_20260927 (endo_T0/endo_T1, hESC ->
definitive-endoderm lineage-tracing): x (a Lambert-catalogue TF) bound within +/-5kb of
the TSS of y, per ChIP-Atlas, max experiment score >= 250, restricted to experiments in a
pluripotent-stem-cell / endoderm-differentiation cell-class (this dataset's own cell
class -- hESC, iPSC, and hESC-derived mesendoderm/endoderm differentiation timecourses --
in place of the PDF's melanoma-Epidermis / LARRY-Blood classes). TFs are never removed
even with zero candidate targets (per the PDF: "TFs are never removed").

Regulators: the Lambert et al. 2018 "Is TF? == Yes" catalogue (humantfs.ccbr.utoronto.ca),
restricted to genes actually detected in this dataset's QC-filtered matrix.
Targets checked: every non-TF-excluded, >=5%-detected gene in the QC matrix (see
pick_hpsc_endoderm_gene_universe.py for the detection + GO/Tirosh/histone exclusion
filter) -- passed in via --targets-json, so this script only needs to run once per
detection-filter revision, not per gene-set.
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
CACHE_DIR = f"{BASE}/analysis_data/hPSC_20260927/data/chipatlas_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

GENOME = "hg38"
DIST = "5"
CHIP_SCORE_MIN = 250
URL = "https://chip-atlas.dbcls.jp/data/{genome}/target/{tf}.{dist}.tsv"
LAMBERT_URL = "https://humantfs.ccbr.utoronto.ca/download/v_1.01/DatabaseExtract_v_1.01.csv"

# hESC/iPSC pluripotency + hESC-derived meso/mesendoderm/endoderm differentiation timecourse
# cell-class keywords, from inspecting real ChIP-Atlas column headers for POU5F1/SOX17
# (e.g. "hESC_H1", "hESC_H9", "hESC_HUES64", "hESC_WIBR2", "iPS_cells", "ES_cells",
# "hESC_derived_mesendodermal_cells", "hESC_derived_endodermal_cells"). Deliberately excludes
# "hESC_derived_ectodermal_cells" (wrong germ layer for this dataset) and any hESC-derived
# neural/cardiac/hemogenic differentiation branch.
CELL_CLASS_KEYWORDS = [
    "hesc", "ips_cells", "es_cells", "ipsc",
    "mesendoderm", "mesendodermal",
    "endoderm", "endodermal",
    "definitive_endoderm", "foregut", "hepatoblast", "hepatocyte_progenitor",
]
EXCLUDE_KEYWORDS = ["ectoderm", "ectodermal", "neural", "cardiac", "hemogenic", "trophoblast"]


def fetch_lambert_tfs():
    r = requests.get(LAMBERT_URL, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    tfs = df.loc[df["Is TF?"] == "Yes", "HGNC symbol"].dropna().astype(str)
    return sorted(set(tfs))


def cell_class_match(header_text):
    h = header_text.lower()
    if any(k in h for k in EXCLUDE_KEYWORDS):
        return False
    return any(k in h for k in CELL_CLASS_KEYWORDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets-json", required=True, help="JSON list of target gene symbols to check")
    ap.add_argument("--limit-tfs", type=int, default=None, help="debug: only process first N TFs")
    args = ap.parse_args()

    targets = set(json.load(open(args.targets_json)))
    print(f"[hPSC_20260927] {len(targets)} candidate target genes loaded", flush=True)

    tfs = fetch_lambert_tfs()
    if args.limit_tfs:
        tfs = tfs[: args.limit_tfs]
    print(f"[hPSC_20260927] {len(tfs)} Lambert-catalogue TFs to query", flush=True)

    cache_path = f"{CACHE_DIR}/chipatlas_pluripotent_endoderm_raw.json"
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    edges = []  # (TF, target)
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
            cols = [c for c in df.columns[1:] if cell_class_match(str(c))]
            if not cols:
                cache[tf] = {"status": "no_matching_experiments", "n_cols_total": int(df.shape[1] - 1)}
                print(f"[{i+1}/{len(tfs)}] {tf}: no pluripotent/endoderm experiments "
                      f"(of {df.shape[1]-1} total cols)", flush=True)
                json.dump(cache, open(cache_path, "w"))
                time.sleep(0.3)
                continue
            scores = df[cols].apply(pd.to_numeric, errors="coerce").max(axis=1)
            bound_genes = sorted(set(df.loc[scores >= CHIP_SCORE_MIN, gene_col].astype(str)))
            cache[tf] = {
                "status": "ok", "n_experiments": len(cols), "experiment_cols": list(cols),
                "n_bound_genes_total": len(bound_genes), "bound_genes": bound_genes,
            }
            print(f"[{i+1}/{len(tfs)}] {tf}: {len(cols)} matching experiment(s), "
                  f"{len(bound_genes)} bound genes (score>={CHIP_SCORE_MIN})", flush=True)
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
    summary.to_csv(f"{OUT_DIR}/hpsc_endoderm_tf_chipatlas_coverage.csv", index=False)

    edges_df = pd.DataFrame(edges, columns=["TF", "target"])
    edges_df.to_csv(f"{OUT_DIR}/hpsc_endoderm_chipatlas_candidate_pairs.csv", index=False)

    print("\n=== SUMMARY ===", flush=True)
    print(f"TFs checked: {len(summary)}", flush=True)
    ok = summary[summary.status == "ok"]
    print(f"TFs with >=1 pluripotent/endoderm ChIP-Atlas experiment: {len(ok)}/{len(summary)}", flush=True)
    print(f"Total candidate (TF, target) edges: {len(edges_df)}", flush=True)
    print(f"Distinct TFs with >=1 candidate edge: {edges_df.TF.nunique() if len(edges_df) else 0}", flush=True)
    print(f"Distinct target genes with >=1 candidate edge: {edges_df.target.nunique() if len(edges_df) else 0}", flush=True)
    print(f"\nwrote {OUT_DIR}/hpsc_endoderm_chipatlas_candidate_pairs.csv", flush=True)
    print(f"wrote {OUT_DIR}/hpsc_endoderm_tf_chipatlas_coverage.csv", flush=True)


if __name__ == "__main__":
    main()
