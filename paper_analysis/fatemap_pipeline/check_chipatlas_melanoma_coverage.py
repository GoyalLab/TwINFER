#!/usr/bin/env python3
"""How many of the FM01/FM06 (WM989 melanoma) panel's TFs have ChIP-Atlas experiments in melanoma-cell-class
lines, and how many of our own panel TARGET genes those experiments actually bind near the TSS -- a coverage
check, not a full ground-truth build (c.f. build_fm06_melanoma_ground_truth.py, the template for that).

For each TF in the union of Watermelon gene-set TFs: fetch chip-atlas.dbcls.jp's per-TF target TSV (hg38, +/-5kb
of TSS), keep only columns whose experiment label matches a melanoma-cell-class keyword, take the max score across
those columns per gene, and call it bound at score >= CHIP_SCORE_MIN (250, matching the FM06 melanoma builder).
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import io
import json
import os
import sys
import time

import pandas as pd
import requests

BASE = f'{TWINFER_PROJECT_ROOT}'
OUT_DIR = f"{BASE}/analysis_data/melanoma_chipatlas"
os.makedirs(OUT_DIR, exist_ok=True)
GENOME = "hg38"
DIST = "5"
CHIP_SCORE_MIN = 250
URL = "https://chip-atlas.dbcls.jp/data/{genome}/target/{tf}.{dist}.tsv"
MELANOMA_KEYWORDS = ["melanoma", "melanocyte", "sk-mel", "skmel", "a375", "501mel", "501-mel", "colo829",
                      "mewo", "wm989", "wm-989", "wm115", "wm266", "malme", "wm239", "wm1341", "sk-mel-28"]

TFS = sorted(json.load(open(f"{BASE}/analysis_data/melanoma_panel_tfs_union_fm01.json")))
# our own panel TARGET genes (union across all 9 gene sets x 3 stages), to check coverage against
TARGETS = set()
gs = json.load(open(f"{BASE}/analysis_data/fm01/data/gene_sets_fm01.json"))
for genes in gs.values():
    TARGETS |= set(genes)

cache_path = f"{OUT_DIR}/chipatlas_melanoma_raw.json"
cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

rows = []
for i, tf in enumerate(TFS):
    if tf in cache:
        entry = cache[tf]
    else:
        url = URL.format(genome=GENOME, tf=tf, dist=DIST)
        try:
            r = requests.get(url, timeout=90)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text), sep="\t")
        except Exception as e:
            print(f"[{i+1}/{len(TFS)}] {tf}: FAILED ({e})", flush=True)
            cache[tf] = {"status": "failed", "error": str(e)}
            json.dump(cache, open(cache_path, "w"))
            time.sleep(0.3)
            continue
        if df.empty or df.shape[1] < 2:
            cache[tf] = {"status": "empty"}
            print(f"[{i+1}/{len(TFS)}] {tf}: empty response", flush=True)
            json.dump(cache, open(cache_path, "w"))
            time.sleep(0.3)
            continue
        gene_col = df.columns[0]
        header_text = [str(c).lower() for c in df.columns[1:]]
        mel_cols = [c for c, h in zip(df.columns[1:], header_text) if any(k in h for k in MELANOMA_KEYWORDS)]
        if not mel_cols:
            cache[tf] = {"status": "no_melanoma_experiments", "n_cols_total": int(df.shape[1] - 1)}
            print(f"[{i+1}/{len(TFS)}] {tf}: no melanoma-cell-class experiments (of {df.shape[1]-1} total cols)", flush=True)
            json.dump(cache, open(cache_path, "w"))
            time.sleep(0.3)
            continue
        scores = df[mel_cols].apply(pd.to_numeric, errors="coerce").max(axis=1)
        genes = sorted(set(df.loc[scores >= CHIP_SCORE_MIN, gene_col].astype(str)))
        cache[tf] = {"status": "ok", "n_melanoma_experiments": len(mel_cols), "experiment_cols": list(mel_cols),
                     "n_bound_genes": len(genes), "bound_genes": genes}
        print(f"[{i+1}/{len(TFS)}] {tf}: {len(mel_cols)} melanoma experiment(s), {len(genes)} genes bound "
              f"(score>={CHIP_SCORE_MIN})", flush=True)
        json.dump(cache, open(cache_path, "w"))
        time.sleep(0.4)
    rows.append(dict(
        TF=tf, status=cache[tf].get("status", "?"),
        n_melanoma_experiments=cache[tf].get("n_melanoma_experiments", 0),
        n_bound_genes_total=cache[tf].get("n_bound_genes", 0),
        n_bound_in_our_targets=len(set(cache[tf].get("bound_genes", [])) & TARGETS),
    ))

summary = pd.DataFrame(rows)
summary.to_csv(f"{OUT_DIR}/melanoma_tf_chipatlas_coverage.csv", index=False)
ok = summary[summary.status == "ok"]
print("\n=== SUMMARY ===", flush=True)
print(f"TFs checked: {len(summary)}", flush=True)
print(f"TFs with >=1 melanoma-cell-class ChIP-Atlas experiment: {len(ok)}/{len(summary)}", flush=True)
print(f"TFs with zero melanoma experiments: {(summary.status=='no_melanoma_experiments').sum()}", flush=True)
print(f"TFs that failed/empty: {(summary.status.isin(['failed','empty'])).sum()}", flush=True)
if len(ok):
    print(f"\nTop 20 TFs by # melanoma experiments:", flush=True)
    print(ok.sort_values("n_melanoma_experiments", ascending=False).head(20)
          [["TF", "n_melanoma_experiments", "n_bound_genes_total", "n_bound_in_our_targets"]].to_string(index=False), flush=True)
    print(f"\nmean panel-target coverage (of {len(TARGETS)} distinct target genes across all panels): "
          f"{ok.n_bound_in_our_targets.mean():.1f} bound genes per TF (median {ok.n_bound_in_our_targets.median():.0f})", flush=True)
print(f"\nwrote {OUT_DIR}/melanoma_tf_chipatlas_coverage.csv", flush=True)
print(f"raw cache: {cache_path}", flush=True)
