#!/usr/bin/env python3
"""ChIP-Atlas ground truth from pancreatic cancer cell lines, for HS054 -- mirrors
build_chipatlas_cutaneous_gt.py (melanoma) exactly, only LINES/OUT differ.

Cell lines (ChIP-Atlas 'Cell type' labels, hg38, track class 'TFs and others'): AsPC-1, BxPC-3, CFPAC-1,
MIA Paca-2, MPanc-96, PANC-1, Panc 10.05, Pancreatic ductal adenocarcinoma.
Writes  <OUT>/chipatlas_pancreatic_tf_network.tsv  and  <OUT>/chipatlas_pancreatic_tf_summary.tsv
usage: build_chipatlas_pancreatic_gt.py [--score-min 250]
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import argparse
import io
import subprocess
import time

import pandas as pd
import requests

OUT = f'{TWINFER_PROJECT_ROOT}/analysis_data/hs054/data/pancreatic_gt_benchmark'
LINES = ["AsPC-1", "BxPC-3", "CFPAC-1", "MIA Paca-2", "MPanc-96", "PANC-1", "Panc 10.05",
         "Pancreatic ductal adenocarcinoma"]
EXCLUDE = {"Epitope tags", "GFP"}
META = "https://chip-atlas.dbcls.jp/data/metadata/experimentList.tab"
TARGET = "https://chip-atlas.dbcls.jp/data/hg38/target/{tf}.5.tsv"


def main():
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-min", type=float, default=250)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    raw = subprocess.run(
        f"curl -s -m 900 '{META}' | awk -F'\\t' '$2==\"hg38\" && $3==\"TFs and others\" {{print $1\"\\t\"$4\"\\t\"$5\"\\t\"$6}}'",
        shell=True, capture_output=True, text=True).stdout
    meta = pd.read_csv(io.StringIO(raw), sep="\t", header=None, names=["srx", "tf", "cls", "cell"])
    meta = meta[meta.cell.isin(LINES) & ~meta.tf.isin(EXCLUDE)]
    print(f"{len(meta)} experiments, {meta.tf.nunique()} TF antigens in {len(LINES)} pancreatic lines", flush=True)
    meta.to_csv(f"{OUT}/chipatlas_pancreatic_experiments.tsv", sep="\t", index=False)

    edges, summ = [], []
    for tf, g in meta.groupby("tf"):
        srx = set(g.srx)
        try:
            r = requests.get(TARGET.format(tf=tf), timeout=300)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"  {tf}: failed ({e})", flush=True)
            summ.append(dict(TF=tf, n_exps=len(srx), n_used=0, n_targets=0, note="download failed"))
            continue
        df = pd.read_csv(io.StringIO(r.text), sep="\t")
        cols = [c for c in df.columns[1:] if c.split("|", 1)[0] in srx]
        if not cols:
            print(f"  {tf}: no matching columns in the target file", flush=True)
            summ.append(dict(TF=tf, n_exps=len(srx), n_used=0, n_targets=0, note="no matching columns"))
            continue
        M = df[cols].apply(pd.to_numeric, errors="coerce")
        mx, ns = M.max(axis=1), (M >= a.score_min).sum(axis=1)
        keep = mx >= a.score_min
        e = pd.DataFrame(dict(TF=tf, target=df.iloc[:, 0].astype(str)[keep], n_supporting=ns[keep].to_numpy(),
                               n_exps_used=len(cols), max_score=mx[keep].to_numpy()))
        edges.append(e)
        summ.append(dict(TF=tf, n_exps=len(srx), n_used=len(cols), n_targets=int(keep.sum()),
                          n_targets_ge2=int((ns >= 2).sum()), cell_lines=";".join(sorted(g.cell.unique())), note=""))
        print(f"  {tf}: {len(cols)} experiments ({';'.join(sorted(g.cell.unique()))}) -> "
              f"{int(keep.sum())} targets at score >= {a.score_min}", flush=True)
        time.sleep(0.3)
    E = pd.concat(edges, ignore_index=True)
    E = E[E.TF != E.target]
    E.to_csv(f"{OUT}/chipatlas_pancreatic_tf_network.tsv", sep="\t", index=False)
    pd.DataFrame(summ).to_csv(f"{OUT}/chipatlas_pancreatic_tf_summary.tsv", sep="\t", index=False)
    print(f"\n{E.TF.nunique()} TFs, {len(E):,} edges (self-loops removed) -> {OUT}")


if __name__ == "__main__":
    main()
