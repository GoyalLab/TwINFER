#!/usr/bin/env python3
"""Step 3 of the duplicate-cell UMI-overlap test: compare the two (or three) copies of each duplicate cell.

Input  (per library, written by umi_overlap_extract.sh): reads_<lib>.tsv.gz = cell barcode, UMI, first 40 bp of the read.
For every duplicate (cell barcode, clone) group the copies A (library L1) and B (library L2) are compared by
  shared_umi        number of distinct UMI sequences present in both copies
  shared_readkey    number of distinct (UMI, first 20 bp) pairs present in both copies = identical read starts
                    with identical UMI (the signature of the same molecule sequenced in both libraries)
Null: A against control cells of L2 (not the duplicate cells, barcode different from B) whose number of distinct UMIs is
within +-25% of B's (nearest 10 if none). usage: umi_overlap_compare.py [dir]  (default the full-run dir)
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"
OUT = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/analysis_data/larry_raw_sra/umi_overlap"
M_NULL = 30
rng = np.random.default_rng(0)
pairs = pd.read_csv(f"{OUT}/pairs.csv")
tg = pd.read_csv(f"{OUT}/targets.csv")
LIBS = sorted(tg.lib.unique())
BASE = {"A": 0, "C": 1, "G": 2, "T": 3}


def umi_code(s):
    v = 0
    for ch in s:
        v = v * 4 + BASE.get(ch, 0)
    return v


# (replaced by the chunked loader below; original kept in umi_overlap_compare.py.before_chunked.bak)
cells = {}   # (lib, cb) -> (umi array, readkey array)
nreads = {}
UMI_MAP = {}
for lib in LIBS:
    fn = f"{OUT}/reads_{lib}.tsv.gz"
    if not os.path.exists(fn):
        continue
    acc = {}     # cb -> [list of u arrays, list of k arrays]; integers only (the earlier all-strings version was killed by the login-node memory limit)
    for r in pd.read_csv(fn, sep="\t", header=None, names=["cb", "umi", "seq"], dtype=str, chunksize=500000):
        r = r[r.umi.str.len() == 6]
        for u in r.umi.unique():
            if u not in UMI_MAP:
                UMI_MAP[u] = umi_code(u)
        u_int = r.umi.map(UMI_MAP).to_numpy(dtype=np.int64)
        k_int = (u_int * 1000003 + pd.util.hash_array(r.seq.str.slice(0, 20).to_numpy(dtype=object)).astype(np.int64)) & 0x7FFFFFFFFFFFFFFF
        cbv = r.cb.to_numpy()
        order = np.argsort(cbv, kind="stable")
        cbs, starts = np.unique(cbv[order], return_index=True)
        ends = list(starts[1:]) + [len(order)]
        for cb, st, en in zip(cbs, starts, ends):
            ix = order[st:en]
            e = acc.setdefault(cb, [[], [], 0])
            e[0].append(np.unique(u_int[ix])); e[1].append(np.unique(k_int[ix])); e[2] += len(ix)
    for cb, (ul, kl, n) in acc.items():
        cells[(lib, cb)] = (np.unique(np.concatenate(ul)), np.unique(np.concatenate(kl)))
        nreads[(lib, cb)] = n
    del acc
print("cells with reads loaded:", len(cells), flush=True)


def stat(a, b):
    return len(np.intersect1d(a[0], b[0], assume_unique=True)), len(np.intersect1d(a[1], b[1], assume_unique=True))


ctrl = {lib: [cb for cb in tg[(tg.lib == lib) & (tg.case == "ctrl")].cb if (lib, cb) in cells] for lib in LIBS}
rows = []
for p in pairs.itertuples():
    libs = p.libraries.split(",")
    for l1, l2 in itertools.combinations(libs, 2):
        A, B = cells.get((l1, p.cb)), cells.get((l2, p.cb))
        if A is None or B is None:
            continue
        su, sk = stat(A, B)
        nb = len(B[0])
        cand = [(abs(len(cells[(l2, c)][0]) - nb), c) for c in ctrl[l2] if c != p.cb]
        cand.sort()
        close = [c for dlt, c in cand if dlt <= 0.25 * nb] or [c for _, c in cand[:10]]
        pick = rng.choice(close, size=min(M_NULL, len(close)), replace=False) if close else []
        nu, nk = zip(*[stat(A, cells[(l2, c)]) for c in pick]) if len(pick) else ([np.nan], [np.nan])
        rows.append(dict(cb=p.cb, clone=p.clone, L1=l1, L2=l2, umiA=len(A[0]), umiB=nb, readsA=nreads[(l1, p.cb)], readsB=nreads[(l2, p.cb)],
                         shared_umi=su, shared_readkey=sk, null_umi_mean=np.mean(nu), null_umi_p95=np.percentile(nu, 95),
                         null_key_mean=np.mean(nk), null_key_p95=np.percentile(nk, 95), n_null=len(pick)))
res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/pair_results.csv", index=False)
if len(res):
    s = dict(n_pairs=len(res),
             shared_umi_mean=float(res.shared_umi.mean()), null_umi_mean=float(res.null_umi_mean.mean()),
             frac_umi_above_null95=float((res.shared_umi > res.null_umi_p95).mean()),
             shared_readkey_mean=float(res.shared_readkey.mean()), null_key_mean=float(res.null_key_mean.mean()),
             frac_key_above_null95=float((res.shared_readkey > res.null_key_p95).mean()),
             frac_pairs_with_any_readkey=float((res.shared_readkey > 0).mean()),
             frac_null_with_any_readkey_mean=float((res.null_key_mean > 0).mean()))
    json.dump(s, open(f"{OUT}/summary.json", "w"), indent=1)
    print(json.dumps(s, indent=1))
    print(res.groupby(["L1", "L2"])[["shared_umi", "null_umi_mean", "shared_readkey", "null_key_mean"]].mean().round(2).to_string())
else:
    print("no pairs had reads in both libraries")
