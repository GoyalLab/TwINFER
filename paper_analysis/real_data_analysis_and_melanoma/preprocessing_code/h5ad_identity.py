#!/usr/bin/env python3
"""Element-wise identity of two .h5ad files: every HDF5 dataset under obs, var, X, layers, obsm, uns (and raw if present) is compared
exactly (same shape, dtype, values; strings decoded). Sparse groups are compared through their data/indices/indptr arrays.
Large arrays are compared in 50M-element blocks so memory stays small.
usage: h5ad_identity.py <a.h5ad> <b.h5ad> [out.json]"""
import json
import sys

import h5py
import numpy as np


def walk(g, prefix=""):
    out = {}
    for k, v in g.items():
        p = f"{prefix}/{k}"
        if isinstance(v, h5py.Group):
            out.update(walk(v, p))
        else:
            out[p] = v
    return out


def same(a, b, block=50_000_000):
    if a.shape != b.shape:
        return False, f"shape {a.shape} vs {b.shape}"
    if a.dtype != b.dtype and not (a.dtype.kind in "OSU" and b.dtype.kind in "OSU"):
        return False, f"dtype {a.dtype} vs {b.dtype}"
    if a.shape == ():
        eq = bool(a[()] == b[()])
        return eq, "identical" if eq else f"scalar differs: {a[()]!r} vs {b[()]!r}"
    n = a.shape[0]
    step = max(1, block // max(1, int(np.prod(a.shape[1:])) if len(a.shape) > 1 else 1))
    for i in range(0, n, step):
        x, y = a[i:i + step], b[i:i + step]
        if x.dtype.kind in "OS":
            x = np.array([e.decode() if isinstance(e, bytes) else e for e in x.ravel()], dtype=object)
            y = np.array([e.decode() if isinstance(e, bytes) else e for e in y.ravel()], dtype=object)
            if not (x == y).all():
                return False, f"strings differ in block {i}"
        elif x.dtype.kind == "f":
            if not np.array_equal(x, y, equal_nan=True):
                d = np.nanmax(np.abs(x.astype(float) - y.astype(float)))
                return False, f"values differ in block {i}, max |diff| {d}"
        elif not np.array_equal(x, y):
            return False, f"values differ in block {i}"
    return True, "identical"


def main(fa, fb, out=None):
    with h5py.File(fa, "r") as A, h5py.File(fb, "r") as B:
        da, db = walk(A), walk(B)
        res = dict(file_a=fa, file_b=fb, only_in_a=sorted(set(da) - set(db)), only_in_b=sorted(set(db) - set(da)), datasets={})
        for k in sorted(set(da) & set(db)):
            ok, why = same(da[k], db[k])
            res["datasets"][k] = why
        res["n_datasets_compared"] = len(res["datasets"])
        res["n_identical"] = sum(v == "identical" for v in res["datasets"].values())
        res["ALL_IDENTICAL"] = (not res["only_in_a"] and not res["only_in_b"] and res["n_identical"] == res["n_datasets_compared"])
    if out:
        json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "datasets"}, indent=1))
    print("non-identical:", {k: v for k, v in res["datasets"].items() if v != "identical"})
    return res


if __name__ == "__main__":
    main(*sys.argv[1:4])
