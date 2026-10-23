#!/usr/bin/env python3
"""Audit of the PeTrace lineage trees (obst) and of the sister-pair definition used in pair_distance.py, per .h5td file.

Checks / counts (all computed, nothing assumed):
  - trees per file; nodes; internal nodes (keys of `edges`); leaves (children that are not internal nodes)
  - every leaf is an obs cell; every obs cell with a non-empty `tree` value is a leaf exactly once; a node never has two parents
  - leaf children per parent: distribution (2 = binary sister pair; >2 = polytomy, all pairs counted as sisters)
  - sister pairs = sum over parents of C(k,2) with k = leaf children present in obs; equals the count pair_distance.py produced
  - n_edits on parent->leaf edges (0 = leaf identical to its parent's characters)
  - the obs `clone` column vs trees: clone == tree name for every cell?
usage: petrace_tree_audit.py <tag> [<tag> ...]     tag like E7.5-R1
Writes analysis_data/dataset_qc_verification/petrace_tree_audit/<tag>.json
"""
import collections
import json
import os
import sys

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qc_common import ROOT, read_h5ad_obs_var

OUT = f"{ROOT}/analysis_data/dataset_qc_verification/petrace_tree_audit"
os.makedirs(OUT, exist_ok=True)


def audit(tag):
    f = f"{ROOT}/real_data/PeTrace/{tag}.h5td"
    obs, _ = read_h5ad_obs_var(f)
    cells = set(obs.index)
    with h5py.File(f, "r") as h:
        trees = json.loads(h["obst"][()].decode())
    r = dict(file=tag, n_obs_cells=len(obs), n_trees=len(trees), trees={})
    parent_of = {}
    multi_parent = 0
    kdist = collections.Counter()
    pairs = 0
    leaf_edits = collections.Counter()
    leaves_all = []
    for tn, t in trees.items():
        E = t["edges"]; N = t["nodes"]
        internal = {p for p, ch in E.items() if ch}          # every node is a key of `edges`; leaves have no children
        ch_all = [c for p in E for c in E[p]]
        leaves = [c for c in ch_all if c not in internal]
        for p, ch in E.items():
            for c in ch:
                if c in parent_of:
                    multi_parent += 1
                parent_of[c] = p
            lv = [c for c in ch if c not in internal]
            lv_obs = [c for c in lv if c in cells]
            if lv_obs:
                kdist[len(lv_obs)] += 1
                pairs += len(lv_obs) * (len(lv_obs) - 1) // 2
            for c in lv:
                leaf_edits[min(int(ch[c].get("n_edits", -1)), 30)] += 1
        roots = internal - set(ch_all)
        r.setdefault("obs_cells_that_have_children", 0)
        r["obs_cells_that_have_children"] += sum(c in cells for c in internal)
        r["trees"][tn] = dict(n_nodes_listed=len(N), n_internal=len(internal), n_leaves=len(leaves), n_roots=len(roots),
                              leaves_not_in_obs=sum(c not in cells for c in leaves))
        leaves_all += leaves
    lc = collections.Counter(leaves_all)
    in_tree = obs["tree"].astype(str).replace({"nan": ""}) != "" if "tree" in obs else pd.Series(False, index=obs.index)
    r.update(
        nodes_with_more_than_one_parent=multi_parent,
        leaves_total=len(leaves_all), leaves_appearing_twice=sum(v > 1 for v in lc.values()),
        leaves_not_in_obs=sum(c not in cells for c in lc),
        obs_cells_with_tree_value=int(in_tree.sum()),
        obs_cells_with_tree_value_that_are_leaves=int(sum(c in lc for c in obs.index[in_tree])),
        obs_cells_that_are_leaves=int(sum(c in lc for c in obs.index)),
        leaf_children_per_parent={int(k): int(v) for k, v in sorted(kdist.items())},
        parents_with_ge2_leaf_children=int(sum(v for k, v in kdist.items() if k >= 2)),
        sister_pairs_all_pairs_within_parent=int(pairs),
        sister_pairs_from_binary_parents_only=int(kdist.get(2, 0)),
        parent_to_leaf_n_edits_hist={int(k): int(v) for k, v in sorted(leaf_edits.items())},
        clone_equals_tree_for_all_cells=bool((obs["clone"].astype(str) == obs["tree"].astype(str)).all()) if "tree" in obs else None,
        n_distinct_clone_values=int(obs["clone"].nunique()),
    )
    json.dump(r, open(f"{OUT}/{tag}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in r.items() if k != "trees"}), flush=True)


if __name__ == "__main__":
    for t in sys.argv[1:]:
        audit(t)
