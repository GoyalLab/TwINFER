#!/usr/bin/env python3
"""Step 8 (graph -> clones) of yscher's michaels_barcodes.py, verbatim, run on the cell-barcode table saved by michaels_rerun_barcodes.py
(final_cell_barcode_umis.csv.gz = after filters 4, 6, 5). Step 7 (>= 50% of the cell's top barcode) is redone here verbatim.
Writes clones to argv[1] and prints whether the result is identical to yscher's cell_to_clone.csv (row by row, including clone ids)."""
import csv, collections, gzip, glob, os, sys
import networkx as nx
Y = "/gpfs/projects/b1255/yscher/Transcriptomic Distance/datasets/michaels2024"
IN = "/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/dataset_qc_verification/michaels_rerun/final_cell_barcode_umis.csv.gz"
WL = {}
for s in ("T1", "T2"):
    with gzip.open(glob.glob(f"{Y}/raw/*_{s}_barcodes.tsv.gz")[0], "rt") as fh:
        for line in fh: WL[line.strip().split("-")[0]] = s
final = {}
with gzip.open(IN, "rt") as fh:
    r = csv.reader(fh); next(r)
    for cb, lb, v in r: final[(cb, lb)] = int(v)
percell = collections.defaultdict(list)
for (cb, lb), v in final.items(): percell[cb].append((lb, v))
edges = collections.defaultdict(set)
for cb, v in percell.items():
    top = max(n for _, n in v)
    for lb, n in v:
        if n >= 0.5 * top: edges[lb].add(cb)
G = nx.Graph()
for lb, cells in edges.items():
    cs = sorted(cells)
    for i in range(len(cs)):
        G.add_node(cs[i])
        for j in range(i + 1, len(cs)): G.add_edge(cs[i], cs[j])
raw = [c for c in nx.connected_components(G)]
def degree_filter(H):
    if H.number_of_nodes() < 2: return H
    mx = max(d for _, d in H.degree())
    keep = [n for n, d in H.degree() if d > 0.55 * mx]
    return H.subgraph(keep).copy()
def try_cut(H):
    if H.number_of_nodes() < 4: return None
    n0 = H.number_of_nodes()
    for cutter in (nx.minimum_edge_cut, nx.minimum_node_cut):
        try: cut = cutter(H)
        except Exception: continue
        K = H.copy()
        if cutter is nx.minimum_edge_cut: K.remove_edges_from(cut)
        else: K.remove_nodes_from(cut)
        parts = [K.subgraph(c).copy() for c in nx.connected_components(K)]
        if len(parts) > 1 and all(p.number_of_nodes() >= 2 and p.number_of_nodes() >= 0.25 * n0 for p in parts):
            return parts
    return None
out = []; stack = [G.subgraph(c).copy() for c in raw]
while stack:
    H = stack.pop()
    p = try_cut(H)
    if p: stack.extend(p); continue
    H = degree_filter(H)
    if H.number_of_nodes() < 2: continue
    comps = list(nx.connected_components(H))
    if len(comps) > 1: stack.extend(H.subgraph(c).copy() for c in comps); continue
    p = try_cut(H)
    if p: stack.extend(p); continue
    out.append(sorted(H.nodes()))
rows = [(cb, WL.get(cb, ""), "clone_%d" % i) for i, c in enumerate(out) for cb in c]
with open(sys.argv[1], "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["cell_barcode", "sample", "clone_id"]); w.writerows(rows)
her = [tuple(x) for x in csv.reader(open(f"{Y}/clones/cell_to_clone.csv"))][1:]
grp = lambda R: {frozenset(c for c, _, k in R if k == key) for key in {k for _, _, k in R}}
print("nx", nx.__version__, "PYTHONHASHSEED", os.environ.get("PYTHONHASHSEED"), "edges/raw", len(edges), len(raw), "clones", len(out), "cells", len(rows),
      "| rows identical incl. ids and order:", rows == her, "| same clone groups:", grp(rows) == grp(her), flush=True)
