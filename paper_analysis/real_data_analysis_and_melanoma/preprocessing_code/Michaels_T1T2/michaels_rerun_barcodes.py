#!/usr/bin/env python3
"""Independent re-run of yscher's Michaels 2024 lineage-barcode -> clone pipeline (helpers/michaels_barcodes.py), instrumented.

The algorithm is a faithful copy of yscher's (same pattern, filter order, thresholds, graph processing). Differences are ONLY (a) counts
logged after every step, (b) validation checks (listed as V1..V8), (c) a faster but provably equivalent neighbour lookup for the
consensus-rescue fallback (checked against yscher's original function on a random sample of barcodes), (d) outputs go to OUT, not to yscher's folder.

usage: michaels_rerun_barcodes.py [--max-reads N]   (N reads per fastq file, for a local smoke test)
Inputs (read-only): raw_data/Michaels_TLB_fastq, raw_data/Michaels_GEO; comparison files (yscher's cell_to_clone.csv in final_data, optionally the lineage h5ad).
Outputs: <REBUILD>/Michaels_T1T2/michaels_rerun/{cell_to_clone_rerun.csv, steps.json, validation.json, final_cell_barcode_umis.csv.gz}
"""
import collections
import csv
import glob
import gzip
import json
import os
import random
import re
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # fig5_paths.py lives in preprocessing_code/
from fig5_paths import MICHAELS_CLONES, MICHAELS_GEO, MICHAELS_TLB_FASTQ, RAW, REBUILD          # noqa: E402
OUT = os.path.join(REBUILD, "Michaels_T1T2", "michaels_rerun")
HER_CSV = MICHAELS_CLONES                                # yscher's clone labels the rerun is compared with
# optional comparison V8 with the authors' combined-singlets lineage h5ad (not available as a link at the moment: V8 is then reported as "not run")
LINEAGE_H5AD = os.environ.get("MICHAELS_LINEAGE_H5AD", os.path.join(RAW, "Michaels_lineage_T1_T2_combined_singlets.h5ad"))
PAT = re.compile(r"AT([ACGT]{4})CT([ACGT]{4})AC([ACGT]{4})TC([ACGT]{4})GT")
CB_LEN, UMI_LEN = 16, 12
MAXR = None
if "--max-reads" in sys.argv:
    MAXR = int(sys.argv[sys.argv.index("--max-reads") + 1])
    OUT = OUT + f"_smoke{MAXR}"
os.makedirs(OUT, exist_ok=True)
STEPS, VAL = {}, {}
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def save():
    json.dump(STEPS, open(f"{OUT}/steps.json", "w"), indent=1, default=str)
    json.dump(VAL, open(f"{OUT}/validation.json", "w"), indent=1, default=str)


# ------------------------------------------------------------------ cell whitelist (10x filtered barcodes of T1 and T2)
WL, WL_S = {}, {}
for s in ("T1", "T2"):
    f = glob.glob(f"{MICHAELS_GEO}/*_{s}_barcodes.tsv.gz")[0]
    with gzip.open(f, "rt") as fh:
        for line in fh:
            WL.setdefault(line.strip().split("-")[0], s)
    WL_S[s] = sum(1 for v in WL.values() if v == s)
VAL["V0_whitelist"] = dict(n_cell_barcodes=len(WL), per_sample=WL_S,
                           note="a barcode present in BOTH T1 and T2 matrices keeps the first sample (T1) in yscher's script -> counted here")
both = 0
for s in ("T2",):
    f = glob.glob(f"{MICHAELS_GEO}/*_{s}_barcodes.tsv.gz")[0]
    with gzip.open(f, "rt") as fh:
        t1 = {b.strip().split("-")[0] for b in gzip.open(glob.glob(f"{MICHAELS_GEO}/*_T1_barcodes.tsv.gz")[0], "rt")}
        both = sum(1 for line in fh if line.strip().split("-")[0] in t1)
VAL["V0_whitelist"]["barcodes_in_both_T1_and_T2"] = both
log("whitelist", len(WL), WL_S, "in both:", both)

# ------------------------------------------------------------------ STEP 1: extraction from reads
counts = collections.Counter()          # (cb, umi, lb) -> reads
per_file, nread, nhit, n_n, n_short = [], 0, 0, 0, 0
wl_hit = umi_wl_hit = 0                 # V2: extracted cell barcodes present in the whitelist; negative control = UMI string in the whitelist
pos_hist = collections.Counter()
for r1 in sorted(glob.glob(f"{MICHAELS_TLB_FASTQ}/*_1.fastq.gz")):
    r2 = r1.replace("_1.fastq.gz", "_2.fastq.gz")
    nr = nh = 0
    with gzip.open(r1, "rt") as f1, gzip.open(r2, "rt") as f2:
        while True:
            h1 = f1.readline()
            if not h1 or (MAXR and nr >= MAXR):
                break
            s1 = f1.readline().strip(); f1.readline(); f1.readline()
            h2 = f2.readline(); s2 = f2.readline().strip(); f2.readline(); f2.readline()
            if h1.split()[0] != h2.split()[0]:
                raise SystemExit(f"read-name mismatch R1/R2 in {r1}: {h1} {h2}")        # V1: pairs in sync
            nr += 1
            if len(s1) < CB_LEN + UMI_LEN:
                n_short += 1
                continue
            m = PAT.search(s2)
            if not m:
                continue
            cb = s1[:CB_LEN]; umi = s1[CB_LEN:CB_LEN + UMI_LEN]
            if "N" in cb or "N" in umi:
                n_n += 1
                continue
            counts[(cb, umi, "".join(m.groups()))] += 1; nh += 1
            wl_hit += cb in WL
            umi_wl_hit += umi in WL
            if nh <= 200000:
                pos_hist[m.start()] += 1
    per_file.append(dict(file=os.path.basename(r1), reads=nr, with_construct=nh, frac=round(nh / max(nr, 1), 4)))
    nread += nr; nhit += nh
    log(per_file[-1])
STEPS["1_extraction"] = dict(per_file=per_file, reads=nread, reads_short_R1=n_short, with_construct_valid_cb_umi=nhit, dropped_N_in_cb_or_umi=n_n,
                             distinct_cb_umi_lb=len(counts))
VAL["V1_pairs_in_sync_and_pattern"] = dict(read_names_matched="all pairs", construct_start_position_top=pos_hist.most_common(5))
VAL["V2_cell_barcode_orientation"] = dict(frac_extracted_cb_in_whitelist=round(wl_hit / max(nhit, 1), 4),
                                          negative_control_frac_umi_string_in_whitelist=round(umi_wl_hit / max(nhit, 1), 6),
                                          note="CB in whitelist >> UMI-as-CB means read 1 layout (16 CB + 12 UMI) is right")
log("step1", STEPS["1_extraction"]["distinct_cb_umi_lb"], VAL["V2_cell_barcode_orientation"])
save()


def lev1(a, b):
    """yscher's original (kept for the equivalence test)"""
    if a == b: return True
    if abs(len(a) - len(b)) > 1: return False
    if len(a) == len(b): return sum(x != y for x, y in zip(a, b)) <= 1
    if len(a) > len(b): a, b = b, a
    for i in range(len(b)):
        if a == b[:i] + b[i + 1:]: return True
    return False


# ------------------------------------------------------------------ STEPS 2: same cell+UMI, several barcodes
byumi = collections.defaultdict(list)
for (cb, umi, lb), n in counts.items():
    byumi[(cb, umi)].append((lb, n))
kept, dropped_umi, merged_umi = {}, 0, 0
for k, v in byumi.items():
    if len(v) == 1:
        kept[k] = v[0]; continue
    v.sort(key=lambda t: -t[1])
    if all(lev1(v[0][0], lb) for lb, _ in v[1:]):
        kept[k] = (v[0][0], sum(n for _, n in v)); merged_umi += 1
    else:
        dropped_umi += 1
STEPS["2_same_UMI_resolution"] = dict(cell_umi_pairs=len(byumi), with_one_barcode=len(byumi) - merged_umi - dropped_umi,
                                      merged_within_1_edit=merged_umi, dropped_incompatible=dropped_umi, kept=len(kept))
VAL["V3_umi_reads"] = dict(reads_per_cell_umi_barcode_quantiles=dict(zip(["q50", "q90", "q99", "max"], map(int, np.quantile(list(counts.values()), [.5, .9, .99, 1])))))
del counts, byumi
log("step2", STEPS["2_same_UMI_resolution"]); save()

# ------------------------------------------------------------------ STEP 3: consensus barcodes and rescue of singletons
lbcount = collections.Counter(lb for (cb, umi), (lb, n) in kept.items())
consensus = [lb for lb, c in lbcount.items() if c > 1]
cset = set(consensus)
corder = {lb: i for i, lb in enumerate(consensus)}


def rescue_orig(lb):
    for i in range(0, len(lb), 4):
        for c in "ACGT":
            alt = lb[:i] + c + lb[i + 1:]
            if alt in cset: return alt
    for cand in consensus:
        if lev1(lb, cand): return cand
    return None


def rescue_fast(lb):
    """same result as rescue_orig: stage 1 identical; fallback = first consensus (list order) within Hamming 1 (all barcodes are 16 nt)."""
    for i in range(0, len(lb), 4):
        for c in "ACGT":
            alt = lb[:i] + c + lb[i + 1:]
            if alt in cset: return alt
    best = None
    for i in range(len(lb)):
        for c in "ACGT":
            alt = lb[:i] + c + lb[i + 1:]
            if alt != lb and alt in cset and (best is None or corder[alt] < corder[best]): best = alt
    return best


def n_neighbours(lb):
    return sum(1 for i in range(len(lb)) for c in "ACGT" if lb[:i] + c + lb[i + 1:] in cset and lb[:i] + c + lb[i + 1:] != lb)


singles = [lb for lb, c in lbcount.items() if c == 1]
random.seed(0)
samp = random.sample(singles, min(300, len(singles)))
same = sum(rescue_orig(x) == rescue_fast(x) for x in samp)         # V4: equivalence of the fast lookup (300 random singleton barcodes; yscher's function is O(N) each)
assert same == len(samp), f"rescue_fast differs from yscher's rescue on {len(samp) - same} barcodes"
amb = sum(n_neighbours(x) > 1 for x in singles[:20000])
final = collections.Counter()           # (cb, lb) -> UMIs
n_resc = n_drop_single = 0
for (cb, umi), (lb, n) in kept.items():
    if lb not in cset:
        lb2 = rescue_fast(lb)
        if lb2 is None:
            n_drop_single += 1; continue
        lb = lb2; n_resc += 1
    final[(cb, lb)] += 1
STEPS["3_consensus"] = dict(distinct_barcodes=len(lbcount), consensus_seen_more_than_once=len(consensus), singleton_barcodes=len(singles),
                            cell_umis_rescued_into_a_consensus=n_resc, cell_umis_dropped_singleton_no_neighbour=n_drop_single,
                            cell_barcode_pairs_after=len(final))
VAL["V4_rescue"] = dict(fast_equals_her_function_on=f"{same}/{len(samp)} random singleton barcodes",
                        frac_of_first_20000_singletons_with_more_than_one_consensus_neighbour=round(amb / max(min(len(singles), 20000), 1), 4),
                        note="if >0 the rescue target depends on list order (dict insertion order)")
del kept
log("step3", STEPS["3_consensus"]); save()

# ------------------------------------------------------------------ STEPS 4-7 filters (yscher's order: matrix, >=2 UMIs, barcode in >1 cell, >=50% of cell's top)
steps = {}
steps["start_pairs"] = len(final)
final = {k: v for k, v in final.items() if k[0] in WL}
steps["4_after_cell_in_10x_matrix"] = len(final)
final = {k: v for k, v in final.items() if v >= 2}
steps["6_after_ge2_UMIs"] = len(final)
ncells = collections.Counter(lb for (cb, lb) in final)
final = {k: v for k, v in final.items() if ncells[k[1]] > 1}
steps["5_after_barcode_in_more_than_1_cell"] = len(final)
percell = collections.defaultdict(list)
for (cb, lb), v in final.items():
    percell[cb].append((lb, v))
edges = collections.defaultdict(set)
for cb, v in percell.items():
    top = max(n for _, n in v)
    for lb, n in v:
        if n >= 0.5 * top: edges[lb].add(cb)
steps["7_after_ge_50pct_of_cell_top"] = sum(len(s) for s in edges.values())
steps["cells_with_a_barcode"] = len({c for s in edges.values() for c in s})
steps["barcodes_used"] = len(edges)
STEPS["4to7_filters"] = steps
pd.DataFrame([(cb, lb, v) for (cb, lb), v in final.items()], columns=["cell_barcode", "lineage_barcode", "umis"]).to_csv(
    f"{OUT}/final_cell_barcode_umis.csv.gz", index=False)
cells_per_cell = collections.Counter(len(v) for v in percell.values())
VAL["V5_filters"] = dict(barcodes_per_cell_after_filters={int(k): v for k, v in sorted(cells_per_cell.items())[:8]},
                         cells_per_barcode_quantiles=dict(zip(["q50", "q90", "q99", "max"], map(int, np.quantile([len(s) for s in edges.values()], [.5, .9, .99, 1])))))
log("steps4-7", steps); save()

# ------------------------------------------------------------------ STEP 8: graph -> clones (yscher's processing, verbatim)
import networkx as nx
G = nx.Graph()
for lb, cells in edges.items():
    cs = sorted(cells)
    for i in range(len(cs)):
        G.add_node(cs[i])
        for j in range(i + 1, len(cs)): G.add_edge(cs[i], cs[j])
raw = [c for c in nx.connected_components(G)]
STEPS["8_graph"] = dict(nodes=G.number_of_nodes(), edges=G.number_of_edges(), connected_components=len(raw),
                        component_size_top10=sorted((len(c) for c in raw), reverse=True)[:10])


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


out = []
stack = [G.subgraph(c).copy() for c in raw]
n_cut = n_deg = 0
while stack:
    H = stack.pop()
    p = try_cut(H)
    if p: stack.extend(p); n_cut += 1; continue
    H2 = degree_filter(H)
    if H2.number_of_nodes() < H.number_of_nodes(): n_deg += 1
    H = H2
    if H.number_of_nodes() < 2: continue
    comps = list(nx.connected_components(H))
    if len(comps) > 1: stack.extend(H.subgraph(c).copy() for c in comps); continue
    p = try_cut(H)
    if p: stack.extend(p); n_cut += 1; continue
    out.append(sorted(H.nodes()))
sizes = collections.Counter(len(c) for c in out)
STEPS["8_graph"].update(cut_operations=n_cut, components_where_degree_filter_removed_cells=n_deg, final_clones=len(out),
                        cells_in_a_clone=sum(len(c) for c in out), cells_dropped_by_cut_or_degree=steps["cells_with_a_barcode"] - sum(len(c) for c in out),
                        size_distribution_first12=dict(sorted(sizes.items())[:12]), max_size=max(sizes))
rows = [(cb, WL.get(cb, ""), f"clone_{i}") for i, c in enumerate(out) for cb in c]
with open(f"{OUT}/cell_to_clone_rerun.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["cell_barcode", "sample", "clone_id"]); w.writerows(rows)
STEPS["8_graph"]["multi_time_clones_T1_and_T2"] = sum(1 for c in out if len({WL.get(cb, "") for cb in c}) > 1)
log("step8", STEPS["8_graph"]); save()

# ------------------------------------------------------------------ V6: internal consistency of the clone table
cl = pd.DataFrame(rows, columns=["cell_barcode", "sample", "clone_id"])
bc_to_clones = collections.defaultdict(set)
cell_clone = dict(zip(cl.cell_barcode, cl.clone_id))
for lb, cs in edges.items():
    for c in cs:
        if c in cell_clone: bc_to_clones[lb].add(cell_clone[c])
VAL["V6_consistency"] = dict(cells_in_two_clones=int(cl.cell_barcode.duplicated().sum()),
                             barcodes_spanning_more_than_one_final_clone=sum(1 for s in bc_to_clones.values() if len(s) > 1),
                             barcodes_total_in_clones=len(bc_to_clones))

# ------------------------------------------------------------------ V7: comparison with yscher's cell_to_clone.csv
from sklearn.metrics import adjusted_rand_score
her = pd.read_csv(HER_CSV)
mine = cl.set_index("cell_barcode").clone_id
hs = her.set_index("cell_barcode").clone_id
allc = sorted(set(mine.index) | set(hs.index))
a = [mine.get(c, f"__none_{c}") for c in allc]; b = [hs.get(c, f"__none_{c}") for c in allc]
both_c = sorted(set(mine.index) & set(hs.index))
# partition identity on shared cells (ids differ in numbering): ARI, and identical-partition check via canonical grouping
def canon(s, cells):
    g = collections.defaultdict(list)
    for c in cells: g[s[c]].append(c)
    return {tuple(sorted(v)) for v in g.values()}
cm, ch = canon(mine, both_c), canon(hs, both_c)
VAL["V7_vs_her_cell_to_clone"] = dict(rerun_cells=len(mine), her_cells=len(hs), shared_cells=len(both_c), only_rerun=len(set(mine.index) - set(hs.index)),
                                      only_her=len(set(hs.index) - set(mine.index)), rerun_clones=int(mine.nunique()), her_clones=int(hs.nunique()),
                                      ARI_shared_cells=round(adjusted_rand_score([mine[c] for c in both_c], [hs[c] for c in both_c]), 5) if both_c else None,
                                      identical_clone_groups_on_shared_cells=len(cm & ch), groups_rerun=len(cm), groups_her=len(ch))
log("V7", VAL["V7_vs_her_cell_to_clone"])

# ------------------------------------------------------------------ V8: comparison with the combined-singlets lineage h5ad
try:
    import anndata as ad
    import h5py
    with h5py.File(LINEAGE_H5AD, "r") as f:
        o = ad.io.read_elem(f["obs"])
    o["cb"] = [i.rsplit("_", 1)[0] for i in o.index]
    lin = o.set_index("cb").clone_id.astype(str)
    sh = sorted(set(lin.index) & set(mine.index))
    VAL["V8_vs_lineage_h5ad"] = dict(lineage_cells=len(lin), shared_with_rerun=len(sh),
                                    ARI_shared=round(adjusted_rand_score([mine[c] for c in sh], [lin[c] for c in sh]), 5) if sh else None)
except Exception as e:                                             # comparison is optional
    VAL["V8_vs_lineage_h5ad"] = f"not run: {e!r}"
log("V8", VAL["V8_vs_lineage_h5ad"])
STEPS["runtime_s"] = round(time.time() - T0)
save()
log("done")
