#!/usr/bin/env python3
"""All-to-all comparison of the robust NMF programs of FM01, FM03, FM05, FM06, FM07, FM08 (gene lists only, no names, no enrichment).
Programs are numbered 1..N in the order dataset, replicate, K, index. Overlap = number of shared genes among the 50.
Outputs (analysis_data/fm_meta_programs_comparison/): program_index.csv, pairwise_overlap_matrix.csv, pairs_long.csv,
components_ge{T}.csv for T in 20, 35 (connected components of the graph 'overlap >= T')."""
import sys
import numpy as np, pandas as pd
from collections import Counter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"
SRC = sys.argv[1] if len(sys.argv) > 1 else "mp_nmf"          # mp_nmf (replicates A/B separate) or mp_nmf_pooled (A+B merged, one sample per dataset)
import os
OUT = f"{ROOT}/analysis_data/fm_meta_programs_comparison" + ("_pooled" if SRC == "mp_nmf_pooled" else ""); os.makedirs(OUT, exist_ok=True)
rows = []
for ds in ["FM01", "FM03", "FM05", "FM06", "FM07", "FM08"]:
    r = pd.read_csv(f"{ROOT}/analysis_data/{ds.lower()}/data/{SRC}/robust_programs.csv").sort_values(["sample", "K", "idx"])
    for x in r.itertuples(): rows.append(dict(dataset=ds, replicate=x.sample, K=x.K, idx=x.idx, genes=x.genes))
P = pd.DataFrame(rows); P.insert(0, "program", np.arange(1, len(P) + 1)); P["sample"] = P.dataset + "_" + P.replicate
P.drop(columns="genes").to_csv(f"{OUT}/program_index.csv", index=False)
S = [set(g.split()) for g in P.genes]; n = len(S)
M = np.array([[len(S[i] & S[j]) for j in range(n)] for i in range(n)]); np.fill_diagonal(M, 50)
pd.DataFrame(M, index=P.program, columns=P.program).to_csv(f"{OUT}/pairwise_overlap_matrix.csv")
L = [dict(a=P.program[i], b=P.program[j], sample_a=P["sample"][i], sample_b=P["sample"][j], dataset_a=P.dataset[i], dataset_b=P.dataset[j], overlap=M[i, j]) for i in range(n) for j in range(i + 1, n)]
L = pd.DataFrame(L); L.to_csv(f"{OUT}/pairs_long.csv", index=False)
X = L[L.sample_a != L.sample_b]   # pairs from different samples
print(f"{n} programs, {len(L)} pairs, {len(X)} from different samples (same-sample pairs excluded below)")
print("expected overlap of two random 50-gene lists from ~10,400 detected genes: 0.24")
for t in (5, 10, 20, 30, 35, 40): print(f"  pairs with overlap >= {t:>2}: {int((X.overlap >= t).sum())}")
print("\nprograms (number: sample K index):"); print(P[["program", "sample", "K", "idx"]].to_string(index=False))
for T in (20, 35):
    A = csr_matrix((M >= T) & ~np.eye(n, dtype=bool)); k, lab = connected_components(A, directed=False)
    sizes = Counter(lab); comp = []
    for c, sz in sizes.most_common():
        idx = np.where(lab == c)[0]; ds = sorted(set(P.dataset[idx]))
        cnt = Counter(g for i in idx for g in S[i]); top = [g for g, _ in cnt.most_common(12)]
        comp.append(dict(component=len(comp) + 1, n_programs=sz, n_datasets=len(ds), datasets=",".join(ds), programs=",".join(map(str, P.program[idx])), min_pairwise_overlap=int(M[np.ix_(idx, idx)].min()) if sz > 1 else 50, top12_genes_by_frequency=" ".join(top)))
    C = pd.DataFrame(comp); C.to_csv(f"{OUT}/components_ge{T}.csv", index=False)
    print(f"\n== components with overlap >= {T}: {int((C.n_programs > 1).sum())} with >1 program; singletons {int((C.n_programs == 1).sum())}")
    print(C[C.n_programs > 1].drop(columns="top12_genes_by_frequency").to_string(index=False))
print("\nper dataset pair: programs of the row dataset having a partner (overlap >= 20) in the column dataset")
ds = list(dict.fromkeys(P.dataset)); T = pd.DataFrame(0, index=ds, columns=ds)
for i in range(n):
    for d in ds:
        j = np.where((P.dataset == d) & (P.dataset != P.dataset[i]))[0]
        if len(j) and M[i, j].max() >= 20: T.loc[P.dataset[i], d] += 1
T["n_programs"] = P.groupby("dataset").size(); print(T.to_string())
