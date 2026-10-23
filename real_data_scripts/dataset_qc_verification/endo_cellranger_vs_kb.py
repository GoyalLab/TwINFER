#!/usr/bin/env python3
"""hPSC_endo_yscher: Cell Ranger counts (endo_counts_mito20_min200genes.h5ad, X) vs the kb-python X of endo_kbnac_spliced_unspliced.h5ad.
Same cells (obs names compared), genes matched by SYMBOL. Streamed in cell chunks (login-node safe). Writes endo_cellranger_vs_kb.json"""
import json
import sys

import anndata as ad
import h5py
import numpy as np
import scipy.sparse as sp

ROOT = "/gpfs/projects/b1255/hzhang/TwINFER_KA"
CR = f"{ROOT}/analysis_data/dataset_qc_verification/rebuild/hPSC_endo_yscher/A/endo_counts_mito20_min200genes.h5ad"   # identical to yscher's (identity_counts_vs_hers.json)
KB = f"{ROOT}/finalized_data/hPSC_endo_yscher_data/endo_kbnac_spliced_unspliced.h5ad"
OUT = f"{ROOT}/analysis_data/dataset_qc_verification/endo_cellranger_vs_kb.json"
CH = 1500


def rows(g, a, b):
    ip = g["indptr"][a:b + 1]
    s, e = int(ip[0]), int(ip[-1])
    return sp.csr_matrix((g["data"][s:e], g["indices"][s:e], ip - ip[0]), shape=(b - a, int(g.attrs["shape"][1])))


fc, fk = h5py.File(CR, "r"), h5py.File(KB, "r")
oc, vc = ad.io.read_elem(fc["obs"]), ad.io.read_elem(fc["var"])
ok, vk = ad.io.read_elem(fk["obs"]), ad.io.read_elem(fk["var"])
sym_c = vc["gene_symbols"].astype(str).to_numpy() if "gene_symbols" in vc else vc.index.astype(str).to_numpy()
sym_k = vk["gene_symbol"].astype(str).to_numpy()
cells_k = pd_idx = ok.index
cpos = {c: i for i, c in enumerate(oc.index)}
sel = np.array([cpos.get(c, -1) for c in cells_k])
print("cells: CR", len(oc), "kb", len(ok), "kb cells found in CR", int((sel >= 0).sum()), flush=True)
# shared genes by symbol (first occurrence on each side)
ic = {}
for i, g in enumerate(sym_c):
    ic.setdefault(g, i)
ik = {}
for i, g in enumerate(sym_k):
    ik.setdefault(g, i)
shared = sorted(set(ic) & set(ik))
gc = np.array([ic[g] for g in shared]); gk = np.array([ik[g] for g in shared])
print("genes: CR", len(sym_c), "kb", len(sym_k), "shared symbols", len(shared), flush=True)

n = len(ok)
r = {k: np.full(n, np.nan) for k in ["tot_cr", "tot_kb", "tot_cr_sh", "tot_kb_sh", "ng_cr", "ng_kb", "pearson_sh", "ident_frac"]}
gsum_c = np.zeros(len(shared)); gsum_k = np.zeros(len(shared))
exact_cells = 0
for a in range(0, n, CH):
    b = min(a + CH, n)
    K = rows(fk["X"], a, b)
    idx = sel[a:b]
    ok_m = idx >= 0
    if not ok_m.any():
        continue
    # CR rows in kb order: read the contiguous min..max block, then index
    lo, hi = idx[ok_m].min(), idx[ok_m].max() + 1
    C = rows(fc["X"], lo, hi)[idx[ok_m] - lo]
    Kk = K[np.flatnonzero(ok_m)]
    ii = np.arange(a, b)[ok_m]
    r["tot_cr"][ii] = np.asarray(C.sum(1)).ravel(); r["tot_kb"][ii] = np.asarray(Kk.sum(1)).ravel()
    r["ng_cr"][ii] = np.diff(C.indptr); r["ng_kb"][ii] = np.diff(Kk.indptr)
    Cs, Ks = C[:, gc].toarray().astype(np.float32), Kk[:, gk].toarray().astype(np.float32)
    r["tot_cr_sh"][ii] = Cs.sum(1); r["tot_kb_sh"][ii] = Ks.sum(1)
    gsum_c += Cs.sum(0); gsum_k += Ks.sum(0)
    r["ident_frac"][ii] = (Cs == Ks).all(1)
    Cz, Kz = Cs - Cs.mean(1, keepdims=True), Ks - Ks.mean(1, keepdims=True)
    den = np.sqrt((Cz ** 2).sum(1) * (Kz ** 2).sum(1))
    r["pearson_sh"][ii] = np.divide((Cz * Kz).sum(1), den, out=np.full(len(ii), np.nan), where=den > 0)
    print(f"chunk {a}-{b}", flush=True)


def q(x):
    x = x[~np.isnan(x)]
    return dict(n=int(len(x)), min=float(x.min()), q25=float(np.quantile(x, .25)), median=float(np.median(x)), q75=float(np.quantile(x, .75)), max=float(x.max()))


res = dict(n_cells_kb=n, n_cells_matched=int((sel >= 0).sum()), n_genes_cr=len(sym_c), n_genes_kb=len(sym_k), n_shared_symbols=len(shared),
           total_counts_CR_all_genes=q(r["tot_cr"]), total_counts_kb_all_genes=q(r["tot_kb"]),
           total_counts_CR_shared_genes=q(r["tot_cr_sh"]), total_counts_kb_shared_genes=q(r["tot_kb_sh"]),
           ratio_kb_over_CR_shared=q(r["tot_kb_sh"] / r["tot_cr_sh"]), ratio_kb_over_CR_all=q(r["tot_kb"] / r["tot_cr"]),
           genes_detected_CR=q(r["ng_cr"]), genes_detected_kb=q(r["ng_kb"]),
           per_cell_pearson_over_shared_genes=q(r["pearson_sh"]),
           cells_with_identical_shared_gene_vector=int(np.nansum(r["ident_frac"])),
           per_gene_total_pearson_across_shared_genes=float(np.corrcoef(gsum_c, gsum_k)[0, 1]),
           per_gene_total_log1p_pearson=float(np.corrcoef(np.log1p(gsum_c), np.log1p(gsum_k))[0, 1]),
           genes_with_counts_CR_only=int(((gsum_c > 0) & (gsum_k == 0)).sum()), genes_with_counts_kb_only=int(((gsum_c == 0) & (gsum_k > 0)).sum()))
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
