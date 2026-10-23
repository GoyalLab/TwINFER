#!/usr/bin/env python3
"""
competitors_candidates.py

Competitor GRN methods on the PDF's candidate-pair universe (TF x ChIP-Atlas-bound target, yscher's chip_candidates.py lists).
Same cells as the TwinScore run: input is the genes x cells log1p(CP10k) table of the scored sample (gene_modules_T1/ for hPSC).
Regulators = the TFs of the candidate list, targets = every candidate gene detected in the scored cells; each method scores every
TF x gene pair, and the candidate pairs are read out at the end (assemble).

  rho        |Spearman| (unweighted ranks over all scored cells), as run_competitors_fatemap.py
  ppcor_shrunk  same, but the rank-correlation matrix is first shrunk toward the identity (Schafer-Strimmer: intensity a = sum Var(r_ij)/sum r_ij^2,
             Var(r_ij) = (1-r_ij^2)^2/(n-1)) and inverted with an eigendecomposition. Well-posed when genes >> cells.
  ppcor      |partial Spearman| exactly as run_competitors_fatemap.py::run_ppcor: pinv(R + 1e-8 I) of the rank-correlation matrix of all genes.
             NOTE: with genes >> cells (hPSC T1: ~12k genes, 885 cells) R is rank-deficient, so this is ill-conditioned; run as in the original on request.
             (A Schafer-Strimmer shrinkage version was written first and is kept commented out in the code.)
  genie3     RandomForest 1000 trees, max_features sqrt, one model per target, regulators = candidate TFs
  grnboost2  GradientBoosting lr 0.01, <=5000 trees, subsample 0.9, max_features 0.1, early stop (window 25)
PIDC (Julia NetworkInference.jl) is run separately: it needs all genes for the third-gene step.

run:       competitors_candidates.py run --method M ...      -> <out>/<M>.npz  (M: TF x gene importance matrix)
assemble:  competitors_candidates.py assemble ...           -> <out>/competitor_scores_candidate_pairs.csv
"""
import argparse, os, sys, time
import numpy as np, pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from competitors_all_genes import fit_target          # per-target RF / GBM fit, self-regulation excluded

log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def load(expr_csv, cand_file, regulators=None, extra_expr=None):
    genes, blocks = [], []
    for ch in pd.read_csv(expr_csv, index_col=0, chunksize=1000):
        genes += list(ch.index); blocks.append(ch.to_numpy(np.float32))
    if extra_expr:                                                       # extra genes x cells rows (e.g. a lowly detected TF not in the >=5% table)
        ex = pd.read_csv(extra_expr, index_col=0); genes += list(ex.index); blocks.append(ex.to_numpy(np.float32))
    X = np.vstack(blocks).T.copy()                                       # cells x genes
    cand = pd.read_csv(cand_file, sep="\t", header=None, names=["TF", "gene"])
    gi = {g: i for i, g in enumerate(genes)}
    cand = cand[cand.TF.isin(gi) & cand.gene.isin(gi)].reset_index(drop=True)
    tfs = sorted(cand.TF.unique()); tgt = sorted(set(cand.gene) | set(tfs))
    if regulators:                                                       # regulator panel = given list; targets = every gene in the table (no ChIP restriction)
        tfs = [t for t in dict.fromkeys(l.strip() for l in open(regulators) if l.strip()) if t in gi]; tgt = sorted(set(genes))
    X = X[:, [gi[g] for g in tgt]]
    return X, tgt, tfs, cand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "assemble"])
    ap.add_argument("--expr", required=True); ap.add_argument("--candidates", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--method", choices=["rho", "ppcor", "ppcor_shrunk", "genie3", "grnboost2"])
    ap.add_argument("--chunk", type=int, default=0); ap.add_argument("--n-chunks", type=int, default=1)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--max-targets", type=int, default=None, help="timing/testing")
    ap.add_argument("--regulators", default=None, help="text file of regulators; if given, targets are ALL genes of the table (pairs file only used by assemble)")
    ap.add_argument("--extra-expr", default=None, help="CSV (genes x cells, same cell order) appended to the table")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    X, genes, tfs, cand = load(a.expr, a.candidates, a.regulators, a.extra_expr)
    G, T = len(genes), len(tfs); gidx = {g: i for i, g in enumerate(genes)}; tf_gene = np.array([gidx[t] for t in tfs])
    log(f"{X.shape[0]:,} cells x {G:,} genes, {T} TFs, {len(cand):,} candidate pairs")

    if a.mode == "run":
        if a.method in ("rho", "ppcor", "ppcor_shrunk"):
            R = np.apply_along_axis(rankdata, 0, X).astype(np.float32); n = R.shape[0]
            Z = (R - R.mean(0)) / (R.std(0) + 1e-12)
            if a.method == "rho":
                M = np.abs((Z[:, tf_gene].T @ Z) / n).astype(np.float32)
            elif a.method == "ppcor_shrunk":
                # variant: Schafer-Strimmer shrinkage of the rank-correlation matrix toward the identity, then the precision matrix
                C = (Z.T @ Z / n).astype(np.float64); np.fill_diagonal(C, 1.0)
                off = ~np.eye(G, dtype=bool); var = (1 - C[off] ** 2) ** 2 / (n - 1)
                lam = float(np.clip(var.sum() / (C[off] ** 2).sum(), 0.0, 1.0)); log(f"ppcor_shrunk: shrinkage intensity {lam:.3f}")
                del off, var
                C = (1 - lam) * C + lam * np.eye(G)
                w, V = np.linalg.eigh(C); P = (V / np.maximum(w, 1e-6)) @ V.T; del V, C
                d = np.sqrt(np.diag(P)); PC = -P / np.outer(d, d); np.fill_diagonal(PC, 0.0)
                M = np.abs(PC[tf_gene]).astype(np.float32)
            else:
                # as run_competitors_fatemap.py::run_ppcor (the original code): rank-correlation matrix, pinv(R + 1e-8 I), abs partial correlation
                Rm = np.nan_to_num(np.corrcoef(np.apply_along_axis(rankdata, 0, X), rowvar=False)); np.fill_diagonal(Rm, 1.0)
                Pm = np.linalg.pinv(Rm + 1e-8 * np.eye(G))
                d = np.sqrt(np.outer(np.diag(Pm), np.diag(Pm))); PC = -Pm / np.where(d > 0, d, 1.0); np.fill_diagonal(PC, 0.0)
                M = np.abs(PC[tf_gene]).astype(np.float32)
                # --- replaced (kept for reference): Schafer-Strimmer shrunk precision matrix instead of pinv
                # C = (Z.T @ Z / n).astype(np.float64); np.fill_diagonal(C, 1.0)
                # off = ~np.eye(G, dtype=bool); var = (1 - C[off] ** 2) ** 2 / (n - 1)
                # lam = float(np.clip(var.sum() / (C[off] ** 2).sum(), 0.0, 1.0)); C = (1 - lam) * C + lam * np.eye(G)
                # w, V = np.linalg.eigh(C); P = (V / np.maximum(w, 1e-6)) @ V.T
                # d = np.sqrt(np.diag(P)); PC = -P / np.outer(d, d); np.fill_diagonal(PC, 0.0); M = np.abs(PC[tf_gene]).astype(np.float32)
            M[np.arange(T), tf_gene] = 0.0
            np.savez_compressed(f"{a.out}/{a.method}.npz", M=M, tfs=tfs, genes=genes); log(f"{a.method} done")
        else:
            from joblib import Parallel, delayed
            Xtf = np.ascontiguousarray(X[:, tf_gene])
            tgt = np.array_split(np.arange(G), a.n_chunks)[a.chunk]
            if a.max_targets:
                tgt = np.random.default_rng(0).choice(tgt, min(a.max_targets, len(tgt)), replace=False)
            t0 = time.time()
            res = Parallel(n_jobs=a.n_jobs)(delayed(fit_target)(Xtf, tf_gene, int(t), X[:, t], a.method) for t in tgt)
            dt = time.time() - t0
            log(f"{a.method}: {len(tgt)} targets in {dt:.0f}s ({dt / len(tgt):.2f}s/target wall, {a.n_jobs} workers) -> all {G:,} targets ~ {dt / len(tgt) * G / 60:.0f} min")
            tag = "_timing" if a.max_targets else ""
            np.savez_compressed(f"{a.out}/{a.method}{tag}_chunk{a.chunk}.npz", M=np.stack(res, 1), targets=tgt, tfs=tfs, genes=genes)
    else:
        tix = {t: i for i, t in enumerate(tfs)}
        out = cand[cand.TF.isin(tix) & cand.gene.isin(gidx)].reset_index(drop=True)      # read out only pairs whose TF is in the regulator panel
        ti = out.TF.map(tix).to_numpy(); gj = out.gene.map(gidx).to_numpy()
        for m in ("rho", "ppcor", "ppcor_shrunk", "genie3", "grnboost2"):
            if m in ("rho", "ppcor", "ppcor_shrunk"):
                p = f"{a.out}/{m}.npz"
                if not os.path.exists(p): continue
                M = np.load(p)["M"]
            else:
                fs = sorted(f for f in os.listdir(a.out) if f.startswith(f"{m}_chunk"))
                if not fs: continue
                M = np.full((T, G), np.nan, np.float32)
                for f in fs:
                    z = np.load(f"{a.out}/{f}", allow_pickle=True); M[:, z["targets"]] = z["M"]
            out[m] = M[ti, gj]
        out.to_csv(f"{a.out}/competitor_scores_candidate_pairs.csv", index=False); log(f"wrote competitor_scores_candidate_pairs.csv {out.shape}")


if __name__ == "__main__":
    main()
