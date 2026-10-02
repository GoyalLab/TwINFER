"""How does the no-filter z_het-fixed twinScore (the real-networks tuning:
`twinScore + heterogeneity_penalty + panel_z(|z_het|)`) do on the 9 LARRY gene
sets, scored the same way as benchmark_methods.ipynb / bench_core?

The analytic twinScore + its component terms are already on disk per gene set in
resources/analytic_infer/{gs}/ranked_edges.csv (from run_analytic_twinfer.py --
shuffle-free z-scores). This just adds the z_het-fix term and re-scores.

    method "analytic_twinfer"            = shipped twinScore, analytic z   (existing)
    method "analytic_twinfer_zhet_fixed" = + heterogeneity_penalty + s(|z_het|)   (new)

Universe = (gene-set TFs) x panel, self excluded; y_true = CollecTRI directed
edge. Metrics: top-k precision (=F1@k), EPR, AUPRC, AUPRC/random (200-draw MC).

Output: resources/benchmark/benchmark_zhet_fixed_results.csv
        resources/benchmark/benchmark_zhet_fixed_heatmap.png
"""
import zlib  # [2026-10-01 added]
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
from twinfer.utils.paths import get_data_root as _res_gdr  # [2026-10-01 LARRY resources/ moved out of the code tree to analysis_data/paper_analysis/larry_hematopoiesis_validation/resources (user)]
RES_HERE = f"{_res_gdr()}/paper_analysis/larry_hematopoiesis_validation"
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] R = os.path.join(HERE, "resources")
R = os.path.join(RES_HERE, "resources")
OUT = os.path.join(R, "benchmark")
GENE_SETS = ["variability_high", "variability_mid", "variability_low",
             "detection_high", "detection_mid", "detection_low",
             "correlation_high", "correlation_mid", "correlation_low"]
N_RANDOM, SEED = 200, 0

ct = pd.read_csv(f"{R}/collectri_mouse.tsv", sep="\t")
ct = ct[ct.source_genesymbol != ct.target_genesymbol]
CE = set(zip(ct.source_genesymbol, ct.target_genesymbol))
gene_sets = json.load(open(f"{R}/gene_sets.json"))
detail = json.load(open(f"{R}/gene_sets_detail.json"))


def panel_z(v):
    v = np.asarray(v, float)
    f = np.isfinite(v)
    o = np.zeros(v.shape)
    if f.any():
        x = v[f]
        sd = x.std(ddof=0)
        o[f] = 0.0 if sd == 0 else (x - x.mean()) / sd
    return o


def universe_for(gs):
    crit, lvl = gs.rsplit("_", 1)
    panel = gene_sets[gs]
    tfs = [tf for tf, _ in detail[crit][lvl] if tf in panel]
    U = [(a, b) for a in tfs for b in panel if a != b]
    y = np.array([1 if p in CE else 0 for p in U], int)
    return U, y, set(tfs)


def align(edge_score, U):
    floor = min(edge_score.values()) if edge_score else 0.0
    return np.array([edge_score.get(p, floor) for p in U], float)


def f1_at_k(y, sc):
    k = int(y.sum()); n = len(y)
    if k == 0 or k >= n:
        return np.nan
    idx = np.argsort(-sc, kind="stable")[:k]
    return float(y[idx].sum()) / k


def metrics(y, sc, seed):
    au = average_precision_score(y, sc)
    f1 = f1_at_k(y, sc)
    k, n = int(y.sum()), len(y)
    ep = f1
    epr = ep / (k / n) if k else np.nan
    rng = np.random.default_rng(seed)
    aur = np.mean([average_precision_score(y, rng.permutation(sc)) for _ in range(N_RANDOM)])
    return dict(early_precision=ep, f1=f1, epr=epr, auprc=au,
                auprc_random=aur, auprc_ratio=au / aur if aur > 0 else np.nan)


rows = []
for gs in GENE_SETS:
    U, y, tfs = universe_for(gs)
    d = pd.read_csv(f"{R}/analytic_infer/{gs}/ranked_edges.csv")
    d = d[d.gene_1.isin(tfs) & (d.gene_1 != d.gene_2)]
    zf = d.twinScore.to_numpy() + d.heterogeneity_penalty.to_numpy() + panel_z(d.z_het.abs())
    for method, col in (("analytic_twinfer", d.twinScore.to_numpy()),
                        ("analytic_twinfer_zhet_fixed", zf)):
        es = {(a, b): float(v) for a, b, v in zip(d.gene_1, d.gene_2, col)}
        # [2026-10-01 commented out: hash() of strings is randomised per process (PYTHONHASHSEED) so this seed was not reproducible; zlib.crc32 of the repr is stable] m = metrics(y, align(es, U), SEED + hash((gs, method)) % 2**31)
        m = metrics(y, align(es, U), SEED + zlib.crc32(repr((gs, method)).encode()) % 2**31)
        rows.append(dict(gene_set=gs, method=method, n_universe=len(U),
                         n_curated=int(y.sum()), density=float(y.mean()), **m))
    print(f"{gs:<18} universe={len(U):>5} curated={int(y.sum()):>4}", flush=True)

new = pd.DataFrame(rows)

# pull the other methods' rows from the committed benchmark for a combined view
old = pd.read_csv(f"{OUT}/benchmark_results.csv")
old = old[old.method.isin(["perm_twinfer", "rho", "ppcor", "pidc", "genie3", "grnboost2"])]
comb = pd.concat([new, old], ignore_index=True)
comb.to_csv(f"{OUT}/benchmark_zhet_fixed_results.csv", index=False)
print(f"\nwrote {OUT}/benchmark_zhet_fixed_results.csv")

order = ["analytic_twinfer", "analytic_twinfer_zhet_fixed", "perm_twinfer",
         "rho", "ppcor", "pidc", "genie3", "grnboost2"]
for metric in ("early_precision", "auprc", "auprc_ratio"):
    piv = comb.pivot(index="gene_set", columns="method", values=metric).reindex(GENE_SETS)[order]
    print(f"\n=== {metric} ===\n{piv.round(3).to_string()}")
print("\n=== mean across gene sets ===")
print(comb.groupby("method")[["early_precision", "epr", "auprc", "auprc_ratio"]]
      .mean().reindex(order).round(3).to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, metric, title in zip(axes, ("epr", "auprc_ratio"), ("EPR", "AUPRC / random")):
    piv = comb.pivot(index="gene_set", columns="method", values=metric).reindex(GENE_SETS)[order]
    A = piv.to_numpy()
    im = ax.imshow(A, aspect="auto", cmap="RdBu_r", vmin=0, vmax=max(2.0, np.nanmax(A)))
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_yticks(range(len(GENE_SETS))); ax.set_yticklabels(GENE_SETS)
    ax.set_title(f"{title} (>1 beats random)")
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if np.isfinite(A[i, j]):
                ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.axvline(1.5, color="k", lw=1.5)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(f"{OUT}/benchmark_zhet_fixed_heatmap.png", dpi=140, bbox_inches="tight")
print(f"wrote {OUT}/benchmark_zhet_fixed_heatmap.png")
