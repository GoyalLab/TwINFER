from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import json
import os
import zlib

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

HERE = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation'
from twinfer.utils.paths import get_data_root as _res_gdr  # [2026-10-01 LARRY resources/ moved out of the code tree to analysis_data/paper_analysis/larry_hematopoiesis_validation/resources (user)]
RES_HERE = f"{_res_gdr()}/paper_analysis/larry_hematopoiesis_validation"
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] NETWORKS_DIR = os.path.join(HERE, "resources", "networks")
NETWORKS_DIR = os.path.join(RES_HERE, "resources", "networks")
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] INFER_DIR = os.path.join(HERE, "resources", "infer_results")
INFER_DIR = os.path.join(RES_HERE, "resources", "infer_results")
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] COLLECTRI_PATH = os.path.join(HERE, "resources", "collectri_mouse.tsv")
COLLECTRI_PATH = os.path.join(RES_HERE, "resources", "collectri_mouse.tsv")
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] GENE_SETS_PATH = os.path.join(HERE, "resources", "gene_sets.json")
GENE_SETS_PATH = os.path.join(RES_HERE, "resources", "gene_sets.json")
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] GENE_SETS_DETAIL_PATH = os.path.join(HERE, "resources", "gene_sets_detail.json")
GENE_SETS_DETAIL_PATH = os.path.join(RES_HERE, "resources", "gene_sets_detail.json")
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] OUT_DIR = os.path.join(HERE, "resources", "benchmark")
OUT_DIR = os.path.join(RES_HERE, "resources", "benchmark")
os.makedirs(OUT_DIR, exist_ok=True)

GENE_SETS = ["variability_high", "variability_mid", "variability_low",
             "detection_high", "detection_mid", "detection_low",
             "correlation_high", "correlation_mid", "correlation_low"]
COMPETITOR_METHODS = ["rho", "ppcor", "pidc", "genie3", "grnboost2"]
TWINFER_METHODS = ["analytic_twinfer", "analytic_twinfer_tuned", "perm_twinfer"]
ALL_METHODS = TWINFER_METHODS + COMPETITOR_METHODS

N_RANDOM = 200   # Monte Carlo draws for the AUPRC/F1 random baseline (matches benchmark_all.py)
SEED = 0
rng = np.random.default_rng(SEED)

collectri = pd.read_csv(COLLECTRI_PATH, sep="\t")
collectri = collectri[collectri["source_genesymbol"] != collectri["target_genesymbol"]]
collectri_edges = set(zip(collectri["source_genesymbol"], collectri["target_genesymbol"]))
print(f"CollecTRI: {len(collectri_edges):,} directed edges (self-pairs stripped)")

gene_sets = json.load(open(GENE_SETS_PATH))
gene_sets_detail = json.load(open(GENE_SETS_DETAIL_PATH))


def build_universe(gene_set):
    """(TFs x panel, self-pairs excluded) for one gene set -- matches the competitor CSVs' own
    edge universe exactly, so every method is scored on identical ground."""
    criterion, level = gene_set.rsplit("_", 1)
    panel = gene_sets[gene_set]
    tfs = [tf for tf, _ in gene_sets_detail[criterion][level] if tf in panel]
    universe = [(a, b) for a in tfs for b in panel if a != b]
    y_true = np.array([1 if pair in collectri_edges else 0 for pair in universe], dtype=int)
    return universe, y_true, tfs


universes = {}
for gs in GENE_SETS:
    universe, y_true, tfs = build_universe(gs)
    universes[gs] = dict(universe=universe, y_true=y_true, tfs=tfs)
    print(f"{gs:<18} universe={len(universe):>5}  curated={int(y_true.sum()):>4}  "
          f"density={y_true.mean():.4f}  TFs={len(tfs)}")

def _align_scores(edge_score, universe):
    """edge_score: dict[(a,b)] -> float. Returns y_score in universe order, missing pairs filled
    with min(edge_score) (least confident -- the pair was never called, not tied for best)."""
    if not edge_score:
        return np.zeros(len(universe), dtype=float)
    floor = min(edge_score.values())
    return np.array([edge_score.get(pair, floor) for pair in universe], dtype=float)


def load_competitor(method, gene_set, universe):
    p = os.path.join(NETWORKS_DIR, f"{method}_{gene_set}.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    edge_score = {(r.TF, r.target): float(r.importance) for r in df.itertuples(index=False)}
    return _align_scores(edge_score, universe)


def _load_ranked_edges(path, universe, tfs):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    tf_set = set(tfs)
    df = df[df["gene_1"].isin(tf_set) & (df["gene_1"] != df["gene_2"])]
    edge_score = {(r.gene_1, r.gene_2): float(r.twinScore) for r in df.itertuples(index=False)}
    return _align_scores(edge_score, universe)


def load_analytic_twinfer(gene_set, universe, tfs):
    # run_analytic_twinfer.py -- shuffle-free z-scores (analytic_zscores.py), cp10k input
    # [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] return _load_ranked_edges(os.path.join(HERE, "resources", "analytic_infer", gene_set,
    return _load_ranked_edges(os.path.join(RES_HERE, "resources", "analytic_infer", gene_set,
                                           "ranked_edges.csv"), universe, tfs)


def load_analytic_twinfer_tuned(gene_set, universe, tfs):
    # run_analytic_twinfer_tuned.py -- unweighted correlations + yscher's tuned TwinScore + gate
    # [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] return _load_ranked_edges(os.path.join(HERE, "resources", "analytic_infer_tuned", gene_set,
    return _load_ranked_edges(os.path.join(RES_HERE, "resources", "analytic_infer_tuned", gene_set,
                                           "ranked_edges.csv"), universe, tfs)


def load_perm_twinfer(gene_set, universe, tfs):
    # 5000-shuffle ALL_PAIRS run (run_infer_correlation_high.py); may be absent for some sets
    return _load_ranked_edges(os.path.join(INFER_DIR, f"{gene_set}_t2_t4_allpairs",
                                           "ranked_edges.csv"), universe, tfs)


scores = {}
for gs in GENE_SETS:
    universe = universes[gs]["universe"]
    tfs = universes[gs]["tfs"]
    got = {}
    for name, loader in (("analytic_twinfer", load_analytic_twinfer),
                         ("analytic_twinfer_tuned", load_analytic_twinfer_tuned),
                         ("perm_twinfer", load_perm_twinfer)):
        s = loader(gs, universe, tfs)
        if s is not None:
            got[name] = s
    for m in COMPETITOR_METHODS:
        s = load_competitor(m, gs, universe)
        if s is not None:
            got[m] = s
    scores[gs] = got
    print(f"{gs:<18} methods present: {list(got)}")

def f1_at_k(y_true, y_score):
    """F1 at the SAME fixed operating point EPR uses: top-k predictions, k = |E_true| (the number
    of curated edges in this universe). Note: when k equals the positive count, precision@k,
    recall@k, and F1@k are all numerically identical (a direct property of that choice of k) --
    so this will equal early_precision below. Reported anyway as its own column since it was asked
    for explicitly and it's still a meaningful, standard operating point (not derived from a
    threshold search like a curve-wide best-F1 would be)."""
    k = int(y_true.sum())
    n = len(y_true)
    if k == 0 or k >= n:
        return float("nan")
    top_k_idx = np.argsort(-y_score, kind="stable")[:k]
    tp = float(y_true[top_k_idx].sum())
    precision = tp / k
    recall = tp / k  # k == n_positives here, so recall@k == precision@k
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def early_precision_ratio(y_true, y_score):
    """BEELINE-style EPR: top-k predictions (k = number of curated edges in this universe),
    precision there, divided by the random-ranking expectation at that k (= edge density, exact
    via the hypergeometric mean -- no Monte Carlo needed for this one)."""
    k = int(y_true.sum())
    n = len(y_true)
    if k == 0 or k >= n:
        return float("nan"), float("nan")
    top_k_idx = np.argsort(-y_score, kind="stable")[:k]
    early_precision = float(y_true[top_k_idx].sum()) / k
    density = k / n
    epr = early_precision / density if density > 0 else float("nan")
    return early_precision, epr


def observed_metrics(y_true, y_score):
    auprc = average_precision_score(y_true, y_score)
    f1 = f1_at_k(y_true, y_score)
    ep, epr = early_precision_ratio(y_true, y_score)
    return dict(auprc=auprc, f1=f1, early_precision=ep, epr=epr)


def random_baseline(y_true, y_score, n_random=N_RANDOM, seed=SEED):
    """Monte Carlo: reshuffle the SAME score values across the SAME universe n_random times (labels
    fixed), recompute AUPRC/F1@k each draw, return their means as the random-ranking expectation.
    EPR's random baseline is exact (see early_precision_ratio) and not reshuffled here -- F1@k's
    Monte Carlo mean should converge to the same density value, which is a useful internal
    consistency check (f1_ratio should land close to epr)."""
    local_rng = np.random.default_rng(seed)
    auprcs = np.empty(n_random)
    f1s = np.empty(n_random)
    for i in range(n_random):
        shuffled = local_rng.permutation(y_score)
        auprcs[i] = average_precision_score(y_true, shuffled)
        f1s[i] = f1_at_k(y_true, shuffled)
    return dict(auprc_random=float(auprcs.mean()), f1_random=float(f1s.mean()))


# smoke test: a perfect ranking should score AUPRC=1, EPR >> 1; a reversed ranking should score low
_yt = np.array([1, 1, 0, 0, 0])
_perfect = np.array([5, 4, 3, 2, 1])
_reversed = np.array([1, 2, 3, 4, 5])
print("perfect ranking:", observed_metrics(_yt, _perfect))
print("reversed ranking:", observed_metrics(_yt, _reversed))
assert observed_metrics(_yt, _perfect)["auprc"] == 1.0
assert observed_metrics(_yt, _perfect)["epr"] > observed_metrics(_yt, _reversed)["epr"]
assert observed_metrics(_yt, _perfect)["f1"] == observed_metrics(_yt, _perfect)["early_precision"]
print("OK")

rows = []
for gs in GENE_SETS:
    y_true = universes[gs]["y_true"]
    for method, y_score in scores[gs].items():
        obs = observed_metrics(y_true, y_score)
        # [2026-10-01 commented out: Python hash() of a tuple of strings changes every process (PYTHONHASHSEED), so the Monte Carlo random baseline was not reproducible; crc32 is stable]
        # rnd = random_baseline(y_true, y_score, seed=SEED + hash((gs, method)) % (2**31))
        rnd = random_baseline(y_true, y_score, seed=SEED + zlib.crc32(f"{gs}|{method}".encode()) % (2**31))
        rows.append(dict(
            gene_set=gs, method=method,
            n_universe=len(y_true), n_curated=int(y_true.sum()),
            density=float(y_true.mean()),
            early_precision=obs["early_precision"],            # top-k precision  (k = n_curated)
            f1=obs["f1"],                                      # top-k F1  (== early_precision at this k)
            epr=obs["epr"],                                    # top-k precision / density
            auprc=obs["auprc"],
            auprc_random=rnd["auprc_random"],
            auprc_ratio=obs["auprc"] / rnd["auprc_random"] if rnd["auprc_random"] > 0 else float("nan"),
            f1_random=rnd["f1_random"],
            f1_ratio=obs["f1"] / rnd["f1_random"] if rnd["f1_random"] > 0 else float("nan"),
        ))
    print(f"{gs}: done")

results = pd.DataFrame(rows)
results.head(len(ALL_METHODS))

out_path = os.path.join(OUT_DIR, "benchmark_results.csv")
results.to_csv(out_path, index=False)
print(f"wrote {out_path}  ({len(results)} rows)")

present = [m for m in ALL_METHODS if m in set(results.method)]
for metric in ("early_precision", "f1", "auprc", "auprc_ratio", "epr"):
    print(f"\n=== {metric} ===")
    piv = results.pivot(index="gene_set", columns="method", values=metric).reindex(GENE_SETS)
    print(piv[[c for c in present if c in piv.columns]].round(3))

print("\n=== mean across gene sets, per method ===")
summary = results.groupby("method")[["early_precision", "f1", "epr", "auprc", "auprc_ratio"]].mean()
print(summary.reindex(present).round(3))
summary.reindex(present).to_csv(os.path.join(OUT_DIR, "benchmark_summary_by_method.csv"))
print("wrote benchmark_summary_by_method.csv")

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, metric, title in zip(axes, ("epr", "auprc_ratio"), ("EPR", "AUPRC / random")):
    pivot = results.pivot(index="gene_set", columns="method", values=metric)[ALL_METHODS].loc[GENE_SETS]
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="RdBu_r",
                    vmin=0, vmax=max(2.0, float(np.nanmax(pivot.to_numpy()))))
    ax.set_xticks(range(len(ALL_METHODS)))
    ax.set_xticklabels(ALL_METHODS, rotation=45, ha="right")
    ax.set_yticks(range(len(GENE_SETS)))
    ax.set_yticklabels(GENE_SETS)
    ax.set_title(f"{title} (>1 beats random)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.to_numpy()[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
fig_path = os.path.join(OUT_DIR, "benchmark_heatmaps.pdf")
plt.savefig(fig_path, bbox_inches="tight")
print(f"wrote {fig_path}")
plt.show()