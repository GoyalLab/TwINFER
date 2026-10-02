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
TWINFER_METHODS = ["analytic_twinfer", "perm_twinfer"]
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
    print(piv[[c for c in present if c in piv.columns]].round(3).to_string())

print("\n=== mean across gene sets, per method ===")
summary = results.groupby("method")[["early_precision", "f1", "epr", "auprc", "auprc_ratio"]].mean()
print(summary.reindex(present).round(3).to_string())
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
NEW_SCORE_GENE_SET = "correlation_high"
GATED_CALL_THRESHOLD = 1.645


def _panel_z(values):
    """Matches calculate_twin_score's own _panel_z in correlation_functions.py exactly."""
    values = np.asarray(values, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return result
    x = values[finite]
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=0))
    if not np.isfinite(sd) or sd == 0:
        result[finite] = 0.0
    else:
        result[finite] = (x - mu) / sd
    return result


def _null_unit(observed, null_values, use_absolute=True):
    """Matches infer.py's own _null_unit exactly."""
    arr = np.asarray(null_values, dtype=float)
    arr = arr[np.isfinite(arr)]
    obs = float(observed)
    if use_absolute:
        obs = abs(obs)
        arr = np.abs(arr)
    if arr.size < 2:
        return np.nan
    mu = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return np.nan
    return float((obs - mu) / sd)


gs = NEW_SCORE_GENE_SET
result_dir = os.path.join(INFER_DIR, f"{gs}_t2_t4_allpairs_50core")

ranked = pd.read_csv(os.path.join(result_dir, "ranked_edges.csv"))
twin_delta_t1 = pd.read_csv(os.path.join(result_dir, "corr_twin_delta_t1.csv"), index_col=0)
random_delta_t1 = pd.read_csv(os.path.join(result_dir, "corr_random_delta_t1.csv"), index_col=0)
random_delta_t2 = pd.read_csv(os.path.join(result_dir, "corr_random_delta_t2.csv"), index_col=0)
raw_nulls = np.load(os.path.join(result_dir, "raw_nulls.npz"))
z_scores_by_step = json.load(open(os.path.join(result_dir, "z_scores_by_step.json")))
z_reg_gated = z_scores_by_step["z_reg_gated"]

universe = universes[gs]["universe"]
undirected_pairs = sorted({tuple(sorted(p)) for p in universe})
print(f"{gs}: {len(universe)} directed pairs, {len(undirected_pairs)} undirected pairs "
      f"for panel standardization")

# ---- z(|rho_hat_Delta(t1)|)
u_abs_rho_delta_t1 = {}
for a, b in undirected_pairs:
    obs = twin_delta_t1.loc[a, b]
    null_key = f"rho_delta_het_t1__{a}__{b}"
    if null_key not in raw_nulls:
        null_key = f"rho_delta_het_t1__{b}__{a}"
    u_abs_rho_delta_t1[(a, b)] = (
        _null_unit(obs, raw_nulls[null_key], use_absolute=True) if null_key in raw_nulls else np.nan)

pairs_order = list(u_abs_rho_delta_t1.keys())
z_vals = _panel_z(np.array([u_abs_rho_delta_t1[p] for p in pairs_order]))
z_abs_rho_delta_t1 = dict(zip(pairs_order, z_vals))
n_missing_null = sum(1 for v in u_abs_rho_delta_t1.values() if not np.isfinite(v))
print(f"z(|rho_delta(t1)|): {n_missing_null}/{len(pairs_order)} pairs missing a matching saved "
      f"null (no Step-2 delta null was drawn for that pair in the ALL_PAIRS run -- left NaN, "
      f"excluded from panel_z's own mean/std, not treated as 0)")

# ---- z(|ref_Delta drift contrast|)
drift_raw = {(a, b): abs(float(random_delta_t2.loc[a, b]) - float(random_delta_t1.loc[a, b]))
             for a, b in undirected_pairs}
z_vals = _panel_z(np.array([drift_raw[p] for p in pairs_order]))
z_drift = dict(zip(pairs_order, z_vals))

print("undirected-pair terms built OK")
def _lookup_undirected(d, a, b):
    return d.get((a, b), d.get((b, a), np.nan))


def _lookup_zreg_gated(a, b):
    for key in (f"{a}__{b}", f"{b}__{a}"):
        if key in z_reg_gated:
            return z_reg_gated[key]
    return np.nan


ranked_idx = ranked.set_index(["gene_1", "gene_2"])

# reg(g): mean SIGNED z_gamma(g, w) over every partner w TwINFER computed for g, from the FULL
# native pair set (not restricted to our benchmarking universe -- see markdown above).
z_gamma_by_pair = {(r.gene_1, r.gene_2): r.z_gamma for r in ranked.itertuples(index=False)}
all_genes = sorted({g for pair in z_gamma_by_pair for g in pair})
reg = {}
for g in all_genes:
    vals = [z_gamma_by_pair[(g, w)] for w in all_genes if w != g and (g, w) in z_gamma_by_pair]
    reg[g] = float(np.nanmean(vals)) if vals else np.nan

# A handful of universe pairs are absent from TwINFER's own ranked_edges.csv entirely -- even
# under ALL_PAIRS, some pairs never produce a usable classification (e.g. NaN correlation from
# near-constant expression). Confirmed by direct check: 10/645 pairs missing (5 undirected pairs,
# both directions) for correlation_high. These get an all-NaN score row, not a crash, and are
# floor-filled like any other missing-method pair when building y_score below.
missing_from_ranked = []

rows = []
for a, b in universe:
    if (a, b) not in ranked_idx.index:
        missing_from_ranked.append((a, b))
        rows.append(dict(gene_1=a, gene_2=b, new_twinscore=np.nan,
                          z_abs_rho_t1=np.nan, z_abs_rho_delta_t1=np.nan,
                          z_abs_rho_change=np.nan, z_drift=np.nan,
                          z_gamma=np.nan, reg_diff=np.nan, z_reg_gated=np.nan,
                          call_bonus=np.nan))
        continue
    row = ranked_idx.loc[(a, b)]
    z_abs_t1 = float(row["z_abs_rho_t1"])
    z_abs_change = float(row["z_abs_rho_change"])
    z_gamma_signed = float(row["z_gamma"])
    z_ddag = z_gamma_signed  # confirmed by the user: z_Ddag IS z(gamma)

    term_rho_delta = _lookup_undirected(z_abs_rho_delta_t1, a, b)
    term_drift = _lookup_undirected(z_drift, a, b)
    zg = _lookup_zreg_gated(a, b)
    call_bonus = 100.0 if (np.isfinite(zg) and abs(zg) > GATED_CALL_THRESHOLD) else 0.0
    reg_diff = reg.get(a, np.nan) - reg.get(b, np.nan)

    score = (z_abs_t1 - term_rho_delta - z_abs_change - term_drift
             - abs(z_gamma_signed) + 0.5 * z_ddag + reg_diff + call_bonus)
    rows.append(dict(gene_1=a, gene_2=b, new_twinscore=score,
                      z_abs_rho_t1=z_abs_t1, z_abs_rho_delta_t1=term_rho_delta,
                      z_abs_rho_change=z_abs_change, z_drift=term_drift,
                      z_gamma=z_gamma_signed, reg_diff=reg_diff, z_reg_gated=zg,
                      call_bonus=call_bonus))

new_score_df = pd.DataFrame(rows)
n_nan = int(new_score_df["new_twinscore"].isna().sum())
print(f"{gs}: {len(new_score_df)} scored pairs, "
      f"{int((new_score_df['call_bonus'] > 0).sum())} called "
      f"(|z_reg_gated|>{GATED_CALL_THRESHOLD})")
print(f"{len(missing_from_ranked)} pairs entirely absent from TwINFER's ranked_edges.csv "
      f"(not classifiable, not just missing a null): {missing_from_ranked}")
print(f"{n_nan} total NaN-score pairs -- floor-filled (least confident) before benchmarking, "
      f"not excluded from the universe")
new_score_df.sort_values("new_twinscore", ascending=False).head(10)
edge_score_new = {(r.gene_1, r.gene_2): r.new_twinscore for r in new_score_df.itertuples(index=False)
                  if np.isfinite(r.new_twinscore)}
y_score_new = _align_scores(edge_score_new, universe)

y_true_gs = universes[gs]["y_true"]
obs_new = observed_metrics(y_true_gs, y_score_new)
rnd_new = random_baseline(y_true_gs, y_score_new, seed=SEED + 999)

print(f"new TwinScore on {gs}:")
print(f"  AUPRC = {obs_new['auprc']:.3f}  (ratio {obs_new['auprc'] / rnd_new['auprc_random']:.2f})")
print(f"  F1@k  = {obs_new['f1']:.3f}  (ratio {obs_new['f1'] / rnd_new['f1_random']:.2f})")
print(f"  EPR   = {obs_new['epr']:.3f}  (early_precision = {obs_new['early_precision']:.3f})")
print(f"\n  For reference, BEST_SCORER.md's own reported 'approved gzu functions' number for "
      f"this exact formula on correlation_high: 3.98x base precision (their own x-base metric, "
      f"not identical in construction to our EPR/AUPRC-ratio, so treat as a ballpark check, not "
      f"an exact-match target).")

comp = results[results["gene_set"] == gs][["method", "auprc", "auprc_ratio", "f1", "f1_ratio", "epr"]].copy()
comp = pd.concat([comp, pd.DataFrame([dict(
    method="new_twinscore", auprc=obs_new["auprc"], auprc_ratio=obs_new["auprc"] / rnd_new["auprc_random"],
    f1=obs_new["f1"], f1_ratio=obs_new["f1"] / rnd_new["f1_random"], epr=obs_new["epr"])])],
    ignore_index=True)
comp = comp.sort_values("epr", ascending=False).reset_index(drop=True)
comp.round(3)
def s(v):
    """Exactly as found in helpers/twinscore_table.py: sample std (ddof=1), NaNs filled to
    min(standardized values) - 1, not 0 or the mean."""
    v = np.asarray(v, dtype=float)
    f = np.isfinite(v)
    o = (v - np.nanmean(v)) / max(np.nanstd(v, ddof=1), 1e-12)
    return np.where(f, o, np.nanmin(o[f]) - 1.0)


direction_z_scores = z_scores_by_step["direction_z_scores"]  # already loaded in Section 7


def _lookup_direction_z(a, b):
    return direction_z_scores.get(f"{a}__{b}", np.nan)


ranked_u = ranked.set_index(["gene_1", "gene_2"])

zr1, zr2, ch_abs, drift_vals, inh_vals, gam_signed = [], [], [], [], [], []
for a, b in universe:
    if (a, b) not in ranked_u.index:
        zr1.append(np.nan); zr2.append(np.nan); ch_abs.append(np.nan)
        drift_vals.append(np.nan); inh_vals.append(np.nan); gam_signed.append(np.nan)
        continue
    row = ranked_u.loc[(a, b)]
    zr1.append(float(row["u_abs_rho_t1"]))
    zr2.append(float(row["u_abs_rho_t2"]))
    ch_abs.append(float(row["u_abs_rho_change"]))
    drift_vals.append(_lookup_undirected(drift_raw, a, b))
    z_xy = _lookup_direction_z(a, b)
    z_yx = _lookup_direction_z(b, a)
    inh_vals.append(np.nanmin([abs(z_xy) if np.isfinite(z_xy) else np.nan,
                                abs(z_yx) if np.isfinite(z_yx) else np.nan]))
    gam_signed.append(float(row["u_gamma"]))

zr1, zr2 = np.array(zr1), np.array(zr2)
min_zr = np.minimum(zr1, zr2)
ch_abs = np.array(ch_abs)
drift_vals = np.array(drift_vals)
inh_vals = np.array(inh_vals)
gam_signed = np.array(gam_signed)

zreg_vals = np.array([_lookup_zreg_gated(a, b) for a, b in universe])
defined = np.isfinite(zr1) & np.isfinite(zr2) & np.isfinite(zreg_vals)
existence_pass = (zr1 > 2.576) | (zr2 > 2.576)
regulation_pass = np.abs(zreg_vals) > 1.645
PASS = defined & existence_pass & regulation_pass

sym_score_raw = s(min_zr) - s(ch_abs) - s(drift_vals) - s(inh_vals) + s(gam_signed)
sym_score = np.where(PASS, sym_score_raw, -1e12)

n_nan_inh = int(np.sum(~np.isfinite(inh_vals)))
print(f"{gs}: SYM computed for {len(universe)} pairs "
      f"({n_nan_inh} with a missing INH -- no direction_z_score for one or both orderings, "
      f"filled via s()'s own min-1 rule)")
print(f"filter (twinscore_table.py's own PASS gate): {int(PASS.sum())}/{len(universe)} pairs pass "
      f"(existence: |z_rho_t1| or |z_rho_t2| > 2.576; regulation: |z_reg_gated| > 1.645) -- "
      f"failing pairs scored -1e12, ranked below everything that passes")

sym_df = pd.DataFrame({"gene_1": [p[0] for p in universe], "gene_2": [p[1] for p in universe],
                        "sym_twinscore": sym_score, "min_zr": min_zr, "ch_abs": ch_abs,
                        "drift": drift_vals, "inh": inh_vals, "gamma_signed": gam_signed})
sym_df.sort_values("sym_twinscore", ascending=False).head(10)
edge_score_sym = {(r.gene_1, r.gene_2): r.sym_twinscore for r in sym_df.itertuples(index=False)
                  if np.isfinite(r.sym_twinscore)}
y_score_sym = _align_scores(edge_score_sym, universe)

obs_sym = observed_metrics(y_true_gs, y_score_sym)
rnd_sym = random_baseline(y_true_gs, y_score_sym, seed=SEED + 4242)

print(f"SYM TwinScore on {gs}:")
print(f"  AUPRC = {obs_sym['auprc']:.3f}  (ratio {obs_sym['auprc'] / rnd_sym['auprc_random']:.2f})")
print(f"  F1@k  = {obs_sym['f1']:.3f}  (ratio {obs_sym['f1'] / rnd_sym['f1_random']:.2f})")
print(f"  EPR   = {obs_sym['epr']:.3f}  (early_precision = {obs_sym['early_precision']:.3f})")

comp2 = comp.copy()
comp2 = pd.concat([comp2, pd.DataFrame([dict(
    method="sym_twinscore", auprc=obs_sym["auprc"], auprc_ratio=obs_sym["auprc"] / rnd_sym["auprc_random"],
    f1=obs_sym["f1"], f1_ratio=obs_sym["f1"] / rnd_sym["f1_random"], epr=obs_sym["epr"])])],
    ignore_index=True)
comp2 = comp2.sort_values("epr", ascending=False).reset_index(drop=True)
comp2.round(3)
def build_universe_allgenes(gene_set):
    panel = gene_sets[gene_set]
    universe = [(a, b) for a in panel for b in panel if a != b]
    y_true = np.array([1 if pair in collectri_edges else 0 for pair in universe], dtype=int)
    return universe, y_true


universes_allgenes = {}
for gene_set in GENE_SETS:
    universe, y_true = build_universe_allgenes(gene_set)
    universes_allgenes[gene_set] = dict(universe=universe, y_true=y_true)
    print(f"{gene_set:<18} universe={len(universe):>5}  curated={int(y_true.sum()):>4}  "
          f"density={y_true.mean():.4f}")


def load_competitor_allgenes(method, gene_set, universe):
    df = pd.read_csv(os.path.join(NETWORKS_DIR, f"{method}_{gene_set}_allgenes.csv"))
    edge_score = {(r.TF, r.target): float(r.importance) for r in df.itertuples(index=False)}
    return _align_scores(edge_score, universe)


def load_twinfer_allgenes(gene_set, universe):
    df = pd.read_csv(os.path.join(INFER_DIR, f"{gene_set}_t2_t4_allpairs_50core", "ranked_edges.csv"))
    edge_score = {(r.gene_1, r.gene_2): float(r.twinScore) for r in df.itertuples(index=False)}
    return _align_scores(edge_score, universe)


scores_allgenes = {}
for gene_set in GENE_SETS:
    universe = universes_allgenes[gene_set]["universe"]
    scores_allgenes[gene_set] = {"twinfer": load_twinfer_allgenes(gene_set, universe)}
    for m in COMPETITOR_METHODS:
        scores_allgenes[gene_set][m] = load_competitor_allgenes(m, gene_set, universe)
    print(f"{gene_set:<18} loaded scores for {len(ALL_METHODS)} methods")
rows_ag = []
for gene_set in GENE_SETS:
    y_true = universes_allgenes[gene_set]["y_true"]
    for method in ALL_METHODS:
        y_score = scores_allgenes[gene_set][method]
        obs = observed_metrics(y_true, y_score)
        # [2026-10-01 commented out: hash() of strings is randomised per process (PYTHONHASHSEED) so this seed was not reproducible; zlib.crc32 of the repr is stable] rnd = random_baseline(y_true, y_score, seed=SEED + hash((gene_set, method, "allgenes")) % (2**31))
        rnd = random_baseline(y_true, y_score, seed=SEED + zlib.crc32(repr((gene_set, method, "allgenes")).encode()) % (2**31))
        rows_ag.append(dict(
            gene_set=gene_set, method=method,
            n_universe=len(y_true), n_curated=int(y_true.sum()),
            auprc=obs["auprc"], auprc_random=rnd["auprc_random"],
            auprc_ratio=obs["auprc"] / rnd["auprc_random"] if rnd["auprc_random"] > 0 else float("nan"),
            f1=obs["f1"], f1_random=rnd["f1_random"],
            f1_ratio=obs["f1"] / rnd["f1_random"] if rnd["f1_random"] > 0 else float("nan"),
            early_precision=obs["early_precision"], epr=obs["epr"],
        ))
    print(f"{gene_set}: done")

results_allgenes = pd.DataFrame(rows_ag)
out_path = os.path.join(OUT_DIR, "benchmark_results_allgenes.csv")
results_allgenes.to_csv(out_path, index=False)
print(f"wrote {out_path}  ({len(results_allgenes)} rows)")

print("\n=== epr (all-genes-to-all-genes universe) ===")
pivot = results_allgenes.pivot(index="gene_set", columns="method", values="epr")[ALL_METHODS].loc[GENE_SETS]
print(pivot.round(3).to_string())

print("\n=== mean across all 9 gene sets, per method (all-genes universe) ===")
summary_ag = results_allgenes.groupby("method")[["auprc", "auprc_ratio", "f1", "f1_ratio",
                                                    "early_precision", "epr"]].mean()
summary_ag = summary_ag.loc[ALL_METHODS]
print(summary_ag.round(3).to_string())

summary_ag_path = os.path.join(OUT_DIR, "benchmark_summary_by_method_allgenes.csv")
summary_ag.to_csv(summary_ag_path)
print(f"wrote {summary_ag_path}")

print("\n=== side by side: TF-restricted (Section 5) vs all-genes (Section 9) mean EPR ===")
side = pd.DataFrame({"epr_tf_restricted": summary.loc[ALL_METHODS, "epr"],
                      "epr_all_genes": summary_ag.loc[ALL_METHODS, "epr"]})
print(side.round(3).to_string())
def compute_sym_score(gene_set, universe):
    """SYM(x,y) = s(min(zr1,zr2)) - s(|CHv|) - s(DRIFT) - s(INH) + s(GAM), generalized from
    Section 8 to any gene set and any universe (here: the full all-genes universe)."""
    rdir = os.path.join(INFER_DIR, f"{gene_set}_t2_t4_allpairs_50core")
    ranked_local = pd.read_csv(os.path.join(rdir, "ranked_edges.csv"))
    rdt1 = pd.read_csv(os.path.join(rdir, "corr_random_delta_t1.csv"), index_col=0)
    rdt2 = pd.read_csv(os.path.join(rdir, "corr_random_delta_t2.csv"), index_col=0)
    zsteps = json.load(open(os.path.join(rdir, "z_scores_by_step.json")))
    dirz = zsteps["direction_z_scores"]

    ranked_local_idx = ranked_local.set_index(["gene_1", "gene_2"])

    def _dz(a, b):
        return dirz.get(f"{a}__{b}", np.nan)

    zr1_l, zr2_l, ch_l, drift_l, inh_l, gam_l = [], [], [], [], [], []
    for a, b in universe:
        if (a, b) not in ranked_local_idx.index:
            zr1_l.append(np.nan); zr2_l.append(np.nan); ch_l.append(np.nan)
            drift_l.append(np.nan); inh_l.append(np.nan); gam_l.append(np.nan)
            continue
        row = ranked_local_idx.loc[(a, b)]
        zr1_l.append(float(row["u_abs_rho_t1"]))
        zr2_l.append(float(row["u_abs_rho_t2"]))
        ch_l.append(float(row["u_abs_rho_change"]))
        try:
            drift_l.append(abs(float(rdt2.loc[a, b]) - float(rdt1.loc[a, b])))
        except KeyError:
            drift_l.append(np.nan)
        z_xy, z_yx = _dz(a, b), _dz(b, a)
        inh_l.append(np.nanmin([abs(z_xy) if np.isfinite(z_xy) else np.nan,
                                 abs(z_yx) if np.isfinite(z_yx) else np.nan]))
        gam_l.append(float(row["u_gamma"]))

    zr1_l, zr2_l = np.array(zr1_l), np.array(zr2_l)
    min_zr_l = np.minimum(zr1_l, zr2_l)
    ch_l, drift_l, inh_l, gam_l = map(np.array, (ch_l, drift_l, inh_l, gam_l))

    zreg = zsteps["z_reg_gated"]
    zreg_l = np.array([zreg.get(f"{a}__{b}", zreg.get(f"{b}__{a}", np.nan)) for a, b in universe])
    defined_l = np.isfinite(zr1_l) & np.isfinite(zr2_l) & np.isfinite(zreg_l)
    pass_l = defined_l & ((zr1_l > 2.576) | (zr2_l > 2.576)) & (np.abs(zreg_l) > 1.645)

    sym_raw = s(min_zr_l) - s(ch_l) - s(drift_l) - s(inh_l) + s(gam_l)
    return np.where(pass_l, sym_raw, -1e12)


sym_scores_allgenes = {}
for gene_set in GENE_SETS:
    universe = universes_allgenes[gene_set]["universe"]
    sym_scores_allgenes[gene_set] = compute_sym_score(gene_set, universe)
    print(f"{gene_set:<18} SYM computed for {len(universe)} pairs (full all-genes universe)")

rows_sym = []
for gene_set in GENE_SETS:
    y_true = universes_allgenes[gene_set]["y_true"]
    y_score = sym_scores_allgenes[gene_set]
    obs = observed_metrics(y_true, y_score)
    # [2026-10-01 commented out: hash() of strings is randomised per process (PYTHONHASHSEED) so this seed was not reproducible; zlib.crc32 of the repr is stable] rnd = random_baseline(y_true, y_score, seed=SEED + hash((gene_set, "sym_allgenes")) % (2**31))
    rnd = random_baseline(y_true, y_score, seed=SEED + zlib.crc32(repr((gene_set, "sym_allgenes")).encode()) % (2**31))
    rows_sym.append(dict(
        gene_set=gene_set, method="twinfer_sym",
        n_universe=len(y_true), n_curated=int(y_true.sum()),
        auprc=obs["auprc"], auprc_random=rnd["auprc_random"],
        auprc_ratio=obs["auprc"] / rnd["auprc_random"] if rnd["auprc_random"] > 0 else float("nan"),
        f1=obs["f1"], f1_random=rnd["f1_random"],
        f1_ratio=obs["f1"] / rnd["f1_random"] if rnd["f1_random"] > 0 else float("nan"),
        early_precision=obs["early_precision"], epr=obs["epr"],
    ))

results_sym = pd.DataFrame(rows_sym)
out_path = os.path.join(OUT_DIR, "benchmark_results_sym_allgenes.csv")
results_sym.to_csv(out_path, index=False)
print(f"wrote {out_path}")

# swap into the Section 9 comparison: twinfer_sym replaces twinfer
ALL_METHODS_SYM = ["twinfer_sym"] + COMPETITOR_METHODS
combined = pd.concat([results_sym, results_allgenes[results_allgenes["method"] != "twinfer"]],
                      ignore_index=True)

print("\n=== epr, twinfer replaced by twinfer_sym (full all-genes universe) ===")
pivot = combined.pivot(index="gene_set", columns="method", values="epr")[ALL_METHODS_SYM].loc[GENE_SETS]
print(pivot.round(3).to_string())

print("\n=== mean across all 9 gene sets ===")
summary_sym = combined.groupby("method")[["auprc", "auprc_ratio", "f1", "f1_ratio",
                                            "early_precision", "epr"]].mean().loc[ALL_METHODS_SYM]
print(summary_sym.round(3).to_string())

summary_sym_path = os.path.join(OUT_DIR, "benchmark_summary_sym_allgenes.csv")
summary_sym.to_csv(summary_sym_path)
print(f"wrote {summary_sym_path}")

print("\n=== old twinScore vs sym_twinscore, mean EPR (all-genes universe) ===")
print(f"  twinScore (original): {summary_ag.loc['twinfer', 'epr']:.3f}")
print(f"  sym_twinscore:        {summary_sym.loc['twinfer_sym', 'epr']:.3f}")

def _fillna_floor(arr):
    arr = np.asarray(arr, dtype=float)
    finite = arr[np.isfinite(arr)]
    floor = finite.min() - 1.0 if finite.size else 0.0
    return np.where(np.isfinite(arr), arr, floor)


RANKED_COLS = ["rho_t1", "rho_t2", "rho_change", "u_abs_rho_t1", "u_abs_rho_t2", "u_abs_rho_change",
               "z_abs_rho_t1", "z_abs_rho_t2", "z_abs_rho_change", "z_het", "z_div", "z_d_het",
               "gamma", "u_gamma", "z_gamma", "twinScore"]

TERMS_TO_TEST = [
    ("rho_t1_abs", lambda df: df["rho_t1"].abs()),
    ("rho_t2_abs", lambda df: df["rho_t2"].abs()),
    ("rho_change_abs", lambda df: df["rho_change"].abs()),
    ("u_abs_rho_t1", lambda df: df["u_abs_rho_t1"]),
    ("u_abs_rho_t2", lambda df: df["u_abs_rho_t2"]),
    ("u_abs_rho_change", lambda df: df["u_abs_rho_change"]),
    ("z_abs_rho_t1", lambda df: df["z_abs_rho_t1"]),
    ("z_abs_rho_t2", lambda df: df["z_abs_rho_t2"]),
    ("z_abs_rho_change", lambda df: df["z_abs_rho_change"]),
    ("z_het_signed", lambda df: df["z_het"]),
    ("z_het_abs", lambda df: df["z_het"].abs()),
    ("z_div_signed", lambda df: df["z_div"]),
    ("z_div_abs", lambda df: df["z_div"].abs()),
    ("z_d_het_signed", lambda df: df["z_d_het"]),
    ("z_d_het_abs", lambda df: df["z_d_het"].abs()),
    ("gamma_abs", lambda df: df["gamma"].abs()),
    ("u_gamma_abs", lambda df: df["u_gamma"].abs()),
    ("z_gamma_abs", lambda df: df["z_gamma"].abs()),
    ("z_gamma_signed", lambda df: df["z_gamma"]),
    ("z_reg_gated_signed", lambda df: df["z_reg_gated"]),
    ("z_reg_gated_abs", lambda df: df["z_reg_gated"].abs()),
    ("drift", lambda df: df["drift"]),
    ("inh", lambda df: df["inh"]),
    ("twinScore (composite)", lambda df: df["twinScore"]),
]

rows_terms = []
for gene_set in GENE_SETS:
    universe = universes_allgenes[gene_set]["universe"]
    y_true = universes_allgenes[gene_set]["y_true"]
    rdir = os.path.join(INFER_DIR, f"{gene_set}_t2_t4_allpairs_50core")

    ranked_local = pd.read_csv(os.path.join(rdir, "ranked_edges.csv")).set_index(["gene_1", "gene_2"])
    idx = pd.MultiIndex.from_tuples(universe, names=["gene_1", "gene_2"])
    df_terms = ranked_local.reindex(idx)[RANKED_COLS].reset_index(drop=True)

    zsteps = json.load(open(os.path.join(rdir, "z_scores_by_step.json")))
    zreg, dirz = zsteps["z_reg_gated"], zsteps["direction_z_scores"]
    rdt1 = pd.read_csv(os.path.join(rdir, "corr_random_delta_t1.csv"), index_col=0)
    rdt2 = pd.read_csv(os.path.join(rdir, "corr_random_delta_t2.csv"), index_col=0)

    zreg_col, drift_col, inh_col = [], [], []
    for a, b in universe:
        zreg_col.append(zreg.get(f"{a}__{b}", zreg.get(f"{b}__{a}", np.nan)))
        try:
            drift_col.append(abs(float(rdt2.loc[a, b]) - float(rdt1.loc[a, b])))
        except KeyError:
            drift_col.append(np.nan)
        z_xy, z_yx = dirz.get(f"{a}__{b}", np.nan), dirz.get(f"{b}__{a}", np.nan)
        inh_col.append(np.nanmin([abs(z_xy) if np.isfinite(z_xy) else np.nan,
                                   abs(z_yx) if np.isfinite(z_yx) else np.nan]))
    df_terms["z_reg_gated"] = zreg_col
    df_terms["drift"] = drift_col
    df_terms["inh"] = inh_col

    for name, fn in TERMS_TO_TEST:
        vals = fn(df_terms).to_numpy(dtype=float)
        y_score_t = _fillna_floor(vals)
        obs = observed_metrics(y_true, y_score_t)
        # [2026-10-01 commented out: hash() of strings is randomised per process (PYTHONHASHSEED) so this seed was not reproducible; zlib.crc32 of the repr is stable] rnd = random_baseline(y_true, y_score_t, seed=SEED + hash((gene_set, name)) % (2**31))
        rnd = random_baseline(y_true, y_score_t, seed=SEED + zlib.crc32(repr((gene_set, name)).encode()) % (2**31))
        mean_true = float(np.nanmean(vals[y_true == 1]))
        mean_false = float(np.nanmean(vals[y_true == 0]))
        rows_terms.append(dict(
            gene_set=gene_set, term=name,
            auprc_ratio=obs["auprc"] / rnd["auprc_random"] if rnd["auprc_random"] > 0 else np.nan,
            epr=obs["epr"], mean_true=mean_true, mean_false=mean_false,
            trend=("higher in true" if mean_true > mean_false else "higher in false")))
    print(f"{gene_set}: done")

results_terms = pd.DataFrame(rows_terms)
out_path = os.path.join(OUT_DIR, "benchmark_per_term.csv")
results_terms.to_csv(out_path, index=False)
print(f"wrote {out_path}")

term_summary = results_terms.groupby("term").agg(
    auprc_ratio_mean=("auprc_ratio", "mean"),
    epr_mean=("epr", "mean"),
    mean_true=("mean_true", "mean"),
    mean_false=("mean_false", "mean"),
).reset_index()
term_summary["trend"] = np.where(term_summary["mean_true"] > term_summary["mean_false"],
                                  "higher in TRUE edges", "higher in FALSE edges")
term_summary = term_summary.sort_values("epr_mean", ascending=False).reset_index(drop=True)

print("=== per-term discriminative power, averaged across all 9 gene sets, sorted by EPR ===")
print("(EPR/auprc_ratio > 1 means this term ALONE beats random at ranking true edges to the top)\n")
print(term_summary.round(3).to_string())

term_summary_path = os.path.join(OUT_DIR, "benchmark_per_term_summary.csv")
term_summary.to_csv(term_summary_path, index=False)
print(f"\nwrote {term_summary_path}")
