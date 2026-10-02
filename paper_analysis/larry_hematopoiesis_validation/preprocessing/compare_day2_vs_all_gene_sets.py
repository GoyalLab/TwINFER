"""One-off: rerun pick_gene_sets.ipynb's exact 9-gene-list recipe restricted to day-2 cells only
(treating day 2 as if it were the whole dataset -- its own detection floor, its own HVGs, its own
correlation structure -- not just a cell-subset of the all-days computation), then diff the result
against the already-saved all-cells resources/gene_sets.json. A sensitivity check: how much does
the gene-set choice depend on which day(s) went into it?

Does NOT touch pick_gene_sets.ipynb or resources/gene_sets.json -- writes resources/gene_sets_day2.json.

Run directly: python3 compare_day2_vs_all_gene_sets.py
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
import os

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io as sio
from scipy import stats
from scipy.stats import rankdata

HERE = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation'
from twinfer.utils.paths import get_data_root as _res_gdr  # [2026-10-01 LARRY resources/ moved out of the code tree to analysis_data/paper_analysis/larry_hematopoiesis_validation/resources (user)]
RES_HERE = f"{_res_gdr()}/paper_analysis/larry_hematopoiesis_validation"
SOURCE = f'{TWINFER_PROJECT_ROOT}/finalized_data/LARRY_data/filtered'
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] COLLECTRI_PATH = os.path.join(HERE, "resources", "collectri_mouse.tsv")
COLLECTRI_PATH = os.path.join(RES_HERE, "resources", "collectri_mouse.tsv")
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] ALL_CELLS_PATH = os.path.join(HERE, "resources", "gene_sets.json")
ALL_CELLS_PATH = os.path.join(RES_HERE, "resources", "gene_sets.json")

MIN_DETECTION_FRAC = 0.05
N_HVG = 2000
SEED = 0
NULL_PERCENTILE = 95
N_HIGH_TFS = 15
N_TARGETS_PER_TF = 4
LOW_BAND_WIDTH = 1.15
MID_BAND = (0.35, 0.65)
# DAYS overrides DAY when set: comma-separated list of days to pool (e.g. "2,4"). DAY stays as the
# single-day default/back-compat knob.
DAYS = [int(d) for d in os.environ["DAYS"].split(",")] if "DAYS" in os.environ \
    else [int(os.environ.get("DAY", "2"))]
DAY = "+".join(str(d) for d in DAYS)  # only used for logging/filenames below
rng = np.random.default_rng(SEED)


def log(m):
    print(m, flush=True)


# ---------------------------------------------------------------- load, subset to day 2, floor
X = sio.mmread(os.path.join(SOURCE, "larry_qc_counts.mtx")).tocsr()
genes = pd.Index(open(os.path.join(SOURCE, "genes.txt")).read().split())
obs = pd.read_csv(os.path.join(SOURCE, "obs_metadata.csv"), index_col=0)
assert X.shape == (len(obs), len(genes))

day = obs["library"].astype(str).str.extract(r"_d(\d+)_", expand=False)
assert day.notna().all(), "could not parse day out of every library name"
day = day.astype(int).to_numpy()
is_day = np.isin(day, DAYS)
X = X[is_day]
obs = obs[is_day].copy()
log(f"day(s) {DAYS} only: {X.shape[0]:,} cells x {X.shape[1]:,} genes")

# detection floor computed on the SELECTED DAY(S) ONLY, pooled together as if that were the whole
# dataset -- no "min across days" step (that convention only applies when comparing against days
# not included here, as in the full all-cells run)
frac_expr = np.asarray((X > 0).sum(axis=0)).ravel() / X.shape[0]
gene_mask = frac_expr >= MIN_DETECTION_FRAC
genes = genes[gene_mask]
X = X[:, gene_mask]
frac_expr = frac_expr[gene_mask]
log(f"day-{DAY} detection floor >= {MIN_DETECTION_FRAC:.0%}: {int(gene_mask.sum()):,} genes")

A = sc.AnnData(X=X.tocsr().astype(np.float32), obs=obs, var=pd.DataFrame(index=genes))
sc.pp.normalize_total(A, target_sum=1e4)
sc.pp.log1p(A)
del X

# ---------------------------------------------------------------- shared CollecTRI + recipe
collectri_raw = pd.read_csv(COLLECTRI_PATH, sep="\t")
collectri = collectri_raw[collectri_raw["source_genesymbol"] != collectri_raw["target_genesymbol"]]
collectri_tf_targets = collectri.groupby("source_genesymbol")["target_genesymbol"].apply(set).to_dict()
log(f"CollecTRI mouse: {collectri['source_genesymbol'].nunique():,} TFs, {len(collectri):,} edges")


def tf_target_subset(gene_pool, score, n_tfs=N_HIGH_TFS, n_targets=N_TARGETS_PER_TF):
    pool_set = set(gene_pool)
    tfs_with_targets = []
    for tf in gene_pool:
        tgts = collectri_tf_targets.get(tf)
        if not tgts:
            continue
        in_pool = sorted((tgts & pool_set) - {tf})
        if in_pool:
            tfs_with_targets.append((tf, in_pool))
    tfs_with_targets.sort(key=lambda kv: score[kv[0]], reverse=True)
    chosen = tfs_with_targets[:n_tfs]
    genes_out = set()
    for tf, in_pool in chosen:
        genes_out.add(tf)
        ranked = sorted(in_pool, key=lambda t: score[t], reverse=True)[:n_targets]
        genes_out.update(ranked)
    return sorted(genes_out)


# ---------------------------------------------------------------- 1. Variability
sc.pp.highly_variable_genes(A, n_top_genes=N_HVG, batch_key="library")
hvg_genes = A.var_names[A.var["highly_variable"]]
hvg_rank = A.var.loc[hvg_genes, "dispersions_norm"].sort_values(ascending=False)
third = N_HVG // 3
variability_high_pool = hvg_rank.index[:third].tolist()
variability_mid_pool = hvg_rank.index[third:2 * third].tolist()
variability_low_pool = hvg_rank.index[2 * third:].tolist()
variability_high = tf_target_subset(variability_high_pool, hvg_rank)
variability_mid = tf_target_subset(variability_mid_pool, hvg_rank)
variability_low = tf_target_subset(variability_low_pool, hvg_rank)
log(f"variability: high={len(variability_high)} mid={len(variability_mid)} low={len(variability_low)}")

# ---------------------------------------------------------------- 2. Detection
detection = pd.Series(frac_expr, index=genes).sort_values(ascending=False)
p50, p75, p90 = np.percentile(detection.to_numpy(), [50, 75, 90])
detection_high_pool = detection[detection >= p90].index.tolist()
detection_mid_pool = detection[(detection >= p75) & (detection < p90)].index.tolist()
detection_low_pool = detection[(detection >= p50) & (detection < p75)].index.tolist()
detection_high = tf_target_subset(detection_high_pool, detection)
detection_mid = tf_target_subset(detection_mid_pool, detection)
detection_low = tf_target_subset(detection_low_pool, detection)
log(f"detection: high={len(detection_high)} mid={len(detection_mid)} low={len(detection_low)}")

# ---------------------------------------------------------------- 3. Correlation
hvg_set = set(hvg_genes)
candidate_tfs = [g for g in hvg_genes if g in collectri_tf_targets]
tf_curated_targets = {tf: sorted((collectri_tf_targets[tf] & hvg_set) - {tf}) for tf in candidate_tfs}
candidate_tfs = [tf for tf in candidate_tfs if tf_curated_targets[tf]]
log(f"candidate TFs (CollecTRI ∩ HVG): {len(candidate_tfs)}")

Xh = A[:, hvg_genes].X
Xh = Xh.toarray() if hasattr(Xh, "toarray") else np.asarray(Xh)
hvg_index = {g: i for i, g in enumerate(hvg_genes)}
tf_idx = np.array([hvg_index[g] for g in candidate_tfs])
candidate_targets = list(hvg_genes)
tgt_idx = np.arange(len(hvg_genes))

curated_mask = np.zeros((len(candidate_tfs), len(candidate_targets)), dtype=bool)
for i, tf in enumerate(candidate_tfs):
    for tgt in tf_curated_targets[tf]:
        curated_mask[i, hvg_index[tgt]] = True


def spearman_matrix(mat, rows_idx, cols_idx):
    Rk = np.apply_along_axis(rankdata, 0, mat).astype(np.float64)
    Rk = (Rk - Rk.mean(axis=0)) / np.maximum(Rk.std(axis=0), 1e-9)
    Rr, Rc = Rk[:, rows_idx], Rk[:, cols_idx]
    return (Rr.T @ Rc) / Rr.shape[0]


R = spearman_matrix(Xh, tf_idx, tgt_idx)
del Xh

det_of = pd.Series(frac_expr, index=genes)
hvg_det = det_of.loc[hvg_genes].to_numpy()
decile_edges = np.quantile(hvg_det, np.linspace(0, 1, 11)[1:-1])
decile = np.digitize(hvg_det, decile_edges)
by_decile = {d: np.flatnonzero(decile == d) for d in range(10)}

real_pairs = {(i, j) for i in range(len(candidate_tfs)) for j in range(len(candidate_targets)) if curated_mask[i, j]}
obs_absr, dec_absr = [], []
for (i, j) in real_pairs:
    obs_absr.append(abs(R[i, j]))
    pool = by_decile[decile[j]]
    for _ in range(20):
        c = int(rng.choice(pool))
        if c != j and (i, c) not in real_pairs:
            dec_absr.append(abs(R[i, c]))
            break
obs_absr, dec_absr = np.array(obs_absr), np.array(dec_absr)
u = stats.mannwhitneyu(obs_absr, dec_absr, alternative="greater")
auc = u.statistic / (len(obs_absr) * len(dec_absr))
NULL_CUTOFF = float(np.percentile(dec_absr, NULL_PERCENTILE))
log(f"correlation: curated_pairs={len(obs_absr)} AUC={auc:.4f} NULL_CUTOFF={NULL_CUTOFF:.4f}")

R_masked = np.where(curated_mask, np.abs(R), np.nan)
tf_mean_absr = pd.Series(np.nanmean(R_masked, axis=1), index=candidate_tfs).sort_values(ascending=False)
high_tfs = tf_mean_absr.index[:N_HIGH_TFS].tolist()
correlation_high = set(high_tfs)
for tf in high_tfs:
    i = candidate_tfs.index(tf)
    row = pd.Series(R_masked[i], index=candidate_targets).dropna().sort_values(ascending=False)
    correlation_high.update(row.index[:N_TARGETS_PER_TF].tolist())
correlation_high = sorted(correlation_high)

all_pairs = pd.DataFrame(
    [(candidate_tfs[i], candidate_targets[j], abs(R[i, j]))
     for i in range(len(candidate_tfs)) for j in range(len(candidate_targets)) if curated_mask[i, j]],
    columns=["tf", "target", "abs_r"])
low_hi = NULL_CUTOFF * LOW_BAND_WIDTH
mid_lo, mid_hi = np.percentile(all_pairs["abs_r"], [MID_BAND[0] * 100, MID_BAND[1] * 100])
low_pairs = all_pairs[all_pairs["abs_r"].between(NULL_CUTOFF, low_hi)]
mid_pairs = all_pairs[all_pairs["abs_r"].between(mid_lo, mid_hi)]


def pick_list(pairs_in_band, n_tfs=N_HIGH_TFS, n_targets=N_TARGETS_PER_TF):
    tf_counts = pairs_in_band["tf"].value_counts()
    chosen_tfs = tf_counts.index[:n_tfs].tolist()
    genes_out = set(chosen_tfs)
    for tf in chosen_tfs:
        sub = pairs_in_band[pairs_in_band["tf"] == tf].sort_values("abs_r", ascending=False)
        genes_out.update(sub["target"].head(n_targets).tolist())
    return sorted(genes_out)


correlation_low = pick_list(low_pairs)
correlation_mid = pick_list(mid_pairs)
log(f"correlation: high={len(correlation_high)} mid={len(correlation_mid)} low={len(correlation_low)}")

day2_gene_sets = {
    "variability_high": variability_high, "variability_mid": variability_mid, "variability_low": variability_low,
    "detection_high": detection_high, "detection_mid": detection_mid, "detection_low": detection_low,
    "correlation_high": correlation_high, "correlation_mid": correlation_mid, "correlation_low": correlation_low,
}
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] out_path = os.path.join(HERE, "resources", f"gene_sets_day{DAY}.json")
out_path = os.path.join(RES_HERE, "resources", f"gene_sets_day{DAY}.json")
json.dump(day2_gene_sets, open(out_path, "w"), indent=2)
log(f"wrote {out_path}")

# ---------------------------------------------------------------- compare vs all-cells
all_gene_sets = json.load(open(ALL_CELLS_PATH))
log(f"\n=== day{DAY}-only vs all-cells: per-list comparison ===")
log(f"{'list':<18}{'all':>6}{f'day{DAY}':>6}{'shared':>8}{'only_all':>10}{f'only_day{DAY}':>11}{'jaccard':>9}")
for name in day2_gene_sets:
    a, d = set(all_gene_sets[name]), set(day2_gene_sets[name])
    shared = a & d
    jacc = len(shared) / len(a | d) if (a | d) else float("nan")
    log(f"{name:<18}{len(a):>6}{len(d):>6}{len(shared):>8}{len(a - d):>10}{len(d - a):>11}{jacc:>9.2f}")

log(f"\n=== genes only in all-cells (dropped when restricted to day {DAY}) ===")
for name in day2_gene_sets:
    only_all = sorted(set(all_gene_sets[name]) - set(day2_gene_sets[name]))
    log(f"{name}: {only_all}")

log(f"\n=== genes only in day-{DAY}-only (not picked by the all-cells run) ===")
for name in day2_gene_sets:
    only_day2 = sorted(set(day2_gene_sets[name]) - set(all_gene_sets[name]))
    log(f"{name}: {only_day2}")
