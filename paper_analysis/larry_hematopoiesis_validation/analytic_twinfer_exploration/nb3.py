from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import os

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io as sio
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath("__file__"))
from twinfer.utils.paths import get_data_root as _res_gdr  # [2026-10-01 LARRY resources/ moved out of the code tree to analysis_data/paper_analysis/larry_hematopoiesis_validation/resources (user)]
RES_HERE = f"{_res_gdr()}/paper_analysis/larry_hematopoiesis_validation"
SOURCE = f'{TWINFER_PROJECT_ROOT}/finalized_data/LARRY_data/filtered'
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] TF_LIST_PATH = os.path.join(HERE, "resources", "mouse_TF_AnimalTFDB3.txt")
TF_LIST_PATH = os.path.join(RES_HERE, "resources", "mouse_TF_AnimalTFDB3.txt")
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] COLLECTRI_PATH = os.path.join(HERE, "resources", "collectri_mouse.tsv")
COLLECTRI_PATH = os.path.join(RES_HERE, "resources", "collectri_mouse.tsv")

# ---- config knobs (named, so retuning means editing one place then rerunning below) ----
MIN_DETECTION_FRAC = 0.05     # matches larry_raw_annotate.py's floor -- drop noise genes first
N_HVG = 2000                  # scanpy seurat-flavor n_top_genes (matches build_panel_set.py); the
                              # dispersions_norm VALUE is what the variability strip bands on, not
                              # this cutoff
SEED = 0

# ---- edge / band selection follows helpers/build_panel_set.py ("THE panel set, one recipe") ----
N_BAND_EDGES = 50            # corr High/Mid/Low = top-50 / middle-50 / bottom-50 CollecTRI edges by |rho|
DETECT_BANDS = ((0.50, 0.75), (0.75, 0.90), (0.90, 1.00))   # detection-quantile strip (low, mid, high)
VAR_BANDS    = ((0.00, 0.33), (0.33, 0.67), (0.67, 1.00))   # dispersion-quantile strip (low, mid, high)

N_HIGH_TFS = 15               # KEPT at 15 (build_panel_set.py uses 10): top TFs per band
N_TARGETS_PER_TF = 4          # KEPT at 4  (build_panel_set.py uses 6): curated targets per TF

# ---- SUPERSEDED: permutation-null / decile-matched-decoy correlation banding ----
# replaced by build_panel_set.py's plain |rho| edge ranking; kept per repo convention
# N_PERMUTATIONS = 5            # independent cell-order shuffles per gene, for the null distribution
# NULL_PERCENTILE = 95          # "average null cutoff for co-expression" = this percentile of |null r|
# LOW_BAND_WIDTH = 1.15         # Low = pairs with |r| in [NULL_CUTOFF, NULL_CUTOFF * this]
# MID_BAND = (0.35, 0.65)       # Mid = pairs whose |r| percentile falls in this range

rng = np.random.default_rng(SEED)


X = sio.mmread(os.path.join(SOURCE, "larry_qc_counts.mtx")).tocsr()
genes = pd.Index(open(os.path.join(SOURCE, "genes.txt")).read().split())
obs = pd.read_csv(os.path.join(SOURCE, "obs_metadata.csv"), index_col=0)
assert X.shape == (len(obs), len(genes))
print(f"loaded {X.shape[0]:,} cells x {X.shape[1]:,} genes")

day = obs["library"].astype(str).str.extract(r"_d(\d+)_", expand=False)
assert day.notna().all(), "could not parse day out of every library name"
day = day.astype(int).to_numpy()
days = sorted(set(day))
print(f"days present: {days}")

# per-day detection fraction, then the WORST (minimum) across days per gene -- not pooled across
# all cells (see markdown above)
per_day_frac = np.stack([
    np.asarray((X[day == d] > 0).sum(axis=0)).ravel() / (day == d).sum()
    for d in days
])
frac_expr = per_day_frac.min(axis=0)
gene_mask = frac_expr >= MIN_DETECTION_FRAC
print(f"per-day detection floor (min across {len(days)} days) >= {MIN_DETECTION_FRAC:.0%}: "
      f"{int(gene_mask.sum()):,} / {len(genes):,} genes")

genes = genes[gene_mask]
X = X[:, gene_mask]
X_day2 = X[day == 2]
obs_day2 = obs[day == 2]

frac_expr = frac_expr[gene_mask]

A = sc.AnnData(X=X.tocsr().astype(np.float32), obs=obs, var=pd.DataFrame(index=genes))
A.layers["counts"] = A.X.copy()
sc.pp.normalize_total(A, target_sum=1e4)
sc.pp.log1p(A)
print("normalized: log1p(CP10k)")

del X, obs, gene_mask, per_day_frac, X_day2, obs_day2
import gc; gc.collect()


collectri_raw = pd.read_csv(COLLECTRI_PATH, sep="\t")

# _collectri.py's exact edge schema: sign (+1 activation/-1 repression/0 both), contested (no
# consensus direction), pmids (distinct PMID count), resources (distinct source-DB count).
def _edge_count(v):
    return len(set(str(v).split(";"))) if pd.notna(v) else 0

s, i = collectri_raw["is_stimulation"].astype(bool), collectri_raw["is_inhibition"].astype(bool)
collectri_raw["sign"] = np.where(s & i, 0, np.where(s, 1, -1))
collectri_raw["contested"] = (~(collectri_raw["consensus_stimulation"].astype(bool) |
                                 collectri_raw["consensus_inhibition"].astype(bool))).astype(int)
collectri_raw["effort"] = collectri_raw["curation_effort"].astype(int)
collectri_raw["pmids"] = collectri_raw["references"].apply(_edge_count)
collectri_raw["resources"] = collectri_raw["sources"].apply(_edge_count)
collectri = collectri_raw[collectri_raw["source_genesymbol"] != collectri_raw["target_genesymbol"]]  # self-pairs stripped
collectri_edge_meta = {
    (r.source_genesymbol, r.target_genesymbol): dict(sign=r.sign, contested=r.contested,
                                                      effort=r.effort, pmids=r.pmids, resources=r.resources)
    for r in collectri.itertuples(index=False)
}
collectri_tf_targets = collectri.groupby("source_genesymbol")["target_genesymbol"].apply(set).to_dict()
print(f"CollecTRI mouse: {collectri['source_genesymbol'].nunique():,} TFs, "
      f"{collectri['target_genesymbol'].nunique():,} distinct targets, {len(collectri):,} edges "
      f"(self-pairs stripped)")


# ---------------------------------------------------------------------------
# build_panel_set.py's shared quantities: the CollecTRI edge set scoped to the
# detection-floored matrix, and |rho| for every such edge (one Spearman pass,
# reused by all three criteria -- matches build_panel_set.py lines 58-70).
# ---------------------------------------------------------------------------
from scipy.stats import rankdata

matrix_genes = set(A.var_names)
collectri_pairs = sorted({
    (r.source_genesymbol, r.target_genesymbol)
    for r in collectri.itertuples(index=False)
    if r.source_genesymbol in matrix_genes and r.target_genesymbol in matrix_genes
})
need = sorted({g for p in collectri_pairs for g in p})
need_idx = {g: k for k, g in enumerate(need)}
print(f"CollecTRI edges with both ends in the detection-floored matrix: {len(collectri_pairs):,} "
      f"over {len(need):,} genes")

# |rho| = |Spearman| = |Pearson on column ranks|, build_panel_set.py's exact formula
_Xs = A[:, need].X
_Xs = np.asarray(_Xs.todense()) if sp.issparse(_Xs) else np.asarray(_Xs)
_Rk = np.apply_along_axis(rankdata, 0, _Xs).astype(np.float32)
del _Xs
_Rk -= _Rk.mean(axis=0)
_S = np.sqrt((_Rk ** 2).sum(axis=0)); _S[_S < 1e-12] = 1.0
_C = np.abs((_Rk.T @ _Rk) / np.outer(_S, _S))
del _Rk
abs_rho = {p: float(_C[need_idx[p[0]], need_idx[p[1]]]) for p in collectri_pairs}
del _C
import gc; gc.collect()
print(f"|rho| over {len(abs_rho):,} curated edges "
      f"[{min(abs_rho.values()):.3f}, {max(abs_rho.values()):.3f}]")


def band_edges(band_genes):
    """CollecTRI edges with BOTH endpoints in `band_genes` (build_panel_set.py's strip step)."""
    b = set(band_genes)
    return [p for p in collectri_pairs if p[0] in b and p[1] in b]


def edges_to_panel(edges, n_tfs=N_HIGH_TFS, n_targets=N_TARGETS_PER_TF,
                   rank_by="count", ascending=False):
    """The 15-TF / 4-target recipe on a set of (tf, target) edges.
      rank_by="count"      -- the `n_tfs` TFs with the MOST edges in the set (ties -> higher mean
                              |rho|). Matches build_panel_set.py's detection / HVG strips; used for
                              the variability and detection criteria.
      rank_by="rho_median" -- the `n_tfs` TFs by the MEDIAN |rho| of their edges in the set
                              (`ascending=True` picks the weakest). Used for the correlation
                              criterion, whose band already IS a per-edge |rho| slice, so "most
                              edges" misses the point -- edge strength does.
    Then each TF gets its `n_targets` highest-|rho| targets in the set. 15/4 vs build_panel_set.py's
    10/6. Returns (sorted gene list, [(tf, [(target, |rho|), ...]), ...])."""
    by_tf = {}
    for tf, tgt in edges:
        by_tf.setdefault(tf, []).append(tgt)
    if rank_by == "rho_median":
        ranked = sorted(by_tf, key=lambda tf: float(np.median([abs_rho[(tf, t)] for t in by_tf[tf]])),
                        reverse=not ascending)[:n_tfs]
    else:
        ranked = sorted(by_tf,
                        key=lambda tf: (len(by_tf[tf]),
                                        float(np.mean([abs_rho[(tf, t)] for t in by_tf[tf]]))),
                        reverse=True)[:n_tfs]
    genes_out, detail = set(), []
    for tf in ranked:
        genes_out.add(tf)
        tgts = sorted(by_tf[tf], key=lambda t: abs_rho[(tf, t)], reverse=True)[:n_targets]
        genes_out.update(tgts)
        detail.append((tf, [(t, round(abs_rho[(tf, t)], 3)) for t in tgts]))
    return sorted(genes_out), detail


# ===== SUPERSEDED: tf_target_subset (pool-score recipe) -> edges_to_panel above =====
#
#
# def tf_target_subset(gene_pool, score, n_tfs=N_HIGH_TFS, n_targets=N_TARGETS_PER_TF):
#     """From `gene_pool` (a band's full gene list), pick the top `n_tfs` CollecTRI TFs by `score`
#     (a Series/dict giving that pool's own ranking value per gene), then for each TF add up to
#     `n_targets` of its CollecTRI-curated targets that are ALSO in `gene_pool` (same-pool only).
#     Falls back to fewer TFs/targets if the pool doesn't have enough. Returns (sorted gene list,
#     detail list of (tf, [(target, score), ...]))."""
#     pool_set = set(gene_pool)
#     tfs_with_targets = []
#     for tf in gene_pool:
#         tgts = collectri_tf_targets.get(tf)
#         if not tgts:
#             continue
#         in_pool = sorted((tgts & pool_set) - {tf})
#         if in_pool:
#             tfs_with_targets.append((tf, in_pool))
#     tfs_with_targets.sort(key=lambda kv: score[kv[0]], reverse=True)
#     chosen = tfs_with_targets[:n_tfs]
#
#     genes_out, detail = set(), []
#     for tf, in_pool in chosen:
#         genes_out.add(tf)
#         ranked = sorted(in_pool, key=lambda t: score[t], reverse=True)[:n_targets]
#         genes_out.update(ranked)
#         detail.append((tf, [(t, float(score[t])) for t in ranked]))
#     return sorted(genes_out), detail


# Original variability design: the top-N_HVG genes (scanpy seurat flavor, batch_key="library"),
# ranked by normalized dispersion and split into three EQUAL-COUNT thirds -- so variability_low is
# the least-variable third *of the HVGs* (still above the HVG cut), not below-average genes.
# Then edges_to_panel() on the CollecTRI edges inside each third (build_panel_set.py's edge recipe).
sc.pp.highly_variable_genes(A, n_top_genes=N_HVG, batch_key="library")
hvg_genes = A.var_names[A.var["highly_variable"]]
assert len(hvg_genes) == N_HVG
hvg_rank = A.var.loc[hvg_genes, "dispersions_norm"].sort_values(ascending=False)

third = N_HVG // 3
variability_pools = {"high": hvg_rank.index[:third].tolist(),
                     "mid":  hvg_rank.index[third:2 * third].tolist(),
                     "low":  hvg_rank.index[2 * third:].tolist()}
print(f"{N_HVG} HVGs, dispersion terciles by rank:  "
      f"high [{hvg_rank.iloc[third-1]:.2f}, {hvg_rank.iloc[0]:.2f}]   "
      f"mid [{hvg_rank.iloc[2*third-1]:.2f}, {hvg_rank.iloc[third]:.2f}]   "
      f"low [{hvg_rank.iloc[-1]:.2f}, {hvg_rank.iloc[2*third]:.2f}]")

variability_high, variability_high_detail = edges_to_panel(band_edges(variability_pools["high"]))
variability_mid,  variability_mid_detail  = edges_to_panel(band_edges(variability_pools["mid"]))
variability_low,  variability_low_detail  = edges_to_panel(band_edges(variability_pools["low"]))

for lvl, gs, dt in (("high", variability_high, variability_high_detail),
                    ("mid", variability_mid, variability_mid_detail),
                    ("low", variability_low, variability_low_detail)):
    print(f"variability_{lvl:<4}: {len(gs)} genes ({len(dt)} TFs + up to {N_TARGETS_PER_TF} targets each)")
print("variability_high TFs:", [tf for tf, _ in variability_high_detail])

# ===== SUPERSEDED (2026-09-08): build_panel_set.py-style dispersion-QUANTILE terciles
# over the CollecTRI-connected genes -- made variability_low = below-average-dispersion
# housekeeping genes, not HVGs. Reverted to the original top-N_HVG design above. =====
# # build_panel_set.py's HVG strip: normalized dispersion (scanpy seurat flavor, no batch key) over
# # the CollecTRI-connected genes, split into dispersion-quantile terciles (VAR_BANDS); High = top
# # tercile. Then edges_to_panel() on the CollecTRI edges inside each tercile.
# sc.pp.highly_variable_genes(A, n_top_genes=N_HVG, flavor="seurat")
# disp_norm = A.var["dispersions_norm"]
# 
# var_pool = [g for g in need if np.isfinite(disp_norm[g])]
# _v = np.array([disp_norm[g] for g in var_pool])
# variability_pools = {}
# for lvl, (q0, q1) in zip(("low", "mid", "high"), VAR_BANDS):
#     lo, hi = np.quantile(_v, [q0, q1])
#     variability_pools[lvl] = [g for g, val in zip(var_pool, _v) if lo <= val <= hi]
#     print(f"variability_{lvl:<4} pool: {len(variability_pools[lvl])} genes, dispersion in [{lo:.3f}, {hi:.3f}]")
# 
# variability_high, variability_high_detail = edges_to_panel(band_edges(variability_pools["high"]))
# variability_mid,  variability_mid_detail  = edges_to_panel(band_edges(variability_pools["mid"]))
# variability_low,  variability_low_detail  = edges_to_panel(band_edges(variability_pools["low"]))
# 
# for lvl, gs, dt in (("high", variability_high, variability_high_detail),
#                     ("mid", variability_mid, variability_mid_detail),
#                     ("low", variability_low, variability_low_detail)):
#     print(f"variability_{lvl:<4}: {len(gs)} genes ({len(dt)} TFs + up to {N_TARGETS_PER_TF} targets each)")
# print("variability_high TFs:", [tf for tf, _ in variability_high_detail])

# ===== SUPERSEDED: top-N_HVG pool split into equal thirds by RANK, tf_target_subset recipe =====
# sc.pp.highly_variable_genes(A, n_top_genes=N_HVG, batch_key="library")
# hvg_genes = A.var_names[A.var["highly_variable"]]
# assert len(hvg_genes) == N_HVG
# hvg_rank = A.var.loc[hvg_genes, "dispersions_norm"].sort_values(ascending=False)
#
# third = N_HVG // 3
# variability_high_pool = hvg_rank.index[:third].tolist()
# variability_mid_pool = hvg_rank.index[third:2 * third].tolist()
# variability_low_pool = hvg_rank.index[2 * third:].tolist()
#
# print(f"variability_high pool: {len(variability_high_pool)} genes, dispersion range "
#       f"[{hvg_rank.iloc[third - 1]:.3f}, {hvg_rank.iloc[0]:.3f}]")
# print(f"variability_mid  pool: {len(variability_mid_pool)} genes, dispersion range "
#       f"[{hvg_rank.iloc[2 * third - 1]:.3f}, {hvg_rank.iloc[third]:.3f}]")
# print(f"variability_low  pool: {len(variability_low_pool)} genes, dispersion range "
#       f"[{hvg_rank.iloc[-1]:.3f}, {hvg_rank.iloc[2 * third]:.3f}]")
#
# # same recipe as Correlation: top N_HIGH_TFS CollecTRI TFs within this tier (ranked by the tier's
# # own dispersion), plus up to N_TARGETS_PER_TF of each TF's curated targets -- restricted to the
# # SAME tier, so the final list stays entirely within its own variability band.
# variability_high, variability_high_detail = tf_target_subset(variability_high_pool, hvg_rank)
# variability_mid, variability_mid_detail = tf_target_subset(variability_mid_pool, hvg_rank)
# variability_low, variability_low_detail = tf_target_subset(variability_low_pool, hvg_rank)
#
# print(f"\nvariability_high: {len(variability_high)} genes "
#       f"({len(variability_high_detail)} TFs + up to {N_TARGETS_PER_TF} same-tier curated targets each)")
# print(f"variability_mid:  {len(variability_mid)} genes ({len(variability_mid_detail)} TFs)")
# print(f"variability_low:  {len(variability_low)} genes ({len(variability_low_detail)} TFs)")
# print("\nvariability_high TFs:", [tf for tf, _ in variability_high_detail])
#


# build_panel_set.py's detection strip: detection fraction over the CollecTRI-connected genes, split
# into detection-quantile bands DETECT_BANDS (50-75 / 75-90 / 90-100); High = top band. Detection
# here stays the notebook's per-day MINIMUM (the retained floor convention), not build_panel_set.py's
# pooled fraction. Then edges_to_panel() on the CollecTRI edges inside each band.
detection_frac = pd.Series(frac_expr, index=genes)

det_pool = list(need)
_d = np.array([detection_frac[g] for g in det_pool])
detection_pools = {}
for lvl, (q0, q1) in zip(("low", "mid", "high"), DETECT_BANDS):
    lo, hi = np.quantile(_d, [q0, q1])
    detection_pools[lvl] = [g for g, val in zip(det_pool, _d) if lo <= val <= hi]
    print(f"detection_{lvl:<4} pool: {len(detection_pools[lvl])} genes, detection in [{lo:.1%}, {hi:.1%}]")

detection_high, detection_high_detail = edges_to_panel(band_edges(detection_pools["high"]))
detection_mid,  detection_mid_detail  = edges_to_panel(band_edges(detection_pools["mid"]))
detection_low,  detection_low_detail  = edges_to_panel(band_edges(detection_pools["low"]))

for lvl, gs, dt in (("high", detection_high, detection_high_detail),
                    ("mid", detection_mid, detection_mid_detail),
                    ("low", detection_low, detection_low_detail)):
    print(f"detection_{lvl:<4}: {len(gs)} genes ({len(dt)} TFs + up to {N_TARGETS_PER_TF} targets each)")
print("detection_high TFs:", [tf for tf, _ in detection_high_detail])

# ===== SUPERSEDED: p50/p75/p90 percentiles over ALL floor-surviving genes, tf_target_subset =====
# detection = pd.Series(frac_expr, index=genes).sort_values(ascending=False)
# p50, p75, p90 = np.percentile(detection.to_numpy(), [50, 75, 90])
#
# detection_high_pool = detection[detection >= p90].index.tolist()
# detection_mid_pool = detection[(detection >= p75) & (detection < p90)].index.tolist()
# detection_low_pool = detection[(detection >= p50) & (detection < p75)].index.tolist()
#
# print(f"detection thresholds: p50={p50:.1%}  p75={p75:.1%}  p90={p90:.1%}")
# print(f"detection_high pool (>=90th pct): {len(detection_high_pool)} genes")
# print(f"detection_mid  pool (75-90th pct): {len(detection_mid_pool)} genes")
# print(f"detection_low  pool (50-75th pct): {len(detection_low_pool)} genes")
#
# # same recipe as Variability/Correlation: top N_HIGH_TFS CollecTRI TFs within this tier (ranked by
# # the tier's own detection fraction), plus up to N_TARGETS_PER_TF of each TF's curated targets --
# # restricted to the SAME tier, so the final list stays entirely within its own detection band.
# detection_high, detection_high_detail = tf_target_subset(detection_high_pool, detection)
# detection_mid, detection_mid_detail = tf_target_subset(detection_mid_pool, detection)
# detection_low, detection_low_detail = tf_target_subset(detection_low_pool, detection)
#
# print(f"\ndetection_high: {len(detection_high)} genes "
#       f"({len(detection_high_detail)} TFs + up to {N_TARGETS_PER_TF} same-tier curated targets each)")
# print(f"detection_mid:  {len(detection_mid)} genes ({len(detection_mid_detail)} TFs)")
# print(f"detection_low:  {len(detection_low)} genes ({len(detection_low_detail)} TFs)")
# print("\ndetection_high TFs:", [tf for tf, _ in detection_high_detail])
#


# build_panel_set.py's correlation method: rank EVERY CollecTRI edge (both ends in the floored
# matrix) by |rho|, then High / Mid / Low = the top / middle / bottom N_BAND_EDGES edges of that
# ranking (build_panel_set.py: o[:50], o[mid-25:mid+25], o[-50:]). NOT HVG-restricted, no null,
# no decoys. Then edges_to_panel(): 15 TFs ranked by the MEDIAN |rho| of their edges in the
# slice (the band is already a |rho| slice, so edge strength -- not count -- is the point) plus
# each TF's 4 highest-|rho| targets. correlation_low ranks TFs by ascending median |rho|.
ranked_edges = sorted(collectri_pairs, key=lambda p: abs_rho[p], reverse=True)
_mid, _h = len(ranked_edges) // 2, N_BAND_EDGES
corr_edges = {
    "high": ranked_edges[:_h],
    "mid":  ranked_edges[_mid - _h // 2: _mid - _h // 2 + _h],
    "low":  ranked_edges[-_h:],
}
for lvl in ("high", "mid", "low"):
    r = [abs_rho[p] for p in corr_edges[lvl]]
    print(f"correlation_{lvl:<4}: {len(r)} edges, |rho| in [{min(r):.3f}, {max(r):.3f}]")

correlation_high, high_detail = edges_to_panel(corr_edges["high"], rank_by="rho_median")
correlation_mid,  mid_detail  = edges_to_panel(corr_edges["mid"],  rank_by="rho_median")
correlation_low,  low_detail  = edges_to_panel(corr_edges["low"],  rank_by="rho_median", ascending=True)

for lvl, gs, dt in (("high", correlation_high, high_detail),
                    ("mid", correlation_mid, mid_detail),
                    ("low", correlation_low, low_detail)):
    print(f"correlation_{lvl:<4}: {len(gs)} genes ({len(dt)} TFs + up to {N_TARGETS_PER_TF} targets each)")
print("correlation_high TFs:", [tf for tf, _ in high_detail])

# ===== SUPERSEDED: HVG-restricted candidate set + TF x HVG Spearman matrix build =====
# # collectri_tf_targets was already loaded in the shared "CollecTRI + TF/target-subset recipe"
# # cell above (before Variability) -- Correlation just reuses it, now scoped to the HVG pool.
# hvg_set = set(hvg_genes)
# candidate_tfs = [g for g in hvg_genes if g in collectri_tf_targets]
# tf_curated_targets = {
#     tf: sorted((collectri_tf_targets[tf] & hvg_set) - {tf})
#     for tf in candidate_tfs
# }
# candidate_tfs = [tf for tf in candidate_tfs if len(tf_curated_targets[tf]) > 0]
# print(f"candidate TFs (CollecTRI ∩ HVG, with >=1 curated target also in HVG): {len(candidate_tfs)}")
#
# Xh = A[:, hvg_genes].X
# Xh = Xh.toarray() if sp.issparse(Xh) else np.asarray(Xh)
# hvg_index = {g: i for i, g in enumerate(hvg_genes)}
# tf_idx = np.array([hvg_index[g] for g in candidate_tfs])
# # Decoy/target universe is the FULL HVG pool, matching curated_vs_random_coexpr.py's use of the
# # whole detectable gene set (their `gl`/`det_all[keep]`) -- NOT restricted to genes that happen to
# # already be a curated target for someone. A decoy drawn only from other curated targets would be
# # a biased comparison; the whole point of their design is "any gene of matched detection".
# candidate_targets = list(hvg_genes)
# tgt_idx = np.arange(len(hvg_genes))
#
# target_index = hvg_index  # candidate_targets == hvg_genes, so this is just an alias
# curated_mask = np.zeros((len(candidate_tfs), len(candidate_targets)), dtype=bool)
# for i, tf in enumerate(candidate_tfs):
#     for tgt in tf_curated_targets[tf]:
#         curated_mask[i, target_index[tgt]] = True
# print(f"curated (TF,target) cells to rank: {curated_mask.sum():,} / {curated_mask.size:,} "
#       f"dense-matrix entries")
#
#
# def spearman_matrix(mat, rows_idx, cols_idx):
#     """Spearman correlation: Pearson on rank-transformed columns, matching
#     curated_vs_random_coexpr.py's Rk = rankdata(X); Rk -= mean; Rk /= std; rho = Rk.T @ Rk / n."""
#     from scipy.stats import rankdata
#     Rk = np.apply_along_axis(rankdata, 0, mat).astype(np.float64)
#     Rk = (Rk - Rk.mean(axis=0)) / np.maximum(Rk.std(axis=0), 1e-9)
#     Rr, Rc = Rk[:, rows_idx], Rk[:, cols_idx]
#     return (Rr.T @ Rc) / Rr.shape[0]
#
#
# R = spearman_matrix(Xh, tf_idx, tgt_idx)  # (n_candidate_tfs, n_hvg_genes), Spearman rho
# print(f"real TF x (full HVG pool) Spearman matrix: {R.shape}, "
#       f"|rho| range over curated cells [{np.abs(R)[curated_mask].min():.3f}, "
#       f"{np.abs(R)[curated_mask].max():.3f}]")
#


# ===== SUPERSEDED: permutation-null / decile-matched-decoy diagnostic (curated_vs_random_coexpr.py) =====
# Kept for reference -- a valid standalone check that curated edges are barely more co-expressed
# than detection-matched decoys -- but it no longer feeds panel construction.

# from scipy import stats
#
# # curated_vs_random_coexpr.py's exact decoy design: for each curated (TF, target) edge, draw a
# # decoy = the SAME TF against a RANDOM gene of the full HVG pool matched on detection DECILE (not a
# # per-gene shuffle, and not restricted to other curated targets) -- so the only difference from a
# # curated edge is whether CollecTRI records it, not expression level or curation status of the decoy.
# det_of = pd.Series(frac_expr, index=genes)
# hvg_det = det_of.loc[hvg_genes].to_numpy()
# decile_edges = np.quantile(hvg_det, np.linspace(0, 1, 11)[1:-1])
# decile = np.digitize(hvg_det, decile_edges)
# by_decile = {d: np.flatnonzero(decile == d) for d in range(10)}
# print("decile pool sizes:", {d: len(v) for d, v in by_decile.items()})
#
# real_pairs = {(i, j) for i in range(len(candidate_tfs)) for j in range(len(candidate_targets)) if curated_mask[i, j]}
# obs_absr, dec_absr = [], []
# for (i, j) in real_pairs:
#     obs_absr.append(abs(R[i, j]))
#     pool = by_decile[decile[j]]
#     for _ in range(20):
#         c = int(rng.choice(pool))
#         if c != j and (i, c) not in real_pairs:
#             dec_absr.append(abs(R[i, c]))
#             break
# obs_absr, dec_absr = np.array(obs_absr), np.array(dec_absr)
# print(f"curated pairs: {len(obs_absr)}  |  matched decoys found: {len(dec_absr)}")
#
# u = stats.mannwhitneyu(obs_absr, dec_absr, alternative="greater")
# auc = u.statistic / (len(obs_absr) * len(dec_absr))
# print(f"curated |rho|: median {np.median(obs_absr):.4f}  mean {obs_absr.mean():.4f}")
# print(f"decoy   |rho|: median {np.median(dec_absr):.4f}  mean {dec_absr.mean():.4f}  "
#       f"(detection-decile-matched, drawn from the full HVG pool)")
# print(f"curated vs matched decoys: AUC {auc:.4f}  P = {u.pvalue:.3e}  "
#       f"(AUC 0.5 = a curated edge is no more co-expressed than a matched non-edge)")
#
# NULL_CUTOFF = float(np.percentile(dec_absr, NULL_PERCENTILE))
# print(f"\nNULL_CUTOFF ({NULL_PERCENTILE}th percentile of the matched-decoy |rho|) = {NULL_CUTOFF:.4f}")
# real_above_null = (obs_absr > NULL_CUTOFF).mean()
# print(f"fraction of REAL, CURATED TF-target pairs clearing NULL_CUTOFF: {real_above_null:.1%}")
#


# ===== SUPERSEDED: null-cutoff / percentile correlation banding -> N_BAND_EDGES slices above =====

# # ---- High: TFs with the best average |r| against their OWN curated targets, plus those top targets ----
# R_masked = np.where(curated_mask, np.abs(R), np.nan)
#
# tf_mean_absr = pd.Series(np.nanmean(R_masked, axis=1), index=candidate_tfs).sort_values(ascending=False)
# high_tfs = tf_mean_absr.index[:N_HIGH_TFS].tolist()
# print("top TFs by mean |r| with their own CollecTRI-curated targets:")
# print(tf_mean_absr.head(N_HIGH_TFS).round(3))
#
# correlation_high = set(high_tfs)
# high_detail = []
# for tf in high_tfs:
#     i = candidate_tfs.index(tf)
#     row = pd.Series(R_masked[i], index=candidate_targets).dropna().sort_values(ascending=False)
#     top_targets = row.index[:N_TARGETS_PER_TF].tolist()
#     correlation_high.update(top_targets)
#     high_detail.append((tf, list(zip(top_targets, row.iloc[:N_TARGETS_PER_TF].round(3).tolist()))))
# correlation_high = sorted(correlation_high)
# print(f"\ncorrelation_high: {len(correlation_high)} genes "
#       f"({len(high_tfs)} TFs + up to {N_TARGETS_PER_TF} curated targets each)")
#
# # ---- Low / Mid: CURATED pairs only, banded by |r|, same TF/target shape as High for comparability ----
# all_pairs = pd.DataFrame(
#     [(candidate_tfs[i], candidate_targets[j], abs(R[i, j]))
#      for i in range(len(candidate_tfs)) for j in range(len(candidate_targets)) if curated_mask[i, j]],
#     columns=["tf", "target", "abs_r"])
#
# low_hi = NULL_CUTOFF * LOW_BAND_WIDTH
# mid_lo, mid_hi = np.percentile(all_pairs["abs_r"], [MID_BAND[0] * 100, MID_BAND[1] * 100])
# print(f"\nNULL_CUTOFF={NULL_CUTOFF:.3f}  low band=[{NULL_CUTOFF:.3f}, {low_hi:.3f}]  "
#       f"mid band=[{mid_lo:.3f}, {mid_hi:.3f}]")
#
# low_pairs = all_pairs[all_pairs["abs_r"].between(NULL_CUTOFF, low_hi)]
# mid_pairs = all_pairs[all_pairs["abs_r"].between(mid_lo, mid_hi)]
# print(f"curated pairs in low band: {len(low_pairs):,}  |  curated pairs in mid band: {len(mid_pairs):,}")
#
#
# def pick_list(pairs_in_band, n_tfs=N_HIGH_TFS, n_targets=N_TARGETS_PER_TF):
#     tf_counts = pairs_in_band["tf"].value_counts()
#     chosen_tfs = tf_counts.index[:n_tfs].tolist()
#     genes_out, detail = set(chosen_tfs), []
#     for tf in chosen_tfs:
#         sub = pairs_in_band[pairs_in_band["tf"] == tf].sort_values("abs_r", ascending=False)
#         tgts = sub["target"].head(n_targets).tolist()
#         genes_out.update(tgts)
#         detail.append((tf, list(zip(tgts, sub["abs_r"].head(n_targets).round(3).tolist()))))
#     return sorted(genes_out), detail
#
#
# correlation_low, low_detail = pick_list(low_pairs)
# correlation_mid, mid_detail = pick_list(mid_pairs)
# print(f"\ncorrelation_low: {len(correlation_low)} genes")
# print(f"correlation_mid: {len(correlation_mid)} genes")
#


import json

gene_sets = {
    "variability_high": variability_high,
    "variability_mid": variability_mid,
    "variability_low": variability_low,
    "detection_high": detection_high,
    "detection_mid": detection_mid,
    "detection_low": detection_low,
    "correlation_high": correlation_high,
    "correlation_mid": correlation_mid,
    "correlation_low": correlation_low,
}

for name, genes_ in gene_sets.items():
    print(f"{name:<18} {len(genes_):>5} genes")

# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] out_path = os.path.join(HERE, "resources", "gene_sets.json")
out_path = os.path.join(RES_HERE, "resources", "gene_sets.json")
json.dump(gene_sets, open(out_path, "w"), indent=2)
print(f"\nwrote {out_path}")

# per-TF detail (which TF pulled in which targets, at what score) for all 9 lists, for inspection
gene_sets_detail = {
    "variability": {"high": variability_high_detail, "mid": variability_mid_detail,
                     "low": variability_low_detail},
    "detection": {"high": detection_high_detail, "mid": detection_mid_detail,
                   "low": detection_low_detail},
    "correlation": {"high": high_detail, "mid": mid_detail, "low": low_detail},
}
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] detail_path = os.path.join(HERE, "resources", "gene_sets_detail.json")
detail_path = os.path.join(RES_HERE, "resources", "gene_sets_detail.json")
json.dump(gene_sets_detail, open(detail_path, "w"), indent=2)
print(f"wrote {detail_path}")
