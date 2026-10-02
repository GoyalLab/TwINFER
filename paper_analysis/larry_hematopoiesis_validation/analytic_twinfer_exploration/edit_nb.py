from twinfer.utils.paths import get_repo_root as _twinfer_get_repo_root  # [2026-09-30 added]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Rewrite pick_gene_sets.ipynb to follow helpers/build_panel_set.py's edge-selection method,
keeping the 15-TF / 4-target recipe and dropping panel12. Old code is commented in place
(repo convention), never deleted."""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json, sys

# [2026-09-30 commented out: pointed into the original code tree; now the clean_code copy] NB = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation/pick_gene_sets.ipynb'
NB = f'{_twinfer_get_repo_root()}/paper_analysis/larry_hematopoiesis_validation/pick_gene_sets.ipynb'
nb = json.load(open(NB))
by_id = {c["id"]: c for c in nb["cells"]}


def as_lines(text):
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


def commented(cell, header):
    old = "".join(cell["source"])
    out = [header + "\n"]
    for l in old.split("\n"):
        out.append(("# " + l).rstrip() + "\n")
    return out


def set_src(cid, text):
    by_id[cid]["source"] = as_lines(text)
    if by_id[cid]["cell_type"] == "code":
        by_id[cid]["outputs"] = []
        by_id[cid]["execution_count"] = None


def append_commented(cid, new_text, header):
    old_block = commented(by_id[cid], header)
    by_id[cid]["source"] = as_lines(new_text) + ["\n\n"] + old_block
    by_id[cid]["outputs"] = []
    by_id[cid]["execution_count"] = None


# ---------------------------------------------------------------- markdown -----
set_src("36819129", '''# Identifying gene sets for inference

Builds 9 gene lists for inference -- 3 criteria x 3 levels (high/mid/low) -- following the method of
`helpers/build_panel_set.py` ("THE panel set, one recipe, every dataset") from **Transcriptomic
Distance**, the closely related project on this same LARRY dataset. Two deliberate departures from
that script, both requested: the per-band recipe keeps **15 TFs x up to 4 curated targets** (the
script uses 10 x 6), and **panel12 is not built**. Normalization stays `log1p(CP10k)` and the
detection floor stays the notebook's per-day-minimum >=5% (from `hvg_tfs_by_detection.py`); every
other choice matches `build_panel_set.py`.

**Shared universe.** All nine lists are built from CollecTRI curated (TF, target) edges whose *both*
endpoints survive the detection floor -- not an HVG subset. One Spearman pass (|rho| = |Pearson on
cell-rank|) scores every such edge once, reused by all three criteria.

1. **Variability** -- rank those genes by scanpy's normalized dispersion (seurat flavor), split into
   dispersion-quantile terciles; within each tercile take the CollecTRI edges with both ends in the
   tercile, then the top 15 TFs by in-tercile edge count + their 4 highest-|rho| targets.
2. **Detection** -- same construction on detection-fraction quantile bands (50-75 / 75-90 / 90-100th
   percentile of the CollecTRI-connected genes).
3. **Correlation** -- rank *all* curated edges by |rho|; High / Mid / Low = the top / middle / bottom
   `N_BAND_EDGES` (=50) edges of that ranking, then the same 15-TF / 4-target recipe on each slice.

TF/non-TF status and each TF's curated targets come from **CollecTRI**
(`resources/collectri_mouse.tsv`, mouse: 1,106 TFs, 39,847 curated edges after self-pairs stripped).
All thresholds are named constants in the config cell below -- rerun downstream cells after changing
them.
''')

set_src("64d96f3b", '''## Shared: CollecTRI + |rho| for every curated edge + the 15-TF / 4-target recipe

`build_panel_set.py` computes one thing up front and reuses it everywhere: the set of CollecTRI
edges whose both endpoints are in the (detection-floored) matrix, and |Spearman rho| for each --
Pearson on per-gene cell ranks, its exact `|R^T R| / (|R| |R|)` formula. `edges_to_panel()` is the
shared recipe: given any set of edges (a dispersion tercile, a detection band, or a |rho| slice),
keep the 15 TFs with the most edges in the set, then each TF's 4 highest-|rho| targets in the set.
''')

set_src("923c5dcd", '''## 1. Variability -- high / mid / low

`build_panel_set.py`'s HVG strip: score the CollecTRI-connected genes by scanpy's normalized
dispersion (seurat flavor, no batch key), split into dispersion-quantile terciles (`VAR_BANDS`),
High = top tercile. Within each tercile, `edges_to_panel()` on the CollecTRI edges that have both
endpoints in the tercile.
''')

set_src("b7e4e540", '''## 2. Detection -- high / mid / low

`build_panel_set.py`'s detection strip: score the CollecTRI-connected genes by detection fraction,
split into quantile bands `DETECT_BANDS` (50-75 / 75-90 / 90-100th percentile), High = top band.
Detection fraction here stays the notebook's **per-day minimum** (the retained floor convention),
where `build_panel_set.py` uses the pooled fraction. Within each band, `edges_to_panel()` on the
CollecTRI edges with both endpoints in the band.
''')

set_src("1c4957b0", '''## 3. Correlation -- high / mid / low

`build_panel_set.py`'s correlation families: rank *every* curated edge (both ends in the floored
matrix) by |rho|, then High / Mid / Low = `o[:N]`, `o[mid-N//2 : mid+N//2]`, `o[-N:]` with
`N = N_BAND_EDGES` (50). No HVG restriction, no permutation null, no decoy matching -- those built
the old bands and are kept commented below (they remain a valid standalone diagnostic, matching
`curated_vs_random_coexpr.py`). `edges_to_panel()` then applies the 15-TF / 4-target recipe to each
50-edge slice; a slice with fewer than 15 distinct TFs yields fewer (as the variability / detection
low bands already do).
''')

# ------------------------------------------------------------------- config ----
set_src("bae0fd56", '''import os

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io as sio
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath("__file__"))
from twinfer.utils.paths import get_data_root as _res_gdr  # [2026-10-01 LARRY resources/ moved out of the code tree to analysis_data/paper_analysis/larry_hematopoiesis_validation/resources (user)]
RES_HERE = f"{_res_gdr()}/paper_analysis/larry_hematopoiesis_validation"
SOURCE = "/home/gzu5140/TwINFER_KA/finalized_data/LARRY_data/filtered"
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
''')

# --------------------------------------------------- shared CollecTRI cell -----
collectri_keep = '''collectri_raw = pd.read_csv(COLLECTRI_PATH, sep="\\t")

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


def edges_to_panel(edges, n_tfs=N_HIGH_TFS, n_targets=N_TARGETS_PER_TF):
    """The 15-TF / 4-target recipe on a set of (tf, target) edges: keep the `n_tfs` TFs with the
    most edges in the set (ties -> higher mean |rho|), then each TF's `n_targets` highest-|rho|
    targets in the set. build_panel_set.py's construction with 15/4 instead of its 10/6. Returns
    (sorted gene list, [(tf, [(target, |rho|), ...]), ...])."""
    by_tf = {}
    for tf, tgt in edges:
        by_tf.setdefault(tf, []).append(tgt)
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
    return sorted(genes_out), detail'''

old_tf_target_subset = '''

def tf_target_subset(gene_pool, score, n_tfs=N_HIGH_TFS, n_targets=N_TARGETS_PER_TF):
    """From `gene_pool` (a band's full gene list), pick the top `n_tfs` CollecTRI TFs by `score`
    (a Series/dict giving that pool's own ranking value per gene), then for each TF add up to
    `n_targets` of its CollecTRI-curated targets that are ALSO in `gene_pool` (same-pool only).
    Falls back to fewer TFs/targets if the pool doesn't have enough. Returns (sorted gene list,
    detail list of (tf, [(target, score), ...]))."""
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

    genes_out, detail = set(), []
    for tf, in_pool in chosen:
        genes_out.add(tf)
        ranked = sorted(in_pool, key=lambda t: score[t], reverse=True)[:n_targets]
        genes_out.update(ranked)
        detail.append((tf, [(t, float(score[t])) for t in ranked]))
    return sorted(genes_out), detail'''

comm_old = "\n".join(
    "# ===== SUPERSEDED: tf_target_subset (pool-score recipe) -> edges_to_panel above =====".splitlines()
    + [("# " + l).rstrip() for l in old_tf_target_subset.split("\n")]
)
set_src("4f75d53a", collectri_keep + "\n\n\n" + comm_old + "\n")

# ---------------------------------------------------------------- variability --
append_commented("2eaebe1a", '''# build_panel_set.py's HVG strip: normalized dispersion (scanpy seurat flavor, no batch key) over
# the CollecTRI-connected genes, split into dispersion-quantile terciles (VAR_BANDS); High = top
# tercile. Then edges_to_panel() on the CollecTRI edges inside each tercile.
sc.pp.highly_variable_genes(A, n_top_genes=N_HVG, flavor="seurat")
disp_norm = A.var["dispersions_norm"]

var_pool = [g for g in need if np.isfinite(disp_norm[g])]
_v = np.array([disp_norm[g] for g in var_pool])
variability_pools = {}
for lvl, (q0, q1) in zip(("low", "mid", "high"), VAR_BANDS):
    lo, hi = np.quantile(_v, [q0, q1])
    variability_pools[lvl] = [g for g, val in zip(var_pool, _v) if lo <= val <= hi]
    print(f"variability_{lvl:<4} pool: {len(variability_pools[lvl])} genes, dispersion in [{lo:.3f}, {hi:.3f}]")

variability_high, variability_high_detail = edges_to_panel(band_edges(variability_pools["high"]))
variability_mid,  variability_mid_detail  = edges_to_panel(band_edges(variability_pools["mid"]))
variability_low,  variability_low_detail  = edges_to_panel(band_edges(variability_pools["low"]))

for lvl, gs, dt in (("high", variability_high, variability_high_detail),
                    ("mid", variability_mid, variability_mid_detail),
                    ("low", variability_low, variability_low_detail)):
    print(f"variability_{lvl:<4}: {len(gs)} genes ({len(dt)} TFs + up to {N_TARGETS_PER_TF} targets each)")
print("variability_high TFs:", [tf for tf, _ in variability_high_detail])''',
    "# ===== SUPERSEDED: top-N_HVG pool split into equal thirds by RANK, tf_target_subset recipe =====")

# ----------------------------------------------------------------- detection ---
append_commented("0fa4be8b", '''# build_panel_set.py's detection strip: detection fraction over the CollecTRI-connected genes, split
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
print("detection_high TFs:", [tf for tf, _ in detection_high_detail])''',
    "# ===== SUPERSEDED: p50/p75/p90 percentiles over ALL floor-surviving genes, tf_target_subset =====")

# --------------------------------------------------------------- correlation ---
append_commented("b58286e0", '''# build_panel_set.py's correlation method: rank EVERY CollecTRI edge (both ends in the floored
# matrix) by |rho|, then High / Mid / Low = the top / middle / bottom N_BAND_EDGES edges of that
# ranking (build_panel_set.py: o[:50], o[mid-25:mid+25], o[-50:]). NOT HVG-restricted, no null,
# no decoys. Then edges_to_panel() -- the 15-TF / 4-target recipe -- on each edge slice.
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

correlation_high, high_detail = edges_to_panel(corr_edges["high"])
correlation_mid,  mid_detail  = edges_to_panel(corr_edges["mid"])
correlation_low,  low_detail  = edges_to_panel(corr_edges["low"])

for lvl, gs, dt in (("high", correlation_high, high_detail),
                    ("mid", correlation_mid, mid_detail),
                    ("low", correlation_low, low_detail)):
    print(f"correlation_{lvl:<4}: {len(gs)} genes ({len(dt)} TFs + up to {N_TARGETS_PER_TF} targets each)")
print("correlation_high TFs:", [tf for tf, _ in high_detail])''',
    "# ===== SUPERSEDED: HVG-restricted candidate set + TF x HVG Spearman matrix build =====")

_old13 = "".join(by_id["e56c69a3"]["source"])
set_src("e56c69a3", "\n".join(
    ["# ===== SUPERSEDED: permutation-null / decile-matched-decoy diagnostic (curated_vs_random_coexpr.py) =====",
     "# Kept for reference -- a valid standalone check that curated edges are barely more co-expressed",
     "# than detection-matched decoys -- but it no longer feeds panel construction.", ""]
    + [("# " + l).rstrip() for l in _old13.split("\n")]
) + "\n")

_old14 = "".join(by_id["3cd37509"]["source"])
set_src("3cd37509", "\n".join(
    ["# ===== SUPERSEDED: null-cutoff / percentile correlation banding -> N_BAND_EDGES slices above =====", ""]
    + [("# " + l).rstrip() for l in _old14.split("\n")]
) + "\n")

json.dump(nb, open(NB, "w"), indent=1)
json.load(open(NB))  # parse check
print("OK: rewrote", NB)
