from twinfer.utils.paths import get_repo_root as _twinfer_get_repo_root  # [2026-09-30 added]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""correlation criterion: rank the 15 TFs by MEDIAN |rho| of their in-band edges, not edge count."""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
# [2026-09-30 commented out: pointed into the original code tree; now the clean_code copy] NB = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation/pick_gene_sets.ipynb'
NB = f'{_twinfer_get_repo_root()}/paper_analysis/larry_hematopoiesis_validation/pick_gene_sets.ipynb'
nb = json.load(open(NB))
by_id = {c["id"]: c for c in nb["cells"]}

OLD_FN = '''def edges_to_panel(edges, n_tfs=N_HIGH_TFS, n_targets=N_TARGETS_PER_TF):
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

NEW_FN = '''def edges_to_panel(edges, n_tfs=N_HIGH_TFS, n_targets=N_TARGETS_PER_TF,
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
    return sorted(genes_out), detail'''

c6 = "".join(by_id["4f75d53a"]["source"])
assert OLD_FN in c6, "edges_to_panel body not found verbatim"
by_id["4f75d53a"]["source"] = (c6.replace(OLD_FN, NEW_FN)).splitlines(keepends=True)

OLD_CALL = '''# no decoys. Then edges_to_panel() -- the 15-TF / 4-target recipe -- on each edge slice.
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
correlation_low,  low_detail  = edges_to_panel(corr_edges["low"])'''

NEW_CALL = '''# no decoys. Then edges_to_panel(): 15 TFs ranked by the MEDIAN |rho| of their edges in the
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
correlation_low,  low_detail  = edges_to_panel(corr_edges["low"],  rank_by="rho_median", ascending=True)'''

c12 = "".join(by_id["b58286e0"]["source"])
assert OLD_CALL in c12, "correlation call block not found verbatim"
by_id["b58286e0"]["source"] = (c12.replace(OLD_CALL, NEW_CALL)).splitlines(keepends=True)
by_id["b58286e0"]["outputs"] = []; by_id["b58286e0"]["execution_count"] = None
by_id["4f75d53a"]["outputs"] = []; by_id["4f75d53a"]["execution_count"] = None

json.dump(nb, open(NB, "w"), indent=1)
json.load(open(NB))
print("OK: edges_to_panel now supports rank_by='rho_median'; correlation uses it (low = ascending)")
