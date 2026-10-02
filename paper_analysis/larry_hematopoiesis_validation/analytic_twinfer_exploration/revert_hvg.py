from twinfer.utils.paths import get_repo_root as _twinfer_get_repo_root  # [2026-09-30 added]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Revert the variability criterion to the original design: top-N_HVG genes split into
equal-count dispersion thirds. Keep the current edges_to_panel(band_edges(...)) recipe."""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
# [2026-09-30 commented out: pointed into the original code tree; now the clean_code copy] NB = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation/pick_gene_sets.ipynb'
NB = f'{_twinfer_get_repo_root()}/paper_analysis/larry_hematopoiesis_validation/pick_gene_sets.ipynb'
nb = json.load(open(NB))
by_id = {c["id"]: c for c in nb["cells"]}

NEW_CODE = '''# Original variability design: the top-N_HVG genes (scanpy seurat flavor, batch_key="library"),
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
'''

old = "".join(by_id["2eaebe1a"]["source"])
# keep everything from the SUPERSEDED marker onward
marker = "\n# ===== SUPERSEDED: top-N_HVG pool split into equal thirds by RANK"
tail = old[old.index(marker):]
# also stash the just-replaced build_panel_set-quantile version as a second superseded block
prev = old[:old.index(marker)]
prev_comment = ("\n# ===== SUPERSEDED (2026-09-08): build_panel_set.py-style dispersion-QUANTILE terciles\n"
                "# over the CollecTRI-connected genes -- made variability_low = below-average-dispersion\n"
                "# housekeeping genes, not HVGs. Reverted to the original top-N_HVG design above. =====\n"
                + "\n".join("# " + l for l in prev.rstrip().split("\n")) + "\n")
by_id["2eaebe1a"]["source"] = (NEW_CODE + prev_comment + tail).splitlines(keepends=True)
by_id["2eaebe1a"]["outputs"] = []; by_id["2eaebe1a"]["execution_count"] = None

by_id["923c5dcd"]["source"] = ('''## 1. Variability -- high / mid / low

The top `N_HVG` highly variable genes (scanpy seurat flavor, `batch_key="library"`), ranked by
normalized dispersion and split into three **equal-count thirds** -- so `variability_low` is the
least-variable third *of the HVGs* (still above the HVG cut), not below-average genes. Then
`edges_to_panel()` on the CollecTRI edges with both endpoints inside each third.

(`VAR_BANDS` in the config cell is now unused -- kept only for the commented build_panel_set-style
quantile version.)
''').splitlines(keepends=True)

json.dump(nb, open(NB, "w"), indent=1)
json.load(open(NB))
print("OK: variability reverted to original top-N_HVG dispersion-tercile design")
