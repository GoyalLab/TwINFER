from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import pandas as pd

t = pd.read_csv(f'{TWINFER_PROJECT_ROOT}/analysis_data/mixed_network_sweep/twinfer_analysis_output.csv')
b = pd.read_csv(f'{TWINFER_PROJECT_ROOT}/analysis_data/mixed_network_sweep/beeline_analysis_output.csv')

print("=== row counts ===")
print("twinfer:", len(t), " beeline:", len(b))
print(b.groupby(["algorithm", "scheme"]).size().unstack())
print()

twinfer_summary = pd.DataFrame([{
    "method": "TwINFER",
    "n_runs": len(t),
    "auprc": t["auprc"].mean(),
    "auprc_undirected": t["auprc_undirected"].mean(),
    "f1_topk": t["f1_topk_all_edges"].mean(),
    "f1_natural": t["f1_natural_after_fan_out"].mean(),
    "precision_natural": t["precision_natural_after_fan_out"].mean(),
    "recall_natural": t["recall_natural_after_fan_out"].mean(),
}])

beeline_by_scheme = (
    b.groupby(["algorithm", "scheme"])
    .agg(n_runs=("auprc", "size"), auprc=("auprc", "mean"),
         auprc_undirected=("auprc_undirected", "mean"),
         f1_topk=("f1_topk", "mean"), f1_natural=("f1_natural", "mean"),
         precision_natural=("precision_natural", "mean"),
         recall_natural=("recall_natural", "mean"))
    .reset_index()
)
beeline_by_scheme["method"] = beeline_by_scheme["algorithm"] + "_" + beeline_by_scheme["scheme"]

beeline_pooled = (
    b.groupby("algorithm")
    .agg(n_runs=("auprc", "size"), auprc=("auprc", "mean"),
         auprc_undirected=("auprc_undirected", "mean"),
         f1_topk=("f1_topk", "mean"), f1_natural=("f1_natural", "mean"),
         precision_natural=("precision_natural", "mean"),
         recall_natural=("recall_natural", "mean"))
    .reset_index().rename(columns={"algorithm": "method"})
)

cols = ["method", "n_runs", "auprc", "auprc_undirected", "f1_topk", "f1_natural",
        "precision_natural", "recall_natural"]

print("=== twin_paired scheme only (fairest vs TwINFER -- same source data) ===")
tp = beeline_by_scheme[beeline_by_scheme.scheme == "twin_paired"][cols]
combo_tp = pd.concat([twinfer_summary[cols], tp], ignore_index=True).sort_values("auprc", ascending=False)
print(combo_tp.to_string(index=False))
print()

print("=== spread scheme only ===")
sp = beeline_by_scheme[beeline_by_scheme.scheme == "spread"][cols]
combo_sp = pd.concat([twinfer_summary[cols], sp], ignore_index=True).sort_values("auprc", ascending=False)
print(combo_sp.to_string(index=False))
print()

print("=== BEELINE pooled across both schemes vs TwINFER ===")
combo_pooled = pd.concat([twinfer_summary[cols], beeline_pooled[cols]], ignore_index=True).sort_values("auprc", ascending=False)
print(combo_pooled.to_string(index=False))
print()

print("=== breakdown by n_genes (6 vs 10) ===")
t6 = t[t.n_genes == 6]["auprc"].mean(); t10 = t[t.n_genes == 10]["auprc"].mean()
print(f"TwINFER auprc: n=6 -> {t6:.3f}, n=10 -> {t10:.3f}")
b2 = b.merge(t[["dataset_id"]].drop_duplicates().assign(_ok=1), on="dataset_id", how="left")
b["n_genes"] = b["dataset_id"].str.extract(r"grn_n(\d+)_").astype(int)
print(b.groupby(["algorithm", "n_genes"])["auprc"].mean().unstack())
