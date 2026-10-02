from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_175bdfed_2026-09-17_todo4v2; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import re
import pandas as pd

# [2026-09-30 commented out: result files now in clean_data/, see REPOINT_LOG.tsv] BENCH = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/work_in_progress/benchmark'
BENCH = f'{TWINFER_PROJECT_ROOT}/clean_data/benchmarks/network_benchmarks/score_results'


def topo_group_ns(ds):
    m = re.match(r"grn_n6_(e\d+_pos\d+_\w+?)_rep\d+", ds)
    return m.group(1) if m else ds


# ---------------- network_sweep (e13_pos100 + broader OFAT sweep combined) ----------------
t4_e13 = pd.read_csv(f"{BENCH}/todo4v2_network_sweep_e13_results.csv")
t4_e13["topo"] = "e13_pos100_center"
t4_broad = pd.read_csv(f"{BENCH}/todo4v2_network_sweep_final_broader_results.csv")
t4_broad["topo"] = t4_broad.dataset_id.apply(topo_group_ns)
t4_ns = pd.concat([t4_e13, t4_broad], ignore_index=True)
t4_ns["rand_row"] = t4_ns.n_true / t4_ns.n_pairs
t4_ns["todo4v2_auprc_raw"] = t4_ns.todo4v2_nogate_auprc_x * t4_ns.rand_row
t4_ns_g = t4_ns.groupby("topo").agg(todo4v2_auprc=("todo4v2_auprc_raw", "mean"),
                                     todo4v2_f1=("todo4v2_nogate_topk_f1", "mean"),
                                     rand=("rand_row", "mean"))

cc_e13 = pd.read_csv(f"{BENCH}/crosscorr_network_sweep_e13_results.csv")
cc_e13["topo"] = "e13_pos100_center"
cc_broad = pd.read_csv(f"{BENCH}/crosscorr_network_sweep_final_broader_results.csv")
cc_broad["topo"] = cc_broad.dataset_id.apply(topo_group_ns)
cc_ns = pd.concat([cc_e13, cc_broad], ignore_index=True)
cc_ns_g = cc_ns.groupby("topo").agg(crosscorr_auprc=("auprc", "mean"), crosscorr_f1=("f1", "mean"))

b_e13 = pd.read_csv(f'{TWINFER_PROJECT_ROOT}/analysis_data/network_sweep_final/e13_pos100/beeline_analysis_output.csv')
b_e13["topo"] = "e13_pos100_center"
aug = pd.read_csv(f'{TWINFER_PROJECT_ROOT}/analysis_data/synthetic_network_benchmark_20260824/summary_metrics_table.csv')
aug = aug[(aug.variant == "directed_unsigned") & (aug.dataset_id != "ALL") & (aug.method != "TwINFER")]
aug["topo"] = aug.dataset_id.apply(topo_group_ns)
aug = aug.rename(columns={"f1_topk": "f1_topk", "auprc": "auprc"})[["topo", "method", "auprc", "f1_topk"]]
aug = aug.rename(columns={"method": "algorithm"})
b_e13 = b_e13[["topo", "algorithm", "auprc", "f1_topk"]]
b_all = pd.concat([b_e13, aug], ignore_index=True)
b_piv = b_all.pivot_table(index="topo", columns="algorithm", values=["auprc", "f1_topk"], aggfunc="mean")

nets = sorted(t4_ns_g.index)
auprc_t = pd.DataFrame(index=nets)
f1_t = pd.DataFrame(index=nets)
auprc_t["TODO4v2"] = t4_ns_g.todo4v2_auprc
auprc_t["cross-corr"] = cc_ns_g.crosscorr_auprc
f1_t["TODO4v2"] = t4_ns_g.todo4v2_f1
f1_t["cross-corr"] = cc_ns_g.crosscorr_f1
for algo in sorted(b_all.algorithm.unique()):
    auprc_t[algo] = b_piv["auprc"][algo] if algo in b_piv["auprc"].columns else None
    f1_t[algo] = b_piv["f1_topk"][algo] if algo in b_piv["f1_topk"].columns else None
auprc_t["random"] = t4_ns_g.rand
f1_t["random"] = t4_ns_g.rand

print("=== network_sweep (all topologies): AUPRC (raw), methods as columns ===")
print(auprc_t.round(3).to_string())
print()
print("=== network_sweep (all topologies): F1-topk (raw), methods as columns ===")
print(f1_t.round(3).to_string())
print()

# ---------------- mixed_network_sweep, grouped by n_genes ----------------
t4_m = pd.read_csv(f"{BENCH}/todo4v2_mixed_network_sweep_results.csv")
t4_m["rand_row"] = t4_m.n_true / t4_m.n_pairs
t4_m["todo4v2_auprc_raw"] = t4_m.todo4v2_nogate_auprc_x * t4_m.rand_row
t4_m["n_genes"] = t4_m.n_pairs.apply(lambda p: 6 if p <= 30 else 10)
t4_m_g = t4_m.groupby("n_genes").agg(todo4v2_auprc=("todo4v2_auprc_raw", "mean"),
                                      todo4v2_f1=("todo4v2_nogate_topk_f1", "mean"),
                                      rand=("rand_row", "mean"), n=("dataset_id", "size"))

cc_m = pd.read_csv(f"{BENCH}/crosscorr_mixed_network_sweep_results.csv")
cc_m = cc_m.merge(t4_m[["dataset_id", "n_genes"]].drop_duplicates(), on="dataset_id", how="left")
cc_m_g = cc_m.groupby("n_genes").agg(crosscorr_auprc=("auprc", "mean"), crosscorr_f1=("f1", "mean"))

bm = pd.read_csv(f'{TWINFER_PROJECT_ROOT}/analysis_data/mixed_network_sweep/beeline_analysis_output.csv')
bm = bm.merge(t4_m[["dataset_id", "n_genes"]].drop_duplicates(), on="dataset_id", how="left")
bm_piv = bm.pivot_table(index="n_genes", columns="algorithm", values=["auprc", "f1_topk"], aggfunc="mean")

groups = sorted(t4_m_g.index, key=str)
auprc_m = pd.DataFrame(index=groups)
f1_m = pd.DataFrame(index=groups)
auprc_m["n"] = t4_m_g.n
f1_m["n"] = t4_m_g.n
auprc_m["TODO4v2"] = t4_m_g.todo4v2_auprc
auprc_m["cross-corr"] = cc_m_g.crosscorr_auprc
f1_m["TODO4v2"] = t4_m_g.todo4v2_f1
f1_m["cross-corr"] = cc_m_g.crosscorr_f1
for algo in sorted(bm.algorithm.dropna().unique()):
    auprc_m[algo] = bm_piv["auprc"][algo] if algo in bm_piv["auprc"].columns else None
    f1_m[algo] = bm_piv["f1_topk"][algo] if algo in bm_piv["f1_topk"].columns else None
auprc_m["random"] = t4_m_g.rand
f1_m["random"] = t4_m_g.rand

print("=== mixed_network_sweep (by gene count): AUPRC (raw), methods as columns ===")
print(auprc_m.round(3).to_string())
print()
print("=== mixed_network_sweep (by gene count): F1-topk (raw), methods as columns ===")
print(f1_m.round(3).to_string())
