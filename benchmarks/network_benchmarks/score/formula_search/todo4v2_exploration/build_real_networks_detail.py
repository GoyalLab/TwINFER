from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_175bdfed_2026-09-17_todo4v2; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import pandas as pd

# [2026-09-30 commented out: result files now in clean_data/, see REPOINT_LOG.tsv] BENCH = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/work_in_progress/benchmark'
BENCH = f'{TWINFER_PROJECT_ROOT}/clean_data/benchmarks/network_benchmarks/score_results'

t4 = pd.read_csv(f"{BENCH}/todo4v2_real_networks_results.csv")
cc = pd.read_csv(f"{BENCH}/crosscorr_real_networks_results.csv")
b = pd.read_csv(f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/real_networks/beeline_scores.csv')

t4["net"] = t4.dataset_id.apply(lambda ds: ds.rsplit("_rep_", 1)[0])
rand_by_net = t4.groupby("net").apply(lambda g: (g.n_true / g.n_pairs).mean(), include_groups=False)

# TODO4v2 raw auprc/f1 aren't stored directly -- reconstruct raw = ratio * rand per row, then average
t4["rand_row"] = t4.n_true / t4.n_pairs
t4["todo4v2_auprc_raw"] = t4.todo4v2_nogate_auprc_x * t4.rand_row
t4_net = t4.groupby("net").agg(
    todo4v2_auprc=("todo4v2_auprc_raw", "mean"),
    todo4v2_f1=("todo4v2_nogate_topk_f1", "mean"),
)

cc_net = cc.groupby("dataset_id").agg(crosscorr_auprc=("auprc", "mean"), crosscorr_f1=("f1", "mean"))
cc_net.index.name = "net"

b_net = b.pivot_table(index="dataset_id", columns="algorithm", values=["auprc", "f1_topk"], aggfunc="mean")

nets = sorted(rand_by_net.index)
auprc_table = pd.DataFrame(index=nets)
f1_table = pd.DataFrame(index=nets)

auprc_table["TODO4v2"] = t4_net.todo4v2_auprc
auprc_table["cross-corr"] = cc_net.crosscorr_auprc
f1_table["TODO4v2"] = t4_net.todo4v2_f1
f1_table["cross-corr"] = cc_net.crosscorr_f1

for algo in sorted(b.algorithm.unique()):
    auprc_table[algo] = b_net["auprc"][algo] if algo in b_net["auprc"].columns else None
    f1_table[algo] = b_net["f1_topk"][algo] if algo in b_net["f1_topk"].columns else None

auprc_table["random"] = rand_by_net
f1_table["random"] = rand_by_net

n_genes = {"mCAD": 5, "Circadian_cycle": 4, "VSC": 8, "B_cell_activation": 10, "HSC": 11,
           "EMT": 17, "GSD": 19, "Pluripotent": 36}
auprc_table.insert(0, "n_genes", [n_genes.get(n, "?") for n in auprc_table.index])
f1_table.insert(0, "n_genes", [n_genes.get(n, "?") for n in f1_table.index])
auprc_table = auprc_table.sort_values("n_genes")
f1_table = f1_table.sort_values("n_genes")

print("=== real_networks: AUPRC (raw), methods as columns ===")
print(auprc_table.round(3).to_string())
print()
print("=== real_networks: F1-topk (raw), methods as columns ===")
print(f1_table.round(3).to_string())
