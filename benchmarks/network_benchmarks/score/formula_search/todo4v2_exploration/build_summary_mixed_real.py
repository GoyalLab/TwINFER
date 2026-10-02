from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_175bdfed_2026-09-17_todo4v2; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import pandas as pd

# [2026-09-30 commented out: result files now in clean_data/, see REPOINT_LOG.tsv] BENCH = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/work_in_progress/benchmark'
BENCH = f'{TWINFER_PROJECT_ROOT}/analysis_data/benchmarks/network_benchmarks/score_results'


def summarize(name, t4_csv, cc_csv, beeline_csv, t4_group_key=None):
    t4 = pd.read_csv(f"{BENCH}/{t4_csv}")
    cc = pd.read_csv(f"{BENCH}/{cc_csv}")

    if t4_group_key:
        t4["group"] = t4.dataset_id.apply(t4_group_key)
    else:
        t4["group"] = t4.dataset_id

    rand_by_ds = t4.groupby("group").apply(lambda g: (g.n_true / g.n_pairs).mean(), include_groups=False)

    t4_auprc_x = t4.groupby("group").todo4v2_nogate_auprc_x.mean()
    t4_f1_raw = t4.groupby("group").todo4v2_nogate_topk_f1.mean()  # already raw, NOT a ratio

    cc_auprc = cc.groupby("dataset_id").auprc.mean()
    cc_auprc_x = cc.groupby("dataset_id").auprc_x.mean()
    cc_f1 = cc.groupby("dataset_id").f1.mean()
    cc_f1_x = cc.groupby("dataset_id").f1_x.mean()

    b = pd.read_csv(beeline_csv)
    b2 = b.copy()
    b2["rand"] = b2.dataset_id.map(rand_by_ds)
    b2 = b2.dropna(subset=["rand"])
    b2["auprc_x"] = b2.auprc / b2.rand
    b2["f1_x"] = b2.f1_topk / b2.rand
    comp = b2.groupby("algorithm")[["auprc", "auprc_x", "f1_topk", "f1_x"]].mean()
    best_algo = comp.auprc_x.idxmax()
    best_row = comp.loc[best_algo]

    print(f"=== {name} (overall mean across all replicates) ===")
    print(f"TODO4v2:      AUPRC(raw/x-rand)={(t4_auprc_x.mean()*rand_by_ds.mean()):.3f}/{t4_auprc_x.mean():.3f}   "
          f"F1-topk(raw/x-rand)={t4_f1_raw.mean():.3f}/{(t4_f1_raw.mean()/rand_by_ds.mean()):.3f}")
    print(f"cross-corr:   AUPRC(raw/x-rand)={cc_auprc.mean():.3f}/{cc_auprc_x.mean():.3f}   "
          f"F1-topk(raw/x-rand)={cc_f1.mean():.3f}/{cc_f1_x.mean():.3f}")
    print(f"best comp ({best_algo}): AUPRC(raw/x-rand)={best_row.auprc:.3f}/{best_row.auprc_x:.3f}   "
          f"F1-topk(raw/x-rand)={best_row.f1_topk:.3f}/{best_row.f1_x:.3f}")
    print()
    print("full competitor table:")
    print(comp.round(3).sort_values("auprc_x", ascending=False).to_string())
    print()


summarize("mixed_network_sweep",
          "todo4v2_mixed_network_sweep_results.csv",
          "crosscorr_mixed_network_sweep_results.csv",
          f'{TWINFER_PROJECT_ROOT}/analysis_data/mixed_network_sweep/beeline_analysis_output.csv')

summarize("real_networks",
          "todo4v2_real_networks_results.csv",
          "crosscorr_real_networks_results.csv",
          f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/real_networks/beeline_scores.csv',
          t4_group_key=lambda ds: ds.rsplit("_rep_", 1)[0])
