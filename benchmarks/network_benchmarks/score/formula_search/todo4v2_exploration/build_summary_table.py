from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_175bdfed_2026-09-17_todo4v2; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import pandas as pd
import re

# [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] t4 = pd.read_csv(f'{TWINFER_PROJECT_ROOT}/code/TwINFER/work_in_progress/benchmark/todo4v2_network_sweep_final_broader_results.csv')
t4 = pd.read_csv(f'{TWINFER_PROJECT_ROOT}/analysis_data/benchmarks/network_benchmarks/score_results/todo4v2_network_sweep_final_broader_results.csv')
# [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] cc = pd.read_csv(f'{TWINFER_PROJECT_ROOT}/code/TwINFER/work_in_progress/benchmark/crosscorr_network_sweep_final_broader_results.csv')
cc = pd.read_csv(f'{TWINFER_PROJECT_ROOT}/analysis_data/benchmarks/network_benchmarks/score_results/crosscorr_network_sweep_final_broader_results.csv')

def topo_group(ds):
    m = re.match(r'grn_n6_(e\d+_pos\d+_\w+?)_rep\d+', ds)
    return m.group(1) if m else ds

t4['topo'] = t4.dataset_id.apply(topo_group)
cc['topo'] = cc.dataset_id.apply(topo_group)

rand = t4.groupby('topo').apply(lambda g: (g.n_true / g.n_pairs).mean(), include_groups=False)

t4g = t4.groupby('topo').agg(todo4v2_auprc_x=('todo4v2_nogate_auprc_x', 'mean'),
                              todo4v2_f1_x=('todo4v2_nogate_topk_f1', 'mean'))
ccg = cc.groupby('topo').agg(cc_auprc=('auprc', 'mean'), cc_auprc_x=('auprc_x', 'mean'),
                              cc_f1=('f1', 'mean'), cc_f1_x=('f1_x', 'mean'))

aug = pd.read_csv(f'{TWINFER_PROJECT_ROOT}/analysis_data/synthetic_network_benchmark_20260824/summary_metrics_table.csv')
aug = aug[(aug.variant == 'directed_unsigned') & (aug.dataset_id != 'ALL') & (aug.method != 'TwINFER')]
aug['topo'] = aug.dataset_id.apply(topo_group)
aug_best = aug.groupby(['topo', 'method'])[['auprc', 'f1_topk']].mean().reset_index()
best_rows = []
for topo, sub in aug_best.groupby('topo'):
    best = sub.loc[sub.auprc.idxmax()]
    best_rows.append(dict(topo=topo, best_method=best.method, best_auprc=best.auprc, best_f1=best.f1_topk))
bestdf = pd.DataFrame(best_rows).set_index('topo')
bestdf['best_auprc_x'] = bestdf.best_auprc / rand
bestdf['best_f1_x'] = bestdf.best_f1 / rand

out = t4g.join(ccg).join(bestdf)
out['todo4v2_auprc_raw'] = out.todo4v2_auprc_x * rand
out['todo4v2_f1_raw'] = out.todo4v2_f1_x * rand
cols = ['todo4v2_auprc_raw', 'todo4v2_auprc_x', 'todo4v2_f1_raw', 'todo4v2_f1_x',
        'cc_auprc', 'cc_auprc_x', 'cc_f1', 'cc_f1_x',
        'best_method', 'best_auprc', 'best_auprc_x', 'best_f1', 'best_f1_x']
print(out[cols].round(3).to_string())
