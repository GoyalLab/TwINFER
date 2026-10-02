from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import sys, glob, os, time
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, '.')
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, '/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/network_sweep_final')
import numpy as np, pandas as pd
from paper_analysis.real_networks.fanout_mutual_z_boxplot import select_views, calculate_all_cross_z, calculate_z_fanout
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, '/home/gzu5140/TwINFER_KA/code/TwINFER/package')
import twinfer.inference.correlation_functions as cf

ROOT = f'{TWINFER_PROJECT_ROOT}'
TOPO_DIR = f'{ROOT}/input_data/network_sweep_final'
SIM_DIR = f'{ROOT}/simulation_data/network_sweep_final/e13_pos100'
N_SHUFFLES, N_CORES, SEED, THR = 200, 1, 101010, 2.326

REP_MAP = {'rep1': 'grn_n6_e13_pos100_center_rep1', 'rep2': 'grn_n6_e13_pos100_center_rep2', 'rep3': 'grn_n6_e13_pos100_center_rep0'}
GENES = [f'gene_{i+1}' for i in range(6)]

rows = []
t0 = time.time()
for tag, topo in REP_MAP.items():
    M = np.loadtxt(f'{TOPO_DIR}/{topo}.txt', delimiter=',', dtype=int)
    true = {(GENES[i], GENES[j]) for i in range(6) for j in range(6) if i != j and M[i, j] != 0}
    poss = [(GENES[i], GENES[j]) for i in range(6) for j in range(6) if i != j]
    sims = sorted(glob.glob(f'{SIM_DIR}/*n6_e13_pos100_{tag}_rep_*.csv'))
    for sim in sims:
        try:
            usecols = ['clone_id','cell_id','time_step','replicate'] + [f'{g}_mRNA' for g in GENES]
            data = pd.read_csv(sim, usecols=usecols)
            views = select_views(data, 1, 20, seed=SEED)
            cross_z = calculate_all_cross_z(views, GENES, 'clone', N_SHUFFLES, SEED, N_CORES, cf)
        except Exception as e:
            print('skip', os.path.basename(sim), e, flush=True)
            continue
        for (gx, gy) in poss:
            res = calculate_z_fanout(gx, gy, GENES, cross_z)
            rows.append(dict(pair=f'{gx},{gy}', z_fanout=res['z_fanout'], is_true_edge=(gx, gy) in true, topo=tag))
        del data, views, cross_z
        import gc; gc.collect()
    print(f'[{time.time()-t0:.0f}s] {tag} ({len(sims)} sims)', flush=True)

df = pd.DataFrame(rows)
df.to_csv('zfanout_e13_pos100_results.csv', index=False)
g = df[np.isfinite(df.z_fanout)]
y = g.is_true_edge.to_numpy()
z = g.z_fanout.to_numpy()
flagged = z > THR
tp = (flagged & ~y).sum()
fp = (flagged & y).sum()
fn = ((~flagged) & ~y).sum()
precision = tp / (tp + fp) if (tp + fp) else float('nan')
recall = tp / (tp + fn) if (tp + fn) else float('nan')
print(f'e13_pos100 ALONE: n={len(g)} n_true={y.sum()} base_rate_true={y.mean():.3f} '
      f'flagged={flagged.sum()} precision={precision:.3f} recall={recall:.3f}')
