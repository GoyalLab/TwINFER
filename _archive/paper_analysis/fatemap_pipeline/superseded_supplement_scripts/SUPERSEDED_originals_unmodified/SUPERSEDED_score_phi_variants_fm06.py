"""Compare phi treatments on FM06 A/B-split outputs (2026-09-23). Caches per-gene bootstrap null SD floors.
Variants: unfloored (NaN if h<=0, penalized), saved (as-run bootstrap floor on positives, h<=0 NaN penalized),
negfloor (h<=0 -> floor, positives untouched), floor_all (max(h, floor) for every gene). Reuses saved w/PAIR/direction."""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import sys, os, numpy as np, pandas as pd
from sklearn.metrics import precision_recall_curve, auc
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apply_twinscore_supplement_fatemap_allpairs as m
s = m.s
ROOT = f'{TWINFER_PROJECT_ROOT}/analysis_data/'; D = ROOT + 'fm06/data/'; OUT = ROOT + 'fatemap_comparison/data/'
def met(sc, y):
    sc = np.nan_to_num(np.asarray(sc, float), nan=np.nanmin(sc) - 1); pr, rc, _ = precision_recall_curve(y, sc); a = auc(rc, pr)
    k = int(y.sum()); tp = int(y[np.argsort(-sc, kind='stable')[:k]].sum()); return a, a / y.mean(), tp
rows = []
for gs in (sys.argv[1:] or ['correlation_high','correlation_mid','correlation_low','detection_high','detection_mid','detection_low','variability_high','variability_mid','variability_low']):
    n = f'twinscore_supplement_fm06_{gs}_absplit_allpairs_'
    p = pd.read_csv(D + n + 'pair_terms.csv'); g = pd.read_csv(D + n + 'gene_terms.csv').set_index('gene'); genes = list(g.index); y = p.collectri_edge.values
    fc = OUT + f'fm06_{gs}_bootstrap_h_floors.csv'
    if os.path.exists(fc): fl = pd.read_csv(fc, index_col=0)
    else:
        full, _, *_ = m.load_raw_ab('FM06', gs)
        b1, _ = m.bootstrap_null_sd_diag(full[full.time_step == 0].reset_index(drop=True), genes, seed=m.SEED)
        b2, _ = m.bootstrap_null_sd_diag(full[full.time_step == 1].reset_index(drop=True), genes, seed=m.SEED + 1)
        fl = pd.DataFrame({'f1': pd.Series(b1), 'f2': pd.Series(b2)}); fl.to_csv(fc)
    g = g.join(fl)
    sp = s(p.PAIR.values); sph = s(p.phi_x.values)
    w = np.median(((p.TwinScore.values - sp - p.direction_term.values) / np.where(np.abs(sph) > 1e-9, sph, np.nan))[np.abs(sph) > 1e-9])
    def ph(f1, f2):
        return np.where((f1 > 0) & (f2 > 0), np.clip(g.rho_dagger_gg / np.sqrt(np.where((f1 > 0) & (f2 > 0), f1 * f2, 1)), -10, 10), np.nan)
    pos = (g.h_t1 > 0) & (g.h_t2 > 0)
    var = {
        'unfloored': ph(np.where(pos, g.h_t1, -1), np.where(pos, g.h_t2, -1)),
        'saved_floor_pos': np.where(pos, ph(np.maximum(g.h_t1, g.f1), np.maximum(g.h_t2, g.f2)), np.nan),
        'neg_only': ph(np.where(g.h_t1 > 0, g.h_t1, g.f1), np.where(g.h_t2 > 0, g.h_t2, g.f2)),
        'floor_all': ph(np.maximum(g.h_t1, g.f1), np.maximum(g.h_t2, g.f2)),
    }
    r = dict(gs=gs, n_true=int(y.sum()), base=round(y.mean(), 4), w=round(w, 3), n_pos_in_floor_zone=int((pos & ((g.h_t1 < g.f1) | (g.h_t2 < g.f2))).sum()), n_h_le0=int((~pos).sum()))
    for name, arr in var.items():
        x = p.gene_1.map(dict(zip(genes, arr))).values
        a, ax, tp = met(w * s(x) + sp + p.direction_term.values, y)
        r[name + '_AUPRC'] = round(a, 4); r[name + '_x'] = round(ax, 3); r[name + '_TP'] = tp
    a, ax, tp = met(sp, y); r['noPhi_AUPRC'] = round(a, 4); r['noPhi_x'] = round(ax, 3); r['noPhi_TP'] = tp
    rows.append(r); print('done', gs, flush=True)
t = pd.DataFrame(rows); t.to_csv(OUT + 'fm06_phi_variants_summary.csv', index=False)
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 50); print(t.to_string(index=False))
for v in ['unfloored', 'saved_floor_pos', 'neg_only', 'floor_all', 'noPhi']:
    print(f'{v:16s} mean AUPRC {t[v + "_AUPRC"].mean():.4f}  mean AUPRC-x {t[v + "_x"].mean():.3f}  mean TP {t[v + "_TP"].mean():.1f}')
