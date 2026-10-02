# SUPERSEDED 2026-09-23 -- DO NOT USE. Use apply_twinscore_supplement_fatemap_gated_bootstrap.py (serial) or ..._gated_bootstrap_parallel.py.
# Kept only so older numbers can be reproduced.
"""TwinFER 'neg-only floor' scoring for FM06 A/B-split outputs (2026-09-23).
phi is left UNFLOORED for h>0; only genes with h<=0 (in either replicate) get that side's h replaced by its
bootstrap null SD (clone-label permutation) so phi is defined instead of NaN. Reuses saved w, PAIR, direction_term.
Writes per-set rank CSVs + summary to analysis_data/fatemap_comparison/data/."""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import sys, os, numpy as np, pandas as pd
from sklearn.metrics import precision_recall_curve, auc
from scipy.stats import spearmanr
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_analysis.fatemap_pipeline import SUPERSEDED_apply_twinscore_supplement_fatemap_allpairs as m
s = m.s
ROOT = f'{TWINFER_PROJECT_ROOT}/analysis_data/'
D = ROOT + 'fm06/data/'; OUT = ROOT + 'fatemap_comparison/data/'
def met(sc, y):
    sc = np.nan_to_num(np.asarray(sc, float), nan=np.nanmin(sc) - 1); pr, rc, _ = precision_recall_curve(y, sc); a = auc(rc, pr)
    k = int(y.sum()); tp = int(y[np.argsort(-sc, kind='stable')[:k]].sum()); return a, a / y.mean(), tp, tp / k
rows = []
for gs in ['correlation_high','correlation_mid','correlation_low','detection_high','detection_mid','detection_low','variability_high','variability_mid','variability_low']:
    n = f'twinscore_supplement_fm06_{gs}_absplit_allpairs_'
    p = pd.read_csv(D + n + 'pair_terms.csv'); g = pd.read_csv(D + n + 'gene_terms.csv').set_index('gene')
    pid = pd.read_csv(D + f'networks/pidc_{gs}_allgenes.csv'); pm = {(a, b): v for a, b, v in zip(pid.TF, pid.target, pid.importance)}
    p['PIDC'] = [pm.get((a, b), 0.0) for a, b in zip(p.gene_1, p.gene_2)]
    genes = list(g.index); y = p.collectri_edge.values
    sp = s(p.PAIR.values); sph = s(p.phi_x.values)
    w = np.median(((p.TwinScore.values - sp - p.direction_term.values) / np.where(np.abs(sph) > 1e-9, sph, np.nan))[np.abs(sph) > 1e-9])
    need1 = (g.h_t1 <= 0).any(); need2 = (g.h_t2 <= 0).any()
    b1 = b2 = None
    if need1 or need2:
        full, _, *_ = m.load_raw_ab('FM06', gs)
        if need1: b1, _ = m.bootstrap_null_sd_diag(full[full.time_step == 0].reset_index(drop=True), genes, seed=m.SEED)
        if need2: b2, _ = m.bootstrap_null_sd_diag(full[full.time_step == 1].reset_index(drop=True), genes, seed=m.SEED + 1)
    phi = {}; nfl = 0
    for gene in genes:
        h1, h2, rd = g.loc[gene, 'h_t1'], g.loc[gene, 'h_t2'], g.loc[gene, 'rho_dagger_gg']
        f1 = h1 if h1 > 0 else b1[gene]; f2 = h2 if h2 > 0 else b2[gene]; nfl += int(h1 <= 0 or h2 <= 0)
        phi[gene] = float(np.clip(rd / np.sqrt(f1 * f2), -10, 10)) if (np.isfinite(f1) and np.isfinite(f2) and f1 > 0 and f2 > 0) else np.nan
    p['phi_negfloor'] = p.gene_1.map(phi)
    p['TwinFER'] = w * s(p.phi_negfloor.values) + sp + p.direction_term.values
    p['pair'] = p.gene_1 + '->' + p.gene_2
    for c in ['TwinFER', 'PIDC']: p['rank_' + c] = p[c].rank(ascending=False, method='min').astype(int)
    p[['pair', 'collectri_edge', 'TwinFER', 'PIDC', 'rank_TwinFER', 'rank_PIDC', 'phi_negfloor']].to_csv(OUT + f'fm06_{gs}_twinfer_negfloor_vs_pidc.csv', index=False)
    out = p[y == 1].groupby('gene_1').size().sort_values(ascending=False)
    aT, xT, tT, pT = met(p.TwinFER, y); aP, xP, tP, pP = met(p.PIDC, y)
    rows.append(dict(gene_set=gs, n_genes=len(genes), n_true=int(y.sum()), base=round(y.mean(), 4), n_h_le0=nfl, w=round(w, 3),
        TwinFER_AUPRC=round(aT, 4), TwinFER_AUPRCx=round(xT, 3), TwinFER_TP=tT, TwinFER_prec=round(pT, 3),
        PIDC_AUPRC=round(aP, 4), PIDC_AUPRCx=round(xP, 3), PIDC_TP=tP, PIDC_prec=round(pP, 3),
        diff_AUPRCx=round(xT - xP, 3), top3_share=round(out.head(3).sum() / y.sum(), 2), HHI=round(((out / out.sum()) ** 2).sum(), 3), top_hub=f'{out.index[0]}({out.iloc[0]})'))
    print('done', gs, flush=True)
t = pd.DataFrame(rows); t.to_csv(OUT + 'fm06_twinfer_negfloor_vs_pidc_summary.csv', index=False)
pd.set_option('display.width', 250); print(t.to_string(index=False))
print('mean AUPRC-x TwinFER %.3f  PIDC %.3f' % (t.TwinFER_AUPRCx.mean(), t.PIDC_AUPRCx.mean()))
print('Spearman(top3_share, diff)=%.2f  Spearman(HHI, diff)=%.2f (n=9)' % (spearmanr(t.top3_share, t.diff_AUPRCx)[0], spearmanr(t.HHI, t.diff_AUPRCx)[0]))
