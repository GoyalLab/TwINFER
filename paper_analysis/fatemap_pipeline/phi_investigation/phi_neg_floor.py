from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import sys, numpy as np, pandas as pd
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0,'.')
import apply_twinscore_supplement_fatemap_allpairs as m
from sklearn.metrics import precision_recall_curve, auc
s=m.s
def ax(sc,y):
    sc=np.nan_to_num(np.asarray(sc,float),nan=np.nanmin(sc)-1); p,r,_=precision_recall_curve(y,sc); return auc(r,p)/y.mean()
ds,gs='FM06','correlation_high'
D=f'{TWINFER_PROJECT_ROOT}/analysis_data/fm06/data/'; n=f'twinscore_supplement_fm06_{gs}_absplit_allpairs_'
p=pd.read_csv(D+n+'pair_terms.csv'); g=pd.read_csv(D+n+'gene_terms.csv').set_index('gene')
full,genes,*_=m.load_raw_ab(ds,gs)
t1=full[full.time_step==0].reset_index(drop=True); t2=full[full.time_step==1].reset_index(drop=True)
b1,_=m.bootstrap_null_sd_diag(t1,genes,seed=m.SEED); b2,_=m.bootstrap_null_sd_diag(t2,genes,seed=m.SEED+1)
y=p.collectri_edge.values; s_pair=s(p.PAIR.values); sp=s(p.phi_x.values)
w=np.median(((p.TwinScore.values-s_pair-p.direction_term.values)/np.where(np.abs(sp)>1e-9,sp,np.nan))[np.abs(sp)>1e-9])
rows=[]; phi_neg={}; phi_unf={}
for gene in genes:
    h1,h2,rd=g.loc[gene,'h_t1'],g.loc[gene,'h_t2'],g.loc[gene,'rho_dagger_gg']
    # unfloored (positive h as-is; NaN if h<=0)
    phi_unf[gene]=rd/np.sqrt(h1*h2) if (h1>0 and h2>0) else np.nan
    # floored ONLY where h<=0 (replace by noise floor); positive h untouched
    h1s=h1 if h1>0 else b1[gene]; h2s=h2 if h2>0 else b2[gene]
    phi_neg[gene]=float(np.clip(rd/np.sqrt(h1s*h2s),-10,10))
for lab,ph in [('unfloored (NaN if h<=0)',phi_unf),('floor ONLY h<=0',phi_neg)]:
    x=p.gene_1.map(ph).values
    print(f'{lab:26s} n_nan_pairs={int(np.isnan(x).sum()):4d}  phi alone={ax(s(x),y):.3f}  TwinScore={ax(w*s(x)+s_pair+p.direction_term.values,y):.3f}  range={np.nanmin(x):.2f}..{np.nanmax(x):.2f}')
print('reference: saved bootstrap-floor TwinScore=%.3f  PAIR=%.3f  w=%.3f'%(ax(p.TwinScore,y),ax(p.PAIR,y),w))
neg=[gn for gn in genes if not (g.loc[gn,'h_t1']>0 and g.loc[gn,'h_t2']>0)]
print('genes with h<=0 (now floored):',len(neg),'of',len(genes)); print(pd.DataFrame({'h1':[g.loc[x,'h_t1'] for x in neg],'h2':[g.loc[x,'h_t2'] for x in neg],'rho_dgg':[g.loc[x,'rho_dagger_gg'] for x in neg],'phi_floor_only_neg':[phi_neg[x] for x in neg]},index=neg).round(3).to_string())
