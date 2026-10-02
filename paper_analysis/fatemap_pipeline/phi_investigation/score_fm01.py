from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import glob, re, sys, numpy as np, pandas as pd
from sklearn.metrics import precision_recall_curve, auc
D=f'{TWINFER_PROJECT_ROOT}/analysis_data/fm01/data/twinscore_supp_gated_bootstrap/'
def met(sc,y):
    sc=np.nan_to_num(np.asarray(sc,float),nan=np.nanmin(sc)-1); pr,rc,_=precision_recall_curve(y,sc); a=auc(rc,pr)
    k=int(y.sum()); tp=int(y[np.argsort(-sc,kind='stable')[:k]].sum()); return a,a/y.mean(),tp,k
rows=[]
for f in sorted(glob.glob(D+'twinscore_supplement_fm01_*_pair_terms.csv')):
    gs=re.search(r'fm01_(.*)_absplit',f).group(1); p=pd.read_csv(f)
    if 'collectri_edge' not in p: print('no collectri_edge col', f, list(p.columns)); continue
    y=p.collectri_edge.values.astype(int)
    r=dict(gene_set=gs,pairs=len(p),n_true=int(y.sum()),prevalence=round(y.mean(),4))
    for term in ['TwinScore','PAIR','D','R','Wz','phi_x']:
        if term in p:
            a,x,tp,k=met(p[term].values,y); r[f'{term}_auprc']=round(a,4); r[f'{term}_x']=round(x,3)
            if term=='TwinScore': r['TwinScore_hits@k']=f'{tp}/{k}'
    rows.append(r)
df=pd.DataFrame(rows); pd.set_option('display.width',250); print(df.to_string(index=False))
df.to_csv(f'{TWINFER_PROJECT_ROOT}/analysis_data/fm01/data/twinscore_supp_gated_bootstrap/fm01_auprc_summary.csv',index=False)
