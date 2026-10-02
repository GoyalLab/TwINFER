from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import re, numpy as np
src=open(f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/fatemap_pipeline/apply_twinscore_supplement_fatemap_allpairs.py').read()
i=src.index('def persistence_no_gate'); j=src.index('def ', src.index('def _finish_phi')+5)
ns={'np':np,'log':print,'bootstrap_phi_noise_floor':lambda *a,**k:{}}
exec(src[i:j],ns)
f=ns['persistence_no_gate']
genes=['ok','h1neg','h2neg','both_neg','pos_tiny','big_rho','neg_rho','nan_rho']
nf=0.03
h1 ={'ok':0.2,'h1neg':-0.05,'h2neg':0.2,'both_neg':-0.02,'pos_tiny':0.001,'big_rho':0.001,'neg_rho':0.2,'nan_rho':0.2}
h2 ={'ok':0.2,'h1neg':0.2,'h2neg':0.0,'both_neg':-0.01,'pos_tiny':0.001,'big_rho':0.001,'neg_rho':0.2,'nan_rho':0.2}
rho={'ok':0.1,'h1neg':0.1,'h2neg':0.1,'both_neg':0.1,'pos_tiny':0.0005,'big_rho':0.05,'neg_rho':-0.05,'nan_rho':np.nan}
n={g:nf for g in genes}
for mode in (True,False):
    phi,w=f(h1,h2,rho,genes,n,n,None,None,neg_floor_only=mode)
    print('neg_floor_only' if mode else 'v3', {g:(None if not np.isfinite(v) else round(v,3)) for g,v in phi.items()})
exp={'ok':0.1/0.2,'h1neg':0.1/np.sqrt(nf*0.2),'h2neg':0.1/np.sqrt(0.2*nf),'both_neg':0.1/nf,'pos_tiny':0.0005/0.001,'big_rho':10.0,'neg_rho':-0.05/0.2,'nan_rho':None}
print('expected neg-only', {k:(None if v is None else round(v,3)) for k,v in exp.items()})
