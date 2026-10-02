# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import sys, time, numpy as np
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0,'/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline')
from paper_analysis.fatemap_pipeline import apply_twinscore_supplement_fatemap_gated_bootstrap as m
full, genes, *_ = m.load_raw_ab('FM06','detection_high')
t1=full[full.time_step==0].reset_index(drop=True); t2=full[full.time_step==1].reset_index(drop=True)
lam=None
import pandas as pd
def run(nc):
    out={}; T={}
    t=time.time(); out['diag'],out['n_diag']=m.bootstrap_null_sd_diag(t1,genes,seed=m.SEED,n_cores=nc); T['diag']=time.time()-t
    t=time.time(); out['off_C'],out['n_offC']=m.bootstrap_null_sd_offdiag(t2,genes,seed=m.SEED+3,kind='C',n_cores=nc); T['offC']=time.time()-t
    lam_=pd.DataFrame(np.full((len(genes),len(genes)),0.5),index=genes,columns=genes)
    t=time.time(); out['off_reg'],out['n_offR']=m.bootstrap_null_sd_offdiag(t2,genes,seed=m.SEED+2,kind='reg',extra=lam_,n_cores=nc); T['offReg']=time.time()-t
    t=time.time(); out['phi']=m.bootstrap_phi_noise_floor_par(t1,t2,genes,0.06,0.06,n_cores=nc); T['phiNoise']=time.time()-t
    return out,T
o1,T1=run(1); o4,T4=run(4)
print('timings serial :',{k:round(v,1) for k,v in T1.items()}); print('timings 4 cores:',{k:round(v,1) for k,v in T4.items()})
same=lambda a,b: all((np.isnan(a[k]) and np.isnan(b[k])) or a[k]==b[k] for k in a)
print('diag identical      :',same(o1['diag'],o4['diag']),'| perms',o1['n_diag'],o4['n_diag'])
print('offdiag C identical :',o1['off_C']==o4['off_C'],o1['off_C'],'| reg identical:',o1['off_reg']==o4['off_reg'])
print('phi noise identical :',o1['phi']==o4['phi'],'(n genes with floor:',len(o1['phi']),')')
print('ALL IDENTICAL:',same(o1['diag'],o4['diag']) and o1['off_C']==o4['off_C'] and o1['off_reg']==o4['off_reg'] and o1['phi']==o4['phi'])
