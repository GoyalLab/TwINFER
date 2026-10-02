from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import sys, numpy as np, pandas as pd
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0,'.')
import apply_twinscore_supplement_fatemap_allpairs as m
s=m.s
ds,gs='FM06','correlation_high'
D=f'{TWINFER_PROJECT_ROOT}/analysis_data/fm06/data/'; n=f'twinscore_supplement_fm06_{gs}_absplit_allpairs_'
p=pd.read_csv(D+n+'pair_terms.csv'); g=pd.read_csv(D+n+'gene_terms.csv').set_index('gene')
pid=pd.read_csv(D+f'networks/pidc_{gs}_allgenes.csv'); pm={(a,b):v for a,b,v in zip(pid.TF,pid.target,pid.importance)}
p['PIDC']=[pm.get((a,b),0.0) for a,b in zip(p.gene_1,p.gene_2)]
full,genes,*_=m.load_raw_ab(ds,gs)
t2=full[full.time_step==1].reset_index(drop=True)
b2,_=m.bootstrap_null_sd_diag(t2,genes,seed=m.SEED+1)
s_pair=s(p.PAIR.values); sp=s(p.phi_x.values)
w=np.median(((p.TwinScore.values-s_pair-p.direction_term.values)/np.where(np.abs(sp)>1e-9,sp,np.nan))[np.abs(sp)>1e-9])
phin={}
for gene in genes:
    h1,h2,rd=g.loc[gene,'h_t1'],g.loc[gene,'h_t2'],g.loc[gene,'rho_dagger_gg']
    b1v=None
    phin[gene]=rd/np.sqrt((h1 if h1>0 else 0.0252)*(h2 if h2>0 else b2[gene])) if True else None
# use unfloored where h>0, negative-only floor for h<=0 (h1 floor not needed: only EZH2 h2<=0 here)
p['TwinScore_negfloor']=w*s(p.gene_1.map(phin).values)+s_pair+p.direction_term.values
y=p.collectri_edge.values; k=int(y.sum())
for c in ['TwinScore','TwinScore_negfloor','PIDC','D','PAIR']:
    p['rank_'+c]=p[c].rank(ascending=False,method='min').astype(int)
p['pair']=p.gene_1+'->'+p.gene_2
for c in ['TwinScore','TwinScore_negfloor','PIDC']: p['top_'+c]=(p['rank_'+c]<=k).astype(int)
out=D.replace('/fm06/data/','/fatemap_comparison/data/')+'fm06_correlation_high_pair_ranks_twinscore_vs_pidc.csv'
cols=['pair','collectri_edge','rank_TwinScore','rank_TwinScore_negfloor','rank_PIDC','rank_D','rank_PAIR','top_TwinScore','top_TwinScore_negfloor','top_PIDC','TwinScore','TwinScore_negfloor','PIDC']
p[cols].sort_values('rank_PIDC').to_csv(out,index=False); print('saved',out,'k=',k)
pos=p[(p.collectri_edge==1)&((p.top_TwinScore==1)|(p.top_PIDC==1)|(p.top_TwinScore_negfloor==1))].sort_values(['top_PIDC','top_TwinScore'],ascending=False)
print(f'\nTRUE edges in top-{k} of TwinScore(bootstrap floor) / TwinScore(neg-only floor) / PIDC; ranks out of {len(p)} (1=best)')
print(pos[['pair','rank_TwinScore','rank_TwinScore_negfloor','rank_PIDC','rank_D']].rename(columns={'rank_TwinScore':'TS_boot','rank_TwinScore_negfloor':'TS_negfl','rank_PIDC':'PIDC','rank_D':'D_pipe'}).to_string(index=False))
for c in ['TwinScore','TwinScore_negfloor','PIDC']: print(c,'true hits in top-k:',int(p.loc[p['top_'+c]==1,'collectri_edge'].sum()))
a=set(p.pair[(p.top_TwinScore==1)&(p.collectri_edge==1)]); b=set(p.pair[(p.top_PIDC==1)&(p.collectri_edge==1)]); print('TS_boot vs PIDC true hits: common',len(a&b),'TS only',len(a-b),'PIDC only',len(b-a))
a=set(p.pair[(p.top_TwinScore_negfloor==1)&(p.collectri_edge==1)]); print('TS_negfloor vs PIDC true hits: common',len(a&b),'TS only',len(a-b),'PIDC only',len(b-a))
