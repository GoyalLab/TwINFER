from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import numpy as np, pandas as pd
R=f'{TWINFER_PROJECT_ROOT}/analysis_data/'
out=[]
for ds,lab in [('FM01','fm01'),('Watermelon_naive','watermelon_naive'),('Watermelon_lag','watermelon_lag'),('Watermelon_late','watermelon_late')]:
    D=f'{R}{lab}/data/'
    p=pd.read_csv(f'{D}twinscore_supp_gated_bootstrap/twinscore_supplement_{lab}_correlation_high_absplit_gated_bootstrap_pair_terms.csv')
    y=p.collectri_edge.astype(int).values; k=int(y.sum())
    p['pair']=p.gene_1+'->'+p.gene_2
    def top(sc):
        sc=np.nan_to_num(np.asarray(sc,float),nan=np.nanmin(sc)-1); o=np.argsort(-sc,kind='stable')[:k]; return o
    to=top(p.TwinScore.values); tw=set(p.pair.values[to]); hits_tw=[q for q,yy in zip(p.pair.values[to],y[to]) if yy]
    comp={}
    for m in ['pidc','ppcor','rho','genie3','grnboost2']:
        c=pd.read_csv(f'{D}networks/{m}_correlation_high_allgenes_absplit.csv'); imp={(t,g):v for t,g,v in zip(c.TF,c.target,c.importance)}
        sc=np.array([imp.get((a,b),0.0) for a,b in zip(p.gene_1,p.gene_2)]); o=top(sc); comp[m]=(set(p.pair.values[o]),[q for q,yy in zip(p.pair.values[o],y[o]) if yy])
    pid_hits=set(comp['pidc'][1]); tw_hits=set(hits_tw)
    print(f"\n=================== {ds}  correlation_high: {len(p)} pairs, {len(p.gene_1.unique())} genes, k = {k} true CollecTRI edges")
    print(f"TwinScore top-{k}: {len(hits_tw)} true edges | PIDC {len(comp['pidc'][1])} | ppcor {len(comp['ppcor'][1])} | rho {len(comp['rho'][1])} | GENIE3 {len(comp['genie3'][1])} | GRNBoost2 {len(comp['grnboost2'][1])}")
    print(f"  TwinScore true-edge hits ({len(hits_tw)}): " + ', '.join(f"{q}{'*' if q in pid_hits else ''}" for q in hits_tw) + "   (* = also a PIDC top-k hit)")
    print(f"  shared TwinScore & PIDC hits: {len(tw_hits&pid_hits)} | PIDC-only hits ({len(pid_hits-tw_hits)}): " + ', '.join(sorted(pid_hits-tw_hits)))
    print(f"  overlap of the two top-{k} lists (any pair): {len(tw & comp['pidc'][0])}/{k}")
    tk=p.iloc[to].assign(rank=range(1,k+1),true_edge=y[to])[['rank','gene_1','gene_2','TwinScore','PAIR','D','R','Wz','phi_x','true_edge']]
    tk.to_csv(f'{D}twinscore_supp_gated_bootstrap/{lab}_correlation_high_topk_pairs.csv',index=False)
    print("  top-10 TwinScore pairs:", ', '.join(f"{a}->{b}{'✓' if t else ''}" for a,b,t in zip(tk.gene_1[:10],tk.gene_2[:10],tk.true_edge[:10])))
