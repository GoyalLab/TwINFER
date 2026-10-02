from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import json, numpy as np, pandas as pd
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
A=f'{TWINFER_PROJECT_ROOT}/analysis_data/'
D=A+'fm06/data/twinscore_supp_gated_bootstrap/'; gs='correlation_high'
p=pd.read_csv(D+f'twinscore_supplement_fm06_{gs}_absplit_gated_bootstrap_pair_terms.csv'); g=pd.read_csv(D+f'twinscore_supplement_fm06_{gs}_absplit_gated_bootstrap_gene_terms.csv').set_index('gene')
prm=json.load(open(D+f'run_params_fm06_{gs}_absplit_gated_bootstrap.json'))['diagnostics']
pid=pd.read_csv(A+f'fm06/data/networks/pidc_{gs}_allgenes.csv'); pm={(a,b):v for a,b,v in zip(pid.TF,pid.target,pid.importance)}
p['PIDC']=[pm.get((a,b),0.0) for a,b in zip(p.gene_1,p.gene_2)]
TF=set(open(f'{TWINFER_PROJECT_ROOT}/real_data/humanTFs/TF_names_v_1.01.txt').read().split())
def met(sc,y):
    sc=np.nan_to_num(np.asarray(sc,float),nan=np.nanmin(sc)-1); pr,rc,_=precision_recall_curve(y,sc); a=auc(rc,pr); k=int(y.sum()); b=y.mean()
    tp=int(y[np.argsort(-sc,kind='stable')[:k]].sum()); return dict(AUPRC=round(a,4),random=round(b,4),AUPRCx=round(a/b,2),AUROC=round(roc_auc_score(y,sc),3),k=k,TP=tp,prec=round(tp/k,3),precx=round((tp/k)/b,2))
y=p.collectri_edge.values; tf=p.gene_1.isin(TF).values
print(f'FM06/{gs} FINAL gated+bootstrap: w={prm["w"]:.4f}  var_phi(pass)={prm.get("var_phi_pass",float("nan")):.5f}  noise_floor={prm.get("phi_noise_floor",float("nan")):.5f}  gated {prm["n_gated"]}/{prm["n_genes"]}: {", ".join(prm["gated_genes"])}  placeholder={prm["phi_placeholder"]:.3f}')
print('boot perms:',prm['boot_perms'],' sd_reg=%.4g sd_twin_t2=%.4g gate_g=%.3f gate_v=%.3f'%(prm['sd_reg'],prm['sd_twin_t2'],prm['gate_g'],prm['gate_v']))
print('NaN in TwinScore/PAIR/phi_x:',int(p.TwinScore.isna().sum()),int(p.PAIR.isna().sum()),int(p.phi_x.isna().sum()))
rows=[('ALL pairs (1122): TwinFER final',met(p.TwinScore,y)),('ALL pairs: PIDC',met(p.PIDC,y)),('ALL pairs: PAIR alone (no phi)',met(p.PAIR,y)),
      ('TF sources (363): TwinFER final',met(p.TwinScore[tf],y[tf])),('TF sources: PIDC',met(p.PIDC[tf],y[tf])),('TF sources: PAIR alone',met(p.PAIR[tf],y[tf]))]
t=pd.DataFrame([dict(method=n,**m) for n,m in rows]); pd.set_option('display.width',200); print(); print(t.to_string(index=False))
print('\nreference (earlier analysis, w=0.946 reused): ALL TwinFER 0.2257 / 2.51x / 28 hits ; TF universe 0.5271 / 2.28x / 40 hits ; PIDC ALL 0.1569 / 1.74x / 18 ; PIDC TF 0.3991 / 1.72x / 33')
print('reference gated genes then: ATAD2, CCNE2, E2F1, EZH2, FOSB, ID1, JUN, MYBL2')
