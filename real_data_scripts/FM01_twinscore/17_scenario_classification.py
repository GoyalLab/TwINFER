# Scenario 1-4 classification of the candidate pairs of one TwinScore run (output of twinscore_pdf_candidates[_strat].py), plus overlap of scenario 2 with the top-S / top-U / top-z_U lists.
# Columns used from pdf_procedure_all_candidate_pairs.csv: rho, S, C, U, z_rho (existence), h (= sign(S)*C/sd(C), the shared-part z), z_U, q_rho, filtered_out.
# Scenario table (handoff 2026-10-07 section 3a, images/1.png):
#   shared not sig & z_rho not sig -> 1 nothing | shared not sig & z_rho sig -> 2 regulation
#   shared sig & z_U sig -> 3 regulation + heterogeneity | shared sig & z_U not sig -> 4 heterogeneity
# --mode repo (default, same rule as twinscore_module_scenarios.py): single pre-specified pair, no BH; shared sig = |h| > 2.326, z_rho sig = |z_rho| > 1.96, z_U sig = |z_U| > 1.96.
# --mode bh   (what the FM01 write-up of 2026-10-13/14 first used): shared sig = h > 2.326 (one-sided, 'inherited'), z_rho sig = Stage I (BH q_rho <= 0.05 over all candidates), z_U sig = BH q <= 0.05 over the shared-significant pairs.
# usage: python 17_scenario_classification.py RUN_DIR [--mode repo|bh] [--no-gene-filter]
import sys, argparse, numpy as np, pandas as pd
from scipy.stats import norm
ap=argparse.ArgumentParser(); ap.add_argument("run_dir"); ap.add_argument("--mode",default="repo",choices=["repo","bh"]); ap.add_argument("--no-gene-filter",action="store_true"); a=ap.parse_args()
def bh(p):
    p=np.asarray(p); m=len(p); o=np.argsort(p); q=np.empty(m); q[o]=np.minimum.accumulate((p[o]*m/np.arange(1,m+1))[::-1])[::-1]; return np.minimum(q,1)
d=pd.read_csv(f"{a.run_dir}/pdf_procedure_all_candidate_pairs.csv")
if not a.no_gene_filter: d=d[~d.filtered_out].copy()
if a.mode=="repo": sh=d.h.abs()>2.326; rs=d.z_rho.abs()>1.96; dl=d.z_U.abs()>1.96
else:
    sh=d.h>2.326; rs=d.q_rho<=0.05; dl=pd.Series(False,index=d.index); q=bh(2*norm.sf(d.loc[sh,"z_U"].abs())); dl[sh]=q<=0.05
d["shared_sig"]=sh; d["rho_sig"]=rs; d["U_sig"]=dl
d["scenario"]=np.select([~sh&~rs,~sh&rs,sh&dl,sh&~dl],[1,2,3,4])
out=f"{a.run_dir}/scenario_classification_{a.mode}.csv"; d.to_csv(out,index=False)
print(f"{a.mode}: {len(d):,} pairs (gene filter {'off' if a.no_gene_filter else 'on'}); counts",d.scenario.value_counts().sort_index().to_dict(),"->",out)
print("scenario 2 pairs: median |S| %.3f, median |U| %.3f; share |S|>0.1 %.2f, |U|>0.1 %.2f"%(d[d.scenario==2].S.abs().median(),d[d.scenario==2].U.abs().median(),(d[d.scenario==2].S.abs()>0.1).mean(),(d[d.scenario==2].U.abs()>0.1).mean()))
rows=[]
for N in (100,1000,5000):
    r={"top N":N}
    for nm,col in (("|S|",d.S.abs()),("|U|",d.U.abs()),("|z_U|",d.z_U.abs())): t=d.assign(k=col).nlargest(N,"k"); r[f"by {nm}: scen 2"]=int((t.scenario==2).sum()); r[f"by {nm}: scen 3/4"]=int(t.scenario.isin([3,4]).sum())
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
t=d.assign(k=d.S.abs()).nlargest(5000,"k"); print("top 5000 by |S|: C/S quantiles (min/5/50/95):",np.round((t.C/t.S).quantile([0,.05,.5,.95]).values,2))
