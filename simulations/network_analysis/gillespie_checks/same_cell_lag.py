from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import sys, numpy as np, pandas as pd, warnings
from sklearn.metrics import average_precision_score as ap
warnings.filterwarnings("ignore")
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0,"/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/work_in_progress/benchmark")
from benchmarks.network_benchmarks.to_beeline import boolode_real_convert as c
R=f'{TWINFER_PROJECT_ROOT}'; RW=f"{R}/input_data/real_world_networks"; OLD=f"{R}/simulation_data/twinfer_format"
TOPO={"B_cell_activation":f"{RW}/B_cell.txt","EMT_real":f"{RW}/EMT.txt","Pluripotent_real":f"{RW}/Pluripotent.txt",**{n:f"{OLD}/{n}/interaction_matrix.txt" for n in ("GSD","HSC","mCAD","VSC")}}
T12={"B_cell_activation":(500,799),"EMT_real":(900,1599),"Pluripotent_real":(900,1599),"GSD":(500,799),"HSC":(500,799),"mCAD":(300,499),"VSC":(300,499)}
def rk(x): return pd.DataFrame(x).rank().to_numpy()
def xcorr(A,B):  # corr of column i of A with column j of B (rows = matched units)
    A=(rk(A)-rk(A).mean(0)); B=(rk(B)-rk(B).mean(0))
    return (A.T@B)/np.sqrt((A**2).sum(0)[:,None]*(B**2).sum(0)[None,:]+1e-12)
def apx(y,x):
    x=np.nan_to_num(np.asarray(x,float),nan=-1e9); return ap(y,x)/y.mean() if 0<y.sum()<len(y) else np.nan
out=[]
for net,genes in c.NETS.items():
    M=np.loadtxt(TOPO[net],delimiter=","); n=len(genes); t1,t2=T12[net]; off=~np.eye(n,dtype=bool)
    for rep in range(3):
        d=pd.read_csv(f"{c.SIM}/{net}/replicate_{rep}/twin_final_states.csv")
        get=lambda src,step: d[(d.source==src)&(d.step==step)].sort_values("pair_id")[genes].to_numpy()
        A1,A2,B1,B2=get("twin_A",t1),get("twin_A",t2),get("twin_B",t1),get("twin_B",t2)
        same=xcorr(A1,A2)          # same cell: gene a @t1 vs gene b @t2   (hypothetical experiment)
        twin=xcorr(A1,B2)          # sister cell: a in twin A @t1 vs b in twin B @t2 (what TwINFER can actually do)
        S2=xcorr(A2,A2)            # same cell, same time, all genes (baseline undirected)
        yt=(M!=0).astype(int)[off]; und=((M!=0)|(M.T!=0)); iu=np.triu_indices(n,1); yu=und[iu].astype(int)
        r=dict(net=net,rep=rep)
        for name,X in (("same_cell_lag",same),("sister_lag",twin)):
            r[f"{name}_exist"]=apx(yu,np.maximum(np.abs(X),np.abs(X.T))[iu])          # undirected existence
            r[f"{name}_ordered"]=apx(yt,np.abs(X)[off])                              # ordered-pair existence |lag_ab|
            g=np.abs(X)-np.abs(X.T)                                                   # asymmetry: a->b stronger than b->a
            oneway=((M!=0)&(M.T==0))|((M==0)&(M.T!=0)); m=oneway[off]                 # one-directional true edges
            r[f"{name}_dir"]=apx(yt[m],g[off][m]) if 0<yt[m].sum()<m.sum() else np.nan  # orientation among one-way edges
        r["samecell_t2_exist"]=apx(yu,np.abs(S2)[iu])
        out.append(r)
    print(net,"done",flush=True)
df=pd.DataFrame(out); df.to_csv(f"{sys.argv[1]}/same_cell_lag.csv",index=False)
order=["B_cell_activation","EMT_real","Pluripotent_real","GSD","HSC","mCAD","VSC"]
print(df.groupby("net").mean(numeric_only=True).drop(columns="rep").reindex(order).round(2).to_string())
print("\nmean of nets:\n",df.groupby("net").mean(numeric_only=True).drop(columns="rep").mean().round(2).to_string())
