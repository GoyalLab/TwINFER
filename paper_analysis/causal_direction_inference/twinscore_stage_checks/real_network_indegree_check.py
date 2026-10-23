"""Does the fan-in effect (targets with more regulators correlate less with a given regulator) show up in the real-network Gillespie simulations?
For each network (one replicate, mRNA at t=20 h, all 12,000 cells): Spearman of each TRUE edge (regulator i -> target j; connectivity_matrix[i,j]!=0) vs the target's in-degree and the edge sign."""
import glob, os, numpy as np, pandas as pd
from scipy.stats import spearmanr
from _paths import DATA
ROOT="/projects/b1255/hzhang/TwINFER_KA/"
NETS={"Pluripotent":("Pluripotent",ROOT+"analysis_data/paper_analysis/Pluripotent/simulate/latest"),
      "EMT":("EMT",ROOT+"analysis_data/paper_analysis/EMT/simulate/latest"),
      "B_cell":("B_cell",ROOT+"analysis_data/paper_analysis/B_cell_activation/simulate/latest"),
      "HSC":("HSC",ROOT+"simulation_data/real_data")}
rows=[]
for name,(mat,folder) in NETS.items():
    C=np.loadtxt(ROOT+f"input_data/real_world_networks/{mat}.txt",delimiter=",").astype(int); n=C.shape[0]
    pat={"HSC":"*HSC_balanced*"}.get(name,"*")
    fs=[f for f in glob.glob(folder+f"/df_rows_{pat}.csv") if "before_division" not in os.path.basename(f)]
    if name=="HSC": fs=[f for f in fs if "HSC_balanced" in f]
    if not fs: print(name,"no sim file"); continue
    f=sorted(fs,key=os.path.getmtime)[0]
    cols=[f"gene_{k+1}_mRNA" for k in range(n)]
    d=pd.read_csv(f,usecols=["time_step"]+cols); tt=20 if 20 in set(d.time_step) else d.time_step.unique()[min(20,len(d.time_step.unique())-1)]
    X=d[d.time_step==tt][cols].values; keep=X.std(0)>0
    R=spearmanr(X)[0] if X.shape[1]>2 else None
    indeg=(C!=0).sum(0)
    for i in range(n):
        for j in range(n):
            if i!=j and C[i,j]!=0 and keep[i] and keep[j]:
                rows.append(dict(network=name,n_genes=n,regulator=i+1,target=j+1,sign=int(np.sign(C[i,j])),target_indegree=int(indeg[j]),rho=R[i,j],abs_rho=abs(R[i,j]),file=os.path.basename(f)[:60]))
    print(name,"genes",n,"edges",int((C!=0).sum()),"time_step",tt,"cells",len(X),"file",os.path.basename(f)[:70])
T=pd.DataFrame(rows); T.to_csv(DATA/"real_network_indegree_edges.csv",index=False)
T["indeg_bin"]=pd.cut(T.target_indegree,[0,1,2,3,100],labels=["1","2","3","4+"])
pd.set_option("display.width",200)
print("\nmedian |Spearman| of true edges by target in-degree (all networks pooled):")
print(T.groupby("indeg_bin",observed=True).abs_rho.agg(["count","median","mean"]).round(3).to_string())
print("\nby network:"); print(T.pivot_table(index="network",columns="indeg_bin",values="abs_rho",aggfunc="median",observed=True).round(3).to_string())
print("\nby sign (pooled):"); print(T.groupby("sign").abs_rho.agg(["count","median"]).round(3).to_string())
print("\nSpearman(|rho|, 1/in-degree) over edges:",round(spearmanr(T.abs_rho,1/T.target_indegree)[0],3))
