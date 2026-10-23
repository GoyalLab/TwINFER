"""Network sweep with ONE time point and ALL clones as twins: t2 = 20 h, 6,000 clones x 2 sister cells (12,000 cells), no singletons, no t1 sample => no direction.
Per gene pair: z_rho (Stage I, twin-only), z_rho_s (shared part, from C), z_rho_hat_Delta (z_U); scenario table from the pictures; same pair classes as network_sweep_scenarios.py.
Output: <data>/network_sweep_final_<set>_twin_only_pairs.csv (one row per file x pair)."""
import argparse, glob, os, numpy as np, pandas as pd, multiprocessing as mp
from u_time_pool import zU
from motif_replicates_zscores import zC, zRho
from network_sweep_scenarios import classes
from _paths import DATA
ROOT="/projects/b1255/hzhang/TwINFER_KA/"
ap=argparse.ArgumentParser(); ap.add_argument("--sweep",default="network_sweep_final"); ap.add_argument("--net",required=True); ap.add_argument("--out",required=True); a,_=ap.parse_known_args()
SIM=ROOT+f"simulation_data/{a.sweep}/"
def one(args):
    f,net,sim=args; C=np.loadtxt(ROOT+f"input_data/{a.sweep}/{a.net}_rep{net}.txt",delimiter=",").astype(int); cl=classes(C)
    d=pd.read_csv(f,usecols=["time_step","clone_id","replicate"]+[f"gene_{k}_mRNA" for k in range(1,7)]); d=d[d.time_step==20]
    X={k:d.pivot(index="clone_id",columns="replicate",values=f"gene_{k}_mRNA").values for k in range(1,7)}; rows=[]
    for (i,j),(cls,x,y) in cl.items():
        gx,gy=x+1,y+1
        if X[gx].std()==0 or X[gy].std()==0: continue
        S,Cv,zc=zC(X[gx],X[gy]); rows.append(dict(network=f"rep{net}",sim=sim,pair=f"{i+1}-{j+1}",cls=cls,x=gx,y=gy,sign=int(np.sign(C[x,y])) if cls=="direct edge" else 0,z_rho=zRho(X[gx],X[gy]),z_C=zc,z_U=zU(X[gx],X[gy]),S=S,C=Cv))
    return rows
if __name__=="__main__":
    jobs=[]
    for net in (0,1,2):
        for f in sorted(glob.glob(SIM+f"df_{a.net}_rep{net}_rep*_*.csv")):
            jobs.append((f,net,int(f.split(f"{a.net}_rep{net}_rep")[1].split("_")[0])))
    with mp.Pool(12) as p: res=sum(p.map(one,jobs),[])
    pd.DataFrame(res).to_csv(DATA/a.out,index=False); print(a.net,len(jobs),"files",len(res),"pairs")
