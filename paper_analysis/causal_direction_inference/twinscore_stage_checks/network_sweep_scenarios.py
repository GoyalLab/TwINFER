"""Scenario table (shared part first, then z_rho / z_rho_hat_Delta) + PDF-style called/directed on the network_sweep 5-edge networks
(input_data/network_sweep/grn_n6_e5_pos50_density_rep{0,1,2}.txt; connectivity_matrix[i,j]!=0 => gene i+1 -> gene j+1). Every unordered gene pair of every file,
design: B = 20 h (1,500 twins + 3,000 singletons), A = 1 h, 3,000 shared clones (Figure-2 analyse()), mRNA only, single pair, |z|>1.96 (shared also 2.326).
Pair classes from the ground truth: direct edge | indirect (path >=2 either way) | co-regulated (common ancestor, no path) | unrelated."""
import glob, os, numpy as np, pandas as pd, multiprocessing as mp
import figure2_1000_zscores as F
from _paths import DATA
ROOT="/projects/b1255/hzhang/TwINFER_KA/"
# [2026-10-08 was hard-coded: SIM=ROOT+"simulation_data/network_sweep/"; NET="grn_n6_e5_pos50_density"; now selectable, default unchanged]
import argparse
_ap=argparse.ArgumentParser(); _ap.add_argument("--sweep",default="network_sweep"); _ap.add_argument("--net",default="grn_n6_e5_pos50_density"); _ap.add_argument("--out",default="network_sweep_e5_pairs.csv")
_a,_=_ap.parse_known_args(); SWEEP=_a.sweep; SIM=ROOT+f"simulation_data/{SWEEP}/"; NET=_a.net
def classes(C):
    n=C.shape[0]; A=(C!=0).astype(int); R=A.copy()
    for _ in range(n): R=((R+R@A)>0).astype(int)
    anc=[set(np.where(R[:,j])[0]) for j in range(n)]; out={}
    for i in range(n):
        for j in range(i+1,n):
            if A[i,j]: out[(i,j)]=("direct edge",i,j)
            elif A[j,i]: out[(i,j)]=("direct edge",j,i)
            elif R[i,j]: out[(i,j)]=("indirect",i,j)
            elif R[j,i]: out[(i,j)]=("indirect",j,i)
            elif anc[i]&anc[j]: out[(i,j)]=("co-regulated",i,j)
            else: out[(i,j)]=("unrelated",i,j)
    return out
def one(a):
    f,net,sim=a; C=np.loadtxt(ROOT+f"input_data/{SWEEP}/{NET}_rep{net}.txt",delimiter=",").astype(int); cl=classes(C)
    d=pd.read_csv(f,usecols=["time_step","clone_id","replicate"]+[f"gene_{k}_mRNA" for k in range(1,7)])
    P={t:{k:d[d.time_step==t].pivot(index="clone_id",columns="replicate",values=f"gene_{k}_mRNA").values for k in range(1,7)} for t in (1,20)}
    rows=[]
    for (i,j),(cls,x,y) in cl.items():
        gx,gy=x+1,y+1; rng=np.random.RandomState(100*net+10*sim+i*6+j)
        if P[20][gx].std()==0 or P[20][gy].std()==0: continue
        o=F.analyse({"gene_1":P[1][gx],"gene_2":P[1][gy]},{"gene_1":P[20][gx],"gene_2":P[20][gy]},rng,200)
        o.update(network=f"rep{net}",sim=sim,pair=f"{i+1}-{j+1}",cls=cls,x=gx,y=gy,sign=int(np.sign(C[x,y])) if cls=="direct edge" else 0); rows.append(o)
    return rows
if __name__=="__main__":
    jobs=[]
    for net in (0,1,2):
        for f in sorted(glob.glob(SIM+f"df_{NET}_rep{net}_rep*_*.csv")):
            sim=int(f.split(f"{NET}_rep{net}_rep")[1].split("_")[0]); jobs.append((f,net,sim))
    print(len(jobs),"files",flush=True)
    with mp.Pool(12) as p: res=sum(p.map(one,jobs),[])
    R=pd.DataFrame(res); R.to_csv(DATA/_a.out,index=False); print("pairs",len(R))
