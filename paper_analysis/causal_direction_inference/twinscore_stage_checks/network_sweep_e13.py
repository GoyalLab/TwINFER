"""e13 networks (network_sweep_final/e13_pos100, 3 networks x 10 sims; sim rep1->grn_n6_e13_pos100_center_rep1, rep2->_rep2, rep3->_rep0, as in zfanout_e13_pos100.py).
Both designs per file and gene pair: (a) 1,500 twins + 3,000 singletons with t1 (Figure-2 analyse(), gives direction) and (b) ALL 6,000 clones as twins at 20 h (no t1, no direction)."""
import glob, os, re, numpy as np, pandas as pd, multiprocessing as mp
import figure2_1000_zscores as F
from u_time_pool import zU
from motif_replicates_zscores import zC, zRho
from network_sweep_scenarios import classes
from _paths import DATA
ROOT="/projects/b1255/hzhang/TwINFER_KA/"; SIM=ROOT+"simulation_data/network_sweep_final/e13_pos100/"
MAP={1:"grn_n6_e13_pos100_center_rep1",2:"grn_n6_e13_pos100_center_rep2",3:"grn_n6_e13_pos100_center_rep0"}
def one(a):
    f,net,sim=a; C=np.loadtxt(ROOT+f"input_data/network_sweep_final/{MAP[net]}.txt",delimiter=",").astype(int); cl=classes(C)
    d=pd.read_csv(f,usecols=["time_step","clone_id","replicate"]+[f"gene_{k}_mRNA" for k in range(1,7)])
    P={t:{k:d[d.time_step==t].pivot(index="clone_id",columns="replicate",values=f"gene_{k}_mRNA").values for k in range(1,7)} for t in (1,20)}; rows=[]
    for (i,j),(cls,x,y) in cl.items():
        gx,gy=x+1,y+1
        if P[20][gx].std()==0 or P[20][gy].std()==0: continue
        rng=np.random.RandomState(1000*net+10*sim+i*6+j)
        o=F.analyse({"gene_1":P[1][gx],"gene_2":P[1][gy]},{"gene_1":P[20][gx],"gene_2":P[20][gy]},rng,200)
        S,Cv,zc=zC(P[20][gx],P[20][gy])
        rows.append(dict(network=f"rep{net}",gt=MAP[net][-4:],sim=sim,pair=f"{i+1}-{j+1}",cls=cls,x=gx,y=gy,mutual=int(C[x,y]!=0 and C[y,x]!=0),
            z_rho=o["z_rho"],z_C=o["z_C"],z_U=o["z_U"],delta=o["delta"],p_delta=o["p_delta"],z_dag_xy=o["z_dag_xy"],z_dag_yx=o["z_dag_yx"],
            t_z_rho=zRho(P[20][gx],P[20][gy]),t_z_C=zc,t_z_U=zU(P[20][gx],P[20][gy])))
    return rows
if __name__=="__main__":
    jobs=[]
    for f in sorted(glob.glob(SIM+"df_rows_*_n6_e13_pos100_rep*_rep_*_*.csv")):
        if "before_division" in f: continue
        m=re.search(r"n6_e13_pos100_rep(\d+)_rep_(\d+)_",f); jobs.append((f,int(m.group(1)),int(m.group(2))))
    print(len(jobs),"files",flush=True)
    with mp.Pool(12) as p: res=sum(p.map(one,jobs),[])
    pd.DataFrame(res).to_csv(DATA/"network_sweep_final_e13_pos100_pairs_both_designs.csv",index=False); print(len(res),"pair rows")
