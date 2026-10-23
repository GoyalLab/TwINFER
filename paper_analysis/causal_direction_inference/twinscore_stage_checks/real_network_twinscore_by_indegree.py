"""TwinScore (design: B=20 h 1,500 twins + 3,000 singletons; A=1 h; 3,000 shared clones; Figure-2 analyse()) on every gene pair of the real-network Gillespie simulations,
grouped by whether the pair is a true edge and by the TARGET's in-degree. One replicate per network (same file as real_network_indegree_check.py), mRNA only.
True edges are analysed with x = regulator, y = target, so delta>0 means the correct direction. Non-edges use x = lower gene index."""
import glob, os, numpy as np, pandas as pd, multiprocessing as mp
import figure2_1000_zscores as F
from real_network_indegree_check import NETS, ROOT
from _paths import DATA
T1,T2,D=1,20,200
def load(name,mat,folder):
    C=np.loadtxt(ROOT+f"input_data/real_world_networks/{mat}.txt",delimiter=",").astype(int); n=C.shape[0]
    fs=[f for f in glob.glob(folder+"/df_rows_*.csv") if "before_division" not in os.path.basename(f)]
    if name=="HSC": fs=[f for f in fs if "HSC_balanced" in f]
    f=sorted(fs,key=os.path.getmtime)[0]; cols=[f"gene_{k+1}_mRNA" for k in range(n)]
    d=pd.read_csv(f,usecols=["time_step","clone_id","replicate"]+cols); out={}
    for t in (T1,T2):
        s=d[d.time_step==t]; out[t]={k+1:s.pivot(index="clone_id",columns="replicate",values=cols[k]).values for k in range(n)}
    return C,out
DATA_={}
def work(a):
    name,x,y,truth,indeg=a; C,P=DATA_[name]; rng=np.random.RandomState(1000*x+y)
    o=F.analyse({"gene_1":P[T1][x],"gene_2":P[T1][y]},{"gene_1":P[T2][x],"gene_2":P[T2][y]},rng,D)
    o.update(network=name,x=x,y=y,truth=truth,target_indegree=indeg); return o
if __name__=="__main__":
    jobs=[]
    for name,(mat,folder) in NETS.items():
        C,P=load(name,mat,folder); DATA_[name]=(C,P); n=C.shape[0]; indeg=(C!=0).sum(0)
        ok=[g for g in range(1,n+1) if P[T2][g].std()>0 and P[T1][g].std()>0]
        for i in ok:
            for j in ok:
                if i==j: continue
                if C[i-1,j-1]!=0: jobs.append((name,i,j,"edge x->y",int(indeg[j-1])))
                elif i<j and C[j-1,i-1]==0: jobs.append((name,i,j,"non-edge",0))
    print(len(jobs),"pairs",flush=True)
    with mp.Pool(12) as p: res=p.map(work,jobs,chunksize=8)
    R=pd.DataFrame(res); R.to_csv(DATA/"real_network_twinscore_by_indegree.csv",index=False); print("saved",len(R))
