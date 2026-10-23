"""Same per-replicate TwinScore z-scores as figure2_1000_zscores.py, on the drift (k_on ramp 0-10 h) simulations.
Each file already holds 3,000 low + 3,000 high clones (6,000), so no source sampling; analyse() is reused unchanged."""
import argparse, glob, os, re, numpy as np, pandas as pd
from figure2_1000_zscores import analyse, load
ROOT="/projects/b1255/hzhang/TwINFER_KA/simulation_data/drift_lowhigh_Kramp/"
def run(a):
    net,rep,out,D=a; dest=f"{out}/{net}_rep_{rep:03d}.csv"
    if os.path.exists(dest): return net,rep
    fs=glob.glob(f"{ROOT}{net}/df_*_ncells_6000_{net}_drift_lowhigh_Kramp_rep_{rep}_*.csv")
    assert len(fs)==1,(net,rep,len(fs))
    s=load(fs[0]); o=analyse(s[1],s[20],np.random.RandomState(rep*10+7),D)
    o.update(rep=rep,scenario=f"{net} (drift)"); pd.DataFrame([o]).to_csv(dest,index=False); return net,rep
if __name__=="__main__":
    import multiprocessing as mp
    ap=argparse.ArgumentParser(); ap.add_argument("--reps",default="0-99"); ap.add_argument("--out",required=True)
    ap.add_argument("--D",type=int,default=500); ap.add_argument("--jobs",type=int,default=8); a=ap.parse_args()
    lo,hi=map(int,a.reps.split("-")); os.makedirs(a.out,exist_ok=True)
    with mp.Pool(a.jobs) as p:
        for r in p.imap_unordered(run,[(n,r,a.out,a.D) for n in ("A_B","A_to_B") for r in range(lo,hi+1)]): print(r,flush=True)
