"""Figure-2 design on the figure_4 motif replicates: sample B (20 h) = 1,500 twin clones + 3,000 singletons; sample A (1 h) = a different 1,500 twins + 3,000 singletons;
the 3,000 shared clones give one cell to A and the sister to B (cross-sample twins for direction). Reuses figure2_1000_zscores.analyse() per gene pair (y,z).
Same first-50 common replicates as motif_replicates_zscores.py."""
import argparse, os, numpy as np, pandas as pd
import figure2_1000_zscores as F
from motif_replicates_zscores import files, load, MOT
from motif_shared_driver import PAIRS, TRUTH
T1,T2=1,20
def run(a):
    m,r,fn,out=a; dest=f"{out}/{m}_rep_{r:04d}.csv"
    if os.path.exists(dest): return
    D=load(fn); rows=[]
    for k,(y,z) in enumerate(PAIRS):
        rng=np.random.RandomState(r*10+k)
        P1={"gene_1":D[T1][y],"gene_2":D[T1][z]}; P2={"gene_1":D[T2][y],"gene_2":D[T2][z]}
        o=F.analyse(P1,P2,rng,500); o.update(motif=m,rep=r,pair=f"{y}-{z}",truth=TRUTH[m][(y,z)]); rows.append(o)
    pd.DataFrame(rows).to_csv(dest,index=False)
if __name__=="__main__":
    import multiprocessing as mp
    ap=argparse.ArgumentParser();ap.add_argument("--nreps",type=int,default=50);ap.add_argument("--out",required=True);ap.add_argument("--jobs",type=int,default=12);a=ap.parse_args()
    Fl=files(); common=sorted(set.intersection(*[{r for (mm,r) in Fl if mm==m} for m in MOT]))[:a.nreps]; os.makedirs(a.out,exist_ok=True)
    with mp.Pool(a.jobs) as p: p.map(run,[(m,r,Fl[(m,r)],a.out) for m in MOT for r in common])
    print("done",len(common))
