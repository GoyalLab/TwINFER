import pandas as pd, numpy as np
from scipy.stats import rankdata
d=pd.read_csv("/home/gzu5140/TwINFER_KA/simulation_data/figure_2/A_B/df_rows_0_1_08082025_093223_ncells_6000_A_B_rep_0.csv")
rng=np.random.RandomState(1)
for t in [1,10,20,48]:
    s=d[d.time_step==t]; P={g:s.pivot(index='clone_id',columns='replicate',values=f"{g}_mRNA").values for g in ["gene_1","gene_2"]}
    R={g:rankdata(P[g].ravel()).reshape(-1,2) for g in P}
    dx=R["gene_1"][:,0]-R["gene_1"][:,1]; dy=R["gene_2"][:,0]-R["gene_2"][:,1]
    rd=(dx*dy).sum()/np.sqrt((dx**2).sum()*(dy**2).sum()); n=len(dx)
    nul=[(lambda p:(dx*dy[p]).sum()/np.sqrt((dx**2).sum()*(dy**2).sum()))(rng.permutation(n)) for _ in range(2000)]
    # also: within-pair sign structure: correlation of |.| etc
    print(f"t={t:2d} all {n} twin clones: rhoDelta={rd:+.4f} z_U={rd*np.sqrt(n):+.2f}  shuffle-null sd*sqrt(n)={np.std(nul)*np.sqrt(n):.2f}  frac zero-diff x={np.mean(dx==0):.2f} y={np.mean(dy==0):.2f}")
