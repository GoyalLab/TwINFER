import pandas as pd, numpy as np
from scipy.stats import rankdata, norm
S="/home/gzu5140/TwINFER_KA/simulation_data/"
files={"A_to_B":S+"figure_2/A_to_B/df_rows_0_1_05082025_160041_ncells_6000_A_to_B_rep_0.csv","A_rep_B":S+"figure_3_1k/A_rep_B/df_rows_4_4_30082026_190919_ncells_6000_A_rep_B_rep_0_4b054658.csv"}
def rdel(d,t):
    s=d[d.time_step==t]; P={g:s.pivot(index='clone_id',columns='replicate',values=f"{g}_mRNA").values for g in ["gene_1","gene_2"]}
    R={g:rankdata(P[g].ravel()).reshape(-1,2) for g in P}
    dx=R["gene_1"][:,0]-R["gene_1"][:,1]; dy=R["gene_2"][:,0]-R["gene_2"][:,1]
    return (dx*dy).sum()/np.sqrt((dx**2).sum()*(dy**2).sum())
pw=lambda r,N: 1-norm.cdf(1.96-abs(r)*np.sqrt(N))+norm.cdf(-1.96-abs(r)*np.sqrt(N))
for name,f in files.items():
    d=pd.read_csv(f); r1,r20=rdel(d,1),rdel(d,20)
    print(f"{name}: rhoDelta 1h={r1:+.3f}  20h={r20:+.3f} | merged (1,500 @1h + 1,500 @20h) expected rhoDelta={(r1+r20)/2:+.3f}, z={(r1+r20)/2*np.sqrt(3000):+.1f}, power={pw((r1+r20)/2,3000):.2f} | 20h only N_t=1500: {pw(r20,1500):.2f} | 3,000 twins all at 20h: {pw(r20,3000):.2f}")
