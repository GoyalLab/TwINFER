import pandas as pd, numpy as np
from scipy.stats import rankdata, norm
S="/home/gzu5140/TwINFER_KA/simulation_data/"
files={"A_to_B":S+"figure_2/A_to_B/df_rows_0_1_05082025_160041_ncells_6000_A_to_B_rep_0.csv","A_rep_B":S+"figure_3_1k/A_rep_B/df_rows_4_4_30082026_190919_ncells_6000_A_rep_B_rep_0_4b054658.csv"}
for name,f in files.items():
    d=pd.read_csv(f)
    for t in [10,20,48]:
        s=d[d.time_step==t]; P={g:s.pivot(index='clone_id',columns='replicate',values=f"{g}_mRNA").values for g in ["gene_1","gene_2"]}
        R={g:rankdata(P[g].ravel()).reshape(-1,2) for g in P}
        dx=R["gene_1"][:,0]-R["gene_1"][:,1]; dy=R["gene_2"][:,0]-R["gene_2"][:,1]
        rd=(dx*dy).sum()/np.sqrt((dx**2).sum()*(dy**2).sum()); n=len(dx)
        pw=lambda N: 1-norm.cdf(1.96-abs(rd)*np.sqrt(N))+norm.cdf(-1.96-abs(rd)*np.sqrt(N))
        print(f"{name} t={t:2d}: rhoDelta(6000 twins)={rd:+.3f} z_U={rd*np.sqrt(n):+.1f} | implied power at N_t=1500: {pw(1500):.2f}  3000: {pw(3000):.2f}  6000: {pw(6000):.2f}  (N_t for 80% power: {int(((1.96+0.84)/abs(rd))**2)})")
