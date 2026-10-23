import pandas as pd, numpy as np
from scipy.stats import rankdata
B="/home/gzu5140/TwINFER_KA/simulation_data/figure_2/"
files={"A_B":B+"A_B/df_rows_0_1_08082025_093223_ncells_6000_A_B_rep_0.csv","A_to_B":B+"A_to_B/df_rows_0_1_05082025_160041_ncells_6000_A_to_B_rep_0.csv"}
rng=np.random.RandomState(0)
def prep(x):  # x: (N,2) twin values -> centred midranks
    r=rankdata(x.ravel()).reshape(x.shape); return r-r.mean()
def h_est(r):
    M=((r**2).sum(1)/2).mean(); return max(0,(r[:,0]*r[:,1]).mean()/M)
def rho(rx,ry):
    a=rx.ravel();b=ry.ravel(); return (a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum())
def z(rx,ry,hx,hy,naive=False):
    N=len(rx); s2=(1/(2*N)) if naive else (1+hx*hy)/(2*N)
    return rho(rx,ry)/np.sqrt(s2)
for name,f in files.items():
    d=pd.read_csv(f); print(name, "time_step max",d.time_step.max())
    for t in [1,10,20,48]:
        s=d[d.time_step==t]
        for col in ["mRNA","protein"]:
            P={g:s.pivot(index='clone_id',columns='replicate',values=f"{g}_{col}").values for g in ["gene_1","gene_2"]}
            rx,ry=prep(P["gene_1"]),prep(P["gene_2"]); hx,hy=h_est(rx),h_est(ry)
            zr=z(rx,ry,hx,hy); zn=z(rx,ry,hx,hy,True)
            nz=[];nn=[]
            for _ in range(2000):
                p=rng.permutation(len(ry)); ryp=ry[p]   # shuffle gene_2 across clones, twins kept together
                nz.append(z(rx,ryp,hx,hy)); nn.append(z(rx,ryp,hx,hy,True))
            nz,nn=np.array(nz),np.array(nn)
            print(f"{name} t={t:2d} {col:7s} hx={hx:.2f} hy={hy:.2f} rho={rho(rx,ry):+.3f} z_obs(h-adj)={zr:+.1f} z_obs(naive)={zn:+.1f} | null sd: h-adj={nz.std():.2f} naive={nn.std():.2f}  frac|z|>1.96: h-adj={np.mean(abs(nz)>1.96):.3f} naive={np.mean(abs(nn)>1.96):.3f}")
