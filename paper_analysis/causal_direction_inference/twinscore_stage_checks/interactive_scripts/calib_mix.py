import pandas as pd, numpy as np
from scipy.stats import rankdata
S="/home/gzu5140/TwINFER_KA/simulation_data/"
files={"A_B":S+"figure_2/A_B/df_rows_0_1_08082025_093223_ncells_6000_A_B_rep_0.csv",
"A_to_B":S+"figure_2/A_to_B/df_rows_0_1_05082025_160041_ncells_6000_A_to_B_rep_0.csv",
"A_rep_B":S+"figure_3_1k/A_rep_B/df_rows_4_4_30082026_190919_ncells_6000_A_rep_B_rep_0_4b054658.csv"}
rng=np.random.RandomState(0)
NT=NS=1500
def wrank(v,w):  # R_i = sum_{v_j<v_i} w_j + 0.5 sum_{v_j=v_i} w_j
    u,inv=np.unique(v,return_inverse=True); sw=np.bincount(inv,weights=w)
    return (np.cumsum(sw)-sw+0.5*sw)[inv]
def wcorr(a,b,w):
    a=a-(w*a).sum()/w.sum(); b=b-(w*b).sum()/w.sum()
    return (w*a*b).sum()/np.sqrt((w*a*a).sum()*(w*b*b).sum())
def h_of(R2):  # R2: (NT,2) weighted ranks of twin cells
    r=R2-R2.mean(); M=((r**2).sum(1)/2).mean(); return max(0.,(r[:,0]*r[:,1]).mean()/M)
def build(v1,v2,tw,sg):   # values: tw (NT,2), sg (NS,)
    v=np.concatenate([tw.ravel(),sg]); w=np.concatenate([np.full(2*NT,.5),np.ones(NS)]); return v,w
N1=NT+NS
def s2(hx,hy): return (NT*(.5+.5*hx*hy)+NS)/N1**2
for name,f in files.items():
    d=pd.read_csv(f)
    for t in [1,10,20,48]:
        s=d[d.time_step==t]
        P={g:s.pivot(index='clone_id',columns='replicate',values=f"{g}_mRNA").values for g in ["gene_1","gene_2"]}
        cl=rng.permutation(P["gene_1"].shape[0]); tc,sc=cl[:NT],cl[NT:NT+NS]; pick=rng.randint(0,2,NS)
        R={};H={}
        for g in P:
            tw=P[g][tc]; sg=P[g][sc,pick]; v,w=build(None,None,tw,sg); r=wrank(v,w); R[g]=r; H[g]=h_of(r[:2*NT].reshape(NT,2))
        w=np.concatenate([np.full(2*NT,.5),np.ones(NS)])
        rho=wcorr(R["gene_1"],R["gene_2"],w)
        z=rho/np.sqrt(s2(H["gene_1"],H["gene_2"])); zn=rho/np.sqrt(s2(0,0))
        a=R["gene_2"][:2*NT].reshape(NT,2); b=R["gene_2"][2*NT:]
        nz=[];nn=[]
        for _ in range(2000):
            r2=np.concatenate([a[rng.permutation(NT)].ravel(),b[rng.permutation(NS)]])
            rr=wcorr(R["gene_1"],r2,w); nz.append(rr/np.sqrt(s2(H["gene_1"],H["gene_2"]))); nn.append(rr/np.sqrt(s2(0,0)))
        nz,nn=np.array(nz),np.array(nn)
        print(f"{name:7s} t={t:2d} h1={H['gene_1']:.2f} h2={H['gene_2']:.2f} rho={rho:+.3f} z_adj={z:+.1f} z_naive={zn:+.1f} | null sd adj={nz.std():.2f} naive={nn.std():.2f} FP adj={np.mean(abs(nz)>1.96):.3f} naive={np.mean(abs(nn)>1.96):.3f}")
