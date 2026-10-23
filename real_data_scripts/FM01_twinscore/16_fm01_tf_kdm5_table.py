# Every TF (human TF catalogue, TF_names_v_1.01.txt, detected in FM01) x KDM5A / KDM5B, from the all-genes tables written by 16_fm01_tf_vs_all_genes.py.
# Columns: rho, z_rho, S, C, U, h_x (TF heritability), h_y (KDM5 heritability), z_U, p_U, BH q_U over all TFs for that KDM5 gene, U/S, rank by |z_U| among TFs.
# usage: python 16_fm01_tf_kdm5_table.py DIR
import sys, numpy as np, pandas as pd
from scipy.stats import norm
D=sys.argv[1]; R0="/gpfs/projects/b1255/hzhang/TwINFER_KA/"
tfs=set(open(R0+"real_data/humanTFs/TF_names_v_1.01.txt").read().split())
def bh(p):
    p=np.asarray(p); m=len(p); o=np.argsort(p); q=np.empty(m); q[o]=np.minimum.accumulate((p[o]*m/np.arange(1,m+1))[::-1])[::-1]; return np.minimum(q,1)
out=[]
for g in ("KDM5A","KDM5B"):
    d=pd.read_csv(f"{D}/{g}_vs_all_genes_US.csv"); d=d[d.gene.isin(tfs)].copy()
    d=d.rename(columns={"gene":"TF"}); d.insert(1,"target",g); d["p_U"]=2*norm.sf(d.z_U.abs()); d["q_U"]=bh(d.p_U); d["U_over_S"]=d.U/d.S
    d["rank_absz_U"]=d.z_U.abs().rank(ascending=False).astype(int); out.append(d.sort_values("rank_absz_U"))
t=pd.concat(out); t.to_csv(f"{D}/all_TFs_x_KDM5A_KDM5B_scores.csv",index=False)
print(len(t),"rows;",{g:int(((t.target==g)&(t.q_U<=0.05)).sum()) for g in ("KDM5A","KDM5B")},"TFs with q_U<=0.05")
print(t.groupby("target").head(10)[["TF","target","rho","S","C","U","z_U","q_U","rank_absz_U"]].round(3).to_string(index=False))
