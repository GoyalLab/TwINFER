import numpy as np, pandas as pd, h5py, sys
from scipy.stats import rankdata
import importlib.util, os
_s=importlib.util.spec_from_file_location("pair_us",os.path.join(os.path.dirname(os.path.abspath(__file__)),"18_pair_us_single_tf_vs_all_genes.py")); pair_us=importlib.util.module_from_spec(_s); _s.loader.exec_module(pair_us); load,R0=pair_us.load,pair_us.R0
path=R0+"finalized_data/FM01_data/FM01_integrated.h5ad"; genes,X,clone=load(path); n=X.shape[1]
o=h5py.File(path,"r")["obs"]; ng=o["n_genes"][:].astype(float); tcn=o["total_counts"][:].astype(float)
rng=np.random.default_rng(0); cols=np.sort(rng.choice(n,6000,replace=False)); X=X[:,cols]; ng=ng[cols]; tcn=tcn[cols]
z=lambda v:(lambda r:(r-r.mean())/r.std())(rankdata(v)); a=z(np.log(ng)); b=z(np.log(tcn)); det=(X>0).mean(1)
r=np.array([(z(X[i])*a).mean() for i in range(len(genes))]); rt=np.array([(z(X[i])*b).mean() for i in range(len(genes))])
print(f"FM01, 6,000-cell sample, {len(genes):,} genes detected in >5% of cells; corr(log genes detected, log total counts) = {np.corrcoef(np.log(ng),np.log(tcn))[0,1]:.2f}")
print("\nSpearman between each gene's NORMALISED expression (log1p CP10k) and the cell's number of genes detected, by gene detection rate:")
rows=[]
for lo,hi in ((0.05,0.10),(0.10,0.20),(0.20,0.40),(0.40,0.70),(0.70,1.01)):
    m=(det>=lo)&(det<hi); rows.append(dict(gene_detected_in=f"{int(lo*100)}-{min(int(hi*100),100)}% of cells",genes=int(m.sum()),median_r=round(float(np.median(r[m])),3),share_positive=f"{100*np.mean(r[m]>0):.0f}%",median_r_with_total_counts=round(float(np.median(rt[m])),3)))
print(pd.DataFrame(rows).to_string(index=False))
rib=pd.Series(genes).str.match(r"^(RPL|RPS|MRPL|MRPS)").values; mt=pd.Series(genes).str.startswith("MT-").values
print(f"\nribosomal-protein genes (n={int(rib.sum())}): median r {np.median(r[rib]):+.2f};  mitochondrial genes (n={int(mt.sum())}): median r {np.median(r[mt]):+.2f};  all other genes: median {np.median(r[~rib&~mt]):+.2f}")
print(f"genes with |r| > 0.1: {100*np.mean(np.abs(r)>0.1):.0f}%;  |r| > 0.2: {100*np.mean(np.abs(r)>0.2):.0f}%;  |r| > 0.3: {100*np.mean(np.abs(r)>0.3):.0f}%")
x=[list(genes).index(g) for g in ("SERPINE1","FOSL1","FN1","CAV1","MITF")]; print("  specific genes (r with genes detected): "+", ".join(f"{genes[i]} {r[i]:+.2f}" for i in x))
# what two unrelated genes would show: random pairs
pr=[]; 
for _ in range(3000):
    i,j=rng.choice(len(genes),2,replace=False); pr.append(np.corrcoef(rankdata(X[i]),rankdata(X[j]))[0,1])
pr=np.abs(np.array(pr)); print(f"\n3,000 random gene pairs: median |Spearman| = {np.median(pr):.3f}; 90th percentile {np.percentile(pr,90):.3f}; 99th percentile {np.percentile(pr,99):.3f}")
# pairs of genes that are both positively linked to library size
pos=np.where(r>0.3)[0]; pp=[]
for _ in range(2000):
    i,j=rng.choice(pos,2,replace=False); pp.append(np.corrcoef(rankdata(X[i]),rankdata(X[j]))[0,1])
print(f"2,000 random pairs among the {len(pos)} genes with r > 0.3 to library size: median Spearman between the two genes = {np.median(pp):+.3f}; 90th percentile {np.percentile(pp,90):+.3f}")
