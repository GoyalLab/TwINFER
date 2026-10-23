"""Fig 2A-style violin (hESCs p2): Pearson correlation between cell pairs, different clones vs same clone, for every SCT-derived matrix the authors' object provides:
 SCT data (log) all genes | SCT data 3,000 variable genes | SCT scale.data (scaled Pearson residuals, 3,000 variable genes; from the h5ad obsm) | SCT corrected counts (no log) | stored PCA (50 PCs).
Paper Fig 2A (read off the figure, approximate): between-clone ~0.55, within-clone ~0.68, Cohen's d = 3.51."""
import numpy as np, pandas as pd, h5py
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
EXP="/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/export"; OUT="/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/paper_reproduction"; H5="/home/gzu5140/TwINFER_KA/real_data/Novo_nordisk_hPSC_endo.h5ad"
rng=np.random.default_rng(0); obs=pd.read_csv(f"{EXP}/obs.csv",index_col=0,low_memory=False)
def load_dense(prefix,cols,nr):
    p=np.fromfile(f"{EXP}/{prefix}_p.bin",dtype=np.float64).astype(np.int64); I=np.memmap(f"{EXP}/{prefix}_i.bin",dtype=np.int32,mode="r"); X=np.memmap(f"{EXP}/{prefix}_x.bin",dtype=np.float64,mode="r")
    D=np.zeros((len(cols),nr),np.float32)
    for s in range(0,len(cols),2000):
        c=cols[s:s+2000]; l=p[c+1]-p[c]; idx=np.repeat(p[c]-np.r_[0,np.cumsum(l)[:-1]],l)+np.arange(int(l.sum())); D[s+np.repeat(np.arange(len(c)),l),np.asarray(I[idx])]=np.asarray(X[idx])
    return D
def pearson_rows(D): D-=D.mean(1,keepdims=True); D/=np.sqrt(np.einsum("ij,ij->i",D,D)/D.shape[1])[:,None]; return (D@D.T)/D.shape[1]
def cohen_d(a,b): return (a.mean()-b.mean())/np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
def pairs_within(l):
    o=np.argsort(l,kind="stable"); ls=l[o]; st=np.flatnonzero(np.r_[True,ls[1:]!=ls[:-1]]); en=np.r_[st[1:],len(ls)]; A,B=[],[]
    for s,e in zip(st,en):
        if e-s>=2: i,j=np.triu_indices(e-s,1); A.append(o[s+i]); B.append(o[s+j])
    return np.concatenate(A),np.concatenate(B)
m=((obs.SampleID.astype(str)=="hESCs_p2_parent")&obs["clone.id"].notna()).to_numpy(); cols=np.flatnonzero(m); clone=obs["clone.id"][m].astype(int).to_numpy()
a,b=pairs_within(clone); ra=rng.integers(0,len(cols),600_000); rb=rng.integers(0,len(cols),600_000); ok=clone[ra]!=clone[rb]; ra,rb=ra[ok][:300_000],rb[ok][:300_000]
sct=pd.read_csv(f"{EXP}/SCT_counts_features.csv").feature.astype(str).values; var=pd.read_csv(f"{EXP}/SCT_variable_features.csv").feature.astype(str).values; keep=np.isin(sct,var)
res,rows={},[]
def rep(name,R):
    w,bt=R[a,b],R[ra,rb]; res[name]=(w,bt)
    rows.append(dict(input=name,within_mean=w.mean(),within_sd=w.std(),between_mean=bt.mean(),between_sd=bt.std(),cohens_d=cohen_d(w,bt),AUC=mannwhitneyu(w,bt,alternative="greater").statistic/(len(w)*len(bt)))); print(rows[-1],flush=True)
D=load_dense("SCT_data",cols,len(sct)); rep("SCT data (log), 3,000 variable genes",pearson_rows(D[:,keep].copy())); rep("SCT data (log), all 34,207 genes",pearson_rows(D)); del D
D=load_dense("SCT_counts",cols,len(sct)); rep("SCT corrected counts (no log), all genes",pearson_rows(D)); del D
with h5py.File(H5) as f:
    names=np.array(f["uns/SCT_scale_data_features"]).astype(str); S=f["obsm/SCT_scale_data"][cols.tolist()].astype(np.float32)   # cells x 3,000 (scaled Pearson residuals)
assert len(names)==S.shape[1]; rep("SCT scale.data (Pearson residuals), 3,000 variable genes",pearson_rows(S.copy())); del S
pc=pd.read_csv(f"{EXP}/red_pca.csv",index_col=0).iloc[:,:50].to_numpy(np.float64)[m]; rep("stored SCT PCA, 50 PCs",np.corrcoef(pc))
T=pd.DataFrame(rows); T.to_csv(f"{OUT}/figA_violin_hESC_p2.csv",index=False)
BLUE,ORANGE,INK,MUTED,GRID,SURF="#2a78d6","#eb6834","#0b0b0b","#52514e","#e7e6e2","#fcfcfb"
names=list(res); fig,axs=plt.subplots(1,len(names),figsize=(15,5),facecolor=SURF)
for ax,nm in zip(axs,names):
    w,bt=res[nm]; ax.set_facecolor(SURF)
    vp=ax.violinplot([rng.choice(bt,20000,replace=False),rng.choice(w,min(len(w),20000),replace=False)],positions=[0,1],showmeans=False,showmedians=False,showextrema=False,widths=0.85)
    for body,col in zip(vp["bodies"],(ORANGE,BLUE)): body.set_facecolor(col); body.set_edgecolor(col); body.set_alpha(0.85)
    for pos,x in ((0,bt),(1,w)):
        q=np.percentile(x,[25,50,75]); ax.plot([pos,pos],[q[0],q[2]],color=INK,linewidth=2.2,solid_capstyle="butt"); ax.plot([pos-0.07,pos+0.07],[q[1],q[1]],color=INK,linewidth=2.2)
    r=T.set_index("input").loc[nm]; ax.set_title(nm.replace(", ",",\n",1),loc="left",fontsize=9,color=INK)
    ax.text(0.5,0.02,f"d = {r.cohens_d:.2f}\nmeans {r.between_mean:.3f} vs {r.within_mean:.3f}",transform=ax.transAxes,ha="center",va="bottom",fontsize=8.5,color=INK)
    ax.set_xticks([0,1]); ax.set_xticklabels(["Between\nclone","Within\nclone"],fontsize=9,color=INK); ax.yaxis.grid(True,color=GRID,linewidth=0.8); ax.set_axisbelow(True); ax.tick_params(colors=MUTED,length=0,labelsize=8)
    for sp in ("top","right","left"): ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
axs[0].set_ylabel("Pearson correlation between two cells",color=INK,fontsize=9.5)
fig.text(0.01,0.97,"hESCs p2: cell-pair correlation within vs between clones, by input",fontsize=12,color=INK,ha="left",va="top")
fig.text(0.01,0.025,"Paper Fig 2A (read off the figure, approximate): between ~0.55, within ~0.68, Cohen's d = 3.51. Bars: quartiles; 17,497 same-clone pairs vs 300,000 random different-clone pairs (violins show 20,000 each).",fontsize=8.5,color=MUTED,ha="left")
fig.subplots_adjust(left=0.05,right=0.995,bottom=0.17,top=0.80,wspace=0.28)
for ext in ("png","pdf"): fig.savefig(f"{OUT}/figA_violin_hESC_p2.{ext}",dpi=200,facecolor=SURF)
pd.set_option("display.width",250); print(T.round(3).to_string(index=False))
