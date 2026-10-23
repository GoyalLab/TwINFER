"""Distribution of Pearson correlations between cell pairs, same clone vs different clones, hESCs p2 (the paper's Fig 2A sample), for six inputs:
raw RNA counts | CP10k (no log) | log1p(CP10k) | SCT corrected counts (no log) | SCT data (log), all genes | SCT data (log), 3,000 variable genes.
Paper Fig 2A (read off the figure, approximate): between-clone ~0.55, within-clone ~0.68, Cohen's d = 3.51. Writes PNG/PDF + CSV to analysis_data/hPSC_Novo_Nordisk/paper_reproduction/."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
EXP="/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/export"; OUT="/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/paper_reproduction"; rng=np.random.default_rng(0)
obs=pd.read_csv(f"{EXP}/obs.csv",index_col=0,low_memory=False)
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
sct=pd.read_csv(f"{EXP}/SCT_counts_features.csv").feature.astype(str).values; var=set(pd.read_csv(f"{EXP}/SCT_variable_features.csv").feature.astype(str)); keep=np.array([g in var for g in sct])
nrna=int(open(f"{EXP}/RNA_counts_dim.txt").read().split()[0]); res={}; rows=[]
def rep(name,R):
    w,bt=R[a,b],R[ra,rb]; res[name]=(w,bt); q=lambda x:np.percentile(x,[5,50,95])
    rows.append(dict(input=name,within_mean=w.mean(),within_p5=q(w)[0],within_p95=q(w)[2],between_mean=bt.mean(),between_sd=bt.std(),between_p5=q(bt)[0],between_p95=q(bt)[2],shift=w.mean()-bt.mean(),cohens_d=cohen_d(w,bt),
                     AUC=mannwhitneyu(w,bt,alternative="greater").statistic/(len(w)*len(bt)))); print(rows[-1],flush=True)
D=load_dense("RNA_counts",cols,nrna); rep("raw RNA counts",pearson_rows(D.copy()))
tot=D.sum(1,keepdims=True); D*=(1e4/tot).astype(np.float32); rep("CP10k (no log)",pearson_rows(D.copy())); np.log1p(D,out=D); rep("log1p(CP10k)",pearson_rows(D)); del D
D=load_dense("SCT_counts",cols,len(sct)); rep("SCT corrected counts (no log)",pearson_rows(D)); del D
D=load_dense("SCT_data",cols,len(sct)); rep("SCT data (log), 3,000 variable genes",pearson_rows(D[:,keep].copy())); rep("SCT data (log), all genes",pearson_rows(D)); del D
T=pd.DataFrame(rows); T.to_csv(f"{OUT}/figA_distribution_hESC_p2.csv",index=False)
BLUE,ORANGE,INK,MUTED,GRID,SURF="#2a78d6","#eb6834","#0b0b0b","#52514e","#e7e6e2","#fcfcfb"
names=list(res); fig,axs=plt.subplots(2,3,figsize=(13,6.6),facecolor=SURF,sharex=True)
for ax,nm in zip(axs.ravel(),names):
    w,bt=res[nm]; bins=np.linspace(0.2,1.0,161); ax.set_facecolor(SURF)
    ax.hist(bt,bins=bins,density=True,color=ORANGE,alpha=0.85,linewidth=0); ax.hist(w,bins=bins,density=True,color=BLUE,alpha=0.75,linewidth=0)
    for x in (0.55,0.68): ax.axvline(x,color=MUTED,linewidth=1,linestyle=(0,(3,3)))
    r=T.set_index("input").loc[nm]; ax.set_title(nm,loc="left",fontsize=10,color=INK); ax.text(0.98,0.95,f"between {r.between_mean:.2f}, within {r.within_mean:.2f}\nCohen's d = {r.cohens_d:.2f}",transform=ax.transAxes,ha="right",va="top",fontsize=8.5,color=INK)
    ax.yaxis.grid(True,color=GRID,linewidth=0.8); ax.set_axisbelow(True); ax.tick_params(colors=MUTED,length=0,labelsize=8)
    for sp in ("top","right","left"): ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.set_yticks([])
for ax in axs[1]: ax.set_xlabel("Pearson correlation between two cells",color=INK,fontsize=9)
fig.legend(handles=[plt.Rectangle((0,0),1,1,color=BLUE,alpha=0.75),plt.Rectangle((0,0),1,1,color=ORANGE,alpha=0.85),plt.Line2D([0],[0],color=MUTED,linestyle=(0,(3,3)))],
           labels=["same clone (17,497 pairs)","different clones (300,000 random pairs)","paper, approximate: between ~0.55, within ~0.68 (Cohen's d = 3.51)"],loc="upper left",bbox_to_anchor=(0.01,0.995),ncol=3,frameon=False,fontsize=9,labelcolor=INK)
fig.text(0.01,0.935,"Distribution of cell-pair correlations at hESCs p2, by input",fontsize=11.5,color=INK,ha="left")
fig.subplots_adjust(left=0.03,right=0.99,bottom=0.09,top=0.86,hspace=0.28,wspace=0.08)
for ext in ("png","pdf"): fig.savefig(f"{OUT}/figA_distribution_hESC_p2.{ext}",dpi=200,facecolor=SURF)
pd.set_option("display.width",250); print(T.round(3).to_string(index=False))
