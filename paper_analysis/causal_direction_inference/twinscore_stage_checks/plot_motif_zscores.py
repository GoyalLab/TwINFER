import os; from _paths import DATA; os.chdir(DATA)  # outputs live in analysis_data
import glob, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
d=pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results_motif_reps_zrho/*.csv"))]); d=d[d.N==6000]
cats=[("Fan_out","1-2"),("Fan_out","1-3"),("Fan_out","2-3"),("Feed_forward","1-2"),("Feed_forward","1-3"),("Feed_forward","2-3"),
      ("Mutual_regulation","1-2"),("Mutual_regulation","1-3"),("Mutual_regulation","2-3")]
kind={("Fan_out","2-3"):"nodirect"}   # no direct edge (shared driver gene_1); all others have a direct edge (2-3 in Feed_forward / Mutual too)
BLUE,ORANGE="#2a78d6","#eb6834"; INK,SUB,GRID="#0b0b0b","#52514e","#e4e3df"
shared={("Fan_out","2-3"),("Feed_forward","2-3"),("Mutual_regulation","2-3")}
panels=[("z_rho","Stage I existence  z$_\\rho$",1.96),("z_C","Stage II inherited state  z$_C$ (h-stat)",2.326),("z_U","Stage III unshared  z$_U$",1.96),
        ("z_xy","Direction  z$^\\dagger$(i→j)",2.576),("z_yx","Direction  z$^\\dagger$(j→i)",2.576)]
plt.rcParams.update({"font.size":10,"axes.edgecolor":SUB,"text.color":INK,"axes.labelcolor":SUB,"xtick.color":SUB,"ytick.color":SUB})
fig,axs=plt.subplots(2,3,figsize=(16,7.8),facecolor="white"); axs=axs.ravel(); rng=np.random.RandomState(0)
for ax,(k,title,thr) in zip(axs,panels):
    ax.set_facecolor("white"); ax.set_yscale("symlog",linthresh=4,linscale=1.2)
    for i,(m,pr) in enumerate(cats):
        v=d[(d.motif==m)&(d.pair==pr)][k].values; c=BLUE if (m,pr)==("Fan_out","2-3") else ORANGE; sh=(m,pr) in shared
        ax.scatter(i+rng.uniform(-.28,.28,len(v)),v,s=7,color=c,alpha=.35,linewidths=0,zorder=2)
        b=ax.boxplot(v,positions=[i],widths=.5,showfliers=False,patch_artist=True,zorder=3,medianprops=dict(color=INK,lw=1.6),
                     whiskerprops=dict(color=c,lw=1.2),capprops=dict(color=c,lw=1.2),boxprops=dict(color=c,lw=1.4))
        p=b["boxes"][0]
        if sh: p.set_hatch("///"); p.set_facecolor("none")
        else: p.set_facecolor(c); p.set_alpha(.30)
    for s in (thr,-thr): ax.axhline(s,color=SUB,lw=1,ls=(0,(4,3)),zorder=1)
    ax.axhline(0,color=GRID,lw=1,zorder=0)
    ax.set_title(title,loc="left",fontsize=11,color=INK); ax.set_xticks(range(9)); ax.set_xticklabels([f"{pr}" for m,pr in cats],fontsize=8.5)
    for j,(m) in enumerate(["Fan_out","Feed_forward","Mutual_regulation"]):
        ax.text(3*j+1,-0.115,m.replace("_"," "),transform=ax.get_xaxis_transform(),ha="center",va="top",fontsize=8.5,color=SUB)
    ax.set_yticks([-30,-10,-4,-2,0,2,4,10,30]); ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid(axis="y",color=GRID,lw=.6); [ax.spines[s].set_visible(False) for s in ("top","right")]
    ax.set_xlim(-.6,8.6); ax.text(8.65,thr,f"{thr:g}",va="center",ha="left",fontsize=8,color=SUB,clip_on=False)
ax=axs[5]; ax.axis("off")
ax.legend(handles=[Patch(fc=ORANGE,alpha=.3,ec=ORANGE,label="direct edge i → j (solid)"),Patch(fc="white",ec=ORANGE,hatch="///",label="direct edge + shared driver gene_1 (hatched)"),
    Patch(fc="white",ec=BLUE,hatch="///",label="NO direct edge, shared driver gene_1 (Fan_out 2-3)")],loc="upper left",frameon=False,fontsize=10)
ax.text(0,.50,"50 replicates per box (figure_4 3-gene motifs, 1k_per_motif_v2).\nEvery clone (N = 6,000) is a twin pair at 20 h and gives one\ncell at 1 h (direction); same clones for all tests.\nBox = IQR, line = median, whiskers = 1.5×IQR, dots = replicates.\nDashed lines: single-pair thresholds (1.96, 2.326, 2.576).\nTruth: Fan_out 1→2, 1→3 | Feed_forward 1→2, 1→3, 2→3 |\nMutual 1→2, 1→3, 2→3, 3→2. mRNA; y axis is symmetric-log.",
        transform=ax.transAxes,fontsize=9,color=SUB,va="top")
fig.suptitle("TwinScore z-scores on the figure_4 motifs: shared-driver check (50 replicates)",x=.01,ha="left",fontsize=13,y=.995)
fig.tight_layout(rect=(0,0.02,1,.97)); fig.savefig("zscore_boxplots_motifs.png",dpi=200); fig.savefig("zscore_boxplots_motifs.pdf")
