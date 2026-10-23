import os; from _paths import DATA; os.chdir(DATA)  # outputs live in analysis_data
import glob, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
d=pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results/rep_*.csv"))+sorted(glob.glob("results_drift/*_rep_*.csv"))])
order=["A_B (single state)","A_B (two states)","A_B (drift)","A_to_B (single state)","A_to_B (two states)","A_to_B (drift)"]
BLUE,ORANGE="#2a78d6","#eb6834"; INK,SUB,GRID="#0b0b0b","#52514e","#e4e3df"
col={"A_B":BLUE,"A_to_B":ORANGE}
panels=[("z_rho","Stage I existence  z$_\\rho$",1.96),("z_C","Stage II inherited state  z$_C$ (h-stat)",2.326),
        ("z_U","Stage III unshared  z$_U$",1.96),("z_dag_xy","Direction  z$^\\dagger$(x→y)",2.576),("z_dag_yx","Direction  z$^\\dagger$(y→x)",2.576)]
plt.rcParams.update({"font.size":10,"axes.edgecolor":SUB,"text.color":INK,"axes.labelcolor":SUB,"xtick.color":SUB,"ytick.color":SUB})
fig,axs=plt.subplots(2,3,figsize=(16,7.6),facecolor="white"); axs=axs.ravel(); rng=np.random.RandomState(0)
for ax,(k,title,thr) in zip(axs,panels):
    ax.set_facecolor("white"); ax.set_yscale("symlog",linthresh=4,linscale=1.2)
    for i,sc in enumerate(order):
        v=d.loc[d.scenario==sc,k].values; c=col[sc.split(" (")[0]]; two=("two" in sc) or ("drift" in sc)
        ax.scatter(i+rng.uniform(-.28,.28,len(v)),v,s=7,color=c,alpha=.35,linewidths=0,zorder=2)
        b=ax.boxplot(v,positions=[i],widths=.5,showfliers=False,patch_artist=True,zorder=3,
                     medianprops=dict(color=INK,lw=1.6),whiskerprops=dict(color=c,lw=1.2),capprops=dict(color=c,lw=1.2),boxprops=dict(color=c,lw=1.4))
        p=b["boxes"][0]; p.set_facecolor("white" if two else c); p.set_alpha(1 if two else .30)
        if two: p.set_hatch("///"); p.set_facecolor("none")
    for s in (thr,-thr): ax.axhline(s,color=SUB,lw=1,ls=(0,(4,3)),zorder=1)
    ax.axhline(0,color=GRID,lw=1,zorder=0)
    ax.set_title(title,loc="left",fontsize=11,color=INK); ax.set_xticks(range(6)); ax.set_xticklabels(["no edge\nsingle","no edge\ntwo states","no edge\ndrift","edge\nsingle","edge\ntwo states","edge\ndrift"],fontsize=8)
    ax.set_yticks([-30,-10,-4,-2,0,2,4,10,30]); ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid(axis="y",color=GRID,lw=.6); [ax.spines[s].set_visible(False) for s in ("top","right")]
    ax.set_xlim(-.6,5.6); ax.text(5.65,thr,f"{thr:g}",va="center",ha="left",fontsize=8,color=SUB,clip_on=False)
ax=axs[5]; ax.axis("off")
ax.legend(handles=[Patch(fc=BLUE,alpha=.3,ec=BLUE,label="A_B: no regulation"),Patch(fc=ORANGE,alpha=.3,ec=ORANGE,label="A_to_B: x → y regulation"),
    Patch(fc="white",ec=SUB,label="solid box: single state"),Patch(fc="none",ec=SUB,hatch="///",label="hatched box: two states / drift (low+high k_on)")],loc="upper left",frameon=False,fontsize=10)
ax.text(0,.40,"100 replicates per box (reps 0–99), one draw each.\nBox = IQR, line = median, whiskers = 1.5×IQR, dots = replicates.\nDashed lines: single-pair thresholds (1.96, 2.326, 2.576).\nB = 20 h (1,500 twin clones + 3,000 singletons);\nA = 1 h; drift = k_on ramped 0–10 h to 0.12/1.66; 3,000 shared clones for direction.\nmRNA, y axis is symmetric-log (linear within ±4).",transform=ax.transAxes,fontsize=9,color=SUB,va="top")
fig.suptitle("TwinScore z-scores across 100 simulated replicates (Figure-2 and k_on-drift simulations)",x=.01,ha="left",fontsize=13,y=.995)
fig.tight_layout(rect=(0,0,1,.97)); fig.savefig("zscore_boxplots_with_drift.png",dpi=200); fig.savefig("zscore_boxplots_with_drift.pdf")
# rates
r=d.copy(); r["covary"]=abs(r.z_rho)>1.96; r["inherited"]=r.z_C>2.326; r["called"]=r.covary&(abs(r.z_U)>1.96)
r["directed"]=r.p_delta<=.05; r["coupled"]=np.maximum(abs(r.z_dag_xy),abs(r.z_dag_yx))>2.576; r["reg_any"]=r.called&r.directed
r["reg_correct"]=r.reg_any&(r.delta>0)&(r.U>0); r["dir_correct"]=r.directed&(r.delta>0)
g=r.groupby("scenario")[["covary","inherited","called","directed","dir_correct","coupled","reg_any","reg_correct"]].mean().loc[order].round(2)
g.to_csv("rates_by_scenario_with_drift.csv"); print(g.to_string())
print(d.groupby("scenario")[["z_rho","z_C","z_U","z_dag_xy","z_dag_yx"]].agg(["median"]).loc[order].round(1).to_string())
