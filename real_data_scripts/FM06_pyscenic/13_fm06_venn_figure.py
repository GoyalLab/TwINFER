import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
OUT="/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/fm06/pyscenic/compare/FM06_venn_TFs_edges.png"
COL={"G":"#0072B2","U":"#E69F00","S":"#009E73"}   # colour-blind-safe (Okabe-Ito)
cent={"G":(-0.55,0.32),"U":(0.55,0.32),"S":(0.0,-0.62)}; R=1.0
pos={"G":(-1.02,0.62),"U":(1.02,0.62),"GU":(0.0,0.88),"GS":(-0.62,-0.42),"US":(0.62,-0.42),"GUS":(0.0,0.06),"S":(0.0,-1.28)}
def venn(ax,v,title,sets):
    for k,c in cent.items(): ax.add_patch(Circle(c,R,facecolor=COL[k],alpha=0.28,edgecolor=COL[k],lw=1.6))
    for k,(x,y) in pos.items(): ax.text(x,y,f"{v[k]:,}",ha="center",va="center",fontsize=11.5,color="#111111")
    ax.text(-1.55,1.42,f"GRNBoost2\n{sets['G']:,}",ha="center",va="bottom",fontsize=10.5,color=COL["G"],fontweight="bold")
    ax.text(1.55,1.42,f"TwINFER U\n{sets['U']:,}",ha="center",va="bottom",fontsize=10.5,color=COL["U"],fontweight="bold")
    ax.text(0.0,-1.78,f"TwINFER S  {sets['S']:,}",ha="center",va="top",fontsize=10.5,color=COL["S"],fontweight="bold")
    ax.set_xlim(-2.3,2.3); ax.set_ylim(-2.1,2.0); ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title,fontsize=11.5,pad=8)
tf={"G":22,"U":22,"S":30,"GU":7,"GS":16,"US":39,"GUS":126}
ed={"G":6228,"U":6145,"S":10889,"GU":1413,"GS":5542,"US":8290,"GUS":11148}
tot=lambda v,k:sum(n for r,n in v.items() if k in r)
fig,axs=plt.subplots(1,2,figsize=(11,5.2))
venn(axs[0],tf,"TFs with a regulon (262 in total)",{k:tot(tf,k) for k in "GUS"})
venn(axs[1],ed,"Regulon edges, TF → target (49,655 in total)",{k:tot(ed,k) for k in "GUS"})
fig.tight_layout(); fig.savefig(OUT,dpi=250,facecolor="white"); print("saved",OUT)
print("totals TFs",{k:tot(tf,k) for k in "GUS"},"edges",{k:tot(ed,k) for k in "GUS"})
