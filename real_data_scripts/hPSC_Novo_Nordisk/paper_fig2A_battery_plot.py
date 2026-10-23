"""Re-plot of figA_battery_hESC_p2.csv with full labels (method + input). Cohen's d, same-clone vs random pairs, hESCs p2."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT="/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/paper_reproduction"; T=pd.read_csv(f"{OUT}/figA_battery_hESC_p2.csv")
T["label"]=T.group+"  |  "+T.measure; T=T.sort_values("cohens_d")
BLUE,INK,MUTED,GRID,SURF="#2a78d6","#0b0b0b","#52514e","#e7e6e2","#fcfcfb"
fig,ax=plt.subplots(figsize=(13,0.24*len(T)+2),facecolor=SURF); ax.set_facecolor(SURF); y=np.arange(len(T))
ax.hlines(y,0,T.cohens_d,color=BLUE,linewidth=1.4,alpha=0.8); ax.plot(T.cohens_d,y,"o",color=BLUE,markersize=4.5)
for yi,v in zip(y,T.cohens_d): ax.text(v+0.04,yi,f"{v:.2f}",va="center",fontsize=7,color=MUTED)
ax.axvline(3.51,color=MUTED,linestyle=(0,(3,3)),linewidth=1.2); ax.text(3.46,1,"paper Fig 2A: d = 3.51",ha="right",color=MUTED,fontsize=9)
ax.set_yticks(y); ax.set_yticklabels(T.label,fontsize=7.2,color=INK); ax.set_xlim(-0.05,3.75); ax.set_ylim(-1,len(T))
ax.set_xlabel("Cohen's d: same-clone pairs vs random different-clone pairs (higher = clone-mates more similar)",color=INK,fontsize=9.5)
ax.xaxis.grid(True,color=GRID,linewidth=0.8); ax.set_axisbelow(True)
for s in ("top","right","left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=MUTED,length=0,labelsize=8)
ax.set_title("hESCs p2: how far from the paper's Fig 2A effect is each measure?",loc="left",color=INK,fontsize=11.5,pad=10)
fig.subplots_adjust(left=0.50,right=0.985,bottom=0.07,top=0.95)
for ext in ("png","pdf"): fig.savefig(f"{OUT}/figA_battery_hESC_p2.{ext}",dpi=170,facecolor=SURF)
print("saved")
