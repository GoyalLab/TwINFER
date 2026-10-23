"""Motif-level call from the 3 pairs of one replicate (genes 1,2,3). Per pair: directed edge by delta test (p<=.05; delta>0 means i->j), inherited state = z_C>2.326 (h-stat), co-varies = |z_rho|>1.96.
Find a root x with directed edges x->y and x->z. Then look at pair (y,z):
  directed y->z or z->y               -> feed-forward
  not directed, inherited (high h)    -> fan-out
  not directed, not inherited, co-varies (high z_rho) -> regulated mutual (x->y, x->z, y<->z)
  not directed, not inherited, no co-variation        -> root only (no relation between y,z)
no root with two directed edges -> no motif."""
import sys, glob, pandas as pd, numpy as np
def classify(g):
    g=g.set_index("pair"); E=set()
    for p,r in g.iterrows():
        i,j=map(int,p.split("-"))
        if r.p_delta<=.05: E.add((i,j) if r.delta>0 else (j,i))
    for x in (1,2,3):
        y,z=sorted({1,2,3}-{x})
        if (x,y) in E and (x,z) in E:
            r=g.loc[f"{y}-{z}"]
            if (y,z) in E or (z,y) in E: return "feed-forward"
            if r.z_C>2.326: return "fan-out"
            if abs(r.z_rho)>1.96: return "regulated mutual"
            return "root only (y,z unrelated)"
    return "no motif found"
def table(d,label):
    d=d.copy(); out=[]
    for (m,rep),g in d.groupby(["motif","rep"]): out.append((m,classify(g)))
    t=pd.DataFrame(out,columns=["true motif","called"]); c=pd.crosstab(t["true motif"],t["called"],normalize="index").round(2)
    print(label); print(c.to_string()); print()
if __name__=="__main__":
    R="/projects/b1255/hzhang/TwINFER_KA/analysis_data/paper_analysis/causal_direction_inference/figure2_1000_zscores/"
    pd.set_option("display.width",200)
    d1=pd.concat([pd.read_csv(f) for f in glob.glob(R+"results_motif_design1500/*.csv")]); table(d1,"1,500 twin clones + 3,000 singletons (B), 50 replicates per motif")
    d2=pd.concat([pd.read_csv(f) for f in glob.glob(R+"results_motif_reps_zrho/*.csv")])
    table(d2[d2.N==6000],"6,000 twin clones (reference), 50 replicates per motif")
    table(d2[d2.N==3000],"3,000 twin clones (reference), 50 replicates per motif")
