"""Share of clone pairs that are the same cell type (clust_annot_02): observed vs chance.
chance 1: two random cells of the sample (different-clone pairs of random cells): sum_t p_t^2 over the sample's clone-labelled cells (analytic, with replacement)
chance 2: clone labels shuffled among the sample's clone-labelled cells (clone sizes kept), 2,000 shuffles -> expected share, sd, z, one-sided p (observed >= null)
"""
import numpy as np, pandas as pd
EXP="/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/export"; OUT="/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/clone_celltype_overlap_chance.csv"
rng=np.random.default_rng(0); NPERM=2000
o=pd.read_csv(f"{EXP}/obs.csv",index_col=0,low_memory=False)
def share_same(c,t):
    od=np.argsort(c,kind="stable"); cs=c[od]; ts=t[od]; st=np.flatnonzero(np.r_[True,cs[1:]!=cs[:-1]]); en=np.r_[st[1:],len(cs)]; same=tot=0
    for s,e in zip(st,en):
        n=e-s
        if n>=2:
            i,j=np.triu_indices(n,1); same+=int((ts[s+i]==ts[s+j]).sum()); tot+=len(i)
    return same,tot
rows=[]
for s in ["hESCs_p2_parent","hESCs_p3_same","hESCs_p3_sis","PS_sis","DE_same","DE_sis"]:
    m=(o.SampleID.astype(str)==s)&o["clone.id"].notna(); c=o.loc[m,"clone.id"].astype(int).to_numpy(); t=pd.Categorical(o.loc[m,"clust_annot_02"].astype(str)).codes
    sm,tt=share_same(c,t); obs=sm/tt
    p=np.bincount(t)/len(t); analytic=float((p**2).sum())
    null=np.array([share_same(c,t[rng.permutation(len(t))])[0]/tt for _ in range(NPERM)])
    rows.append(dict(sample=s,clone_pairs=tt,observed_same_type=obs,chance_random_cells=analytic,chance_shuffled_clones=null.mean(),null_sd=null.std(),
                     enrichment_obs_over_chance=obs/null.mean(),z=(obs-null.mean())/null.std(),p_one_sided=(1+(null>=obs).sum())/(NPERM+1),
                     expected_same_pairs=null.mean()*tt,observed_same_pairs=sm))
T=pd.DataFrame(rows); T.to_csv(OUT,index=False); pd.set_option("display.width",250); print(T.round(4).to_string(index=False))
