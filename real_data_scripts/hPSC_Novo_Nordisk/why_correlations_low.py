"""Why are gene-gene correlations low in hESCs p2? Spearman across all cells of hESCs_p2_parent (13,148 cells), CP10k from raw counts.
(1) positive controls: ribosomal, S-phase, G2M, mitochondrial gene sets -> median pairwise |rho| and detection;
(2) the analysed TF x target candidate pairs: detection fraction of both genes and |rho| by detection bin (rho taken from the TwinScore base run)."""
import json, numpy as np, pandas as pd
from scipy.stats import rankdata
EXP="/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/export"; TS="/home/gzu5140/TwINFER_KA/analysis_data/hPSC_Novo_Nordisk/twinscore/base"
EXCL="/home/gzu5140/TwINFER_KA/clean_data/paper_analysis/fatemap_pipeline/hpsc_20260927/gene_exclusion_list.json"
obs=pd.read_csv(f"{EXP}/obs.csv",index_col=0,low_memory=False); cols=np.flatnonzero((obs.SampleID.astype(str)=="hESCs_p2_parent").to_numpy()); n=len(cols)
genes=pd.read_csv(f"{EXP}/RNA_counts_features.csv").feature.astype(str).values; gi={g:i for i,g in enumerate(genes)}; nr=len(genes)
ex=json.load(open(EXCL)); sets={"ribosomal (RPS/RPL)":[g for g in genes if g.startswith(("RPS","RPL"))],"S-phase (Tirosh)":ex["tirosh_S"],"G2M (Tirosh)":ex["tirosh_G2M"],"mitochondrial (MT-)":[g for g in genes if g.startswith("MT-")]}
cand=pd.read_csv(f"{TS}/hESCs_p2_parent/pdf_procedure_all_candidate_pairs.csv",usecols=["TF","gene","rho"]).dropna(); need=set(cand.TF)|set(cand.gene)
for v in sets.values(): need|=set(v)
need=sorted(g for g in need if g in gi); sel=np.full(nr,-1,np.int32); sel[[gi[g] for g in need]]=np.arange(len(need))
p=np.fromfile(f"{EXP}/RNA_counts_p.bin",dtype=np.float64).astype(np.int64); I=np.memmap(f"{EXP}/RNA_counts_i.bin",dtype=np.int32,mode="r"); X=np.memmap(f"{EXP}/RNA_counts_x.bin",dtype=np.float64,mode="r")
D=np.zeros((n,len(need)),np.float32); tot=np.zeros(n)
for s in range(0,n,2000):
    c=cols[s:s+2000]; l=p[c+1]-p[c]; idx=np.repeat(p[c]-np.r_[0,np.cumsum(l)[:-1]],l)+np.arange(int(l.sum())); ii=np.asarray(I[idx]); xx=np.asarray(X[idx]); row=np.repeat(np.arange(len(c)),l)
    tot[s:s+len(c)]=np.bincount(row,weights=xx,minlength=len(c)); g=sel[ii]; k=g>=0; D[s+row[k],g[k]]=xx[k]
D*= (1e4/tot)[:,None].astype(np.float32); det=(D>0).mean(0); R=np.apply_along_axis(rankdata,0,D).astype(np.float32); R-=R.mean(0); R/=np.sqrt((R**2).mean(0)); C=(R.T@R)/n
name={g:i for i,g in enumerate(need)}
print(f"hESCs_p2_parent: {n} cells, median total UMI {np.median(tot):.0f}\n"); rows=[]
for nm,gs in sets.items():
    ix=[name[g] for g in gs if g in name and det[name[g]]>=0.05]
    if len(ix)<5: continue
    sub=np.abs(C[np.ix_(ix,ix)][np.triu_indices(len(ix),1)]); rows.append(dict(gene_set=nm,genes=len(ix),median_detection=np.median(det[ix]),median_abs_rho=np.median(sub),p95_abs_rho=np.percentile(sub,95),median_signed_rho=np.median(C[np.ix_(ix,ix)][np.triu_indices(len(ix),1)])))
cs=cand[cand.TF.isin(name)&cand.gene.isin(name)].copy(); a=cs.TF.map(name).to_numpy(); b=cs.gene.map(name).to_numpy(); cs["rho_here"]=C[a,b]; cs["det_min"]=np.minimum(det[a],det[b]); cs["det_max"]=np.maximum(det[a],det[b])
rows.append(dict(gene_set="candidate TF x target pairs (the analysed pairs)",genes=len(cs),median_detection=np.median(np.r_[det[a],det[b]]),median_abs_rho=cs.rho_here.abs().median(),p95_abs_rho=cs.rho_here.abs().quantile(.95),median_signed_rho=cs.rho_here.median()))
pd.set_option("display.width",250); print(pd.DataFrame(rows).round(3).to_string(index=False))
cs["bin"]=pd.cut(cs.det_min,[0,0.1,0.2,0.4,0.7,1.0]); print("\ncandidate pairs by detection of the LESS-detected gene (fraction of cells with nonzero):")
print(cs.groupby("bin",observed=True).agg(pairs=("rho_here","size"),median_abs_rho=("rho_here",lambda x:x.abs().median()),p95_abs_rho=("rho_here",lambda x:x.abs().quantile(.95))).round(3).to_string())
print("\nshare of candidate pairs by that bin:",(cs.groupby('bin',observed=True).size()/len(cs)).round(2).to_dict())
