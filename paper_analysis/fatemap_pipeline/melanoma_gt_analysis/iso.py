from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import sys, numpy as np, pandas as pd
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0,"/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline")
from paper_analysis.fatemap_pipeline import twinscore_spec_fm06 as T
YROOT=T.YROOT
gs="correlation_high_clean_30x6"
df,Xc,tot,genes,pidx,gf=T.load_raw(gs)
clone=df.barcode.to_numpy(); m=T.clone_filter(pd.DataFrame(dict(c=clone)),"c"); r=np.flatnonzero(m)
X=Xc[r]; t=tot[r]; uni=T.covariate_universe(X,gf)
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0,f"{YROOT}/hive_fm06/work/fm06-builder-contrarian/scripts")
from paper_analysis.fatemap_pipeline.external_yscher.gene_lists import G2M_GENES,S_GENES
Vp=np.log1p(X[:,pidx].toarray().astype(float)/np.where(t==0,1,t)[:,None]*1e4)
VB=np.log1p(X[:,uni].toarray().astype(float)/np.where(t==0,1,t)[:,None]*1e4)
idx={gf[i].upper():k for k,i in enumerate(uni)}
IEG=["FOS","FOSB","JUN","JUNB","JUND","EGR1","EGR2","EGR3","IER2","IER3","ATF3","DUSP1","ZFP36","NR4A1"]
def score(gl):
    cols=[idx[n.upper()] for n in gl if n.upper() in idx]; sd=VB[:,cols].std(0)
    return ((VB[:,cols]-VB[:,cols].mean(0))/np.where(sd>0,sd,1)).mean(1), [gf[uni[c]] for c in cols]
cov={"logtot":np.log1p(np.expm1(VB).sum(1))}
cov["S"],_=score(S_GENES); cov["G2M"],_=score(G2M_GENES); cov["IEG"],used=score(IEG)
print("IEG genes present in universe:",used)
cov["IEG_noAP1"],_=score([g for g in IEG if g not in("FOS","FOSB","JUN","JUNB","JUND")])
def resid(names):
    if not names: return Vp
    Z=np.column_stack([np.ones(len(Vp))]+[cov[n] for n in names]); b=np.linalg.lstsq(Z,Vp,rcond=None)[0]
    return Vp-Z@b+Vp.mean(0)
G=list(genes); cl=clone[m]; cid=df.cell_id.to_numpy()[m]
pairs=[("JUN","IL11"),("JUN","CXCL1"),("JUN","SERPINE1"),("JUN","DKK1"),("MITF","MLANA"),("SOX10","MLANA")]
variants={"none":[],"logtot":["logtot"],"S":["S"],"G2M":["G2M"],"IEG":["IEG"],"IEG_noAP1":["IEG_noAP1"],"all4 (default)":["logtot","S","G2M","IEG"]}
rows=[];store={}
for name,ns in variants.items():
    smp=T.Sample(resid(ns),cl,cid,np.random.default_rng(0)); sd=smp.null_sds("analytic")
    H=-((smp.RD-smp.CEN)/sd["sd_het"])*np.sign(smp.CEN); store[name]=H
    for x,y in pairs:
        i,j=G.index(x),G.index(y)
        rows.append(dict(variant=name,pair=f"{x}->{y}",rho=smp.RHO[i,j],C=smp.C[i,j],rhoD=smp.RD[i,j],rhoD_rand=smp.CEN[i,j],h=H[i,j],lam=min(1,max(0,H[i,j])/2.33)))
    jn=G.index("JUN"); print(name,"JUN row mean lam %.2f mean rho %.3f"%(np.mean([min(1,max(0,H[jn,k])/2.33) for k in range(len(G)) if k!=jn]),np.mean([abs(smp.RHO[jn,k]) for k in range(len(G)) if k!=jn])),flush=True)
o=pd.DataFrame(rows); o.to_csv(f"{sys.argv[1]}/iso.csv",index=False)
print(o.pivot(index="variant",columns="pair",values="h").round(2).loc[list(variants)].to_string())
print(o.pivot(index="variant",columns="pair",values="rho").round(3).loc[list(variants)].to_string())
# validate against stored merged run, all ordered pairs
st=pd.read_csv(f'{TWINFER_PROJECT_ROOT}/analysis_data/fm06/data/twinscore_spec/twinscore_spec_merged_analytic_zreg_corrected_fm06_correlation_high_clean_30x6_pair_terms.csv')
H=store["all4 (default)"]; gi={g:i for i,g in enumerate(G)}
mine=np.array([H[gi[a],gi[b]] for a,b in zip(st.gene_1,st.gene_2)])
print("\nvalidation vs stored merged run (analytic_h, all %d pairs): max|diff| %.4f, corr %.5f"%(len(st),np.nanmax(np.abs(mine-st.analytic_h)),np.corrcoef(mine,st.analytic_h)[0,1]))
Hn=store["none"]; mn=np.array([Hn[gi[a],gi[b]] for a,b in zip(st.gene_1,st.gene_2)])
print("no-covariate h vs stored: corr %.3f"%np.corrcoef(mn,st.analytic_h)[0,1])
