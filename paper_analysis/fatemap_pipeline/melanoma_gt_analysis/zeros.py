import sys, numpy as np, pandas as pd
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0,"/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline")
from paper_analysis.fatemap_pipeline import twinscore_spec_fm06 as T
from scipy.stats import spearmanr
df,Xc,tot,genes,pidx,gf=T.load_raw("correlation_high_clean_30x6")
clone=df.barcode.to_numpy(); m=T.clone_filter(pd.DataFrame(dict(c=clone)),"c"); r=np.flatnonzero(m)
X=Xc[r]; t=tot[r]; G=list(genes)
Vp=np.log1p(X[:,pidx].toarray().astype(float)/np.where(t==0,1,t)[:,None]*1e4)
print("detection fraction:",{g:round(float((Vp[:,G.index(g)]>0).mean()),2) for g in ["JUN","IL11","CXCL1","SERPINE1","DKK1","MITF","MLANA","SOX10"]})
ltot=np.log1p(t)
def res(v): Z=np.column_stack([np.ones(len(v)),ltot]); return v-Z@np.linalg.lstsq(Z,v,rcond=None)[0]
for a,b in [("JUN","IL11"),("JUN","CXCL1"),("MITF","MLANA")]:
    x,y=Vp[:,G.index(a)],Vp[:,G.index(b)]
    both0=(x==0)&(y==0)
    print(a,b,"raw rho %.3f | after log-total regression rho %.3f | rho among cells zero in both (n=%d): raw %.3f -> resid %.3f"%(
        spearmanr(x,y)[0],spearmanr(res(x),res(y))[0],both0.sum(),np.nan if both0.sum()<3 else spearmanr(x[both0],y[both0])[0],spearmanr(res(x)[both0],res(y)[both0])[0]))
