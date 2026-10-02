import sys, numpy as np, pandas as pd
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0,"/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline")
from paper_analysis.fatemap_pipeline import twinscore_spec_fm06 as T
gs="correlation_high_clean_30x6"
df,Xc,tot,genes,pidx,gf=T.load_raw(gs)
clone=df.barcode.to_numpy(); m=T.clone_filter(pd.DataFrame(dict(c=clone)),"c"); r=np.flatnonzero(m)
V=T.panel_expression(Xc[r],tot[r],pidx,None,gf)   # no covariate regression
smp=T.Sample(V,clone[m],df.cell_id.to_numpy()[m],np.random.default_rng(0))
sd=smp.null_sds("analytic"); G=list(genes)
rows=[]
for x,y in [("JUN","IL11"),("JUN","CXCL1"),("JUN","SERPINE1"),("JUN","DKK1"),("JUN","JUND"),("MITF","MLANA"),("MITF","PMEL"),("SOX10","MLANA"),("SOX10","TYR")]:
    i,j=G.index(x),G.index(y)
    zh=(smp.RD[i,j]-smp.CEN[i,j])/sd["sd_het"][i,j]; h=-zh*np.sign(smp.CEN[i,j])
    rows.append(dict(pair=f"{x}->{y}",rho=smp.RHO[i,j],S=smp.S[i,j],C=smp.C[i,j],rhoDelta_obs=smp.RD[i,j],rhoDelta_random=smp.CEN[i,j],sd_het=sd["sd_het"][i,j],h=h,lam=min(1,max(0,h)/2.33)))
print(pd.DataFrame(rows).round(3).to_string(index=False))
