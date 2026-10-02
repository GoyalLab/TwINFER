from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import json, glob, os, re, sys, warnings, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
warnings.filterwarnings("ignore")
R=f'{TWINFER_PROJECT_ROOT}'; RW=f"{R}/input_data/real_world_networks"; OLD=f"{R}/simulation_data/twinfer_format"
BO=f"{R}/analysis_data/boolode_sims_real_networks/twinfer_inference_allpairs"
GI=f"{R}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs"
TOPO_BO={"B_cell_activation":f"{RW}/B_cell.txt","EMT_real":f"{RW}/EMT.txt","Pluripotent_real":f"{RW}/Pluripotent.txt",**{n:f"{OLD}/{n}/interaction_matrix.txt" for n in ("GSD","HSC","mCAD","VSC")}}
TOPO_GI={"B_cell_activation":"B_cell.txt","Pluripotent":"Pluripotent.txt","GSD":"GSD.txt","HSC":"HSC.txt","mCAD":"mCAD.txt","VSC":"VSC.txt"}
def load(path,M):
    d=json.load(open(path)); t=d["twin_score_inputs"]; dd=pd.DataFrame(t["data"],columns=t["columns"])
    dd=dd[dd.gene_1!=dd.gene_2].reset_index(drop=True)
    gn=d["gene_names"]; gi={g:i for i,g in enumerate(gn)}
    ia=dd.gene_1.map(gi).to_numpy(); ib=dd.gene_2.map(gi).to_numpy()
    dd["fwd"]=(M[ia,ib]!=0).astype(int); dd["rev"]=(M[ib,ia]!=0).astype(int)
    zr=d.get("gated_regulation",{}).get("z_reg_gated",{})
    dd["z_reg_gated"]=[zr.get(f"{a}__{b}",np.nan) if isinstance(zr,dict) else np.nan for a,b in zip(dd.gene_1,dd.gene_2)]
    rn=d.get("direction",{}).get("rho_cross_null",{})
    dd["z_dagger"]=[rn.get(f"{a}__{b}",{}).get("z_rho_cross",np.nan) if isinstance(rn.get(f"{a}__{b}",{}),dict) else np.nan for a,b in zip(dd.gene_1,dd.gene_2)]
    return dd
def auc(y,x):
    ok=np.isfinite(x)
    if ok.sum()<5 or y[ok].min()==y[ok].max(): return np.nan
    return average_precision_score(y[ok],x[ok])/y[ok].mean()  # AUPRC x base rate (1.0 = chance)
recs=[]
for src,files,topo in (("BoolODE",glob.glob(f"{BO}/*_all_results.json"),TOPO_BO),("Gillespie",glob.glob(f"{GI}/*_all_results.json"),TOPO_GI)):
    for f in sorted(files):
        b=os.path.basename(f); net=re.match(r"(.+?)_rep_",b).group(1)
        if net not in topo: continue
        p=topo[net]; p=p if os.path.isabs(p) else f"{RW}/{p}"
        M=np.loadtxt(p,delimiter=",")
        try: dd=load(f,M)
        except Exception as e: print("skip",b,repr(e)[:80]); continue
        zc=[c for c in dd.columns if c.startswith("z_")]+["rho_t1","rho_t2","rho_change","gamma"]
        # existence (undirected): one row per unordered pair
        u=dd[dd.gene_1<dd.gene_2]; ex=((u.fwd+u.rev)>0).astype(int).to_numpy()
        # orientation: one-directional true edges only; score >0 should mean gene_1 -> gene_2
        o=dd[(dd.fwd+dd.rev)==1]; yo=o.fwd.to_numpy()
        for c in zc:
            x=u[c].to_numpy(float)
            r=dict(src=src,net=net,rep=b,col=c,auc_signed=auc(ex,x),auc_abs=auc(ex,np.abs(x)),
                   auc_dir=auc(yo,o[c].to_numpy(float)) if len(np.unique(yo))==2 else np.nan)
            recs.append(r)
df=pd.DataFrame(recs); df.to_csv(f"{sys.argv[1]}/zscore_usefulness_auprc_raw.csv",index=False); print(len(df),"rows")
