from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import glob, sys, numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score
R=f'{TWINFER_PROJECT_ROOT}'
import warnings; warnings.filterwarnings("ignore")
NETS={
 "B_cell_activation":dict(g=["Ikaros","PU_1","Flk2","IL_7R","GATA_1","E2A","EBF","C_EBPa","PAX5","Notch_1"],mat="B_cell.txt",
    gil=f"{R}/analysis_data/paper_analysis/B_cell_activation/simulate/20260812_153622"),
 "EMT_real":dict(g=["Cdh1","Cldn7","Foxc2","Grhl2","Gsc","Klf8","Np63a","Ovol2","Snai1","Snai2","Tcf3","Tgfbeta","Twist1","Twist2","Vim","Zeb1","Zeb2"],mat="EMT.txt",
    gil=f"{R}/analysis_data/paper_analysis/EMT/simulate/20260825_224653"),
 "Pluripotent_real":dict(g=["ARID3B","CEBPZ","ETV4","FOXH1","HIC2","HMGB3","JARID2","LIN28B","MIS18BP1","MYCN","NANOG","POU5F1","POU5F1B","PRDM14","REST","SALL2","SALL4","SMARCC1","SOX2","TEAD2","TERF1","TGIF1","WDHD1","ZBTB12","ZBTB39","ZNF281","ZNF286A","ZNF286B","ZNF322","ZNF398","ZNF462","ZNF730","ZNF90","ZNF92","ZSCAN10","ZSCAN2"],mat="Pluripotent.txt",
    gil=f"{R}/analysis_data/paper_analysis/Pluripotent/simulate/20260825_224511"),
}
NREP=5
rng=np.random.default_rng(0)
def z(X): return (X-X.mean(0))/(X.std(0)+1e-9)
def metrics(A,B,label):
    """A,B: (n_pairs, genes) twin A/B final states. Returns per-gene twin corr, gene-gene corr (all cells), twin vs random pearson/euclid over genes (z-scored)."""
    allc=np.vstack([A,B]); mu,sd=allc.mean(0),allc.std(0)+1e-9
    Az,Bz=(A-mu)/sd,(B-mu)/sd
    tw_pe=np.array([pearsonr(a,b)[0] for a,b in zip(Az,Bz)]); tw_eu=np.linalg.norm(Az-Bz,axis=1)
    perm=rng.permutation(len(A)); 
    rn_pe=np.array([pearsonr(a,b)[0] for a,b in zip(Az,Bz[perm])]); rn_eu=np.linalg.norm(Az-Bz[perm],axis=1)
    gene_twin=np.array([spearmanr(A[:,j],B[:,j])[0] for j in range(A.shape[1])])
    C=np.nan_to_num(spearmanr(allc)[0]); print("   const genes:",int((allc.std(0)<1e-9).sum()),flush=True) if (allc.std(0)<1e-9).any() else None
    return dict(tw_pe=tw_pe.mean(),rn_pe=rn_pe.mean(),tw_eu=tw_eu.mean(),rn_eu=rn_eu.mean(),gene_twin=gene_twin,C=C)
def load_boolode(net,g,rep):
    d=pd.read_csv(f"{R}/simulation_data/boolode_sims_replicates/{net}/replicate_{rep}/twin_final_states.csv")
    d=d[d.source!="trunk"]; last=d.step.max(); d=d[d.step==last]
    a=d[d.source=="twin_A"].sort_values("pair_id")[g].values; b=d[d.source=="twin_B"].sort_values("pair_id")[g].values
    return a,b
def load_gil(net,dirn,rep,ng,kind):
    f=glob.glob(f"{dirn}/df_rows*_rep_{rep}_*.csv")[0]
    cols=["cell_id","time_step","clone_id"]+[f"gene_{i+1}_{kind}" for i in range(ng)]
    d=pd.read_csv(f,usecols=cols); d=d[d.time_step==d.time_step.max()]
    d=d.sort_values(["clone_id","cell_id"]); 
    grp=d.groupby("clone_id"); first=grp.nth(0); second=grp.nth(1)
    vals=lambda x: np.log1p(x[[f"gene_{i+1}_{kind}" for i in range(ng)]].values.astype(float))
    return vals(first),vals(second)
out=[]
for net,cfg in [(k,v) for k,v in NETS.items() if k=="Pluripotent_real"]:
    g=cfg["g"]; ng=len(g); M=np.loadtxt(f"{R}/input_data/real_world_networks/{cfg['mat']}",delimiter=",")
    iu=np.triu_indices(ng,1); undirected=((M!=0)|(M.T!=0))[iu].astype(int)
    res={}
    for sim,loader in [("boolode",lambda r:load_boolode(net,g,r)),("gil_mRNA",lambda r:load_gil(net,cfg["gil"],r,ng,"mRNA")),("gil_prot",lambda r:load_gil(net,cfg["gil"],r,ng,"protein"))]:
        ms=[]
        for r in range(NREP):
            try: A,B=loader(r)
            except Exception as e: print(net,sim,r,"skip",repr(e)[:80]); continue
            ms.append(metrics(A,B,sim))
        res[sim]=ms
        print(f"{net} {sim}: reps={len(ms)}",flush=True)
    print(f"\n=== {net} ({ng} genes, {int(undirected.sum())} undirected edges of {len(iu[0])} pairs) ===")
    Cs={}
    for sim,ms in res.items():
        if not ms: continue
        C=np.mean([m["C"] for m in ms],0); Cs[sim]=C[iu]
        auc=roc_auc_score(undirected,np.abs(C[iu])) if 0<undirected.sum()<len(undirected) else np.nan
        gt=np.nanmean([m["gene_twin"] for m in ms],0)
        print(f"{sim:9s} twin_pe={np.mean([m['tw_pe'] for m in ms]):.3f} rand_pe={np.mean([m['rn_pe'] for m in ms]):.3f} twin_eu={np.mean([m['tw_eu'] for m in ms]):.2f} rand_eu={np.mean([m['rn_eu'] for m in ms]):.2f} | per-gene twin rho mean={np.nanmean(gt):.3f} | |rho| AUROC vs edges={auc:.3f}")
        Cs[sim+"_gt"]=gt
    for a,b in [("boolode","gil_mRNA"),("boolode","gil_prot"),("gil_mRNA","gil_prot")]:
        if a in Cs and b in Cs:
            print(f"  gene-gene rho agreement {a} vs {b}: spearman={spearmanr(Cs[a],Cs[b],nan_policy='omit')[0]:.3f} (signed), |rho|: {spearmanr(np.abs(Cs[a]),np.abs(Cs[b]),nan_policy='omit')[0]:.3f}; per-gene twin rho agreement: {spearmanr(Cs[a+'_gt'],Cs[b+'_gt'],nan_policy='omit')[0]:.3f}")
