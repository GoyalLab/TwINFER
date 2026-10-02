"""What separates detected/undetected and true/false ChIP edges for TwinFER (spec, merged, no covariate regression) on the random-target panel.
usage: features_rand.py <out_dir>"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import os, sys, numpy as np, pandas as pd
exec(open(os.path.dirname(os.path.abspath(__file__)) + "/iso_rand.py").read().split("G=list(genes)")[0])
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
OUT = sys.argv[1]; B = f'{TWINFER_PROJECT_ROOT}/analysis_data/fm06/data'
G = list(genes); cl = clone[m]; cid = df.cell_id.to_numpy()[m]
smp = T.Sample(Vp, cl, cid, np.random.default_rng(0)); r = T.steps(smp, smp.null_sds("analytic"))
TFS = ["MITF","SOX10","JUN","JUND","FOSL1","FOSL2","TEAD1","TEAD2","TEAD4"]
gt = pd.read_csv(f"{B}/melanoma_tf_network_fm06.tsv", sep="\t"); gt = gt[(gt.TF != gt.target) & gt.evidence.isin(["chip_only","both"])]
pos = {(a, b): e for a, b, e in zip(gt.TF, gt.target, gt.evidence)}
tier = gt.groupby("TF").chip_tier.first().to_dict(); nbind = gt.groupby("target").TF.nunique().to_dict(); tfsize = gt.groupby("TF").size().to_dict()
kd_tfs = set(gt[gt.evidence == "both"].TF)
det = (Vp > 0).mean(0); mu = Vp.mean(0); sdv = Vp.std(0); hgene = np.diag(smp.C)
rows = []
for x in TFS:
    i = G.index(x)
    for j, y in enumerate(G):
        if j == i or y in TFS and False: continue
        rows.append(dict(src=x, tgt=y, true=int((x, y) in pos), evidence=pos.get((x, y), ""),
            called=int(bool(r["stage1"][i, j] and r["called"][i, j])), R=r["R"][i, j],
            abs_rho=abs(smp.RHO[i, j]), z_rho=abs(r["z_rho"][i, j]), S=smp.S[i, j], C=smp.C[i, j], h_pair=r["h"][i, j], lam=r["lam"][i, j], z_star=abs(r["zstar"][i, j]),
            det_src=det[i], det_tgt=det[j], mean_tgt=mu[j], sd_tgt=sdv[j], h_src=hgene[i], h_tgt=hgene[j],
            tgt_is_TF=int(y in TFS), n_TFs_binding_tgt=nbind.get(y, 0), tf_chip_set_size=tfsize.get(x, 0), tf_chip_tier=tier.get(x, "")))
D = pd.DataFrame(rows); D["R"] = D.R.where(D.called == 1)
k = int(D.true.sum()); D["topk"] = D.R.fillna(-1e9).rank(ascending=False, method="first") <= k
D.to_csv(f"{OUT}/features_rand_pairs.csv", index=False)
print(f"pairs {len(D)} | true {k} ({D.true.mean():.3f}) | called {D.called.sum()} | recall of true among called {D[D.true==1].called.mean():.3f} | precision among called {D[D.called==1].true.mean():.3f} | top-k precision {D[D.topk].true.mean():.3f}")
print("\nby source TF:"); print(D.groupby("src").agg(pairs=("true","size"), true=("true","sum"), called=("called","sum"), true_called=("true", lambda s: int(s[D.loc[s.index,'called']==1].sum())), true_topk=("true", lambda s: int(s[D.loc[s.index,'topk']].sum()))).to_string())
feats = ["abs_rho","z_rho","S","C","h_pair","lam","det_src","det_tgt","mean_tgt","sd_tgt","h_src","h_tgt","tgt_is_TF","n_TFs_binding_tgt","tf_chip_set_size"]
def uni(sub, lab, name):
    o = []
    for f in feats:
        v = sub[f].to_numpy(float)
        if sub[lab].nunique() < 2 or np.nanstd(v) == 0: continue
        o.append(dict(feature=f, AUC=roc_auc_score(sub[lab], np.nan_to_num(v)), median_pos=np.nanmedian(v[sub[lab] == 1]), median_neg=np.nanmedian(v[sub[lab] == 0])))
    t = pd.DataFrame(o); t["sep"] = (t.AUC - 0.5).abs(); t = t.sort_values("sep", ascending=False).drop(columns="sep")
    print(f"\n== {name}: n={len(sub)} (label=1: {int(sub[lab].sum())}) -- univariate AUC of each feature (0.5 = none)"); print(t.round(3).to_string(index=False)); t.to_csv(f"{OUT}/features_rand_{name}.csv", index=False)
uni(D[D.true == 1], "called", "TRUE_edges_detected_vs_undetected")
uni(D[D.called == 1], "true", "DETECTED_edges_true_vs_false")
uni(D, "true", "ALL_pairs_true_vs_false")
Dc = D[D.called == 1]; Xs = ((Dc[feats] - Dc[feats].mean()) / Dc[feats].std().replace(0, 1)).fillna(0)
lr = LogisticRegression(max_iter=2000, C=1.0).fit(Xs, Dc.true)
print("\nmultivariate logistic on detected edges (standardized coefs; + = more likely true):"); print(pd.Series(lr.coef_[0], index=feats).sort_values(key=abs, ascending=False).round(2).to_string())
