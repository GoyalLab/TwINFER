"""TwinFER without the calling gate: rank ALL source-TF pairs by ungated R (or |z*|, |rho|), top-k with k = number of true ChIP edges. Spec, merged, no covariate regression.
usage: topk_ungated.py <gene_set> <iso_file> <out_dir>   (iso_file: iso_mel.py or iso_rand.py, which set the gene set)"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import os, sys, numpy as np, pandas as pd
gs_arg, iso_file, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
exec(open(os.path.dirname(os.path.abspath(__file__)) + "/" + iso_file).read().split("G=list(genes)")[0])
from sklearn.metrics import auc, precision_recall_curve
B = f'{TWINFER_PROJECT_ROOT}/analysis_data/fm06/data'
G = list(genes); cl = clone[m]; cid = df.cell_id.to_numpy()[m]
smp = T.Sample(Vp, cl, cid, np.random.default_rng(0)); r = T.steps(smp, smp.null_sds("analytic"))
TFS = ["MITF","SOX10","JUN","JUND","FOSL1","FOSL2","TEAD1","TEAD2","TEAD4"]
gt = pd.read_csv(f"{B}/melanoma_tf_network_fm06.tsv", sep="\t"); gt = gt[(gt.TF != gt.target) & gt.evidence.isin(["chip_only","both"])]
pos = set(zip(gt.TF, gt.target))
rows = []
for x in TFS:
    if x not in G: continue
    i = G.index(x)
    for j, y in enumerate(G):
        if j != i:
            rows.append(dict(gene_1=x, gene_2=y, true=int((x, y) in pos), R_ungated=r["R"][i, j], R_called=r["R"][i, j] if (r["stage1"][i, j] and r["called"][i, j]) else np.nan,
                             abs_zstar=abs(r["zstar"][i, j]), abs_rho=abs(smp.RHO[i, j]), zreg=r["zreg"][i, j]))
q = pd.DataFrame(rows)
for mth in ["rho", "ppcor", "pidc", "genie3", "grnboost2"]:
    f = f"{B}/networks/{mth}_{gs_arg}_allgenes.csv"
    if os.path.exists(f):
        c = pd.read_csv(f); dd = {(a, b): v for a, b, v in zip(c.TF, c.target, c.importance)}
        q[mth] = [dd.get((a, b), 0.0) for a, b in zip(q.gene_1, q.gene_2)]
y = q.true.values; k = int(y.sum())
def ev(s):
    s = np.nan_to_num(np.asarray(s, float), nan=-1e9); pr, rc, _ = precision_recall_curve(y, s); a = auc(rc, pr)
    o = np.argsort(-s, kind="stable"); return a, a / y.mean(), int(y[o[:k]].sum()), y[o[:20]].sum(), y[o[:50]].sum()
rows = []
for c, l in [("R_called", "TwinFER R, gated (called only)"), ("R_ungated", "TwinFER R, ungated (top-k)"), ("abs_zstar", "|z*| ungated"), ("abs_rho", "|rho| (all-cell Spearman)"), ("zreg", "z_reg (signed)")] + [(m_, m_) for m_ in ["rho","ppcor","pidc","genie3","grnboost2"] if m_ in q]:
    a, ax, tp, t20, t50 = ev(q[c]); rows.append(dict(score=l, AUPRC=a, AUPRC_x=ax, TP_at_k=tp, true_top20=int(t20), true_top50=int(t50)))
print(f"{gs_arg}: pairs {len(q)}, true {k} (random {y.mean():.3f}), k={k}; expected true in top20 {20*y.mean():.1f}, top50 {50*y.mean():.1f}, top-k {k*y.mean():.0f}")
t = pd.DataFrame(rows); print(t.round(3).to_string(index=False)); t.to_csv(f"{OUT}/topk_ungated_{gs_arg}.csv", index=False); q.to_csv(f"{OUT}/topk_ungated_pairs_{gs_arg}.csv", index=False)
