"""TwinFER with the direction term but the merged sample everywhere except the cross-sample step.
Steps I-IV / R: merged A+B sample (clone = barcode over both dishes), no covariate regression, analytic null (spec).
Cross-sample terms: rho_dagger(x->y) = corr(x in A cell, y in B clone-mate) (twinscore_spec_fm06.cross_terms), z_dagger = rho_dagger/sd_dag, coupled = max(|z_xy|,|z_yx|)>2.576,
gamma(x->y) = |rho_dagger(x->y)| - |rho_dagger(y->x)|.  score = s(R) + coupled*s(gamma) over called pairs; fan-out (called+heterogeneous+bidirectional pair with another TF called with both genes) dropped.
usage: ab_gamma_merged.py <gene_set> <out_dir>"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import os, sys, numpy as np, pandas as pd
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline")
from paper_analysis.fatemap_pipeline import twinscore_spec_fm06 as T
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
gs, OUT = sys.argv[1], sys.argv[2]; B = f'{TWINFER_PROJECT_ROOT}/analysis_data/fm06/data'
Z = T.Z
df, Xc, tot, genes, pidx, gf = T.load_raw(gs); G = list(genes); n = len(G)
TFS = [g for g in ["MITF","SOX10","JUN","JUND","FOSL1","FOSL2","TEAD1","TEAD2","TEAD4"] if g in G]
clone = df.barcode.to_numpy(); mk = T.clone_filter(pd.DataFrame(dict(c=clone)), "c"); r_ = np.flatnonzero(mk)
V = T.panel_expression(Xc[r_], tot[r_], pidx, None, gf)
smp = T.Sample(V, clone[mk], df.cell_id.to_numpy()[mk], np.random.default_rng(0)); st = T.steps(smp, smp.null_sds("analytic"))
parts, Vd = {}, {}
for d_ in ("A", "B"):
    sel = np.flatnonzero((df.dish == d_).to_numpy()); parts[d_] = df.iloc[sel].reset_index(drop=True); Vd[d_] = T.panel_expression(Xc[sel], tot[sel], pidx, None, gf)
both = set(parts["A"].barcode) & set(parts["B"].barcode)
kA, kB = parts["A"].barcode.isin(both).to_numpy(), parts["B"].barcode.isin(both).to_numpy()
rdag, sd_dag = T.cross_terms(parts["A"][kA].rename(columns={"barcode": "clone_id"}), Vd["A"][kA], parts["B"][kB].rename(columns={"barcode": "clone_id"}), Vd["B"][kB], genes)
rdag = np.nan_to_num(rdag); zxy = rdag / sd_dag; zyx = zxy.T
coup = np.maximum(np.abs(zxy), np.abs(zyx)) > Z; gam = np.abs(rdag) - np.abs(rdag.T)
ok = st["stage1"] & st["called"]; R = st["R"]
sR = np.zeros((n, n)); sG = np.zeros((n, n)); sR[ok] = T.s(R[ok]); sG[ok] = T.s(gam[ok])
score = np.where(ok, sR + coup * sG, -np.inf)
bidir = (np.abs(zxy) > Z) & (np.abs(zyx) > Z); tfi = [G.index(t) for t in TFS]; drop = np.zeros((n, n), bool)
for x in range(n):
    for y in range(n):
        if x != y and ok[x, y] and st["hetero"][x, y] and bidir[x, y] and any(ok[x, w] and ok[y, w] for w in tfi if w not in (x, y)): drop[x, y] = True
score_drop = np.where(drop, -np.inf, score)
gt = pd.read_csv(f"{B}/melanoma_tf_network_fm06.tsv", sep="\t"); gt = gt[(gt.TF != gt.target) & gt.evidence.isin(["chip_only", "both"])]; pos = set(zip(gt.TF, gt.target))
rows = []
for x in TFS:
    i = G.index(x)
    for j, y in enumerate(G):
        if j != i:
            rows.append(dict(gene_1=x, gene_2=y, true=int((x, y) in pos), R_called=R[i, j] if ok[i, j] else np.nan, R_ungated=R[i, j], S=smp.S[i, j], called=int(ok[i, j]), coupled=int(coup[i, j]),
                             gamma=gam[i, j], z_xy=zxy[i, j], z_yx=zyx[i, j], bidir=int(bidir[i, j]), dropped=int(drop[i, j]), score_R_gamma=score[i, j], score_R_gamma_fanout=score_drop[i, j]))
q = pd.DataFrame(rows)
for m_ in ["rho", "ppcor", "pidc", "genie3", "grnboost2"]:
    f = f"{B}/networks/{m_}_{gs}_allgenes.csv"
    if os.path.exists(f):
        c = pd.read_csv(f); dd = {(a, b): v for a, b, v in zip(c.TF, c.target, c.importance)}; q[m_] = [dd.get((a, b), 0.0) for a, b in zip(q.gene_1, q.gene_2)]
q.to_csv(f"{OUT}/ab_gamma_merged_pairs_{gs}.csv", index=False)
y = q.true.values; k = int(y.sum())
def ev(s):
    s = np.nan_to_num(np.asarray(s, float), nan=-1e9, neginf=-1e9); pr, rc, _ = precision_recall_curve(y, s); a = auc(rc, pr); o = np.argsort(-s, kind="stable")
    return a, a / y.mean(), int(y[o[:k]].sum()), int(y[o[:20]].sum()), int(y[o[:50]].sum())
print(f"{gs}: {len(q)} pairs, true {k} (base {y.mean():.3f}); called {int(q.called.sum())}; coupled among called {int(q[q.called==1].coupled.sum())}; bidirectional among called {int(q[q.called==1].bidir.sum())}; fan-out dropped {int(q.dropped.sum())}; clones in both {len(both)}, sd_dag {sd_dag:.4f}")
rows = []
for c, l in [("R_called", "R only (merged, gated)"), ("score_R_gamma", "s(R)+coupled*s(gamma)"), ("score_R_gamma_fanout", "s(R)+coupled*s(gamma), fan-out dropped")] + [(m_, m_) for m_ in ["rho","ppcor","pidc","genie3","grnboost2"] if m_ in q]:
    a, ax, tp, t20, t50 = ev(q[c]); rows.append(dict(score=l, AUPRC=a, AUPRC_x=ax, TP_at_k=tp, true_top20=t20, true_top50=t50))
t = pd.DataFrame(rows); print(t.round(3).to_string(index=False)); t.to_csv(f"{OUT}/ab_gamma_merged_{gs}.csv", index=False)
cc = q[(q.called == 1) & (q.coupled == 1)]
print(f"\ncalled & coupled: {len(cc)} pairs, true {cc.true.mean():.3f} vs called&uncoupled {q[(q.called==1)&(q.coupled==0)].true.mean():.3f} (base {y.mean():.3f})")
if cc.true.nunique() > 1: print("AUC of gamma (true vs false) among called&coupled: %.3f; of |z_xy|: %.3f" % (roc_auc_score(cc.true, cc.gamma), roc_auc_score(cc.true, cc.z_xy.abs())))
