"""Score TwinFER (spec, merged sample, analytic null, no covariate regression; R for called pairs and ungated R) and the competitors on every panel against the cutaneous-melanoma
ChIP-Atlas truth (chipatlas_cutaneous_melanoma_tf_network.tsv). Universes: all antigens as sources, and sequence-specific TFs only.
usage: score_all_panels_newgt.py <out_dir> gene_set [gene_set ...]"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import os, sys, numpy as np, pandas as pd
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline")
from paper_analysis.fatemap_pipeline import twinscore_spec_fm06 as T
from sklearn.metrics import auc, precision_recall_curve
OUT = sys.argv[1]; B = f'{TWINFER_PROJECT_ROOT}/analysis_data/fm06/data'
E = pd.read_csv(f"{B}/melanoma_gt_benchmark/chipatlas_cutaneous_melanoma_tf_network.tsv", sep="\t"); pos = set(zip(E.TF, E.target)); ALLTF = set(E.TF)
SEQ = {"AR","ARNT","ATF3","E2F1","EGR1","EPAS1","ETV1","FOS","FOSL1","FOSL2","HIF1A","JUN","JUNB","JUND","MITF","NR2F2","REST","SMAD1","SMAD2","SOX10","SREBF2","TEAD4","TFAP2A","TFAP2C","TP53","ZEB1"}
MIN_TRUE = 5; rows_all = []
def ev(s, y):
    s = np.nan_to_num(np.asarray(s, float), nan=-1e9, neginf=-1e9); pr, rc, _ = precision_recall_curve(y, s); a = auc(rc, pr); k = int(y.sum()); o = np.argsort(-s, kind="stable")
    return a / y.mean(), int(y[o[:k]].sum()), int(y[o[:20]].sum())
for gs in sys.argv[2:]:
    df, Xc, tot, genes, pidx, gf = T.load_raw(gs); G = list(genes); n = len(G)
    clone = df.barcode.to_numpy(); mk = T.clone_filter(pd.DataFrame(dict(c=clone)), "c"); r_ = np.flatnonzero(mk)
    V = T.panel_expression(Xc[r_], tot[r_], pidx, None, gf)
    smp = T.Sample(V, clone[mk], df.cell_id.to_numpy()[mk], np.random.default_rng(0)); st = T.steps(smp, smp.null_sds("analytic"))
    ok = st["stage1"] & st["called"]
    rows = []
    for x in [g for g in G if g in ALLTF]:
        i = G.index(x)
        for j, y in enumerate(G):
            if j != i: rows.append(dict(gene_1=x, gene_2=y, true=int((x, y) in pos), seq=int(x in SEQ), R_gated=st["R"][i, j] if ok[i, j] else np.nan, R_ungated=st["R"][i, j], abs_rho=abs(smp.RHO[i, j])))
    q = pd.DataFrame(rows)
    for m_ in ["rho", "ppcor", "pidc", "genie3", "grnboost2"]:
        f = f"{B}/networks/{m_}_{gs}_allgenes.csv"
        if os.path.exists(f):
            c = pd.read_csv(f); dd = {(a, b): v for a, b, v in zip(c.TF, c.target, c.importance)}; q[m_] = [dd.get((a, b), 0.0) for a, b in zip(q.gene_1, q.gene_2)]
    q.to_csv(f"{OUT}/newgt_pairs_{gs}.csv", index=False)
    for uni, sub in [("all_antigens", q), ("seq_specific_TFs", q[q.seq == 1])]:
        y = sub.true.values
        if y.sum() < MIN_TRUE or y.sum() == len(y): print(f"{gs} [{uni}]: {int(y.sum())} true of {len(y)} -> skipped", flush=True); continue
        r = dict(gene_set=gs, universe=uni, sources=sub.gene_1.nunique(), pairs=len(sub), true=int(y.sum()), base=round(y.mean(), 3))
        for c, l in [("R_gated", "TwinFER"), ("R_ungated", "TwinFER_ungated"), ("abs_rho", "abs_rho")] + [(m_, m_) for m_ in ["rho","ppcor","pidc","genie3","grnboost2"] if m_ in sub]:
            ax, tp, t20 = ev(sub[c], y); r[l + "_x"] = round(ax, 3); r[l + "_TP"] = tp; r[l + "_t20"] = t20
        rows_all.append(r); print(f"{gs} [{uni}] sources {r['sources']} pairs {r['pairs']} true {r['true']} base {r['base']} | TwinFER x {r['TwinFER_x']} pidc x {r.get('pidc_x')} ppcor x {r.get('ppcor_x')}", flush=True)
pd.DataFrame(rows_all).to_csv(f"{OUT}/newgt_summary_{'_'.join(sys.argv[2:3])}_and_{len(sys.argv)-2}_panels.csv", index=False)
