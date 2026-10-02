from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from tmp_2026-09-22_phi_v2; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import pandas as pd, numpy as np, glob, os
from sklearn.metrics import auc, precision_recall_curve

# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] ct = pd.read_csv(f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation/resources/collectri_human.tsv', sep="\t")
ct = pd.read_csv(f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/larry_hematopoiesis_validation/resources/collectri_human.tsv', sep="\t")
ct = ct[ct.source_genesymbol != ct.target_genesymbol]
CE = set(zip(ct.source_genesymbol, ct.target_genesymbol))
METHODS = ["rho", "ppcor", "pidc", "genie3", "grnboost2"]

def report(sc, y):
    sc = np.asarray(sc, float)
    finite = np.isfinite(sc)
    v, yy = sc[finite], y[finite]
    if yy.sum()==0 or len(v)==0:
        return dict(auprc_x=np.nan, topk=np.nan, tp=0, k=0)
    prec, rec, _ = precision_recall_curve(yy, v)
    auprc = auc(rec, prec)
    rand = yy.mean()
    k = int(yy.sum())
    order = np.argsort(-v, kind="stable")
    tp = int((yy[order[:k]]==1).sum())
    return dict(auprc_x=auprc/rand if rand>0 else np.nan, topk=tp/k if k else np.nan, tp=tp, k=k)

def load_competitor(method, label, gs, univ):
    path = f"{TWINFER_PROJECT_ROOT}/analysis_data/{label}/data/networks/{method}_{gs}_allgenes.csv"
    if not os.path.exists(path): return None
    df = pd.read_csv(path)
    m = {(r[0], r[1]): float(r[2]) for r in df[["TF","target","importance"]].itertuples(index=False)}
    return np.array([m.get(p, 0.0) for p in univ])

rows = []
for dataset in ["fm06", "fm08"]:
    for f in sorted(glob.glob(f"{TWINFER_PROJECT_ROOT}/analysis_data/{dataset}/data/twinscore_supplement_{dataset}_*_absplit_allpairs_pair_terms.csv")):
        gs = os.path.basename(f).replace(f"twinscore_supplement_{dataset}_","").replace("_absplit_allpairs_pair_terms.csv","")
        d = pd.read_csv(f)
        y = d.collectri_edge.to_numpy()
        U = list(zip(d.gene_1, d.gene_2))
        rep_ts = report(d.TwinScore.to_numpy(), y)
        best_x, best_hits, best_name = np.nan, None, None
        for method in METHODS:
            sc = load_competitor(method, dataset, gs, U)
            if sc is None: continue
            rep_c = report(sc, y)
            if np.isfinite(rep_c["auprc_x"]) and (not np.isfinite(best_x) or rep_c["auprc_x"] > best_x):
                best_x, best_hits, best_name = rep_c["auprc_x"], f"{rep_c['tp']}/{rep_c['k']}", method
        rows.append(dict(dataset=dataset.upper(), gene_set=gs, n_pairs=len(d), n_true=int(y.sum()),
                          TwinScore_auprc_x=rep_ts["auprc_x"], TwinScore_hits=f"{rep_ts['tp']}/{rep_ts['k']}",
                          best_comp_auprc_x=best_x, best_comp_hits=best_hits, best_competitor=best_name))

t = pd.DataFrame(rows).sort_values(["dataset","n_true"], ascending=[True,False])
with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format):
    print(t.to_string(index=False))
print(f"\nFM06 mean TwinScore AUPRC-x: {t[t.dataset=='FM06'].TwinScore_auprc_x.mean():.3f}")
print(f"FM08 mean TwinScore AUPRC-x: {t[t.dataset=='FM08'].TwinScore_auprc_x.mean():.3f}")
print(f"FM06 mean best-competitor AUPRC-x: {t[t.dataset=='FM06'].best_comp_auprc_x.mean():.3f}")
print(f"FM08 mean best-competitor AUPRC-x: {t[t.dataset=='FM08'].best_comp_auprc_x.mean():.3f}")
