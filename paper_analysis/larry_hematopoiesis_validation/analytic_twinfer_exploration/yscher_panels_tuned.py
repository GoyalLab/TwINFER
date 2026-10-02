# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Analytic TwINFER (unweighted + tuned + gate) on yscher's EXACT 9 panels, cp10k input.
Score by yscher's rule (genes x genes, all CollecTRI edges) and also the TFs x panel rule."""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json, os, sys, time
import numpy as np, pandas as pd, scipy.io as sio
from sklearn.metrics import average_precision_score

HERE = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation'
from twinfer.utils.paths import get_data_root as _res_gdr  # [2026-10-01 LARRY resources/ moved out of the code tree to analysis_data/paper_analysis/larry_hematopoiesis_validation/resources (user)]
RES_HERE = f"{_res_gdr()}/paper_analysis/larry_hematopoiesis_validation"
S = f'{TWINFER_PROJECT_ROOT}/clean_code/paper_analysis/larry_hematopoiesis_validation/analytic_twinfer_exploration'
INP = f"{S}/yscher_inputs_cp10k"; os.makedirs(INP, exist_ok=True)
OUTR = f"{S}/yscher_analytic_tuned"; os.makedirs(OUTR, exist_ok=True)
PANELS = {"corrhigh": "yscher_corrhigh", "corrmid": "yscher_corrmid", "corrlow": "yscher_corrlow",
          "detect_q5075": "yscher_detlow", "detect_q7590": "yscher_detmid", "detect_q90100": "yscher_dethigh",
          "hvg_q0033": "yscher_hvglow", "hvg_q3367": "yscher_hvgmid", "hvg_q67100": "yscher_hvghigh"}

# ---- build cp10k inputs for the 9 yscher panels -------------------------------------------
SRC = f'{TWINFER_PROJECT_ROOT}/finalized_data/LARRY_data/filtered'
X = sio.mmread(f"{SRC}/larry_qc_counts.mtx").tocsr()
allg = pd.Index(open(f"{SRC}/genes.txt").read().split())
obs = pd.read_csv(f"{SRC}/obs_metadata.csv", index_col=0)
gi = {g: i for i, g in enumerate(allg)}
day = obs["library"].astype(str).str.extract(r"_d(\d+)_", expand=False).astype(int).to_numpy()
tot = np.asarray(X.sum(1)).ravel().astype(float)
for name, jf in PANELS.items():
    G = sorted(json.load(open(f"{S}/{jf}_panel.json"))["panel"])
    G = [g for g in G if g in gi]
    norm = np.log1p(X[:, [gi[g] for g in G]].toarray() / tot[:, None] * 1e4)
    df = pd.DataFrame(norm, columns=[f"{g}_mRNA" for g in G], index=obs.index)
    df.insert(0, "time_step", day); df.insert(0, "cell_id", obs.index)
    df.insert(0, "clone_id", obs["larry_clone_singletcode"].to_numpy())
    df.reset_index(drop=True).to_csv(f"{INP}/{name}.csv", index=False)
print("built 9 cp10k inputs")

# ---- run the tuned analytic ----------------------------------------------------------------
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, HERE)
os.environ["INPUT_DIR"] = INP
os.environ["OUT_DIR"] = OUTR
os.environ["GENE_SETS"] = ",".join(PANELS)
from paper_analysis.larry_hematopoiesis_validation import run_analytic_twinfer_tuned as R
for name in PANELS:
    t = time.time(); n, ng = R.run(name)
    print(f"  {name:14s} {n:5d} pairs, {ng} pass gate  ({time.time()-t:.0f}s)", flush=True)

# ---- score ------------------------------------------------------------------------------
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] ct = pd.read_csv(f"{HERE}/resources/collectri_mouse.tsv", sep="\t")
ct = pd.read_csv(f"{RES_HERE}/resources/collectri_mouse.tsv", sep="\t")
ct = ct[ct.source_genesymbol != ct.target_genesymbol]
CE = set(zip(ct.source_genesymbol, ct.target_genesymbol))


def metrics(univ, smap):
    y = np.array([1 if p in CE else 0 for p in univ]); R_ = int(y.sum()); base = y.mean()
    fl = min(smap.values()); s = np.array([smap.get(p, fl) for p in univ])
    idx = np.argsort(-s, kind="stable")
    au = average_precision_score(y, s)
    rng = np.random.default_rng(0)
    au_r = np.mean([average_precision_score(y, rng.permutation(s)) for _ in range(300)])
    pk = y[idx[:R_]].sum() / R_
    return dict(n_univ=len(univ), R=R_, base=round(100 * base, 1),
                prec_at_R=round(pk, 3), epr=round(pk / base, 2),
                auprc=round(au, 3), auprc_ratio=round(au / au_r, 2))


rows = []
for name, jf in PANELS.items():
    pj = json.load(open(f"{S}/{jf}_panel.json"))
    G = sorted(pj["panel"]); TF = sorted(pj["tfs"])
    d = pd.read_csv(f"{OUTR}/{name}/ranked_edges.csv")
    d = d[d.gene_1 != d.gene_2]
    sm = {(r.gene_1, r.gene_2): float(r.twinScore) for r in d.itertuples(index=False)}
    for tag, univ in (("genes_x_genes", [(a, b) for a in G for b in G if a != b]),
                      ("TFs_x_panel", [(a, b) for a in TF for b in G if a != b])):
        rows.append(dict(panel=name, universe=tag, **metrics(univ, sm)))
res = pd.DataFrame(rows)
res.to_csv(f"{S}/yscher_panels_tuned_results.csv", index=False)
pd.set_option("display.width", 200)
print("\n=== analytic TwINFER (unweighted + tuned + gate) on yscher's 9 panels ===")
print(res.to_string(index=False))
print("\n--- mean by universe ---")
print(res.groupby("universe")[["prec_at_R", "epr", "auprc", "auprc_ratio"]].mean().round(2).to_string())
