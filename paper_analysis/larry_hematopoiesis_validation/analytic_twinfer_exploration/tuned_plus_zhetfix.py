# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""z_het-fix ON TOP OF the yscher-tuned analytic twinScore, for
  (a) our 9 LARRY gene sets      -> TFs x panel
  (b) yscher's 9 exact panels    -> genes x genes  AND  TFs x panel

tuned TS  = run_analytic_twinfer_tuned.py  (E1 - CHG - RD1 - DR - XS - GAM
            + 0.5*ZDD + dREG, then |z_reg_gated|>1.645 gate else -E1)
tuned+zf  = tuned TS  +  s(|z_het|)          # s() = panel z over all directed pairs
            z_het from the clone-weighted analytic run (analytic_infer / yscher_analytic_zhetfix)
"""
import zlib  # [2026-10-01 added]
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json, os, sys, time
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score

HERE = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation'
from twinfer.utils.paths import get_data_root as _res_gdr  # [2026-10-01 LARRY resources/ moved out of the code tree to analysis_data/paper_analysis/larry_hematopoiesis_validation/resources (user)]
RES_HERE = f"{_res_gdr()}/paper_analysis/larry_hematopoiesis_validation"
S = f'{TWINFER_PROJECT_ROOT}/clean_code/paper_analysis/larry_hematopoiesis_validation/analytic_twinfer_exploration'
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, HERE)
N_RAND = int(os.environ.get("N_RAND", "30"))
N_MC = 200
GS_OURS = ["variability_high", "variability_mid", "variability_low",
           "detection_high", "detection_mid", "detection_low",
           "correlation_high", "correlation_mid", "correlation_low"]
YP = {"corrhigh": "yscher_corrhigh", "corrmid": "yscher_corrmid", "corrlow": "yscher_corrlow",
      "detlow": "yscher_detlow", "detmid": "yscher_detmid", "dethigh": "yscher_dethigh",
      "hvglow": "yscher_hvglow", "hvgmid": "yscher_hvgmid", "hvghigh": "yscher_hvghigh"}
san = lambda g: g.replace("-", "_")

# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] ct = pd.read_csv(f"{HERE}/resources/collectri_mouse.tsv", sep="\t")
ct = pd.read_csv(f"{RES_HERE}/resources/collectri_mouse.tsv", sep="\t")
ct = ct[ct.source_genesymbol != ct.target_genesymbol]
CE = set(zip(ct.source_genesymbol.map(san), ct.target_genesymbol.map(san)))
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] gene_sets = json.load(open(f"{HERE}/resources/gene_sets.json"))
gene_sets = json.load(open(f"{RES_HERE}/resources/gene_sets.json"))
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] detail = json.load(open(f"{HERE}/resources/gene_sets_detail.json"))
detail = json.load(open(f"{RES_HERE}/resources/gene_sets_detail.json"))


def pz(v):
    v = np.asarray(v, float); f = np.isfinite(v); o = np.zeros(v.shape)
    if f.any():
        x = v[f]; sd = x.std(ddof=0)
        o[f] = 0.0 if sd == 0 else (x - x.mean()) / sd
    return o


def score(univ, smap, seed):
    y = np.array([1 if p in CE else 0 for p in univ], int)
    R = int(y.sum())
    if R == 0 or R == len(univ):
        return None
    fl = min(smap.values())
    s = np.array([smap.get(p, fl) for p in univ], float)
    au = average_precision_score(y, s)
    pk = y[np.argsort(-s, kind="stable")[:R]].sum() / R
    rng = np.random.default_rng(seed)
    aur = np.mean([average_precision_score(y, rng.permutation(s)) for _ in range(N_MC)])
    return dict(R=R, prec_at_R=round(pk, 3), auprc=round(au, 3),
                auprc_ratio=round(au / aur, 2), epr=round(pk / (R / len(univ)), 2))


def merged(tuned_csv, plain_csv):
    t = pd.read_csv(tuned_csv)[["gene_1", "gene_2", "twinScore"]].rename(columns={"twinScore": "tuned"})
    p = pd.read_csv(plain_csv)[["gene_1", "gene_2", "z_het"]]
    m = t.merge(p, on=["gene_1", "gene_2"], how="inner")
    m = m[m.gene_1 != m.gene_2].copy()
    m["tuned_zf"] = m.tuned + pz(m.z_het.abs())
    return m


rows = []

# ---- (a) our 9 panels -------------------------------------------------------
for gs in GS_OURS:
    # [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] m = merged(f"{HERE}/resources/analytic_infer_tuned/{gs}/ranked_edges.csv",
    m = merged(f"{RES_HERE}/resources/analytic_infer_tuned/{gs}/ranked_edges.csv",
               # [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] f"{HERE}/resources/analytic_infer/{gs}/ranked_edges.csv")
               f"{RES_HERE}/resources/analytic_infer/{gs}/ranked_edges.csv")
    crit, lvl = gs.rsplit("_", 1)
    panel = gene_sets[gs]
    tfs = set(tf for tf, _ in detail[crit][lvl] if tf in panel)
    U = [(a, b) for a in tfs for b in panel if a != b]
    for vn in ("tuned", "tuned_zf"):
        sm = dict(zip(zip(m.gene_1, m.gene_2), m[vn]))
        # [2026-10-01 commented out: hash() of strings is randomised per process (PYTHONHASHSEED) so this seed was not reproducible; zlib.crc32 of the repr is stable] r = score(U, sm, hash((gs, vn)) % 2**31)
        r = score(U, sm, zlib.crc32(repr((gs, vn)).encode()) % 2**31)
        if r:
            rows.append(dict(dataset="ours", panel=gs, universe="TFs_x_panel", variant=vn, **r))

# ---- (b) yscher panels: run tuned analytic on the cp10k inputs -------------
from paper_analysis.larry_hematopoiesis_validation import run_analytic_twinfer_tuned as RT
RT.INPUT_DIR = f"{S}/yscher_inputs_cp10k"
RT.OUT_DIR = f"{S}/yscher_analytic_tuned"
RT.N_RAND = N_RAND
os.makedirs(RT.OUT_DIR, exist_ok=True)
for name in YP:
    if os.path.exists(f"{RT.OUT_DIR}/{name}/ranked_edges.csv"):
        continue
    t = time.time(); n, ng = RT.run(name)
    print(f"  tuned {name:9s} {n:5d} pairs, {ng} gated  ({time.time()-t:.0f}s)", flush=True)

genes_all = set(open(f'{TWINFER_PROJECT_ROOT}/finalized_data/LARRY_data/filtered/genes.txt').read().split())
for name, jf in YP.items():
    d = json.load(open(f"{S}/{jf}_panel.json"))
    G = [san(g) for g in d["panel"] if g in genes_all]
    TF = set(san(g) for g in d["tfs"] if san(g) in G)
    m = merged(f"{RT.OUT_DIR}/{name}/ranked_edges.csv",
               f"{S}/yscher_analytic_zhetfix/{name}/ranked_edges.csv")
    gg = [(a, b) for a in G for b in G if a != b]
    tp = [(a, b) for a in TF for b in G if a != b]
    for vn in ("tuned", "tuned_zf"):
        sm = dict(zip(zip(m.gene_1, m.gene_2), m[vn]))
        for un, U in (("yscher_genes_x_genes", gg), ("TFs_x_panel", tp)):
            # [2026-10-01 commented out: hash() of strings is randomised per process (PYTHONHASHSEED) so this seed was not reproducible; zlib.crc32 of the repr is stable] r = score(U, sm, hash((name, vn, un)) % 2**31)
            r = score(U, sm, zlib.crc32(repr((name, vn, un)).encode()) % 2**31)
            if r:
                rows.append(dict(dataset="yscher", panel=name, universe=un, variant=vn, **r))

res = pd.DataFrame(rows)
res.to_csv(f"{S}/tuned_plus_zhetfix_results.csv", index=False)
pd.set_option("display.width", 240)
print("\n=== per panel ===")
print(res.to_string(index=False))
print("\n=== mean by (dataset, universe, variant) ===")
print(res.groupby(["dataset", "universe", "variant"])[["prec_at_R", "epr", "auprc", "auprc_ratio"]]
      .mean().round(3).to_string())
