# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""z_het-fixed analytic twinScore on yscher's EXACT 9 panels.

1. build log1p-CP10k TwINFER inputs for the 9 yscher panels (LARRY counts)
2. run_analytic_twinfer.analytic_for_geneset  -> analytic z-scores + shipped twinScore
3. score  twinScore  and  twinScore + heterogeneity_penalty + panel_z(|z_het|)
   on BOTH universes:
     yscher    : genes x genes, positives = every CollecTRI directed edge in it
     TFs_panel : (panel TFs) x panel
"""
import zlib  # [2026-10-01 added]
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.metrics import average_precision_score

HERE = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation'
from twinfer.utils.paths import get_data_root as _res_gdr  # [2026-10-01 LARRY resources/ moved out of the code tree to analysis_data/paper_analysis/larry_hematopoiesis_validation/resources (user)]
RES_HERE = f"{_res_gdr()}/paper_analysis/larry_hematopoiesis_validation"
S = f'{TWINFER_PROJECT_ROOT}/clean_code/paper_analysis/larry_hematopoiesis_validation/analytic_twinfer_exploration'
INP = f"{S}/yscher_inputs_cp10k"
OUTR = f"{S}/yscher_analytic_zhetfix"
os.makedirs(INP, exist_ok=True)
os.makedirs(OUTR, exist_ok=True)
SRC = f'{TWINFER_PROJECT_ROOT}/finalized_data/LARRY_data/filtered'
PANELS = {"corrhigh": "yscher_corrhigh", "corrmid": "yscher_corrmid", "corrlow": "yscher_corrlow",
          "detlow": "yscher_detlow", "detmid": "yscher_detmid", "dethigh": "yscher_dethigh",
          "hvglow": "yscher_hvglow", "hvgmid": "yscher_hvgmid", "hvghigh": "yscher_hvghigh"}
N_RAND = int(os.environ.get("N_RAND", "30"))

# ------------------------------------------------------------------ 1. inputs
X = sio.mmread(f"{SRC}/larry_qc_counts.mtx").tocsr()
allg = pd.Index(open(f"{SRC}/genes.txt").read().split())
obs = pd.read_csv(f"{SRC}/obs_metadata.csv", index_col=0)
gi = {g: i for i, g in enumerate(allg)}
day = obs["library"].astype(str).str.extract(r"_d(\d+)_", expand=False).astype(int).to_numpy()
tot = np.asarray(X.sum(1)).ravel().astype(float)

san = lambda g: g.replace("-", "_")            # H2-Aa / H2-Q7 -> H2_Aa / H2_Q7
panel_genes, panel_tfs = {}, {}
if not os.path.exists(f"{INP}/hvghigh.csv"):
    for name, jf in PANELS.items():
        d = json.load(open(f"{S}/{jf}_panel.json"))
        G = [g for g in d["panel"] if g in gi]
        norm = np.log1p(X[:, [gi[g] for g in G]].toarray() / tot[:, None] * 1e4)
        df = pd.DataFrame(norm, columns=[f"{san(g)}_mRNA" for g in G])
        df.insert(0, "time_step", day)
        df.insert(0, "cell_id", np.arange(len(df)))
        df.insert(0, "clone_id", obs["larry_clone_singletcode"].to_numpy())
        df.to_csv(f"{INP}/{name}.csv", index=False)
    print("built 9 cp10k inputs", flush=True)
for name, jf in PANELS.items():
    d = json.load(open(f"{S}/{jf}_panel.json"))
    panel_genes[name] = [san(g) for g in d["panel"] if g in gi]
    panel_tfs[name] = [san(g) for g in d["tfs"] if san(g) in panel_genes[name]]

# ------------------------------------------------------------------ 2. analytic
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, HERE)
from paper_analysis.larry_hematopoiesis_validation import run_analytic_twinfer as RA
RA.INPUT_DIR = INP
RA.OUT_DIR = OUTR
for name in PANELS:
    rk = f"{OUTR}/{name}/ranked_edges.csv"
    if os.path.exists(rk):
        continue
    t = time.time()
    RA.N_RAND = N_RAND
    n, _ = RA.analytic_for_geneset(name)
    print(f"  {name:9s} {n:5d} edges  ({time.time()-t:.0f}s)", flush=True)

# ------------------------------------------------------------------ 3. score
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] ct = pd.read_csv(f"{HERE}/resources/collectri_mouse.tsv", sep="\t")
ct = pd.read_csv(f"{RES_HERE}/resources/collectri_mouse.tsv", sep="\t")
ct = ct[ct.source_genesymbol != ct.target_genesymbol]
CE = set(zip(ct.source_genesymbol.map(san), ct.target_genesymbol.map(san)))
N_MC = 200


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
    idx = np.argsort(-s, kind="stable")
    pk = y[idx[:R]].sum() / R
    rng = np.random.default_rng(seed)
    aur = np.mean([average_precision_score(y, rng.permutation(s)) for _ in range(N_MC)])
    dens = R / len(univ)
    return dict(n_univ=len(univ), R=R, prec_at_R=round(pk, 3), epr=round(pk / dens, 2),
                auprc=round(au, 3), auprc_ratio=round(au / aur, 2))


rows = []
for name in PANELS:
    G, TF = panel_genes[name], set(panel_tfs[name])
    d = pd.read_csv(f"{OUTR}/{name}/ranked_edges.csv")
    d = d[d.gene_1 != d.gene_2]
    base = d.twinScore.to_numpy()
    zf = base + d.heterogeneity_penalty.to_numpy() + pz(d.z_het.abs())
    variants = {"twinScore": dict(zip(zip(d.gene_1, d.gene_2), base)),
                "twinScore+zhetfix": dict(zip(zip(d.gene_1, d.gene_2), zf))}
    gg = [(a, b) for a in G for b in G if a != b]
    tp = [(a, b) for a in TF for b in G if a != b]
    for vn, sm in variants.items():
        for un, U in (("yscher_genes_x_genes", gg), ("TFs_x_panel", tp)):
            r = score(U, {k: sm.get(k, min(sm.values())) for k in U} if False else sm,
                      # [2026-10-01 commented out: hash() of strings is randomised per process (PYTHONHASHSEED) so this seed was not reproducible; zlib.crc32 of the repr is stable] hash((name, vn, un)) % 2**31)
                      zlib.crc32(repr((name, vn, un)).encode()) % 2**31)
            if r:
                rows.append(dict(panel=name, variant=vn, universe=un, **r))

res = pd.DataFrame(rows)
res.to_csv(f"{S}/yscher_zhet_fixed_results.csv", index=False)
pd.set_option("display.width", 220)
print("\n=== per panel ===")
print(res.to_string(index=False))
print("\n=== mean by (universe, variant) ===")
print(res.groupby(["universe", "variant"])[["prec_at_R", "epr", "auprc", "auprc_ratio"]]
      .mean().round(3).to_string())
