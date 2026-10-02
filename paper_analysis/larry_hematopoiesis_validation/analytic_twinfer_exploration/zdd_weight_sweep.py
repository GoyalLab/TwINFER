# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Sweep the coefficient on the signed-cross-corr-difference term
  +C * s(rho_cross_xy - rho_cross_yx)
in the yscher-tuned analytic twinScore (default C=0.5), for our 9 LARRY panels
and yscher's 9 exact panels. Everything else in TS unchanged; gate unchanged.

All raw pieces (rho_t1, rho_t2, rho_delta_t1, drr, rxy, ryx, gamma, z_reg_gated)
are already in the cached tuned ranked_edges.csv -- no correlation recompute.
"""
import zlib  # [2026-10-01 added]
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json, os
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score

HERE = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation'
from twinfer.utils.paths import get_data_root as _res_gdr  # [2026-10-01 LARRY resources/ moved out of the code tree to analysis_data/paper_analysis/larry_hematopoiesis_validation/resources (user)]
RES_HERE = f"{_res_gdr()}/paper_analysis/larry_hematopoiesis_validation"
S = f'{TWINFER_PROJECT_ROOT}/clean_code/paper_analysis/larry_hematopoiesis_validation/analytic_twinfer_exploration'
COEFS = [0.5, 1.0, 2.0, 4.0]
GATE_Z = 1.645
N_MC = 200
san = lambda g: g.replace("-", "_")

GS_OURS = ["variability_high", "variability_mid", "variability_low",
           "detection_high", "detection_mid", "detection_low",
           "correlation_high", "correlation_mid", "correlation_low"]
YP = {"corrhigh": "yscher_corrhigh", "corrmid": "yscher_corrmid", "corrlow": "yscher_corrlow",
      "detlow": "yscher_detlow", "detmid": "yscher_detmid", "dethigh": "yscher_dethigh",
      "hvglow": "yscher_hvglow", "hvgmid": "yscher_hvgmid", "hvghigh": "yscher_hvghigh"}

# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] ct = pd.read_csv(f"{HERE}/resources/collectri_mouse.tsv", sep="\t")
ct = pd.read_csv(f"{RES_HERE}/resources/collectri_mouse.tsv", sep="\t")
ct = ct[ct.source_genesymbol != ct.target_genesymbol]
CE = set(zip(ct.source_genesymbol.map(san), ct.target_genesymbol.map(san)))
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] gene_sets = json.load(open(f"{HERE}/resources/gene_sets.json"))
gene_sets = json.load(open(f"{RES_HERE}/resources/gene_sets.json"))
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] detail = json.load(open(f"{HERE}/resources/gene_sets_detail.json"))
detail = json.load(open(f"{RES_HERE}/resources/gene_sets_detail.json"))
genes_all = set(open(f'{TWINFER_PROJECT_ROOT}/finalized_data/LARRY_data/filtered/genes.txt').read().split())


def s(v):
    v = np.asarray(v, float); f = np.isfinite(v); o = np.zeros(v.shape)
    if f.sum() > 1:
        o[f] = (v[f] - v[f].mean()) / max(v[f].std(ddof=1), 1e-12)
    o[~f] = (o[f].min() - 1.0) if f.any() else 0.0
    return o


def ts_for_coef(d, C):
    d = d.copy()
    d["gene_1"] = d.gene_1.map(san); d["gene_2"] = d.gene_2.map(san)
    genes = sorted(set(d.gene_1) | set(d.gene_2))
    reg_raw = d.assign(asym=d.rxy - d.ryx).groupby("gene_1")["asym"].mean()
    rv = reg_raw.to_numpy()
    REG = {g: (reg_raw.get(g, np.nan) - rv.mean()) / max(rv.std(ddof=1), 1e-12) for g in genes}
    E1 = s(np.abs(d.rho_t1))
    TS = (E1
          - s(np.abs(d.rho_t2 - d.rho_t1))
          - s(np.abs(d.rho_delta_t1))
          - s(np.abs(d.drr))
          - s(np.minimum(np.abs(d.rxy), np.abs(d.ryx)))
          - s(np.abs(d.gamma))
          + C * s(d.rxy - d.ryx)
          + np.array([REG.get(a, 0.0) - REG.get(b, 0.0) for a, b in zip(d.gene_1, d.gene_2)]))
    gate = np.abs(d.z_reg_gated.to_numpy()) > GATE_Z
    twin = np.where(gate, TS, -E1)
    return dict(zip(zip(d.gene_1, d.gene_2), twin))


def score(univ, smap, seed):
    y = np.array([1 if p in CE else 0 for p in univ], int)
    R = int(y.sum())
    if R == 0 or R == len(univ):
        return None
    fl = min(smap.values())
    sc = np.array([smap.get(p, fl) for p in univ], float)
    au = average_precision_score(y, sc)
    pk = y[np.argsort(-sc, kind="stable")[:R]].sum() / R
    rng = np.random.default_rng(seed)
    aur = np.mean([average_precision_score(y, rng.permutation(sc)) for _ in range(N_MC)])
    return dict(R=R, prec_at_R=round(pk, 3), epr=round(pk / (R / len(univ)), 2),
                auprc=round(au, 3), auprc_ratio=round(au / aur, 2))


rows = []
for gs in GS_OURS:
    # [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] d = pd.read_csv(f"{HERE}/resources/analytic_infer_tuned/{gs}/ranked_edges.csv")
    d = pd.read_csv(f"{RES_HERE}/resources/analytic_infer_tuned/{gs}/ranked_edges.csv")
    crit, lvl = gs.rsplit("_", 1)
    panel = [san(g) for g in gene_sets[gs]]
    tfs = set(san(tf) for tf, _ in detail[crit][lvl] if tf in gene_sets[gs])
    U = [(a, b) for a in tfs for b in panel if a != b]
    for C in COEFS:
        sm = ts_for_coef(d, C)
        # [2026-10-01 commented out: hash() of strings is randomised per process (PYTHONHASHSEED) so this seed was not reproducible; zlib.crc32 of the repr is stable] r = score(U, sm, hash((gs, C)) % 2**31)
        r = score(U, sm, zlib.crc32(repr((gs, C)).encode()) % 2**31)
        if r:
            rows.append(dict(dataset="ours", panel=gs, universe="TFs_x_panel", C=C, **r))

for name, jf in YP.items():
    d = pd.read_csv(f"{S}/yscher_analytic_tuned/{name}/ranked_edges.csv")
    pj = json.load(open(f"{S}/{jf}_panel.json"))
    G = [san(g) for g in pj["panel"] if g in genes_all]
    TF = set(san(g) for g in pj["tfs"] if san(g) in G)
    gg = [(a, b) for a in G for b in G if a != b]
    tp = [(a, b) for a in TF for b in G if a != b]
    for C in COEFS:
        sm = ts_for_coef(d, C)
        for un, U in (("yscher_gxg", gg), ("TFs_x_panel", tp)):
            # [2026-10-01 commented out: hash() of strings is randomised per process (PYTHONHASHSEED) so this seed was not reproducible; zlib.crc32 of the repr is stable] r = score(U, sm, hash((name, C, un)) % 2**31)
            r = score(U, sm, zlib.crc32(repr((name, C, un)).encode()) % 2**31)
            if r:
                rows.append(dict(dataset="yscher", panel=name, universe=un, C=C, **r))

res = pd.DataFrame(rows)
res.to_csv(f"{S}/zdd_weight_sweep_results.csv", index=False)
pd.set_option("display.width", 240)
print("\n=== mean by (dataset, universe, C) ===")
print(res.groupby(["dataset", "universe", "C"])[["prec_at_R", "epr", "auprc", "auprc_ratio"]]
      .mean().round(3).to_string())
print("\n=== per-panel auprc_ratio, C=0.5 vs C=2.0 ===")
for uni in res.universe.unique():
    p = res[res.universe == uni].pivot_table(index="panel", columns="C", values="auprc_ratio")
    print(f"\n-- {uni} --"); print(p.round(2).to_string())
