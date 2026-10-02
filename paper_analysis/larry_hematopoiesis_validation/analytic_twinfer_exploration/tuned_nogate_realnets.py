# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Tuned analytic twinScore WITHOUT the z_reg_gated gate -- the raw composition
    TS = E1 - s|drho| - s|rho_d1| - s|drr| - s(min|rho_x|) - s|gamma| + 0.5 s(rho_xy-rho_yx) + dREG
scored for every directed pair, on the NO-FILTER correlation matrices (full coverage),
for the original 7 real networks + the 8 multistate/seeded datasets.
"""
import zlib  # [2026-10-01 added]
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob, json
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, auc, precision_recall_curve

RN = f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/real_networks'
TOPO = f'{TWINFER_PROJECT_ROOT}/input_data/real_world_networks'
S2PI, VH = np.sqrt(2 / np.pi), 1 - 2 / np.pi

ORIG = {"GSD": "GSD.txt", "HSC": "HSC.txt", "VSC": "VSC.txt", "mCAD": "mCAD.txt",
        "EMT": "EMT.txt", "Pluripotent": "Pluripotent.txt", "B_cell_activation": "B_cell.txt"}
NEW = ["GSD_multistate", "GSD_seeded", "HSC_multistate", "HSC_seeded",
       "EMT_multistate", "EMT_seeded", "VSC_seeded", "mCAD_seeded"]
BASE = {"GSD": "GSD.txt", "HSC": "HSC.txt", "EMT": "EMT.txt", "VSC": "VSC.txt", "mCAD": "mCAD.txt"}


def s(v):
    v = np.asarray(v, float); f = np.isfinite(v); o = np.zeros(v.shape)
    if f.sum() > 1:
        o[f] = (v[f] - v[f].mean()) / max(v[f].std(ddof=1), 1e-12)
    o[~f] = (o[f].min() - 1.0) if f.any() else 0.0
    return o


def true_poss(topo):
    M = np.loadtxt(f"{TOPO}/{topo}", delimiter=",", dtype=int); n = M.shape[0]
    g = [f"gene_{i+1}" for i in range(n)]
    true = {(g[i], g[j]) for i in range(n) for j in range(n) if i != j and M[i, j] != 0}
    poss = [(g[i], g[j]) for i in range(n) for j in range(n) if i != j]
    return true, poss


def tuned_nogate(jf):
    d = json.load(open(jf)); C = d["correlations"]
    need = ["gene_t1", "gene_t2", "twin_delta_t1", "random_delta_t1", "random_delta_t2", "direction"]
    if any(C.get(k) is None for k in need):
        return None
    m = lambda k: pd.DataFrame(C[k]["data"], index=C[k]["index"], columns=C[k]["columns"])
    R1, R2, D1, RR1, RR2, XC = (m(k) for k in need)
    genes = list(R1.columns)
    reg_raw = np.array([np.nanmean((XC.loc[g] - XC[g]).drop(g).to_numpy()) for g in genes])
    REG = dict(zip(genes, (reg_raw - reg_raw.mean()) / max(reg_raw.std(ddof=1), 1e-12)))
    ai, bi, rt1, rt2, rd1, drr, rxy, ryx = [], [], [], [], [], [], [], []
    for a in genes:
        for b in genes:
            if a == b:
                continue
            ai.append(a); bi.append(b)
            rt1.append(R1.loc[a, b]); rt2.append(R2.loc[a, b]); rd1.append(D1.loc[a, b])
            drr.append(RR2.loc[a, b] - RR1.loc[a, b])
            rxy.append(XC.loc[a, b]); ryx.append(XC.loc[b, a])
    rt1, rt2, rd1, drr, rxy, ryx = map(np.array, (rt1, rt2, rd1, drr, rxy, ryx))
    TS = (s(np.abs(rt1))
          - s(np.abs(rt2) - np.abs(rt1))
          - s(np.abs(rd1))
          - s(np.abs(drr))
          - s(np.minimum(np.abs(rxy), np.abs(ryx)))
          - s(np.abs(rxy) - np.abs(ryx))
          + 0.5 * s(rxy - ryx)
          + np.array([REG[a] - REG[b] for a, b in zip(ai, bi)]))
    return {(a, b): abs(float(v)) for a, b, v in zip(ai, bi, TS) if np.isfinite(v)}


def scoreit(sc, true, poss, seed):
    y = np.array([1 if p in true else 0 for p in poss])
    fl = min(sc.values()) - 1
    x = np.array([sc.get(p, fl) for p in poss])
    prec, rec, _ = precision_recall_curve(y, x); au = auc(rec, prec)
    rng = np.random.default_rng(seed)
    aur = np.mean([average_precision_score(y, rng.permutation(x)) for _ in range(150)])
    k = int(y.sum()); top = np.argsort(-x, kind="stable")[:k]
    f1 = y[top].sum() / k
    return au, au / aur, f1


rows = []
for grp, items, srcdir in [("orig7", ORIG.items(), "twinfer_inference_nofilter"),
                           ("new8", [(n, BASE[n.split("_")[0]]) for n in NEW],
                            "twinfer_inference_multistate_nofilter")]:
    for name, topo in items:
        true, poss = true_poss(topo)
        jfs = sorted(glob.glob(f"{RN}/{srcdir}/{name}_rep_*_all_results.json"))
        rr = []
        for jf in jfs:
            sc = tuned_nogate(jf)
            if not sc:
                continue
            # [2026-10-01 commented out: hash() of strings is randomised per process (PYTHONHASHSEED) so this seed was not reproducible; zlib.crc32 of the repr is stable] rr.append(scoreit(sc, true, poss, hash(jf) % 2**31))
            rr.append(scoreit(sc, true, poss, zlib.crc32(repr(jf).encode()) % 2**31))
        if rr:
            au, ar, f1 = np.mean(rr, axis=0)
            rows.append(dict(group=grp, dataset=name, n=len(rr),
                             auprc=round(au, 3), ratio=round(ar, 2), f1=round(f1, 3)))

df = pd.DataFrame(rows)
print(df.to_string(index=False))
print("\n=== mean by group (tuned, NO gate, no-filter) ===")
print(df.groupby("group")[["auprc", "ratio", "f1"]].mean().round(3).to_string())
