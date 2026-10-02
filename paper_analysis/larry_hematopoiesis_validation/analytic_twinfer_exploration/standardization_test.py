# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Does the per-term panel standardization s() help?  z_het-fixed twinScore
computed 4 ways, scored AUPRC/random on the real-net sims (orig 7 + multistate 8):

  std   : s(z|rt1|) + s(z|rt2|) - s(z|drho|) + pi - div_pen + s(|z_het|) + gb   (current)
  raw   : z|rt1|   + z|rt2|   - z|drho|   + pi - div_pen +   |z_het| + gb       (no per-term s)
  std_noconst : std  without pi / div_pen / gb  (only the standardized z terms)
  raw_noconst : raw  without pi / div_pen / gb
"""
import zlib  # [2026-10-01 added]
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob, json, sys
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, auc, precision_recall_curve

H = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/synthetic_network_analysis'
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, H)
from benchmarks.network_benchmarks.score import score_real_networks_analytic_zhet_fixed as Z

RN = Z.REAL_NET_ROOT
GRP = [("orig7", Z.OLD_TOPO.items(), RN / "twinfer_inference_nofilter"),
       ("new8", [(n, Z.NEW_BASE_TOPO[n.split("_")[0]]) for n in Z.NEW_DATASETS],
        RN / "twinfer_inference_multistate_nofilter")]
S2PI, VH = Z.S2PI, Z.VH


def variants(jf):
    d = json.load(open(jf)); C = d["correlations"]
    need = ["gene_t1", "gene_t2", "twin_delta_t1", "random_delta_t1", "random_delta_t2", "direction"]
    if any(C.get(k) is None for k in need):
        return None
    m = lambda k: pd.DataFrame(C[k]["data"], index=C[k]["index"], columns=C[k]["columns"])
    R1, R2, D1, RR1, RR2, XC = (m(k) for k in need); D2 = m("twin_delta_t2")
    genes = list(R1.columns)
    red = pd.DataFrame(d["ranked_edges"]["data"], columns=d["ranked_edges"]["columns"])
    fin = lambda x, dflt: x if (x is not None and np.isfinite(x)) else dflt
    sd1 = fin(Z._calib(red.rho_t1.abs(), red.z_abs_rho_t1, 0.7), 0.026)
    sd2 = fin(Z._calib(red.rho_t2.abs(), red.z_abs_rho_t2, 0.7), sd1)
    sdc = fin(Z._calib((red.rho_t2 - red.rho_t1).abs(), red.z_abs_rho_change, 0.7), sd1 * np.sqrt(2))
    nc = [v["null_std"] for v in d["direction"]["rho_cross_null"].values()]
    sdx = float(np.median(nc)) if nc else 0.018
    rk = red.set_index(["gene_1", "gene_2"])
    sd_het = fin(Z._calib_signed([D1.loc[a, b] - RR1.loc[a, b] for a, b in rk.index], red.z_het.values, 0.5), 0.030)
    sd_div = fin(Z._calib_signed([D1.loc[a, b] for a, b in rk.index], red.z_div.values, 0.3), sd_het)
    do = np.array([D2.loc[a, b] - D1.loc[a, b] for a, b in rk.index])
    dc = np.array([RR2.loc[a, b] - RR1.loc[a, b] for a, b in rk.index])
    sd_d = fin(Z._calib_signed(do - dc, red.z_d_het.values, 0.5), 0.030)

    def pz(v):
        v = np.asarray(v, float); f = np.isfinite(v); o = np.full(v.shape, np.nan)
        if f.any():
            x = v[f]; s = x.std(ddof=0); o[f] = 0.0 if s == 0 else (x - x.mean()) / s
        return o

    A, B, za1, za2, zac, zh, zd, zdh, gam = [], [], [], [], [], [], [], [], []
    for a in genes:
        for b in genes:
            if a == b:
                continue
            r1, r2 = R1.loc[a, b], R2.loc[a, b]
            cxy, cyx = XC.loc[a, b], XC.loc[b, a]
            A.append(a); B.append(b)
            za1.append((abs(r1) - sd1 * S2PI) / (sd1 * np.sqrt(VH)))
            za2.append((abs(r2) - sd2 * S2PI) / (sd2 * np.sqrt(VH)))
            zac.append((abs(r2 - r1) - sdc * S2PI) / (sdc * np.sqrt(VH)))
            zh.append((D1.loc[a, b] - RR1.loc[a, b]) / sd_het)
            zd.append(D1.loc[a, b] / sd_div)
            zdh.append(((D2.loc[a, b] - D1.loc[a, b]) - (RR2.loc[a, b] - RR1.loc[a, b])) / sd_d)
            gam.append(abs(cxy) - abs(cyx))
    za1, za2, zac, zh, zd, zdh, gam = map(np.array, (za1, za2, zac, zh, zd, zdh, gam))
    dp = np.where((zh < -2.326) & np.isfinite(zd), np.abs(zd), 0.0)
    gb = 0.5 * np.sign(gam) * (np.abs(pz(gam)) >= 1)   # gamma bonus (gamma z ~ pz of gamma)
    z0 = lambda v: np.where(np.isfinite(v), v, 0.0)   # non-finite -> 0 so a term drop-out doesn't kill the row
    out = {}
    out["std"] = pz(za1) + pz(za2) - pz(zac) + np.pi - dp + pz(np.abs(zh)) + gb
    out["raw"] = z0(za1) + z0(za2) - z0(zac) + np.pi - dp + z0(np.abs(zh)) + gb
    out["std_noconst"] = pz(za1) + pz(za2) - pz(zac) + pz(np.abs(zh))
    out["raw_noconst"] = z0(za1) + z0(za2) - z0(zac) + z0(np.abs(zh))
    return {k: {(a, b): abs(float(v)) for a, b, v in zip(A, B, arr) if np.isfinite(v)}
            for k, arr in out.items()}


def ratio(sc, true, poss, seed):
    if not sc:
        return np.nan
    y = np.array([1 if p in true else 0 for p in poss])
    fl = min(sc.values()) - 1
    x = np.array([sc.get(p, fl) for p in poss])
    prec, rec, _ = precision_recall_curve(y, x); a = auc(rec, prec)
    rng = np.random.default_rng(seed)
    ar = np.mean([average_precision_score(y, rng.permutation(x)) for _ in range(150)])
    return a / ar


rows = []
for grp, items, sd in GRP:
    for name, topo in items:
        true, poss, _ = Z.true_edges_from_topo(Z.TOPO_ROOT / topo)
        acc = {k: [] for k in ["std", "raw", "std_noconst", "raw_noconst"]}
        for jf in sorted(glob.glob(str(sd / f"{name}_rep_*_all_results.json"))):
            v = variants(jf)
            if v is None:
                continue
            for k in acc:
                # [2026-10-01 commented out: hash() of strings is randomised per process (PYTHONHASHSEED) so this seed was not reproducible; zlib.crc32 of the repr is stable] acc[k].append(ratio(v[k], true, poss, hash(jf) % 2**31))
                acc[k].append(ratio(v[k], true, poss, zlib.crc32(repr(jf).encode()) % 2**31))
        rows.append(dict(group=grp, dataset=name, **{k: np.mean(x) if x else np.nan for k, x in acc.items()}))

df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(df.round(2).to_string(index=False))
print("\n=== mean AUPRC/random by group ===")
print(df.groupby("group")[["std", "raw", "std_noconst", "raw_noconst"]].mean().round(3).to_string())
