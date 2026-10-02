# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Signed z_het-fixed twinScore with a NEW edge-sign rule:

  sign(x->y) = sign( rho_cross(x->y) )      if |z_cross(x->y)| >= T   (directed cross-corr significant)
             = sign( rho_xy(t2) )           elif |z_coexp(x,y)| >= T  (co-expression significant)
             = sign( rho_xy(t2) )           else  (fallback)

vs the OLD rule (sign from rho_t2 -> rho_t1).  Signed AUPRC / random on the
real-network sims (orig 7 + multistate/seeded 8), both sign rules, T in {1.96, 2.576}.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob, sys
import numpy as np, pandas as pd

H = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/synthetic_network_analysis'
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, H)
from benchmarks.network_benchmarks.score import score_real_networks_analytic_zhet_fixed as Z

RN = Z.REAL_NET_ROOT
OLD_DIR = RN / "twinfer_inference_nofilter"
NEW_DIR = RN / "twinfer_inference_multistate_nofilter"
ORIG = Z.OLD_TOPO
NEW = {n: Z.NEW_BASE_TOPO[n.split("_")[0]] for n in Z.NEW_DATASETS}
S2PI, VH = Z.S2PI, Z.VH


def signed_variants(json_path, T):
    """returns mag, {rule: sign_dict}. Recompute like analytic_zhet_fixed but keep the
    per-pair cross / co-expression z so we can build the new sign rule."""
    import json
    d = json.load(open(json_path))
    C = d["correlations"]
    need = ["gene_t1", "gene_t2", "twin_delta_t1", "random_delta_t1", "random_delta_t2", "direction"]
    if any(C.get(k) is None for k in need):
        return None
    m = lambda k: pd.DataFrame(C[k]["data"], index=C[k]["index"], columns=C[k]["columns"])
    R1, R2, D1, RR1, RR2, XC = (m(k) for k in need)
    D2 = m("twin_delta_t2")
    genes = list(R1.columns)
    red = pd.DataFrame(d["ranked_edges"]["data"], columns=d["ranked_edges"]["columns"])
    sd1 = Z._calib(red.rho_t1.abs(), red.z_abs_rho_t1, 0.7)
    sd2 = Z._calib(red.rho_t2.abs(), red.z_abs_rho_t2, 0.7)
    sdc = Z._calib((red.rho_t2 - red.rho_t1).abs(), red.z_abs_rho_change, 0.7)
    ncross = [v["null_std"] for v in d["direction"]["rho_cross_null"].values()]
    sdx = float(np.median(ncross)) if ncross else np.nan
    rk = red.set_index(["gene_1", "gene_2"])
    sd_het = Z._calib_signed([D1.loc[a, b] - RR1.loc[a, b] for a, b in rk.index], red.z_het.values, 0.5)
    sd_div = Z._calib_signed([D1.loc[a, b] for a, b in rk.index], red.z_div.values, 0.3)
    do = np.array([D2.loc[a, b] - D1.loc[a, b] for a, b in rk.index])
    dc = np.array([RR2.loc[a, b] - RR1.loc[a, b] for a, b in rk.index])
    sd_d = Z._calib_signed(do - dc, red.z_d_het.values, 0.5)
    sd1 = sd1 if np.isfinite(sd1) else 0.026
    sd2 = sd2 if np.isfinite(sd2) else sd1
    sdc = sdc if np.isfinite(sdc) else sd1 * np.sqrt(2)
    sdx = sdx if np.isfinite(sdx) else 0.018
    sd_het = sd_het if np.isfinite(sd_het) else 0.030
    sd_div = sd_div if np.isfinite(sd_div) else sd_het
    sd_d = sd_d if np.isfinite(sd_d) else 0.030

    def pz(v):
        v = np.asarray(v, float); f = np.isfinite(v); o = np.full(v.shape, np.nan)
        if f.any():
            x = v[f]; s = x.std(ddof=0)
            o[f] = 0.0 if s == 0 else (x - x.mean()) / s
        return o

    A, Bx, rt1, rt2, rxy, zc, ze = [], [], [], [], [], [], []
    za1, za2, zac, zg, zh, zd, zdh = [], [], [], [], [], [], []
    for a in genes:
        for b in genes:
            if a == b:
                continue
            r1, r2 = R1.loc[a, b], R2.loc[a, b]
            cxy, cyx = XC.loc[a, b], XC.loc[b, a]
            A.append(a); Bx.append(b); rt1.append(r1); rt2.append(r2); rxy.append(cxy)
            zc.append(cxy / sdx); ze.append(r2 / sd2)
            za1.append((abs(r1) - sd1 * S2PI) / (sd1 * np.sqrt(VH)))
            za2.append((abs(r2) - sd2 * S2PI) / (sd2 * np.sqrt(VH)))
            zac.append((abs(r2 - r1) - sdc * S2PI) / (sdc * np.sqrt(VH)))
            zg.append((abs(cxy) - abs(cyx)) / (sdx * np.sqrt(2 * VH)))
            zh.append((D1.loc[a, b] - RR1.loc[a, b]) / sd_het)
            zd.append(D1.loc[a, b] / sd_div)
            zdh.append(((D2.loc[a, b] - D1.loc[a, b]) - (RR2.loc[a, b] - RR1.loc[a, b])) / sd_d)
    o = pd.DataFrame(dict(a=A, b=Bx, rt1=rt1, rt2=rt2, rxy=rxy, zc=zc, ze=ze,
                          za1=za1, za2=za2, zac=zac, zg=zg, zh=zh, zd=zd, zdh=zdh))
    s1, s2, scc, sgg = pz(o.za1), pz(o.za2), pz(o.zac), pz(o.zg)
    dp = np.where((o.zh < -2.326) & np.isfinite(o.zd), np.abs(o.zd), 0.0)
    hp = np.where((o.zdh > 0) & np.isfinite(o.zh), o.zh, 0.0)
    gb = 0.5 * np.sign((np.abs(o.rxy) - np.abs([XC.loc[b, a] for a, b in zip(o.a, o.b)])).to_numpy()) * (np.abs(sgg) >= 1)
    twin = s1 + s2 - scc + np.pi - dp - hp + gb + hp + pz(np.abs(o.zh))
    mag = {(a, b): abs(float(v)) for a, b, v in zip(o.a, o.b, twin) if np.isfinite(v)}

    old_sign, new_sign = {}, {}
    for a, b, r1, r2, cxy, zci, zei in zip(o.a, o.b, o.rt1, o.rt2, o.rxy, o.zc, o.ze):
        sv = r2 if np.isfinite(r2) else r1
        old_sign[(a, b)] = "+" if (np.isfinite(sv) and sv >= 0) else "-"
        if np.isfinite(zci) and abs(zci) >= T and np.isfinite(cxy):
            new_sign[(a, b)] = "+" if cxy >= 0 else "-"
        elif np.isfinite(zei) and abs(zei) >= T:
            new_sign[(a, b)] = "+" if r2 >= 0 else "-"
        else:
            new_sign[(a, b)] = old_sign[(a, b)]
    return mag, {"old": old_sign, "new": new_sign}


def signed_ratio(mag, sign, tsign, poss):
    gated = Z._sign_gate(mag, sign, tsign, poss)
    if not gated:
        return np.nan
    au = Z.auprc_from_scored_pairs(gated, set(tsign), poss)
    y = np.array([1 if p in tsign else 0 for p in poss])
    fl = min(gated.values()) - 1
    x = np.array([gated.get(p, fl) for p in poss])
    rng = np.random.default_rng(0)
    from sklearn.metrics import average_precision_score
    aur = np.mean([average_precision_score(y, rng.permutation(x)) for _ in range(150)])
    return au / aur if aur > 0 else np.nan


rows = []
for T in (1.96, 2.576):
    for grp, items, sd in [("orig7", ORIG.items(), OLD_DIR), ("new8", NEW.items(), NEW_DIR)]:
        for name, topo in items:
            true, poss, tsign = Z.true_edges_from_topo(Z.TOPO_ROOT / topo)
            r = {"old": [], "new": []}
            for jf in sorted(glob.glob(str(sd / f"{name}_rep_*_all_results.json"))):
                res = signed_variants(jf, T)
                if res is None:
                    continue
                mag, signs = res
                for rule in ("old", "new"):
                    r[rule].append(signed_ratio(mag, signs[rule], tsign, poss))
            rows.append(dict(T=T, group=grp, dataset=name,
                             old=np.nanmean(r["old"]) if r["old"] else np.nan,
                             new=np.nanmean(r["new"]) if r["new"] else np.nan))

df = pd.DataFrame(rows)
df.to_csv(f"{RN}/signed_rule_test.csv", index=False)
pd.set_option("display.width", 200)
print(df.round(2).to_string(index=False))
print("\n=== mean signed AUPRC/random by (T, group) ===")
print(df.groupby(["T", "group"])[["old", "new"]].mean().round(3).to_string())
