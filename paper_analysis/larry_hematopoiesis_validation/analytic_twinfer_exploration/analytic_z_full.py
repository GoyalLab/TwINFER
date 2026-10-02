# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Analytic approximation of EVERY z-score infer_with_twinfer returns, vs the 5000-shuffle
permutation z, for the NEW LARRY gene sets. Runs on whatever infer_results dirs are complete.

Analytic model per statistic:
  signed, 0-centred null (step1, z_div, z_d_div, z_rho_cross):  z = rho_obs / (1/sqrt(m_eff-1))
  signed, structure-offset null (z_het, z_d_het):               z = (rho_obs - rho_random) / (1/sqrt(m_eff-1))
  |.|-statistics (z_abs_rho_t1/t2/change):  null draw ~ N(0, s), s=1/sqrt(m_eff-1);
      |X| has mean s*sqrt(2/pi), sd s*sqrt(1-2/pi)  ->  z = (|rho| - s*sqrt(2/pi)) / (s*sqrt(1-2/pi))
  z_gamma = |rho_xy| - |rho_yx|:  null = |Nxy| - |Nyx| on the paired direction draws;
      analytic sd ~= s*sqrt(2*(1-2/pi))  (treats the two direction nulls as ~independent)
m_eff = Kish effective sample size under clone weighting, from the clone-size distribution.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob, json, os, sys
import numpy as np, pandas as pd
from scipy import stats

# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] RES = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation/resources'
RES = f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/larry_hematopoiesis_validation/resources'
INP = f"{RES}/twinfer_input"
S2PI, V = np.sqrt(2 / np.pi), (1 - 2 / np.pi)

# ---- clone structure -> Kish m_eff (gene-set independent: use any input CSV) ----
cl = pd.read_csv(f"{INP}/detection_high.csv", usecols=["clone_id", "time_step"])
def kish_cell(t):
    g = cl[cl.time_step == t].groupby("clone_id").size()
    return g.nunique() ** 2 / (1 / g).sum() if False else cl[cl.time_step == t].clone_id.nunique() ** 2 / (1 / g).sum()
def kish_twin(t):
    g = cl[cl.time_step == t].groupby("clone_id").size(); g = g[g >= 2]; c = g * (g - 1) / 2
    return len(g) ** 2 / (1 / c).sum()
d2, d4 = set(cl[cl.time_step == 2].clone_id), set(cl[cl.time_step == 4].clone_id)
g2 = cl[cl.time_step == 2].groupby("clone_id").size(); g4 = cl[cl.time_step == 4].groupby("clone_id").size()
both = d2 & d4
# cross-time: clone weight 1 split over n2*n4 pairs -> Kish = nclones^2 / sum(1/(n2*n4))
mcross = len(both) ** 2 / sum(1 / (g2[c] * g4[c]) for c in both)
M = dict(step1_t1=kish_cell(2), step1_t2=kish_cell(4),
         twin_t1=kish_twin(2), twin_t2=kish_twin(4),
         d=1 / (1 / kish_twin(2) + 1 / kish_twin(4)), cross=mcross)
print("Kish m_eff:", {k: round(v) for k, v in M.items()})


def sd_an(m):
    return 1.0 / np.sqrt(m - 1.0)


def metrics(z_an, z_emp):
    z_an, z_emp = np.asarray(z_an, float), np.asarray(z_emp, float)
    ok = np.isfinite(z_an) & np.isfinite(z_emp) & (np.abs(z_emp) < 1e4)
    z_an, z_emp = z_an[ok], z_emp[ok]
    if len(z_an) < 5:
        return None
    e = z_an - z_emp
    A = np.vstack([z_an, np.ones(len(z_an))]).T
    slope = np.linalg.lstsq(A, z_emp, rcond=None)[0][0]
    return dict(n=len(z_an), corr=np.corrcoef(z_an, z_emp)[0, 1],
               med=np.median(np.abs(e)), rms=np.sqrt(np.mean(e ** 2)),
               mx=np.abs(e).max(), slope=slope)


def run_geneset(gsdir):
    npz = np.load(f"{gsdir}/raw_nulls.npz")
    dirnpz_p = f"{gsdir}/direction_rho_cross_null.npz"
    dirnpz = np.load(dirnpz_p) if os.path.exists(dirnpz_p) else None
    C = {f: pd.read_csv(f"{gsdir}/corr_{f}.csv", index_col=0)
         for f in ("gene_t1", "gene_t2", "twin_delta_t1", "twin_delta_t2",
                   "random_delta_t1", "random_delta_t2", "rho_cross")}
    J = json.load(open(f"{gsdir}/z_scores_by_step.json"))
    tsi = pd.read_csv(f"{gsdir}/twin_score_inputs.csv")
    out = []

    def emp(nulltype, a, b):
        k = f"{nulltype}__{a}__{b}"
        if k not in npz.files:
            return None
        d = npz[k].astype(float); d = d[np.isfinite(d)]
        return d if len(d) >= 50 else None

    for k in [x for x in npz.files if x.startswith("step1__")]:
        _, a, b = k.split("__"); d = emp("step1", a, b)
        if d is None or a not in C["gene_t1"].index:
            continue
        rho = C["gene_t1"].loc[a, b]
        out.append(("step1", a, b, (rho - d.mean()) / d.std(ddof=1),
                    rho / sd_an(M["step1_t1"]), float(stats.skew(d))))
    for nt, cf, mk, cen in [("rho_delta_div_t1", "twin_delta_t1", "twin_t1", None),
                            ("rho_delta_div_t2", "twin_delta_t2", "twin_t2", None),
                            ("rho_delta_het_t1", "twin_delta_t1", "twin_t1", "random_delta_t1")]:
        for k in [x for x in npz.files if x.startswith(nt + "__")]:
            _, a, b = k.split("__"); d = emp(nt, a, b)
            if d is None or a not in C[cf].index:
                continue
            rho = C[cf].loc[a, b]
            c = 0.0 if cen is None else C[cen].loc[a, b]
            lbl = "z_div_t1" if nt.endswith("div_t1") else "z_div_t2" if nt.endswith("div_t2") else "z_het_t1"
            out.append((lbl, a, b, (rho - d.mean()) / d.std(ddof=1),
                        (rho - c) / sd_an(M[mk]), float(stats.skew(d))))
    # step3 d_het / d_div
    stage3 = J.get("stage3", {})
    for nt, cen_sign in [("d_het", 1), ("d_div", 0)]:
        for k in [x for x in npz.files if x.startswith(nt + "__")]:
            _, a, b = k.split("__"); d = emp(nt, a, b)
            if d is None:
                continue
            key = f"{a}__{b}"
            s3 = stage3.get(key) or stage3.get(f"{b}__{a}")
            if not s3:
                continue
            dobs = s3["d"]
            c = 0.0 if cen_sign == 0 else (C["random_delta_t2"].loc[a, b] - C["random_delta_t1"].loc[a, b])
            out.append((nt, a, b, (dobs - d.mean()) / d.std(ddof=1),
                        (dobs - c) / sd_an(M["d"]), float(stats.skew(d))))
    # step4 direction z_rho_cross
    if dirnpz is not None:
        dz = J.get("direction_z_scores", {})
        for k in dirnpz.files:
            a, b = k.split("__")
            if a not in C["rho_cross"].index:
                continue
            d = dirnpz[k].astype(float); d = d[np.isfinite(d)]
            if len(d) < 50:
                continue
            rho = C["rho_cross"].loc[a, b]
            out.append(("z_rho_cross", a, b, (rho - d.mean()) / d.std(ddof=1),
                        rho / sd_an(M["cross"]), float(stats.skew(d))))
        # z_gamma from paired direction draws
        for _, r in tsi.iterrows():
            a, b = r["gene_1"], r["gene_2"]
            kxy, kyx = f"{a}__{b}", f"{b}__{a}"
            if kxy not in dirnpz.files or kyx not in dirnpz.files:
                continue
            nxy, nyx = dirnpz[kxy].astype(float), dirnpz[kyx].astype(float)
            n = min(len(nxy), len(nyx))
            gn = np.abs(nxy[:n]) - np.abs(nyx[:n])
            g_emp = (r["gamma"] - gn.mean()) / gn.std(ddof=1)
            g_an = (r["gamma"] - 0.0) / (sd_an(M["cross"]) * np.sqrt(2 * V))
            out.append(("z_gamma", a, b, g_emp, g_an, float(stats.skew(gn))))
    # |.| magnitude z-scores from twin_score_inputs + rho_*_for_twinscore nulls
    for col, nt, mk in [("rho_t1", "rho_t1_for_twinscore", "cross"),
                        ("rho_t2", "rho_t2_for_twinscore", "cross"),
                        ("rho_change", "rho_change_for_twinscore", "cross")]:
        for _, r in tsi.iterrows():
            a, b = r["gene_1"], r["gene_2"]
            d = emp(nt, a, b) or emp(nt, b, a)
            if d is None or not np.isfinite(r[col]):
                continue
            z_emp = (abs(r[col]) - np.abs(d).mean()) / np.abs(d).std(ddof=1)
            s = d.std(ddof=1)  # use the null's own SD for a fair |.| moment mapping
            z_an = (abs(r[col]) - s * S2PI) / (s * np.sqrt(V))
            out.append((f"z_abs_{col}", a, b, z_emp, z_an, float(stats.skew(d))))
    return out


DIRS = sorted(glob.glob(f"{RES}/infer_results/*_t2_t4_allpairs"))
DIRS = [d for d in DIRS if os.path.exists(f"{d}/raw_nulls.npz")
        and os.path.getmtime(f"{d}/z_scores_by_step.json") > 1757300000]  # new run only (Sep 8+)
print("gene sets available (new run):", [os.path.basename(d).replace("_t2_t4_allpairs", "") for d in DIRS])

rows = []
for d in DIRS:
    gs = os.path.basename(d).replace("_t2_t4_allpairs", "")
    for lbl, a, b, ze, za, sk in run_geneset(d):
        rows.append(dict(gs=gs, stat=lbl, a=a, b=b, z_emp=ze, z_an=za, nskew=sk))
df = pd.DataFrame(rows)
# [2026-10-01 commented out: ephemeral Claude scratchpad; the data was copied to clean_data/_rescued_scratchpad] df.to_csv("/gpfs/home/gzu5140/.claude-tmp/claude-2000104/-gpfs-projects-b1255-hzhang-TwINFER-KA/8c755a86-0074-4569-970e-28157d9bb13a/scratchpad/analytic_z_full_rows.csv", index=False)
df.to_csv(f"{TWINFER_PROJECT_ROOT}/clean_data/_rescued_scratchpad/8c755a86/analytic_z_full_rows.csv", index=False)
print(f"\n{len(df)} rows over {df.gs.nunique()} gene set(s)\n")

order = ["step1", "z_div_t1", "z_div_t2", "z_het_t1", "d_div", "d_het",
         "z_rho_cross", "z_gamma", "z_abs_rho_t1", "z_abs_rho_t2", "z_abs_rho_change"]
print(f"{'statistic':16s} {'n':>5s} {'corr':>7s} {'med|dz|':>8s} {'RMS':>6s} {'max|dz|':>8s} {'slope':>6s} {'|skew|med':>9s}")
for st in order:
    g = df[df.stat == st]
    if not len(g):
        continue
    m = metrics(g.z_an, g.z_emp)
    if m is None:
        continue
    print(f"{st:16s} {m['n']:5d} {m['corr']:7.4f} {m['med']:8.3f} {m['rms']:6.3f} {m['mx']:8.2f} {m['slope']:6.3f} {g.nskew.abs().median():9.3f}")
