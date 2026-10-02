# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Analytic z vs the saved permutation null for the LARRY superseded (old) gene sets.
For each pair/step: reconstruct z_emp = (rho_obs - mean(null_draws)) / std(null_draws) from the
raw_nulls.npz + corr_*.csv, then compare to z_analytic = (rho_obs - center)/(1/sqrt(m_eff-1)).
Also checks whether the null is actually Gaussian (skew / tail quantiles)."""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob, json, os, sys
import numpy as np, pandas as pd
from scipy import stats

BK = f'{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/larry_hematopoiesis_validation/analytic_twinfer_exploration/infer_results_bak_20260908_1034'
# [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] INP = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation/resources/twinfer_input/correlation_high.csv'
INP = f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/larry_hematopoiesis_validation/resources/twinfer_input/correlation_high.csv'

# ---- LARRY clone structure -> effective m per step (clone = ~1 independent unit) ----
cl = pd.read_csv(INP, usecols=["clone_id", "cell_id", "time_step"])
def struct(t):
    s = cl[cl.time_step == t]; g = s.groupby("clone_id").size(); mult = g[g >= 2]
    return dict(n_clone=s.clone_id.nunique(),
                kish_step1=s.clone_id.nunique() ** 2 / (1 / g).sum(),
                n_twin_clone=len(mult))
S2, S4 = struct(2), struct(4)
d2 = set(cl[cl.time_step == 2].clone_id); d4 = set(cl[cl.time_step == 4].clone_id)
N_BOTH = len(d2 & d4)
print(f"day2: {S2}   day4: {S4}   both-day clones: {N_BOTH}")

# step -> (corr file, m candidates dict, center)
STEPMAP = {
    "step1":            ("corr_gene_t1.csv",        {"n_clone": S2["n_clone"], "kish": S2["kish_step1"]}, "zero"),
    "rho_delta_div_t1": ("corr_twin_delta_t1.csv",  {"n_twin_clone": S2["n_twin_clone"]},                "zero"),
    "rho_delta_div_t2": ("corr_twin_delta_t2.csv",  {"n_twin_clone": S4["n_twin_clone"]},                "zero"),
    "rho_delta_het_t1": ("corr_twin_delta_t1.csv",  {"n_twin_clone": S2["n_twin_clone"]},                "empmean"),
}

def load_corr(d, fn):
    m = pd.read_csv(os.path.join(d, fn), index_col=0)
    return m

allrows = []
GS = [d for d in sorted(glob.glob(f"{BK}/*_t2_t4_allpairs")) if os.path.exists(os.path.join(d,"raw_nulls.npz"))]
for gsdir in GS:
    gs = os.path.basename(gsdir).replace("_t2_t4_allpairs", "")
    npz = np.load(os.path.join(gsdir, "raw_nulls.npz"))
    corrcache = {}
    for step, (cfn, mcand, center) in STEPMAP.items():
        if cfn not in corrcache:
            corrcache[cfn] = load_corr(gsdir, cfn)
        C = corrcache[cfn]
        keys = [k for k in npz.files if k.startswith(step + "__")]
        for k in keys:
            _, a, b = k.split("__")
            if a not in C.index or b not in C.columns:
                continue
            draws = npz[k].astype(float)
            draws = draws[np.isfinite(draws)]
            if len(draws) < 50:
                continue
            rho = float(C.loc[a, b])
            mu, sd = draws.mean(), draws.std(ddof=1)
            if sd <= 0:
                continue
            z_emp = (rho - mu) / sd
            m_impl = 1 + 1 / sd ** 2
            cen = 0.0 if center == "zero" else mu
            row = dict(gs=gs, step=step, a=a, b=b, rho=rho, mu=mu, sd=sd,
                       z_emp=z_emp, m_impl=m_impl,
                       nskew=float(stats.skew(draws)), nkurt=float(stats.kurtosis(draws)),
                       q99_emp=float(np.quantile(draws, 0.99)),
                       q99_gauss=float(mu + 2.326 * sd),
                       q01_emp=float(np.quantile(draws, 0.01)),
                       q01_gauss=float(mu - 2.326 * sd))
            for mname, mval in mcand.items():
                row[f"z_an_{mname}"] = (rho - cen) / (1.0 / np.sqrt(mval - 1))
            allrows.append(row)

df = pd.DataFrame(allrows).replace([np.inf, -np.inf], np.nan)
OUTCSV = f'{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/larry_hematopoiesis_validation/analytic_twinfer_exploration/analytic_z_larry_rows.csv'
df.to_csv(OUTCSV, index=False)
print(f"\n{len(df)} pair-step rows across {df.gs.nunique()} gene sets -> {OUTCSV}")

pd.set_option("display.width", 220)
for step, g0 in df.groupby("step"):
    g = g0.dropna(subset=["z_emp", "rho", "sd", "m_impl"])
    g = g[np.isfinite(g.z_emp) & (g.z_emp.abs() < 1e4)]
    mcols = [c for c in g.columns if c.startswith("z_an_")]
    print(f"\n=== {step}   n={len(g)} ===")
    print(f"  m_implied  median {g.m_impl.median():.0f}  iqr [{g.m_impl.quantile(.25):.0f}, {g.m_impl.quantile(.75):.0f}]  cv {100*g.m_impl.std()/g.m_impl.mean():.1f}%")
    print(f"  null shape  |skew| med {g.nskew.abs().median():.3f}   excess kurt med {g.nkurt.median():+.3f}   frac |skew|>0.2 = {100*(g.nskew.abs()>0.2).mean():.0f}%")
    r99 = (g.q99_emp / g.q99_gauss).replace([np.inf, -np.inf], np.nan).dropna()
    if len(r99):
        print(f"  99th pctile empirical/gaussian: median {r99.median():.3f}   [5%,95%]=[{r99.quantile(.05):.2f}, {r99.quantile(.95):.2f}]")
    for mc in mcols:
        gg = g.dropna(subset=[mc])
        gg = gg[np.isfinite(gg[mc])]
        if len(gg) < 5:
            continue
        err = gg[mc] - gg.z_emp
        cc = np.corrcoef(gg[mc], gg.z_emp)[0, 1]
        A = np.vstack([gg[mc].to_numpy(), np.ones(len(gg))]).T
        sl = np.linalg.lstsq(A, gg.z_emp.to_numpy(), rcond=None)[0][0]
        print(f"    {mc:16s} median|z_an-z_emp| {err.abs().median():.3f}   RMS {np.sqrt((err**2).mean()):.3f}   "
              f"max {err.abs().max():.2f}   corr {cc:.4f}   slope(z_emp~z_an) {sl:.3f}")
