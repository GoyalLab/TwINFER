# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Test the analytic z approximation  z ~= rho_obs * sqrt(m_eff - 1)  (null mean 0, null
SD = 1/sqrt(m_eff-1))  against the 5000-shuffle z-scores already saved for the drift/6-scenario
sims.  m_eff comes from the KNOWN clone structure of infer.py's partition, not from the shuffles."""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob, json, os, sys
import numpy as np, pandas as pd

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/home/gzu5140/TwINFER_KA/code/TwINFER/package")
from twinfer.inference.correlation_functions import split_and_merge_simulations

ROOT = f'{TWINFER_PROJECT_ROOT}/analysis_data/drift_inference/eight_scenario'
YS = "/projects/b1255/yscher/Transcriptomic Distance/simulation_data/figure_2_simulations_1000"
LM = f'{TWINFER_PROJECT_ROOT}/simulation_data/drift_lowmid'
SEED = 101010


def partition_counts(sim):
    """Replicate run_6scenario_zscores._partition and report the unit counts each step sees."""
    rng = np.random.default_rng(SEED)
    cl = sim["clone_id"].drop_duplicates().to_numpy()
    sh = rng.permutation(cl)
    n = len(sh) // 4
    t1c, t2c, ac = sh[:n], sh[n:2 * n], sh[2 * n:]
    # within-time twin pairs (clone size 2 at that timepoint)
    def npairs(clones, t):
        s = sim[sim.clone_id.isin(clones) & (sim.time_step == t)]
        g = s.groupby("clone_id").size()
        return int((g >= 2).sum())
    # step1 rho_t1 pool = t1 twins (size 2) + ac_left (size 1)  -> Kish m_eff, clone-weighted
    n_tw_t1 = npairs(t1c, 1)  # timepoint value doesn't matter for counts (structure identical)
    n_ac = len(ac)
    # size-2 clones weight 1 (0.5+0.5), size-1 clones weight 1
    sumw = n_tw_t1 + n_ac
    sumw2 = n_tw_t1 * 2 * 0.25 + n_ac * 1.0
    m_step1 = sumw ** 2 / sumw2
    return dict(n_twin=n_tw_t1, n_ac=n_ac, m_step1=m_step1)


# one representative sim per family, for the structural counts
FAM = {}
for name, loader in {
    "single": lambda: pd.read_csv(sorted(glob.glob(f"{YS}/A_to_B/df_rows_0_1_*_rep_1_*.csv"))[0]),
    "multistate": lambda: split_and_merge_simulations([
        sorted(glob.glob(f"{YS}/A_to_B_high_k_on/df_*rep_1_*.csv"))[0],
        sorted(glob.glob(f"{YS}/A_to_B_low_k_on/df_*rep_1_*.csv"))[0]]),
    "lowmid": lambda: pd.read_csv(sorted(glob.glob(f"{LM}/K_ramp/df_rows_0_0_*A_to_B_lowmid_K_ramp_*1*.csv"))[0]),
}.items():
    try:
        FAM[name] = partition_counts(loader())
        print(f"{name:11s}: {FAM[name]}")
    except Exception as e:
        print(f"{name}: FAILED {type(e).__name__}: {e}")


def fam_of(scen):
    if scen.startswith("multistate"):
        return "multistate"
    if scen.startswith(("kramp", "kfrozen")):
        return "lowmid"
    return "single"


rows = []
for sub in ("t1_1_t2_20", "t1_10_t2_20"):
    for jf in sorted(glob.glob(f"{ROOT}/{sub}/*_rep_*.json")):
        d = json.load(open(jf))
        scen = os.path.basename(jf).rsplit("_rep_", 1)[0]
        c = FAM[fam_of(scen)]
        m_tw = c["n_twin"]                       # div / het:  n twin pairs, clone-weighted (size 2 -> weight1)
        m_s1 = c["m_step1"]                      # step1 Kish m_eff
        m_x = c["n_ac"]                          # step4 cross-time pairs
        m_d = 1.0 / (1.0 / m_tw + 1.0 / m_tw)    # step3: var(d)=var_t1+var_t2, disjoint clones
        for tp in ("t1", "t2"):
            rows.append(dict(scen=scen, sub=sub, tp=tp, step="step1",
                             rho=d[f"rho_{tp}"], zc=d[f"step1_z_{tp}"],
                             nm=d[f"step1_null_mean_{tp}"], ns=d[f"step1_null_std_{tp}"], m=m_s1))
            rows.append(dict(scen=scen, sub=sub, tp=tp, step="z_div",
                             rho=d[f"rho_delta_{tp}"], zc=d.get(f"z_div_{tp}"),
                             nm=d[f"div_null_mean_{tp}"], ns=d[f"div_null_std_{tp}"], m=m_tw))
            rows.append(dict(scen=scen, sub=sub, tp=tp, step="z_het",
                             rho=d[f"rho_delta_{tp}"], zc=d[f"step2_z_het_{tp}"],
                             nm=d[f"het_null_mean_{tp}"], ns=d[f"het_null_std_{tp}"], m=m_tw))
        rows.append(dict(scen=scen, sub=sub, tp="-", step="step3_d",
                         rho=d["step3_d"], zc=d["step3_z_d"],
                         nm=d["step3_null_mean"], ns=d["step3_null_std"], m=m_d))
        # step4: no null_std saved -> reconstruct implied from z, treat mean 0
        z4 = d.get("step4_z_1to2"); r4 = d.get("step4_rho_cross_1to2")
        rows.append(dict(scen=scen, sub=sub, tp="-", step="step4",
                         rho=r4, zc=z4, nm=0.0,
                         ns=(abs(r4 / z4) if z4 else np.nan), m=m_x))

df = pd.DataFrame(rows).dropna(subset=["rho", "zc", "ns"])
df = df[df["ns"] > 0]
df["ns_analytic"] = 1.0 / np.sqrt(df["m"] - 1.0)
df["m_implied"] = 1.0 + 1.0 / df["ns"] ** 2
df["z_analytic"] = (df["rho"] - 0.0) / df["ns_analytic"]                 # mean 0 + analytic SD
df["z_emp_mean_analytic_sd"] = (df["rho"] - df["nm"]) / df["ns_analytic"]  # empirical mean, analytic SD

pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
print("\n================  per-step summary  ================")
g = df.groupby("step")
summ = pd.DataFrame({
    "n": g.size(),
    "m_structural": g["m"].first().round(0),
    "m_implied_mean": g["m_implied"].mean().round(0),
    "m_implied_cv%": (100 * g["m_implied"].std() / g["m_implied"].mean()).round(1),
    "nullSD_emp/analytic": (g.apply(lambda x: (x["ns"] / x["ns_analytic"]).mean())).round(3),
    "null_mean(|mean|/SD)": (g.apply(lambda x: (x["nm"].abs() / x["ns"]).mean())).round(3),
    "corr(z_analytic,z_5k)": g.apply(lambda x: np.corrcoef(x["z_analytic"], x["zc"])[0, 1]).round(4),
    "median|z_an - z_5k|": g.apply(lambda x: (x["z_analytic"] - x["zc"]).abs().median()).round(3),
    "median|z_empMean_analSD - z_5k|": g.apply(lambda x: (x["z_emp_mean_analytic_sd"] - x["zc"]).abs().median()).round(3),
})
print(summ.to_string())

print("\n================  all rows (z_analytic vs 5k)  ================")
show = df[["scen", "sub", "tp", "step", "rho", "m", "ns", "ns_analytic", "nm", "zc", "z_analytic", "z_emp_mean_analytic_sd"]].copy()
show.columns = ["scen", "sub", "tp", "step", "rho", "m", "nullSD_5k", "nullSD_an", "nullMean_5k", "z_5k", "z_an", "z_an(empMean)"]
for c in ["rho", "nullSD_5k", "nullSD_an", "nullMean_5k"]:
    show[c] = show[c].round(4)
for c in ["z_5k", "z_an", "z_an(empMean)"]:
    show[c] = show[c].round(2)
print(show.sort_values(["step", "scen", "sub", "tp"]).to_string(index=False))
