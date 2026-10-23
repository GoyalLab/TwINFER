"""Compare two scenario callers on the same replicates (output of scenario_flowchart_clone_weighted.py) and check the calls against the S / C / U ranks.
FLOWCHART (paper Fig. 2a, infer.py clone-weighted):  I |step1_z_t1|>thrI -> else 1 | II |step2_z_het_t1|>thrII (multi-state) else 2 | III step3_z_d>2.326: 3 else 4
PICTURE   (z_C, z_rho, z_U; two-sided |z|>1.96, single pair):  z_C ns & z_rho ns:1 | z_C ns & z_rho sig:2 | z_C sig & z_U sig:3 | z_C sig & z_U ns:4   (at t1 and at t2)
Usage: python scenario_flowchart_vs_picture_summary.py DIR   (DIR contains flowchart_zscores.csv; tables go to DIR/summary_*.csv)"""
import sys, os, numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
D = sys.argv[1]
r = pd.read_csv(os.path.join(D, "flowchart_zscores.csv"))
NAME = {"no_regulation": "A_B", "A_to_B": "A_to_B", "multistate_A_B": "A_B two states", "multistate_A_to_B": "A_to_B two states"}
ORD = [NAME[k] for k in NAME]
r["sim"] = r.scenario.map(NAME)


def flow(thr1, thr2, d):
    s = np.where(d.step1_z_t1.abs() <= thr1, 1, np.where(d.step2_z_het_t1.abs() <= thr2, 2, np.where(d.step3_z_d > 2.326, 3, 4)))
    return pd.Series(s, index=d.index)


def pic(d, t, thr=1.96):
    sh = d[f"z_C_{t}"].abs() > thr; rh = d[f"z_rho_{t}"].abs() > thr; u = d[f"z_U_{t}"].abs() > thr
    return pd.Series(np.where(~sh & ~rh, 1, np.where(~sh & rh, 2, np.where(sh & u, 3, 4))), index=d.index)


calls = {"flow_p01_het5": flow(2.576, 5, r), "flow_p01_het10": flow(2.576, 10, r), "flow_onetail_het10": flow(2.326, 10, r),
         "pic_t1": pic(r, "t1"), "pic_t2": pic(r, "t2")}
for k, v in calls.items(): r[k] = v
out = {}
print("== 1. scenario counts per simulation (columns 1 2 3 4) ==")
for k in calls:
    t = pd.crosstab(r.sim, r[k]).reindex(index=[o for o in ORD if o in set(r.sim)], columns=[1, 2, 3, 4], fill_value=0)
    print(f"\n{k}"); print(t.to_string()); out[f"counts_{k}"] = t
print("\n== 2. agreement flowchart vs picture (rows = flowchart call, columns = picture call), all replicates ==")
for f in ("flow_p01_het5", "flow_p01_het10"):
    for p in ("pic_t1", "pic_t2"):
        t = pd.crosstab(r[f], r[p]).reindex(index=[1, 2, 3, 4], columns=[1, 2, 3, 4], fill_value=0)
        print(f"\n{f} vs {p}: same call in {int((r[f] == r[p]).sum())}/{len(r)}"); print(t.to_string()); out[f"agree_{f}_{p}"] = t
print("\n== 3. same-call fraction per simulation ==")
t = pd.DataFrame({f"{f} vs {p}": r.groupby("sim").apply(lambda d: (d[f] == d[p]).mean()) for f in ("flow_p01_het5", "flow_p01_het10") for p in ("pic_t1", "pic_t2")}).reindex([o for o in ORD if o in set(r.sim)]).round(2)
print(t.to_string()); out["same_call_fraction"] = t

print("\n== 4. S, C, U, rho_hat_Delta medians per simulation ==")
cols = [f"{m}_{t}" for t in ("t1", "t2") for m in ("S", "C", "U", "rho_delta", "h_x", "h_y", "z_rho", "z_C", "z_U")]
t = r.groupby("sim")[cols + ["step1_z_t1", "step2_z_het_t1", "step3_z_d"]].median().reindex([o for o in ORD if o in set(r.sim)]).round(3).T
print(t.to_string()); out["medians"] = t

print("\n== 5. rank overlap: pooled replicates ranked by |metric| (1 = largest); median rank of each flowchart / picture scenario, and scenario mix of the top 50 ==")
for tm in ("t1", "t2"):
    for m in ("S", "C", "U", "rho_delta", "z_U"):
        r[f"rk_{m}_{tm}"] = r[f"{m}_{tm}"].abs().rank(ascending=False)
for cl in ("flow_p01_het5", "pic_t2"):
    rows = []
    for sc in (1, 2, 3, 4):
        d = r[r[cl] == sc]
        row = {"scenario": sc, "n": len(d)}
        for tm in ("t1", "t2"):
            for m in ("S", "C", "U", "rho_delta"):
                row[f"med_rank_{m}_{tm}"] = d[f"rk_{m}_{tm}"].median() if len(d) else np.nan
        rows.append(row)
    t = pd.DataFrame(rows).set_index("scenario").round(0); print(f"\nmedian rank by {cl} scenario"); print(t.to_string()); out[f"rank_{cl}"] = t
    rows = []
    for tm in ("t1", "t2"):
        for m in ("S", "C", "U", "rho_delta"):
            top = r[r[f"rk_{m}_{tm}"] <= 50]
            rows.append({"ranked_by": f"|{m}|_{tm}", **{f"n_scen{s}": int((top[cl] == s).sum()) for s in (1, 2, 3, 4)}})
    t = pd.DataFrame(rows).set_index("ranked_by"); print(f"\nscenario mix of the top 50 by {cl}"); print(t.to_string()); out[f"top50_{cl}"] = t
for k, v in out.items(): v.to_csv(os.path.join(D, f"summary_{k}.csv"))
r.to_csv(os.path.join(D, "flowchart_zscores_with_calls.csv"), index=False)
