"""Compare scenario calls on the 3-gene motif pairs: flowchart (|z_het|>5 and >2.576), rho,C,U at t2, combined OR (shared = |z_het|>5 OR |z_C|>1.96; regulation = |z_U|>1.96 OR z_d>2.326; existence = |z_rho_t2|>1.96)
and combined AND (shared = |z_het|>5 AND |z_C|>1.96; regulation = |z_U|>1.96). Heterogeneous = scenario 3 or 4; regulation = 2 or 3. usage: python motif_flowchart_compare.py CSV"""
import sys, numpy as np, pandas as pd
pd.set_option("display.width", 220)
m = pd.read_csv(sys.argv[1]); m["pos"] = ~m.truth.str.startswith("none"); print(len(m), "pair-replicates", m.groupby("motif").rep.nunique().to_dict())
ex_f = m.step1_z_t1.abs() > 2.576; ex2 = m.z_rho_t2.abs() > 1.96; h5 = m.step2_z_het_t1.abs() > 5; h2 = m.step2_z_het_t1.abs() > 2.576
C2 = m.z_C_t2.abs() > 1.96; U2 = m.z_U_t2.abs() > 1.96; d = m.step3_z_d > 2.326
sc = lambda e, s, g: pd.Series(np.where(~e, 1, np.where(~s, 2, np.where(g, 3, 4))), index=m.index)
calls = {"flowchart |z_het|>5": sc(ex_f, h5, d), "flowchart |z_het|>2.576": sc(ex_f, h2, d),
         "rho,C,U at t2": pd.Series(np.where(~C2 & ~ex2, 1, np.where(~C2 & ex2, 2, np.where(C2 & U2, 3, 4))), index=m.index),
         "combined OR": sc(ex2, h5 | C2, U2 | d), "combined AND": sc(ex2, h5 & C2, U2)}
for k, c in calls.items():
    m["call_" + k] = c; print(f"\n=== {k}: scenario counts 1/2/3/4 per pair"); print(pd.crosstab([m.motif, m.pair, m.truth], c).reindex(columns=[1, 2, 3, 4], fill_value=0).to_string())
ff = (m.motif == "Feed_forward") & (m.pair == "2-3"); fo = (m.motif == "Fan_out") & (m.pair == "2-3"); other = m.pos & ~ff
rows = []
for k, c in calls.items():
    het = c.isin([3, 4]); reg = c.isin([2, 3])
    rows.append({"call": k, "FF 2->3 heterogeneous": f"{int((het & ff).sum())}/{int(ff.sum())}", "FF 2->3 called regulation": f"{int((reg & ff).sum())}/{int(ff.sum())}",
                 "Fan_out 2-3 (no edge) heterogeneous": f"{int((het & fo).sum())}/{int(fo.sum())}", "Fan_out 2-3 called regulation": f"{int((reg & fo).sum())}/{int(fo.sum())}",
                 "other edges heterogeneous": f"{int((het & other).sum())}/{int(other.sum())}", "other edges called regulation": f"{int((reg & other).sum())}/{int(other.sum())}"})
print("\n=== summary"); print(pd.DataFrame(rows).set_index("call").to_string()); m.to_csv(sys.argv[1].replace(".csv", "_with_calls.csv"), index=False)
