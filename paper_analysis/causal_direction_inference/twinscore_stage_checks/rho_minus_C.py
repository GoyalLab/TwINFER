"""|z_rho| - |z_C| (no z_U) vs |z_rho|, |z_U| and |z_rho|-|z_C|+|z_U| on designs X (1500 twins + 3000 singletons), Y3 (3000 twins), Y6 (6000 twins); AUPRC x base per group, split by network type."""
exec(open("twin_design_compare.py").read().split("SC = {")[0])
SC = {"|z_rho|": A_.azr, "|z_U|": A_.azU, "|z_rho|-|z_C|": A_.azr - A_.azC, "|z_rho|-|z_C|+|z_U|": A_.new3, "|z_rho|-0.5|z_C|": A_.azr - 0.5 * A_.azC}
for k, v in SC.items(): A_[k] = v
rows = []
for (raw, sim, des), x in A_.groupby(["raw", "sim", "design"]):
    if x.edge.sum() in (0, len(x)): continue
    r = dict(group=raw.rsplit("_rep", 1)[0], fam=x.fam.iloc[0], design=des)
    for k in SC: r[k] = average_precision_score(x.edge, x[k]) / x.edge.mean()
    rows.append(r)
G = pd.DataFrame(rows).groupby(["fam", "group", "design"]).mean(numeric_only=True).reset_index()
G["type"] = np.where(G.group.str.contains("_a[1-9]\\d*_pos"), "mixed autoreg", np.where(G.fam == "mixed", "mixed no autoreg", "sweep"))
pd.set_option("display.width", 220); cols = list(SC)
print(G.groupby("design")[cols].mean().round(2).to_string()); print()
print(G.groupby(["design", "type"])[cols].mean().round(2).to_string())
for des, x in G.groupby("design"):
    print(des, "| rho-C better than |z_rho| in %d/%d groups; vs full formula: full better in %d/%d" % ((x["|z_rho|-|z_C|"] > x["|z_rho|"]).sum(), len(x), (x["|z_rho|-|z_C|+|z_U|"] > x["|z_rho|-|z_C|"]).sum(), len(x)))
