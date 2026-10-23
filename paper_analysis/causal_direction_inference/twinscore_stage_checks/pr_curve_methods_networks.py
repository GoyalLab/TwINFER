"""Precision-recall of every method as a direct-edge-pair score on the network sweeps (positive = direct-edge gene pair; negatives = all other pairs), same pairs as pr_curve_U_C_networks.py.
Per gene pair: |U|, |C|, |S|, |rho| (twin design, sample B 20 h) | BEELINE methods GENIE3 GRNBOOST2 PEARSON PIDC PPCOR SCODE SCSGL (rankedEdges.csv, |EdgeWeight|, pair score = max of the two directions, pairs a method did not
return get 0; LEAP / SINCERITIES produced no rankedEdges) | TwinScore (mixed sweep only; twinfer_inference_allpairs ranked_edges, max of the two directions).
BEELINE input: twin_paired (same 1500/1500/3000 clone split) = main, spread = second set of columns. Reports raw AUPRC (maximum 1, chance = base rate).
usage: python pr_curve_methods_networks.py OUTDIR [zrho_threshold]  (threshold keeps only pairs with |z_rho| > threshold, as in the U/C figure)"""
import sys, os, json, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
O = sys.argv[1]; THR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0; SUF = "" if THR == 0 else f"_zrho{THR:g}"
A = "/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/"; ALGOS = ["GENIE3", "GRNBOOST2", "PEARSON", "PIDC", "PPCOR", "SCODE", "SCSGL"]
d5 = pd.read_csv(A + "paper_analysis/causal_direction_inference/figure2_1000_zscores/network_sweep_e5_pairs.csv"); d5["net"] = "grn_n6_e5_pos50_density_" + d5.network
dm = pd.read_csv(A + "mixed_network_sweep/twinscore_pairs_SCU.csv")
cfg = {"network_sweep 5-edge networks": (d5, A + "network_sweep/beeline_inference/", False), "mixed_network_sweep": (dm, A + "mixed_network_sweep/beeline_inference/", True)}
def add_scores(d, root, has_ts):
    d = d.dropna(subset=["S", "C", "U", "z_rho"]).copy(); d[["gi", "gj"]] = d.pair.str.split("-", expand=True).astype(int)
    for scheme in ("twin_paired", "spread"):
        for algo in ALGOS:
            col = f"{algo}|{scheme}"; d[col] = np.nan
            for (net, sim), g in d.groupby(["net", "sim"]):
                f = f"{root}{net}/simrep{sim}_{scheme}/{algo}/rankedEdges.csv"
                if not os.path.exists(f): continue
                e = pd.read_csv(f, sep="\t"); w = {}
                for a, b, v in zip(e.Gene1, e.Gene2, e.EdgeWeight.abs()): k = tuple(sorted((a, b))); w[k] = max(w.get(k, 0.0), float(v))
                d.loc[g.index, col] = [w.get(tuple(sorted((f"gene_{i}", f"gene_{j}"))), 0.0) for i, j in zip(g.gi, g.gj)]
    if has_ts:
        d["TwinScore"] = np.nan
        for (net, sim), g in d.groupby(["net", "sim"]):
            f = A + f"mixed_network_sweep/twinfer_inference_allpairs/{net}_rep{sim}_all_results.json"
            if not os.path.exists(f): continue
            r = json.load(open(f))["ranked_edges"]; e = pd.DataFrame(r["data"], columns=r["columns"]); w = {}
            for a, b, v in zip(e.gene_1, e.gene_2, e.twinScore): k = tuple(sorted((a, b))); w[k] = max(w.get(k, -np.inf), float(v))
            d.loc[g.index, "TwinScore"] = [w.get(tuple(sorted((f"gene_{i}", f"gene_{j}"))), np.nan) for i, j in zip(g.gi, g.gj)]
    for c in [c for c in d.columns if "|twin_paired" in c or "|spread" in c or c == "TwinScore"]:   # +/-inf weights: keep the ranking, make finite
        v = d[c].replace([np.inf, -np.inf], np.nan); hi, lo = v.max(), v.min()
        d[c] = d[c].replace(np.inf, hi + 1).replace(-np.inf, lo - 1)
    d["|U|"], d["|C|"], d["|S|"], d["|rho|"] = d.U.abs(), d.C.abs(), d.S.abs(), d.rho.abs(); d["pos"] = d.cls == "direct edge"
    return d
COL = {"|U|": "#2a78d6", "|C|": "#eb6834", "TwinScore": "#1baf7a"}; GREY = "#9a9a94"
fig, axs = plt.subplots(1, 2, figsize=(13, 5.2)); rows = []
for ax, (sn, (d, root, ts)) in zip(axs, cfg.items()):
    d = add_scores(d, root, ts); d.to_csv(f"{O}/method_scores_{sn.split()[0]}.csv", index=False)
    if THR: d = d[d.z_rho.abs() > THR]
    base = d.pos.mean(); ax.axhline(base, color="#e3e2dd", lw=1.2); res = {}
    scores = ["|U|", "|C|", "|S|", "|rho|"] + (["TwinScore"] if ts else []) + [f"{a}|twin_paired" for a in ALGOS]
    for sc in scores:
        dd = d.dropna(subset=[sc]); 
        if dd.pos.nunique() < 2: continue
        p, rc, _ = precision_recall_curve(dd.pos, dd[sc]); ap = average_precision_score(dd.pos, dd[sc]); res[sc] = ap
        lab = sc.replace("|twin_paired", ""); ax.plot(rc, p, color=COL.get(sc, GREY), lw=2.2 if sc in COL else 0.9, alpha=1 if sc in COL else 0.75, zorder=3 if sc in COL else 2)
        rows.append(dict(sweep=sn, score=lab, input="twin_paired" if "|twin_paired" in sc else "twin design", n_pairs=len(dd), n_pos=int(dd.pos.sum()), base_rate=base, AUPRC=ap, AUPRC_over_base=ap / base))
    for algo in ALGOS:                                                  # spread input, AUPRC only (table)
        sc = f"{algo}|spread"; dd = d.dropna(subset=[sc])
        if len(dd) and dd.pos.nunique() == 2: rows.append(dict(sweep=sn, score=algo, input="spread", n_pairs=len(dd), n_pos=int(dd.pos.sum()), base_rate=base, AUPRC=average_precision_score(dd.pos, dd[sc]), AUPRC_over_base=average_precision_score(dd.pos, dd[sc]) / base))
    order = sorted(res, key=res.get, reverse=True)
    for k, sc in enumerate(order):                                      # ranked legend, coloured text only for U / C / TwinScore
        ax.text(1.02, 0.93 - 0.047 * k, f"{res[sc]:.2f}  {sc.replace('|twin_paired', '')}", transform=ax.transAxes, fontsize=8.3, va="top", color=COL.get(sc, "#5b5a55"), fontweight="bold" if sc in COL else "normal")
    ax.text(1.02, 0.99, "AUPRC (chance = " + f"{base:.2f})", transform=ax.transAxes, fontsize=8.3, va="top", color="#1a1a19")
    ax.set_title(f"{sn}\ndirect edge vs all other pairs (n = {len(d)})", fontsize=10, loc="left", color="#1a1a19"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.03); ax.grid(color="#e3e2dd", lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color("#5b5a55")
    ax.tick_params(colors="#5b5a55", labelsize=9); ax.set_xlabel("recall (direct edges found)", color="#1a1a19"); ax.set_ylabel("precision", color="#1a1a19")
fig.suptitle("Edge-pair score comparison" + (f" (only pairs with |z_rho| > {THR:g})" if THR else "") + ": U, C and TwinScore (colour) vs |S|, |rho| and BEELINE methods (grey)", fontsize=11, x=0.01, ha="left", color="#1a1a19")
fig.tight_layout(rect=(0, 0, 0.99, 0.95)); fig.subplots_adjust(right=0.80, wspace=0.62)
fig.savefig(f"{O}/pr_curve_methods{SUF}.png", dpi=140, facecolor="#fcfcfb"); t = pd.DataFrame(rows); t.to_csv(f"{O}/pr_curve_methods_auprc{SUF}.csv", index=False)
print(t[t.input != "spread"].pivot_table(index="score", columns="sweep", values="AUPRC").round(2).sort_values(t.sweep.unique()[-1], ascending=False).to_string())
print("\nspread input:"); print(t[t.input == "spread"].pivot_table(index="score", columns="sweep", values="AUPRC").round(2).to_string())
