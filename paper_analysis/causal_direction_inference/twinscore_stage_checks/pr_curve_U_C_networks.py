"""Precision-recall of |U| and |C| (sample B, 20 h, Figure-2 design) for direct-edge pairs on (a) network_sweep 5-edge networks (3 nets x 3 sims, 6 genes) and
(b) mixed_network_sweep (48 nets x 3 sims, 6/10 genes). Positive = direct-edge gene pair (either direction); negatives = all other pairs, or only indirect / co-regulated / unrelated pairs.
Reports raw AUPRC (maximum 1, chance = base rate) and AUPRC / base rate (1.0 = chance). usage: python pr_curve_U_C_networks.py OUTDIR"""
import sys, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
O = sys.argv[1]; THR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0; SUF = "" if THR == 0 else f"_zrho{THR:g}"; A = "/gpfs/projects/b1255/hzhang/TwINFER_KA/analysis_data/"
d5 = pd.read_csv(A + "paper_analysis/causal_direction_inference/figure2_1000_zscores/network_sweep_e5_pairs.csv"); dm = pd.read_csv(A + "mixed_network_sweep/twinscore_pairs_SCU.csv")
sets = {"network_sweep 5-edge networks": d5, "mixed_network_sweep": dm}
negs = {"direct edge vs all other pairs": None, "direct edge vs indirect": "indirect", "direct edge vs co-regulated": "co-regulated", "direct edge vs unrelated": "unrelated"}
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1a1a19", "#5b5a55", "#e3e2dd"
fig, axs = plt.subplots(2, 4, figsize=(15, 7.2), sharey=False); rows = []
for i, (sn, d) in enumerate(sets.items()):
    d = d.dropna(subset=["S", "C", "U", "z_rho"]).copy(); d = d[d.z_rho.abs() > THR].copy(); d["pos"] = d.cls == "direct edge"
    for j, (nn, c) in enumerate(negs.items()):
        ax = axs[i, j]; dd = d if c is None else d[d.pos | (d.cls == c)]
        if c is not None and (dd.cls == c).sum() == 0: ax.axis("off"); continue
        base = dd.pos.mean(); ax.axhline(base, color=GRID, lw=1.2); ax.text(0.01, base + 0.015, f"chance = base rate = {base:.2f}", color=MUTED, fontsize=8)
        for sc, col in (("U", BLUE), ("C", ORANGE)):
            s = dd[sc].abs().to_numpy(); p, rc, _ = precision_recall_curve(dd.pos, s); ap = average_precision_score(dd.pos, s)
            ax.plot(rc, p, color=col, lw=2, label=f"|{sc}|  AUPRC = {ap:.2f}")
            rows.append(dict(sweep=sn, comparison=nn, score=f"|{sc}|", n_pairs=len(dd), n_pos=int(dd.pos.sum()), base_rate=base, AUPRC=ap, AUPRC_over_base=ap / base))
        ax.set_title(f"{sn}\n{nn}  (n = {len(dd)})", fontsize=9.5, color=INK, loc="left"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.03); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8.5); ax.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=INK)
        if i == 1: ax.set_xlabel("recall (direct edges found)", color=INK)
        if j == 0: ax.set_ylabel("precision", color=INK)
fig.suptitle("U vs C as an edge score on network simulations (sample B, 20 h)" + (f"; only pairs with |z_rho| > {THR:g}" if THR else ""), fontsize=11.5, color=INK, x=0.01, ha="left"); fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(f"{O}/pr_curve_U_C_networks{SUF}.png", dpi=140, facecolor="#fcfcfb"); t = pd.DataFrame(rows); t.to_csv(f"{O}/pr_curve_U_C_networks_auprc{SUF}.csv", index=False)
print("threshold |z_rho| >", THR); print(t.pivot_table(index=["sweep", "comparison", "n_pairs", "n_pos"], columns="score", values="AUPRC_over_base").round(2).to_string())
for sn, d in sets.items():
    d = d.dropna(subset=["S", "C", "U", "z_rho"]); d = d[d.z_rho.abs() > THR]; print(f"\n{sn}: median |S| |C| |U| by class"); print(d.assign(aS=d.S.abs(), aC=d.C.abs(), aU=d.U.abs()).groupby("cls")[["aS", "aC", "aU"]].median().round(3).to_string())
