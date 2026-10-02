from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_a2a3302f_2026-09-28_why_twinfer_fails; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

SC = f'{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/real_networks/why_twinfer_fails'
OUT = f'{TWINFER_PROJECT_ROOT}/analysis_data/boolode_sims_real_networks/diagnostics/gillespie_dynamics_fidelity.pdf'

df = pd.read_csv(f"{SC}/dynamics_fidelity_results.csv")
order = ["mCAD", "GSD", "HSC", "VSC", "B_cell_activation", "EMT", "Pluripotent"]
algos = ["GENIE3", "GRNBOOST2", "PEARSON", "PIDC", "PPCOR", "SCODE", "SCSGL", "TwINFER"]

gt = df.drop_duplicates("net")[["net", "n_genes", "n_true_edges", "gt_fixed_points", "gt_exact", "gt_reachable", "gt_conv_rate"]].set_index("net").reindex(order)

with PdfPages(OUT) as pdf:
    # Page 1: title / methodology
    fig = plt.figure(figsize=(11, 8.5)); fig.text(0.06, 0.94, "Gillespie real-network dynamics-fidelity validation", fontsize=16, weight="bold")
    text = (
        "Mirrors BEELINE's own supplementary dynamics-fidelity analysis (Pratapa et al. 2020), applied to TwINFER's\n"
        "Gillespie-simulated real networks instead of BoolODE's synthetic toy topologies.\n\n"
        "METHOD (matches BEELINE's supplementary text exactly, except noted):\n"
        "  1. Self-loops removed from each algorithm's ranked edge list (twin_paired scheme, rep 0).\n"
        "  2. Walk down the ranking, adding edges, until every gene has >=1 incoming edge (BEELINE's own\n"
        "     adaptive threshold rule -- not a fixed top-k).\n"
        "  3. Sign: use the algorithm's own signed weight if it predicts one (PEARSON/PPCOR/SCODE/TwINFER);\n"
        "     otherwise borrow PPCOR's sign for that pair (BEELINE's own stated convention for unsigned methods\n"
        "     GENIE3/GRNBoost2/PIDC/SCSGL).\n"
        "  4. Rule per gene: (OR of activators) AND NOT (OR of repressors). Self-loops added back from ground\n"
        "     truth if present (matches BEELINE exactly).\n"
        "  5. Ground truth dynamics: the network's ACTUAL rule file --\n"
        "       mCAD/GSD/HSC/VSC: literal BoolODE .txt rule (real AND-combination logic preserved, not the\n"
        "         simplified reconstruction -- e.g. HSC's real Gata2 rule has a genuine 'Gata1 AND Fog1' term).\n"
        "       B_cell_activation/EMT/Pluripotent: no hand-built rule exists (these were only ever simulated as\n"
        "         a continuous Gillespie promoter model) -- ground truth uses the SAME simplified OR/AND-NOT\n"
        "         reconstruction applied to the TRUE signed edges, so it is an approximation for these three,\n"
        "         not literal ground truth. Flagged in each panel.\n"
        "  6. Steady states: a state s is a fixed point iff applying every gene's rule to s reproduces s exactly\n"
        "     (this is sync/async-invariant, so no simulation is needed for the COUNT). Exact for n<=20 genes\n"
        "     (full 2^n enumeration); Pluripotent (n=36) uses a 500,000-state random sample -- an approximate\n"
        "     LOWER BOUND only, flagged 'not exact' throughout.\n"
        "  7. Reachability: 40 independent asynchronous simulations (random update order each sweep) from the\n"
        "     all-OFF state, capped at 200 sweeps; a run that never stops changing (a genuine limit cycle, e.g.\n"
        "     EMT's Snai1 self-repression) is excluded from the reached set, not miscounted as a fixed point --\n"
        "     convergence rate is reported alongside every reachable-state count.\n\n"
        "CAVEATS:\n"
        "  - Single replicate (rep 0) per algorithm/network -- these are point estimates, not averaged.\n"
        "  - Edge-selection threshold (in-degree >=1 for every gene) can pull in far more edges than the true\n"
        "    count on dense/large networks (see Pluripotent) -- this is BEELINE's own convention, not a bug.\n"
        "  - EMT and Pluripotent's ground truth is a simplified reconstruction, not literal truth -- read their\n"
        "    fixed-point/reachability ratios as 'how well did the algorithm match OUR reconstruction', not\n"
        "    'how well did it match the real biology'.\n"
    )
    fig.text(0.06, 0.06, text, fontsize=8.3, family="monospace", va="bottom")
    pdf.savefig(fig); plt.close(fig)

    # Page 2: ground truth summary table
    fig, ax = plt.subplots(figsize=(11, 4)); ax.axis("off")
    gt_disp = gt.copy()
    gt_disp["gt_fixed_points"] = [f"{v}{'' if e else ' (approx, not exact)'}" for v, e in zip(gt.gt_fixed_points, gt.gt_exact)]
    gt_disp["gt_reachable (conv rate)"] = [f"{v} ({c:.0%})" for v, c in zip(gt.gt_reachable, gt.gt_conv_rate)]
    gt_disp = gt_disp[["n_genes", "n_true_edges", "gt_fixed_points", "gt_reachable (conv rate)"]]
    gt_disp.columns = ["genes", "true edges", "ground-truth\nfixed points", "reachable from\nall-OFF (conv. rate)"]
    tbl = ax.table(cellText=gt_disp.reset_index().values, colLabels=["network"] + list(gt_disp.columns),
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 2.2)
    ax.set_title("Ground-truth dynamics per network\n(EMT's true 0 fixed points is exact -- consistent with its near-total lack of twin memory found earlier this session)", fontsize=10)
    pdf.savefig(fig); plt.close(fig)

    # Page 3-4: edge ratio and fixed-point ratio heatmaps
    for metric, title, cmap, center in [
        ("edge_ratio", "Edge ratio (inferred edges kept / true edges); BEELINE found most methods stay <2x", "RdBu_r", 1.0),
        ("fp_ratio", "Fixed-point (attractor) ratio: inferred / ground-truth count", "RdBu_r", 1.0),
        ("reach_ratio", "Reachable-steady-state ratio from all-OFF: inferred / ground-truth\n"
     "(trivially 1.0 almost everywhere -- both ground truth and every inferred network converge to exactly\n"
     "ONE attractor from all-OFF, in 40/40 runs; only EMT shows a real difference, see note below)", "RdBu_r", 1.0),
    ]:
        piv = df.pivot(index="net", columns="algorithm", values=metric).reindex(order)[algos]
        fig, ax = plt.subplots(figsize=(11, 5.5))
        vmax = np.nanpercentile(piv.values, 90) if np.isfinite(piv.values).any() else 2
        im = ax.imshow(np.log2(piv.values.astype(float) + 1e-9), cmap=cmap, vmin=-3, vmax=3, aspect="auto")
        ax.set_xticks(range(len(algos))); ax.set_xticklabels(algos, rotation=45, ha="right")
        ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
        for i in range(len(order)):
            for j in range(len(algos)):
                v = piv.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                             color="white" if abs(np.log2(v+1e-9)) > 1.5 else "black")
        fig.colorbar(im, ax=ax, label="log2(ratio)  (0 = matches ground truth exactly)")
        ax.set_title(title, fontsize=10)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

    # Page 5: edge ratio vs fixed-point ratio scatter (BEELINE's own correlation check)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for algo in algos:
        sub = df[df.algorithm == algo]
        ax.scatter(sub.edge_ratio, sub.fp_ratio, label=algo, s=40, alpha=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.axhline(1, color="gray", lw=0.5); ax.axvline(1, color="gray", lw=0.5)
    ax.set_xlabel("edge ratio (inferred/true)"); ax.set_ylabel("fixed-point ratio (inferred/true)")
    ax.set_title("Do over-predicted edges create spurious attractors?\n(BEELINE found yes on their synthetic networks)")
    ax.legend(fontsize=7, ncol=2)
    from scipy.stats import spearmanr
    r, p = spearmanr(df.edge_ratio, df.fp_ratio)
    direction = "MORE edges -> FEWER extra attractors" if r < 0 else "MORE edges -> MORE extra attractors (matches BEELINE)"
    ax.text(0.02, 0.02, f"Spearman r={r:.2f}, p={p:.3f} (pooled)\n{direction}", transform=ax.transAxes, fontsize=8)
    ax.set_title("Do over-predicted edges create spurious attractors?\n"
                 "NEGATIVE correlation here -- opposite of what BEELINE found on their synthetic toy networks:\n"
                 "on these real curated networks, UNDER-covering edges (ratio<1, e.g. SCSGL/PIDC/GENIE3) leaves\n"
                 "genes under-constrained and creates MORE spurious attractors, not fewer.", fontsize=9.5)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # Page 6: TwINFER vs best competitor, per network, per metric
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, metric, label in zip(axes, ["edge_ratio", "fp_ratio", "reach_ratio"],
                                  ["edge ratio", "fixed-point ratio", "reachable-state ratio"]):
        tw = df[df.algorithm == "TwINFER"].set_index("net")[metric].reindex(order)
        comp = df[df.algorithm != "TwINFER"].groupby("net")[metric].median().reindex(order)
        x = np.arange(len(order))
        ax.bar(x - 0.18, np.abs(np.log2(tw.values.astype(float) + 1e-9)), width=0.36, label="TwINFER |log2 ratio|")
        ax.bar(x + 0.18, np.abs(np.log2(comp.values.astype(float) + 1e-9)), width=0.36, label="competitor median |log2 ratio|")
        ax.set_xticks(x); ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("|log2(ratio)|  (0 = perfect match to ground truth)")
        ax.set_title(label + ("" if label != "reachable-state ratio" else "\n(trivially ~0 for everyone -- see page 5 note)"), fontsize=10)
        ax.legend(fontsize=7)
    fig.suptitle("TwINFER vs. competitor median: distance from ground truth on each dynamics metric (lower = better)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); pdf.savefig(fig); plt.close(fig)

    # Page 7: full results table
    fig, ax = plt.subplots(figsize=(11, 11)); ax.axis("off")
    disp = df[["net", "algorithm", "n_inferred_edges", "n_true_edges", "edge_ratio",
               "inferred_fixed_points", "gt_fixed_points", "fp_ratio",
               "inferred_reachable", "gt_reachable", "reach_ratio", "inferred_conv_rate"]].copy()
    disp.columns = ["net", "algo", "inf_edges", "true_edges", "edge_ratio", "inf_fp", "gt_fp", "fp_ratio",
                    "inf_reach", "gt_reach", "reach_ratio", "conv_rate"]
    for c in ["edge_ratio", "fp_ratio", "reach_ratio", "conv_rate"]:
        disp[c] = disp[c].round(2)
    disp = disp.sort_values(["net", "algo"])
    tbl = ax.table(cellText=disp.values, colLabels=disp.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(6.3); tbl.scale(1, 1.25)
    ax.set_title("Full results table (all networks x algorithms)", fontsize=10)
    pdf.savefig(fig); plt.close(fig)

print("saved", OUT)
