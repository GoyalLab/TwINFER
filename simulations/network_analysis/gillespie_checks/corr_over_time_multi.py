from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import sys, numpy as np, pandas as pd, warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0,"/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/work_in_progress/benchmark")
from benchmarks.network_benchmarks.to_beeline import boolode_real_convert as c
R=f'{TWINFER_PROJECT_ROOT}'; RW=f"{R}/input_data/real_world_networks"; OLD=f"{R}/simulation_data/twinfer_format"
TOPO={"B_cell_activation":f"{RW}/B_cell.txt","EMT_real":f"{RW}/EMT.txt","Pluripotent_real":f"{RW}/Pluripotent.txt",
      **{n:f"{OLD}/{n}/interaction_matrix.txt" for n in ("GSD","HSC","mCAD","VSC")}}
order=["B_cell_activation","EMT_real","Pluripotent_real","GSD","HSC","mCAD","VSC"]
N_PAIRS_PER_NET = 8
rng = np.random.default_rng(0)

def series(trunk, gene, steps):
    return np.array([trunk[trunk.step == t].sort_values("pair_id")[gene].to_numpy() for t in steps])

with PdfPages(f"{sys.argv[1]}/corr_over_time_multi.pdf") as pdf:
    for net in order:
        genes = c.NETS[net]; M = np.loadtxt(TOPO[net], delimiter=","); n = len(genes)
        reg_idx, tgt_idx = np.where((M != 0) & ~np.eye(n, dtype=bool))
        edges = list(zip(reg_idx, tgt_idx))
        rng.shuffle(edges)
        edges = edges[:N_PAIRS_PER_NET]

        d = pd.read_csv(f"{c.SIM}/{net}/replicate_0/twin_final_states.csv")
        trunk = d[d.source == "trunk"].sort_values(["step", "pair_id"])
        steps = sorted(trunk.step.unique())
        t0 = steps[1] if len(steps) > 1 else steps[0]
        ts = [trunk[trunk.step == t]["t"].iloc[0] for t in steps]

        ncols = 2; nrows = int(np.ceil(len(edges) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.1 * nrows), squeeze=False)
        axes = axes.ravel()
        for ax, (ri, ti) in zip(axes, edges):
            reg, tgt = genes[ri], genes[ti]
            X_tgt = series(trunk, tgt, steps)   # (n_steps, n_cells)
            X_reg = series(trunk, reg, steps)
            i0 = steps.index(t0)
            self_corr = [spearmanr(X_tgt[i0], X_tgt[k])[0] for k in range(len(steps))]
            reg_corr = [spearmanr(X_reg[i0], X_tgt[k])[0] for k in range(len(steps))]
            ax.plot(ts, self_corr, "o-", color="tab:blue", label=f"self: {tgt}(t0) vs {tgt}(t)")
            ax.plot(ts, reg_corr, "s--", color="tab:orange", label=f"reg: {reg}(t0) vs {tgt}(t)")
            ax.axhline(0, color="gray", lw=0.5)
            ax.set_title(f"{reg} -> {tgt}", fontsize=9)
            ax.set_ylim(-1.05, 1.05); ax.legend(fontsize=6.5, loc="best")
        for ax in axes[len(edges):]:
            ax.axis("off")
        fig.suptitle(f"{net}  (n={n} genes, {len(reg_idx)} non-self true edges, showing {len(edges)}, t0={ts[i0]:.2f})", fontsize=11)
        fig.supxlabel("t"); fig.supylabel("Spearman corr with t0 reference")
        fig.tight_layout(rect=[0.01, 0.01, 1, 0.95])
        pdf.savefig(fig); plt.close(fig)
        print(net, "done", flush=True)
print("saved PDF")
