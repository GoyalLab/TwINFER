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

def series(df, gene, steps):
    return {t: df[df.step == t].sort_values("pair_id")[gene].to_numpy() for t in steps}

with PdfPages(f"{sys.argv[1]}/corr_over_time_multi_div_ref.pdf") as pdf:
    for net in order:
        genes = c.NETS[net]; M = np.loadtxt(TOPO[net], delimiter=","); n = len(genes)
        reg_idx, tgt_idx = np.where((M != 0) & ~np.eye(n, dtype=bool))
        edges = list(zip(reg_idx, tgt_idx))
        rng.shuffle(edges)
        edges = edges[:N_PAIRS_PER_NET]

        d = pd.read_csv(f"{c.SIM}/{net}/replicate_0/twin_final_states.csv")
        tA = d[d.source == "twin_A"].sort_values(["step", "pair_id"])
        tB = d[d.source == "twin_B"].sort_values(["step", "pair_id"])
        post_steps = sorted(tA.step.unique())
        t_div = post_steps[0]  # exact division (branch) step -- A and B are identical here by construction
        ts = [tA[tA.step == t]["t"].iloc[0] for t in post_steps]

        ncols = 2; nrows = int(np.ceil(len(edges) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.1 * nrows), squeeze=False)
        axes = axes.ravel()
        for ax, (ri, tidx) in zip(axes, edges):
            reg, tgt = genes[ri], genes[tidx]
            At = series(tA, tgt, post_steps); Ar = series(tA, reg, post_steps); Bt = series(tB, tgt, post_steps)
            self_sc = [spearmanr(At[t_div], At[t])[0] for t in post_steps]   # same-cell: twin A vs itself later
            reg_sc = [spearmanr(Ar[t_div], At[t])[0] for t in post_steps]
            self_ss = [spearmanr(At[t_div], Bt[t])[0] for t in post_steps]  # sister: twin A(t_div) vs twin B(t)
            reg_ss = [spearmanr(Ar[t_div], Bt[t])[0] for t in post_steps]

            ax.plot(ts, self_sc, "o-", color="tab:blue", label=f"self, same-cell: {tgt}(A,t_div) vs {tgt}(A,t)")
            ax.plot(ts, reg_sc, "s--", color="tab:orange", label=f"reg, same-cell: {reg}(A,t_div) vs {tgt}(A,t)")
            ax.plot(ts, self_ss, "o-", color="tab:green", alpha=0.75, label=f"self, sister: {tgt}(A,t_div) vs {tgt}(B,t)")
            ax.plot(ts, reg_ss, "s--", color="tab:red", alpha=0.75, label=f"reg, sister: {reg}(A,t_div) vs {tgt}(B,t)")
            ax.axhline(0, color="gray", lw=0.5)
            ax.set_title(f"{reg} -> {tgt}", fontsize=9)
            ax.set_ylim(-1.05, 1.05); ax.legend(fontsize=5.8, loc="best")
        for ax in axes[len(edges):]:
            ax.axis("off")
        fig.suptitle(f"{net}  (n={n} genes, {len(reg_idx)} non-self true edges, showing {len(edges)})\n"
                     f"t_div (division/branch time) = {ts[0]:.2f} -- all four lines anchored here",
                     fontsize=10)
        fig.supxlabel("t"); fig.supylabel("Spearman corr with t_div reference")
        fig.tight_layout(rect=[0.01, 0.01, 1, 0.93])
        pdf.savefig(fig); plt.close(fig)
        print(net, "done", flush=True)
print("saved PDF")
