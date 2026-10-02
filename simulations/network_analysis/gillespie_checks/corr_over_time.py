from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import sys, numpy as np, pandas as pd, warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0,"/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/work_in_progress/benchmark")
from benchmarks.network_benchmarks.to_beeline import boolode_real_convert as c
R=f'{TWINFER_PROJECT_ROOT}'; RW=f"{R}/input_data/real_world_networks"; OLD=f"{R}/simulation_data/twinfer_format"
TOPO={"B_cell_activation":f"{RW}/B_cell.txt","EMT_real":f"{RW}/EMT.txt","Pluripotent_real":f"{RW}/Pluripotent.txt",
      **{n:f"{OLD}/{n}/interaction_matrix.txt" for n in ("GSD","HSC","mCAD","VSC")}}
order=["B_cell_activation","EMT_real","Pluripotent_real","GSD","HSC","mCAD","VSC"]

fig, axes = plt.subplots(4, 2, figsize=(11, 15))
axes = axes.ravel()
for ax, net in zip(axes, order):
    genes = c.NETS[net]; M = np.loadtxt(TOPO[net], delimiter=","); n = len(genes)
    gi = {g: i for i, g in enumerate(genes)}
    # first true directed edge reg->target, in gene order
    reg_idx, tgt_idx = np.where((M != 0) & ~np.eye(n, dtype=bool))
    reg, tgt = genes[reg_idx[0]], genes[tgt_idx[0]]

    d = pd.read_csv(f"{c.SIM}/{net}/replicate_0/twin_final_states.csv")
    trunk = d[d.source == "trunk"].sort_values(["step", "pair_id"])
    steps = sorted(trunk.step.unique())
    t0 = steps[1] if len(steps) > 1 else steps[0]  # first point after t=0 (skip exact IC, near-zero variance)
    ref_self = trunk[trunk.step == t0].sort_values("pair_id")[tgt].to_numpy()
    ref_reg = trunk[trunk.step == t0].sort_values("pair_id")[reg].to_numpy()

    self_corr, reg_corr, ts = [], [], []
    for t in steps:
        row = trunk[trunk.step == t].sort_values("pair_id")
        x_tgt = row[tgt].to_numpy(); x_reg = row[reg].to_numpy()
        self_corr.append(spearmanr(ref_self, x_tgt)[0])
        reg_corr.append(spearmanr(ref_reg, x_tgt)[0])
        ts.append(row["t"].iloc[0])

    ax.plot(ts, self_corr, "o-", label=f"self: {tgt}(t0) vs {tgt}(t)", color="tab:blue")
    ax.plot(ts, reg_corr, "s--", label=f"reg: {reg}(t0) vs {tgt}(t)  [{reg}->{tgt} true edge]", color="tab:orange")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_title(f"{net}  (n={n} genes, t0={ts[1] if len(ts)>1 else ts[0]:.2f})", fontsize=10)
    ax.set_xlabel("t"); ax.set_ylabel("Spearman corr with t0 value")
    ax.legend(fontsize=7, loc="best")
    ax.set_ylim(-1.05, 1.05)
fig.delaxes(axes[-1])
fig.suptitle("Population-level correlation with the t0 reference value, vs time\n(self = same gene, reg = t0 value of a true regulator)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(f"{sys.argv[1]}/corr_over_time.png", dpi=130)
print("saved")
