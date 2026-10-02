# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""
Trajectory / steady-state comparison: BoolODE HSC vs TwINFER's seeded
(multi-state) Gillespie HSC.

BoolODE HSC: simulation_data/boolode_sims_replicates/HSC/replicate_{k}/twin_final_states.csv
  -- trunk (source=='trunk') then branch (source!='trunk') phase per replicate.
  Checks whether independent replicates land in different final states (the only
  way BoolODE, which starts every replicate from one fixed IC + noise, could show
  multistability -- via path-dependent divergence across replicates).

Gillespie seeded HSC: analysis_data/paper_analysis/HSC/simulate/multistate_2000steps_seeded/
  -- pre-division trunk (simulation_before_division_*, 800 steps, 6000 cells split into
  3 blocks of 2000 seeded into 3 different stable states) then post-division twin data
  (df_rows_*, 49 steps, replicate 1/2 per clone_id). Checks whether the 3 seeded
  sub-populations actually stay separated (reach/hold 3 distinct steady states) through
  both phases.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import numpy as np
import pandas as pd
from pathlib import Path

BOOLODE_DIR = Path(f'{TWINFER_PROJECT_ROOT}/simulation_data/boolode_sims_replicates/HSC')
SEEDED_DIR = Path(f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/HSC/simulate/multistate_2000steps_seeded')
GENE_ORDER = (Path(f'{TWINFER_PROJECT_ROOT}/simulation_data/twinfer_format/HSC/gene_order.txt')
              .read_text().split())

PRE_FILES = {
    0: "simulation_before_division_df_rows_0_0_0_0_0_0_0_0_0_0_0_04092026_003703_ncells_6000_HSC_seeded_rep_0_cdfd31ed.csv",
    1: "simulation_before_division_df_rows_0_0_0_0_0_0_0_0_0_0_0_04092026_003657_ncells_6000_HSC_seeded_rep_1_fddf88c2.csv",
    2: "simulation_before_division_df_rows_0_0_0_0_0_0_0_0_0_0_0_04092026_003728_ncells_6000_HSC_seeded_rep_2_743f3a5c.csv",
}
POST_FILES = {
    0: "df_rows_0_0_0_0_0_0_0_0_0_0_0_04092026_003703_ncells_6000_HSC_seeded_rep_0_cdfd31ed.csv",
    1: "df_rows_0_0_0_0_0_0_0_0_0_0_0_04092026_003657_ncells_6000_HSC_seeded_rep_1_fddf88c2.csv",
    2: "df_rows_0_0_0_0_0_0_0_0_0_0_0_04092026_003728_ncells_6000_HSC_seeded_rep_2_743f3a5c.csv",
}

print("=" * 80)
print("BOOLODE HSC: trunk -> branch, across replicates 0-4")
print("=" * 80)
for rep in range(5):
    df = pd.read_csv(BOOLODE_DIR / f"replicate_{rep}" / "twin_final_states.csv")
    genes = [c for c in df.columns if c in GENE_ORDER]
    trunk = df[df["source"] == "trunk"].sort_values("t")
    branch = df[df["source"] != "trunk"]
    t0 = trunk.iloc[0][genes].values.astype(float)
    t_end = trunk.iloc[-1][genes].values.astype(float)
    b_end = branch.groupby("pair_id")[genes].last().mean().values.astype(float)
    print(f"rep{rep}: trunk t0->t_end (n={len(trunk)} steps, t={trunk['t'].min():.1f}-{trunk['t'].max():.1f}):")
    for g, a, b in zip(genes, t0, t_end):
        print(f"    {g:8s} {a:7.3f} -> {b:7.3f}")
    print(f"  mean branch-phase endpoint (avg over {branch['pair_id'].nunique()} pairs): "
          f"{np.round(b_end, 3).tolist()}")

print()
print("=" * 80)
print("GILLESPIE SEEDED HSC: pre-division trunk, 3 seeded sub-populations, rep 0")
print("=" * 80)
mrna_cols = [f"gene_{i+1}_mRNA" for i in range(len(GENE_ORDER))]
df = pd.read_csv(SEEDED_DIR / PRE_FILES[0], usecols=["cell_id", "time_step"] + mrna_cols)
# 3 seed blocks: cell_id in [0,2000), [2000,4000), [4000,6000)
n_cells = df["cell_id"].max() + 1
bounds = np.linspace(0, n_cells, 4).astype(int)
df["block"] = pd.cut(df["cell_id"], bins=bounds, labels=[0, 1, 2], include_lowest=True)

for tp in sorted(df["time_step"].unique())[:: max(1, len(df["time_step"].unique()) // 8)]:
    sub = df[df["time_step"] == tp]
    means = sub.groupby("block", observed=True)[mrna_cols].mean()
    means_log = np.log1p(means)
    print(f"t={tp:4d}  log1p(mRNA) mean per block (rows=block, cols=genes):")
    print(means_log.round(2).to_string())
    print()

print("=" * 80)
print("GILLESPIE SEEDED HSC: post-division (twin) phase, rep 0 -- do the 3 states persist?")
print("=" * 80)
dfp = pd.read_csv(SEEDED_DIR / POST_FILES[0], usecols=["cell_id", "time_step", "clone_id", "replicate"] + mrna_cols)
dfp["block"] = pd.cut(dfp["clone_id"], bins=bounds, labels=[0, 1, 2], include_lowest=True)
for tp in [dfp["time_step"].min(), dfp["time_step"].max()]:
    sub = dfp[dfp["time_step"] == tp]
    means = np.log1p(sub.groupby("block", observed=True)[mrna_cols].mean())
    print(f"t={tp}  log1p(mRNA) mean per block:")
    print(means.round(2).to_string())
    print()

print("Checking within-block variance (do the 3 blocks stay distinct or re-merge?)")
final = dfp[dfp["time_step"] == dfp["time_step"].max()]
between_block_var = np.log1p(final.groupby("block", observed=True)[mrna_cols].mean()).var().mean()
within_block_var = np.log1p(final[mrna_cols]).groupby(final["block"], observed=True).var().mean().mean()
print(f"mean between-block variance (of block means): {between_block_var:.3f}")
print(f"mean within-block variance (of cells within a block): {within_block_var:.3f}")
print(f"ratio (between/within): {between_block_var / within_block_var:.3f}  "
      f"(>>1 means the 3 seeded states remain well separated)")
