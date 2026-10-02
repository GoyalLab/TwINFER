# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""
Test the hypothesis: HSC (legacy, 6000 pre-division steps) is closer to full
steady state, and its twins re-converge faster post-division, than EMT (new
track, 1000 steps) -- which would explain why PEARSON beats TwINFER on HSC/GSD/
VSC/mCAD but not on B_cell/Circadian/EMT/Pluripotent.

Two direct measurements, no modeling:
  (A) Population mean trajectory over the FULL before-division window --
      is it still trending (transient) or flat (steady) near the end?
  (B) Twin-pair (replicate 1 vs 2 of the same clone_id) correlation at t=1 and
      t=20 post-division -- high & stable = twins snap back together fast
      (little divergence signal for TwINFER); lower / more spread = real
      post-division drift for TwINFER's cross-correlation to read.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import numpy as np
import pandas as pd

CASES = {
    "HSC (legacy, 6000 steps)": dict(
        before=f'{TWINFER_PROJECT_ROOT}/simulation_data/real_data/simulation_before_division_df_rows_0_0_0_0_0_0_0_0_0_0_0_12072026_020848_ncells_6000_HSC_balanced_0_0_6bff4482.csv',
        twin=f'{TWINFER_PROJECT_ROOT}/simulation_data/real_data/df_rows_0_0_0_0_0_0_0_0_0_0_0_12072026_020848_ncells_6000_HSC_balanced_0_0_6bff4482.csv',
        n_genes=11, t_end=6000,
    ),
    "EMT (new track, 1000 steps)": dict(
        before=f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/EMT/simulate/20260825_224653/simulation_before_division_df_rows_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_25082026_234904_ncells_6000_EMT_rep_0_839e1156.csv',
        twin=f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/EMT/simulate/20260825_224653/df_rows_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_25082026_234904_ncells_6000_EMT_rep_0_839e1156.csv',
        n_genes=17, t_end=1000,
    ),
}

for label, cfg in CASES.items():
    print(f"\n===== {label} =====")
    n = cfg["n_genes"]
    mrna_cols = [f"gene_{i+1}_mRNA" for i in range(n)]

    # --- (A) population mean trajectory: sample sparse timepoints across the whole window ---
    checkpoints = sorted(set(int(x) for x in np.linspace(0, cfg["t_end"] - 1, 12)))
    means = {c: [] for c in mrna_cols}
    for ch in pd.read_csv(cfg["before"], usecols=["time_step"] + mrna_cols, chunksize=1_000_000):
        sub = ch[ch["time_step"].isin(checkpoints)]
        if len(sub):
            g = sub.groupby("time_step")[mrna_cols].mean()
            for c in mrna_cols:
                for t, v in g[c].items():
                    means[c].append((t, v))
    print(f"  (A) population mean mRNA at {len(checkpoints)} timepoints across [0,{cfg['t_end']}) (first 4 genes):")
    for c in mrna_cols[:4]:
        pts = sorted(set(means[c]))
        vals = [v for _, v in pts]
        rel_change_last_half = abs(vals[-1] - vals[len(vals)//2]) / (abs(vals[len(vals)//2]) + 1e-9)
        print(f"      {c}: {[round(v,2) for v in vals]}  -> rel. change over 2nd half = {rel_change_last_half:.2%}")

    # --- (B) twin-pair correlation at t=1 and t=20 post-division ---
    prot_cols = [f"gene_{i+1}_protein" for i in range(n)]
    twin = pd.read_csv(cfg["twin"], usecols=["clone_id", "replicate", "time_step"] + mrna_cols)
    for t in [1, 20]:
        sub = twin[twin["time_step"] == t]
        r1 = sub[sub["replicate"] == 1].set_index("clone_id")[mrna_cols]
        r2 = sub[sub["replicate"] == 2].set_index("clone_id")[mrna_cols]
        common = r1.index.intersection(r2.index)
        r1, r2 = r1.loc[common], r2.loc[common]
        corrs = [np.corrcoef(r1[c], r2[c])[0, 1] for c in mrna_cols]
        print(f"  (B) t={t:>2}: twin1-vs-twin2 mRNA correlation per gene, mean={np.nanmean(corrs):.3f}, "
              f"min={np.nanmin(corrs):.3f}, max={np.nanmax(corrs):.3f}")
