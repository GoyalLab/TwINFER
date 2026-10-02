# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""
Convert the seeded (multi-state) HSC Gillespie twin data into the same
infer_with_twinfer input schema (clone_id, cell_id, time_step, {gene}_mRNA,
replicate) used for the plain HSC/mCAD conversions.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import pandas as pd
from pathlib import Path

SRC_DIR = Path(f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/HSC/simulate/multistate_2000steps_seeded')
TWINFER_FORMAT_DIR = Path(f'{TWINFER_PROJECT_ROOT}/simulation_data/twinfer_format')
OUT_DIR = Path(f'{TWINFER_PROJECT_ROOT}/clean_data/benchmarks/boolode/gillespie_comparison/gillespie_twinfer_input/HSC_seeded')
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    "df_rows_0_0_0_0_0_0_0_0_0_0_0_04092026_003703_ncells_6000_HSC_seeded_rep_0_cdfd31ed.csv",
    "df_rows_0_0_0_0_0_0_0_0_0_0_0_04092026_003657_ncells_6000_HSC_seeded_rep_1_fddf88c2.csv",
    "df_rows_0_0_0_0_0_0_0_0_0_0_0_04092026_003728_ncells_6000_HSC_seeded_rep_2_743f3a5c.csv",
]

gene_order = (TWINFER_FORMAT_DIR / "HSC" / "gene_order.txt").read_text().split()

for rep_idx, fname in enumerate(FILES):
    df = pd.read_csv(SRC_DIR / fname)
    mrna_cols = {f"gene_{i+1}_mRNA": f"{g}_mRNA" for i, g in enumerate(gene_order)}
    keep = ["cell_id", "time_step", "clone_id", "replicate"] + list(mrna_cols.keys())
    df = df[keep].rename(columns=mrna_cols)
    out_path = OUT_DIR / f"replicate_{rep_idx}_simulation.csv"
    df.to_csv(out_path, index=False)
    print(f"HSC_seeded rep{rep_idx}: {fname} -> {out_path} "
          f"({df['clone_id'].nunique()} clones, replicate values {sorted(df['replicate'].unique())}, "
          f"time_step range {df['time_step'].min()}-{df['time_step'].max()})")
