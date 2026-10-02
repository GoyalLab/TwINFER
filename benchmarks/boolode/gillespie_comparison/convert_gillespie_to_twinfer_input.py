# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""
Convert TwINFER's own Gillespie SSA twin-pair output (post-division only,
simulation_data/real_data/df_rows_*_<net>*.csv) into the same input schema
infer_with_twinfer expects (clone_id, cell_id, time_step, {gene}_mRNA,
replicate) -- same schema as simulation_data/twinfer_format/<net>/replicate_*_simulation.csv
(which is BoolODE data reformatted).

Per user instruction: ignore the pre-division ("before_division") trunk phase
entirely -- only use the post-division twin data.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import pandas as pd
from pathlib import Path

REAL_DATA_DIR = Path(f'{TWINFER_PROJECT_ROOT}/simulation_data/real_data')
TWINFER_FORMAT_DIR = Path(f'{TWINFER_PROJECT_ROOT}/simulation_data/twinfer_format')
OUT_DIR = Path(f'{TWINFER_PROJECT_ROOT}/clean_data/benchmarks/boolode/gillespie_comparison/gillespie_twinfer_input')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (network, source csv basename, replicate label used in output filename)
FILES = {
    "HSC": [
        "df_rows_0_0_0_0_0_0_0_0_0_0_0_12072026_020848_ncells_6000_HSC_balanced_0_0_6bff4482.csv",
        "df_rows_0_0_0_0_0_0_0_0_0_0_0_12072026_033541_ncells_6000_HSC_balanced_0_1_d38e9655.csv",
        "df_rows_0_0_0_0_0_0_0_0_0_0_0_12072026_050342_ncells_6000_HSC_balanced_0_2_d7e88bea.csv",
    ],
    "mCAD": [
        "df_rows_0_0_0_0_0_12072026_005933_ncells_6000_mCAD_1_0_c733c5af.csv",
        "df_rows_0_0_0_0_0_12072026_011354_ncells_6000_mCAD_1_1_984e3851.csv",
        "df_rows_0_0_0_0_0_12072026_012802_ncells_6000_mCAD_1_2_9c719c8a.csv",
    ],
}

for net, files in FILES.items():
    gene_order = (TWINFER_FORMAT_DIR / net / "gene_order.txt").read_text().split()
    for rep_idx, fname in enumerate(files):
        df = pd.read_csv(REAL_DATA_DIR / fname)
        mrna_cols = {f"gene_{i+1}_mRNA": f"{g}_mRNA" for i, g in enumerate(gene_order)}
        keep = ["cell_id", "time_step", "clone_id", "replicate"] + list(mrna_cols.keys())
        df = df[keep].rename(columns=mrna_cols)
        out_path = OUT_DIR / net / f"replicate_{rep_idx}_simulation.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"{net} rep{rep_idx}: {fname} -> {out_path}  "
              f"({df['clone_id'].nunique()} clones, replicate values {sorted(df['replicate'].unique())}, "
              f"time_step range {df['time_step'].min()}-{df['time_step'].max()})")
