from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import pandas as pd
import numpy as np

BOOLODE_DIR = f'{TWINFER_PROJECT_ROOT}/simulation_data/boolode_sims_replicates/HSC'

for rep in range(3):
    df = pd.read_csv(f"{BOOLODE_DIR}/replicate_{rep}/twin_final_states.csv")
    genes = [c for c in df.columns if c not in ("branch_tp","pair_id","source","step","t")]
    trunk = df[df["source"] == "trunk"]
    print(f"rep{rep}: trunk rows={len(trunk)}, unique pair_id={trunk['pair_id'].nunique()}, unique t={trunk['t'].nunique()}")
    t_max = trunk["t"].max()
    final_trunk = trunk[trunk["t"] == t_max]
    print(f"  at final trunk t={t_max}: n_pairs={len(final_trunk)}")
    print(f"  population mean: {final_trunk[genes].mean().round(3).to_dict()}")
    print(f"  population std:  {final_trunk[genes].std().round(3).to_dict()}")
    # bimodality check on a couple of genes
    for g in ["Cebpa","Pu1","Gfi1","cJun"]:
        vals = final_trunk[g].values
        print(f"    {g}: min={vals.min():.3f} p25={np.percentile(vals,25):.3f} median={np.median(vals):.3f} p75={np.percentile(vals,75):.3f} max={vals.max():.3f}")
    print()
