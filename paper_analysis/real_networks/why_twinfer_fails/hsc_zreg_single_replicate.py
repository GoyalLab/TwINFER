#!/usr/bin/env python
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_a2a3302f_2026-09-28_why_twinfer_fails; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Does zreg's OWN bootstrap-calibrated noise floor (sd_reg, from bootstrap_null_sd_offdiag -- clone-label
permutation) recover the Gata1->Fog1 single-input S-C signal on a SINGLE replicate, the way raw 10-replicate
averaging did (2.7e)? This is what the production pipeline would actually rely on in practice.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import sys
import numpy as np
import pandas as pd

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/fatemap_pipeline")
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/package")
from paper_analysis.fatemap_pipeline import twinscore_supp_helpers as supp
from paper_analysis.fatemap_pipeline import apply_twinscore_supplement_fatemap_gated_bootstrap as gb

R = f'{TWINFER_PROJECT_ROOT}'
FILE = f"{R}/simulation_data/real_data/df_rows_0_0_0_0_0_0_0_0_0_0_0_12072026_020848_ncells_6000_HSC_balanced_0_0_6bff4482.csv"
genes = [l.strip() for l in open(f"{R}/simulation_data/twinfer_format/HSC/gene_order.txt")]
gi = {g: i + 1 for i, g in enumerate(genes)}
reg, tgt = "Gata1", "Fog1"
native_genes = [f"gene_{i+1}" for i in range(len(genes))]  # helpers key by these column-mapped names

df_full = pd.read_csv(FILE)

for t in [10, 20, 30, 40, 48]:
    t_raw = df_full[df_full.time_step == t].reset_index(drop=True)
    # rename columns to native gene_N_mRNA (already are) -- use only the 2 genes needed for speed,
    # but same_cell_matrix/sister_matrix_from_frame expect the full "genes" list name-matched to columns
    sub_genes = [reg, tgt]
    # helpers use f"{g}_mRNA" directly against the dataframe's own column names, which here are gene_N_mRNA;
    # remap by renaming just these two columns to their biological names for this call
    rc, tc = f"gene_{gi[reg]}_mRNA", f"gene_{gi[tgt]}_mRNA"
    tmp = t_raw.rename(columns={rc: f"{reg}_mRNA", tc: f"{tgt}_mRNA"})

    S = supp.same_cell_matrix(tmp, sub_genes)
    C, meff = supp.sister_matrix_from_frame(tmp, sub_genes)
    lam_scalar = 1.0  # no zhet-based shrinkage for this quick 2-gene check; use raw S - C
    sd_reg, nperm = gb.bootstrap_null_sd_offdiag(tmp, sub_genes, seed=42, kind="reg",
                                                  extra=pd.DataFrame(lam_scalar, index=sub_genes, columns=sub_genes))
    s_val = S.loc[reg, tgt]
    c_val = C.loc[reg, tgt]
    zreg = (s_val - lam_scalar * c_val) / sd_reg if sd_reg and np.isfinite(sd_reg) and sd_reg > 0 else np.nan
    print(f"t={t:3d}  S={s_val:+.4f}  C={c_val:+.4f}  S-C={s_val-c_val:+.4f}  sd_reg={sd_reg:.5f} (n_perm={nperm})  zreg={zreg:+.2f}")
