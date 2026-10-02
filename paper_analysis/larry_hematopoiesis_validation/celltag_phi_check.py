"""phi/heritability for the 68-gene HSC/myeloid/panel12 list, on CellTag (day2.5 -> day5) instead
of LARRY. CellTag's day5 population (49,729 cells, clones up to 78 cells) makes
sister_matrix_from_frame alone take ~134s, and persistence()'s internal 20-replicate bootstrap
calls it repeatedly -- ~45+ min total, hence SLURM with a generous time budget rather than an
interactive/backgrounded shell call."""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import sys
import time

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, ".")
from paper_analysis.larry_hematopoiesis_validation import apply_twinscore_supplement_larry as m
import pandas as pd
import numpy as np
import anndata as ad

genes = pd.read_csv("all_hematopoiesis_regulators_phi.csv").gene.tolist()
A = ad.read_h5ad(f'{TWINFER_PROJECT_ROOT}/real_data/cellTag3_data/celltag_clones_repaired_nopad.h5ad')
obs = A.obs
has_clone = obs["repaired_clone"].notna().to_numpy()
print(f"cells with a clone call: {has_clone.sum()} / {len(obs)}", flush=True)

gidx = {g: i for i, g in enumerate(A.var_names)}
col = [gidx[g] for g in genes]
Xg = A.X[:, col]
Xg = np.asarray(Xg.todense()) if hasattr(Xg, "todense") else np.asarray(Xg)

df = pd.DataFrame(Xg, columns=[f"{g}_mRNA" for g in genes])
df.insert(0, "time_step", obs["Time point"].to_numpy())
df.insert(0, "cell_id", obs.index.to_numpy())
df.insert(0, "clone_id", obs["repaired_clone"].to_numpy())
df = df[has_clone].reset_index(drop=True)

t1_raw = df[df.time_step == 2].reset_index(drop=True)
t2_raw = df[df.time_step == 4].reset_index(drop=True)
print(f"t1 (day2.5) {t1_raw.shape}  t2 (day5) {t2_raw.shape}", flush=True)

t0 = time.time()
C1, meff_t1 = m.sister_matrix_from_frame(t1_raw, genes)
print(f"C1 done ({time.time()-t0:.0f}s)", flush=True)
t0 = time.time()
C2, meff_t2 = m.sister_matrix_from_frame(t2_raw, genes)
print(f"C2 done ({time.time()-t0:.0f}s)", flush=True)
h1 = {g: C1.loc[g, g] for g in genes}
h2 = {g: C2.loc[g, g] for g in genes}

t0 = time.time()
rho_dag, meff_cross = m.cross_matrix(t1_raw, t2_raw, genes)
print(f"cross_matrix done ({time.time()-t0:.0f}s)", flush=True)
rho_dag_diag = {g: rho_dag.loc[g, g] for g in genes}

t0 = time.time()
phi, w = m.persistence(h1, h2, rho_dag_diag, genes, meff_t1, meff_t2, meff_cross, t1_raw, t2_raw)
print(f"persistence (with bootstrap w) done ({time.time()-t0:.0f}s)", flush=True)

out = pd.DataFrame({"gene": genes, "h_t1_celltag": [h1[g] for g in genes],
                     "h_t2_celltag": [h2[g] for g in genes], "phi_celltag": [phi[g] for g in genes]})
out.to_csv("celltag_hematopoiesis_regulators_phi.csv", index=False)
print("w (panel reliability) =", w, flush=True)
print(out.sort_values("phi_celltag", ascending=False).to_string(index=False), flush=True)
