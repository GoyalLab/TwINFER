from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_a2a3302f_2026-09-28_why_twinfer_fails; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import numpy as np, pandas as pd, warnings
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

R = f'{TWINFER_PROJECT_ROOT}'
FILES = {
    "mCAD": (f"{R}/simulation_data/real_data/df_rows_0_0_0_0_0_12072026_005933_ncells_6000_mCAD_1_0_c733c5af.csv", 5),
    "GSD": (f"{R}/simulation_data/real_data/df_rows_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_12072026_062944_ncells_6000_GSD_0_0_117c4801.csv", 19),
    "HSC": (f"{R}/simulation_data/real_data/df_rows_0_0_0_0_0_0_0_0_0_0_0_12072026_020848_ncells_6000_HSC_balanced_0_0_6bff4482.csv", 11),
    "VSC": (f"{R}/simulation_data/real_data/df_rows_0_0_0_0_0_0_0_0_12072026_005406_ncells_6000_VSC_2_0_2befd3f9.csv", 8),
    "B_cell_activation": (f"{R}/analysis_data/paper_analysis/B_cell_activation/simulate/20260812_153622/df_rows_0_0_0_0_0_0_0_0_0_0_12082026_155308_ncells_6000_B_cell_activation_rep_0_08a5aeab.csv", 10),
    "EMT": (f"{R}/analysis_data/paper_analysis/EMT/simulate/20260825_224653/df_rows_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_25082026_234904_ncells_6000_EMT_rep_0_839e1156.csv", 17),
    "Pluripotent": (f"{R}/analysis_data/paper_analysis/Pluripotent/simulate/20260825_224511/df_rows_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_26082026_021918_ncells_6000_Pluripotent_rep_0_eb8c3e1d.csv", 36),
}

def best_k_clusters(X, kmax=8):
    bics = []
    for k in range(1, kmax + 1):
        gm = GaussianMixture(n_components=k, n_init=2, random_state=0, reg_covar=1e-3).fit(X)
        bics.append(gm.bic(X))
    k = int(np.argmin(bics)) + 1
    labels = GaussianMixture(n_components=k, n_init=2, random_state=0, reg_covar=1e-3).fit_predict(X)
    sizes = pd.Series(labels).value_counts()
    real_k = int((sizes / len(labels) > 0.02).sum())
    return k, real_k

for net, (f, ng) in FILES.items():
    cols = ["cell_id", "time_step"] + [f"gene_{i+1}_protein" for i in range(ng)]
    df = pd.read_csv(f, usecols=cols)
    tmax = df.time_step.max()
    checkpoints = sorted(set(int(round(tmax * frac)) for frac in [0.25, 0.5, 0.75, 1.0]))
    row = [net]
    for t in checkpoints:
        sub = df[df.time_step == t].drop_duplicates("cell_id")
        if len(sub) < 50:
            row.append("n/a"); continue
        X = StandardScaler().fit_transform(np.log1p(sub[[f"gene_{i+1}_protein" for i in range(ng)]].to_numpy()))
        bick, realk = best_k_clusters(X)
        row.append(f"t={t}:k={bick}(real={realk})")
    print(" | ".join(row), flush=True)
