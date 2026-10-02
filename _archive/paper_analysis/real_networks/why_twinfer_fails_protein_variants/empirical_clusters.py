from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_a2a3302f_2026-09-28_why_twinfer_fails; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import glob, numpy as np, pandas as pd, warnings
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

R = f'{TWINFER_PROJECT_ROOT}'
FILES = {
    "mCAD": f"{R}/simulation_data/real_data/df_rows_0_0_0_0_0_12072026_005933_ncells_6000_mCAD_1_0_c733c5af.csv",
    "GSD": f"{R}/simulation_data/real_data/df_rows_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_12072026_062944_ncells_6000_GSD_0_0_117c4801.csv",
    "HSC": f"{R}/simulation_data/real_data/df_rows_0_0_0_0_0_0_0_0_0_0_0_12072026_020848_ncells_6000_HSC_balanced_0_0_6bff4482.csv",
    "VSC": f"{R}/simulation_data/real_data/df_rows_0_0_0_0_0_0_0_0_12072026_005406_ncells_6000_VSC_2_0_2befd3f9.csv",
    "B_cell_activation": f"{R}/analysis_data/paper_analysis/real_networks/beeline_inference/../../B_cell_activation/simulate/20260812_153622/df_rows_0_0_0_0_0_0_0_0_0_0_12082026_155308_ncells_6000_B_cell_activation_rep_0_08a5aeab.csv",
    "EMT": f"{R}/analysis_data/paper_analysis/real_networks/beeline_inference/../../EMT/simulate/20260825_224653/df_rows_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_25082026_234904_ncells_6000_EMT_rep_0_839e1156.csv",
    "Pluripotent": f"{R}/analysis_data/paper_analysis/real_networks/beeline_inference/../../Pluripotent/simulate/20260825_224511/df_rows_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_26082026_021918_ncells_6000_Pluripotent_rep_0_eb8c3e1d.csv",
}
N_GENES = {"mCAD": 5, "GSD": 19, "HSC": 11, "VSC": 8, "B_cell_activation": 10, "EMT": 17, "Pluripotent": 36}

boolean_fp = {"mCAD": 2, "GSD": 3, "HSC": 6, "VSC": 5, "B_cell_activation": 6, "EMT": 0, "Pluripotent": "N/A (approx)"}
meanfield_fp = {"mCAD": 2, "GSD": 2, "HSC": 3, "VSC": 5, "B_cell_activation": "not computed", "EMT": "not computed", "Pluripotent": "not computed"}

results = []
for net, f in FILES.items():
    ng = N_GENES[net]
    cols = ["cell_id", "time_step"] + [f"gene_{i+1}_protein" for i in range(ng)]
    df = pd.read_csv(f, usecols=cols)
    last = df[df.time_step == df.time_step.max()].drop_duplicates("cell_id")
    X = StandardScaler().fit_transform(np.log1p(last[[f"gene_{i+1}_protein" for i in range(ng)]].to_numpy()))
    bics = []
    for k in range(1, 9):
        gm = GaussianMixture(n_components=k, n_init=3, random_state=0, reg_covar=1e-3).fit(X)
        bics.append(gm.bic(X))
    best_k = int(np.argmin(bics)) + 1
    labels = GaussianMixture(n_components=best_k, n_init=3, random_state=0, reg_covar=1e-3).fit_predict(X)
    sizes = pd.Series(labels).value_counts().sort_values(ascending=False)
    # merge near-duplicate/tiny clusters (<2% of cells) into "noise", report the real count
    real_clusters = int((sizes / len(labels) > 0.02).sum())
    results.append(dict(net=net, n_cells=len(last), gmm_bic_k=best_k, real_clusters_2pct=real_clusters,
                         cluster_sizes=sizes.head(8).to_dict(), boolean_fp=boolean_fp[net], meanfield_fp=meanfield_fp[net]))
    print(f"{net}: n_cells={len(last)}, GMM-BIC-best-k={best_k}, clusters>2%={real_clusters}, "
          f"sizes={dict(sizes.head(6))}, boolean_fp={boolean_fp[net]}, meanfield_fp={meanfield_fp[net]}", flush=True)

pd.DataFrame(results).to_csv(f"{__import__('os').path.dirname(__import__('os').path.abspath(__file__))}/empirical_clusters.csv", index=False)
print("saved")
