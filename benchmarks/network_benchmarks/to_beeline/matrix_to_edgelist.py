from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import numpy as np, sys, csv

def matrix_to_edges(matrix_path, gene_names, out_csv):
    mat = np.loadtxt(matrix_path, delimiter=',')
    n = len(gene_names)
    assert mat.shape == (n, n), f"{matrix_path}: shape {mat.shape} != ({n},{n})"
    rows = []
    for i in range(n):       # source
        for j in range(n):   # target
            v = mat[i, j]
            if v > 0:
                rows.append((gene_names[i], gene_names[j], '+'))
            elif v < 0:
                rows.append((gene_names[i], gene_names[j], '-'))
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Source', 'Target', 'Type'])
        w.writerows(rows)
    print(f"{matrix_path}: {n} genes, {len(rows)} edges -> {out_csv}")

B_CELL_GENES = ["Ikaros","PU.1","Flk2","IL-7R","GATA-1","E2A","EBF","C/EBPa","PAX5","Notch-1"]
EMT_GENES = ["Cdh1","Cldn7","Foxc2","Grhl2","Gsc","Klf8","Np63a","Ovol2","Snai1",
             "Snai2","Tcf3","Tgfbeta","Twist1","Twist2","Vim","Zeb1","Zeb2"]

if __name__ == '__main__':
    base = sys.argv[1]
    matrix_to_edges(f"{base}/input_data/real_world_networks/B_cell.txt", B_CELL_GENES,
                     f'{TWINFER_PROJECT_ROOT}/clean_data/benchmarks/network_benchmarks/to_beeline/B_cell_activation_edges.csv')
    matrix_to_edges(f"{base}/input_data/real_world_networks/EMT.txt", EMT_GENES,
                     f'{TWINFER_PROJECT_ROOT}/clean_data/benchmarks/network_benchmarks/to_beeline/EMT_edges.csv')
