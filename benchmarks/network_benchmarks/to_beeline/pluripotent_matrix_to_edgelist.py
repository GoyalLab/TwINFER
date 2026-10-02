from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import numpy as np, sys, csv

PLURIPOTENT_GENES = ["ARID3B","CEBPZ","ETV4","FOXH1","HIC2","HMGB3","JARID2","LIN28B",
    "MIS18BP1","MYCN","NANOG","POU5F1","POU5F1B","PRDM14","REST","SALL2","SALL4",
    "SMARCC1","SOX2","TEAD2","TERF1","TGIF1","WDHD1","ZBTB12","ZBTB39","ZNF281",
    "ZNF286A","ZNF286B","ZNF322","ZNF398","ZNF462","ZNF730","ZNF90","ZNF92","ZSCAN10","ZSCAN2"]

def matrix_to_edges(matrix_path, gene_names, out_csv):
    mat = np.loadtxt(matrix_path, delimiter=',')
    n = len(gene_names)
    assert mat.shape == (n, n), f"{matrix_path}: shape {mat.shape} != ({n},{n})"
    rows = []
    for i in range(n):
        for j in range(n):
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

if __name__ == '__main__':
    base = sys.argv[1]
    matrix_to_edges(f"{base}/input_data/real_world_networks/Pluripotent.txt", PLURIPOTENT_GENES,
                     f'{TWINFER_PROJECT_ROOT}/clean_data/benchmarks/network_benchmarks/to_beeline/Pluripotent_real_edges.csv')
