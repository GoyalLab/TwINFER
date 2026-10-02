from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_a2a3302f_2026-09-28_why_twinfer_fails; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import sys
import glob
import numpy as np
import pandas as pd

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/paper_analysis/larry_hematopoiesis_validation")
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/work_in_progress/benchmark")
from benchmarks.network_benchmarks.analysis_plots import summary_table_todo4v2 as S

gt_path = f'{TWINFER_PROJECT_ROOT}/input_data/real_world_networks/EMT.txt'
M = np.loadtxt(gt_path, delimiter=",")

files = sorted(glob.glob(f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs/EMT_rep_*_all_results.json'))
print(f"scoring {len(files)} completed EMT replicates (today's run):")

aurpcs = []
for f in files:
    import json
    d = json.load(open(f))
    gene_names = d["gene_names"]
    n = len(gene_names)
    recs = [(gene_names[i], gene_names[j], "+" if M[i, j] > 0 else "-")
            for i in range(n) for j in range(n) if i != j and M[i, j] != 0]
    gt_df = pd.DataFrame(recs, columns=["Gene1", "Gene2", "Type"])
    row = S.todo4v2_row_for_dataset(f, gt_df)
    if row is None:
        print(f"  {f.split('/')[-1]}: SKIPPED (no valid twin_score_inputs)")
        continue
    print(f"  {f.split('/')[-1]}: auprc={row['auprc']:.4f}")
    aurpcs.append(row["auprc"])

print(f"\nmean AUPRC over {len(aurpcs)} completed replicates: {np.mean(aurpcs):.4f}")
