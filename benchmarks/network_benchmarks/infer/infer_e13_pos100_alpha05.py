"""
One-off rerun of infer_e13_pos100.py with alpha_gene_gene_corr=0.05 (instead
of the package default 0.01), to test whether loosening Stage 1's
significance threshold recovers real signal on e13_pos100 -- see the
Stage-1-power investigation in this session (correlations here are real but
marginal: z~2.1-2.2 vs the alpha=0.01 critical value of 2.576; alpha=0.05's
critical value of ~1.96 should clear several of them).

Output: analysis_data/network_sweep_final/e13_pos100/twinfer_inference_alpha05/
(kept separate from the alpha=0.01 results, not overwriting them).

    python infer_e13_pos100_alpha05.py
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import sys
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, ".")
from benchmarks.network_benchmarks.infer import infer_e13_pos100 as base

base.OUTPUT_DIR = f'{TWINFER_PROJECT_ROOT}/analysis_data/network_sweep_final/e13_pos100/twinfer_inference_alpha05'
base.BASE_CONFIG["output_folder"] = f"{base.OUTPUT_DIR}/_scratch/"
base.BASE_CONFIG["log_file"] = f"{base.OUTPUT_DIR}/_scratch/logs/e13_pos100.jsonl"

_orig_run_one = base.run_one


def run_one(label, sim_rep, source_path, n_cores):
    import os
    out_path = os.path.join(base.OUTPUT_DIR, f"{label}_rep{sim_rep}_all_results.json")
    if os.path.exists(out_path):
        return out_path, "skipped (already exists)"

    matrix_path = base.LABEL_TO_GT[label]
    base_config = dict(base.BASE_CONFIG)
    base_config["path_to_connectivity_matrix"] = matrix_path

    results = base.infer_with_twinfer(
        source_path,
        merge_to_multiple_states=False,
        base_config=base_config,
        t1=base.T1, t2=base.T2,
        check_for_steady_state=False,
        match_sim_details=False,
        seed=101010,
        n_cores=n_cores,
        z_score_threshold_two_states=4.501,
        ranked_list=True,
        separate_fan_outs_from_mutual_regulation_flag=True,
        fan_out_z_score_threshold=3.2,
        use_scramble_cross_correlation=False,
        alpha_gene_gene_corr=0.05,
    )

    gene_names = [f"gene_{i+1}" for i in range(base.N_GENES)]
    record = {
        "dataset_id": label, "label": sim_rep, "source_file": source_path,
        "gene_names": gene_names, "n_genes": base.N_GENES,
        "ground_truth_matrix": matrix_path, "alpha_gene_gene_corr": 0.05,
        **base.make_json_safe(results),
    }
    import os as _os, json as _json
    _os.makedirs(base.OUTPUT_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        _json.dump(record, f, cls=base.NumpyEncoder, indent=2)
    return out_path, "computed"


base.run_one = run_one

if __name__ == "__main__":
    import os
    os.makedirs(f"{base.OUTPUT_DIR}/_scratch/logs", exist_ok=True)
    tasks = base.discover_tasks()
    print(f"{len(tasks)} simulation(s) discovered", flush=True)
    import time
    for i, (label, k, src) in enumerate(tasks):
        t0 = time.time()
        path, status = run_one(label, k, src, 8)
        print(f"[{i+1}/{len(tasks)}] {label} rep {k}: {status} ({time.time()-t0:.1f}s)", flush=True)
