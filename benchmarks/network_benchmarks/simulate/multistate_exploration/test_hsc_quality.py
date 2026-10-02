from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import warnings, time, sys
warnings.filterwarnings("ignore")
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/package")
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/synthetic_network_analysis")
import pandas as pd, numpy as np
from twinfer.inference.infer import infer_with_twinfer
from benchmarks.network_benchmarks.score.score_multistate_benchmark import true_edges_from_topo, auprc_from_scored_pairs
from benchmarks.network_benchmarks.score.score_original_benchmark_twinscore import f1_topk

INPUT_DATA = f'{TWINFER_PROJECT_ROOT}/input_data'
CASES = {
    "HSC_multistate": (f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/HSC/simulate/multistate_2000steps/df_rows_0_0_0_0_0_0_0_0_0_0_0_04092026_004008_ncells_6000_HSC_multistate_rep_0_2e5177df.csv', 2000),
    "HSC_seeded": (f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/HSC/simulate/multistate_2000steps_seeded/df_rows_0_0_0_0_0_0_0_0_0_0_0_04092026_003703_ncells_6000_HSC_seeded_rep_0_cdfd31ed.csv', 800),
}

topo = f"{INPUT_DATA}/real_world_networks/HSC.txt"
true_edges, possible_edges = true_edges_from_topo(topo)

for label, (path, simtime) in CASES.items():
    base_config = {
        "n_cells": 6000, "simulation_time_before_division": simtime,
        "twin_simulation_time_after_division": 48, "twin_measurement_resolution": 1,
        "path_to_connectivity_matrix": topo,
        "param_csv": f"{INPUT_DATA}/network_sweep/parameters.csv",
        "rows_to_use": [[0]*11], "type": "HSC",
    }
    print(f"===== {label} =====")
    for mode in ["filtered", "nofilter"]:
        kwargs = dict(z_score_threshold_two_states=12) if mode == "filtered" else dict(
            alpha_gene_gene_corr=0.999999, alpha_stage3=0.999999,
            z_score_threshold_two_states=0, z_score_threshold_cross_correlation=0,
            corr_threshold_cross_correlation=0, fan_out_z_score_threshold=0,
        )
        results = infer_with_twinfer(
            path, merge_to_multiple_states=False, base_config=base_config, t1=1, t2=20,
            match_sim_details=False, check_for_steady_state=False, seed=101010, n_cores=15,
            ranked_list=True, **kwargs,
        )
        red = results["ranked_edges"]
        # hard classification quality: which pairs TwINFER actually CALLS an edge
        called = {(r.gene_1, r.gene_2) for r in red.itertuples() if bool(r.is_final_directed_edge)}
        tp = len(called & true_edges)
        fp = len(called - true_edges)
        fn = len(true_edges - called)
        precision = tp/len(called) if called else 0.0
        recall = tp/len(true_edges)
        f1 = 0 if precision+recall==0 else 2*precision*recall/(precision+recall)
        tw = {(r.gene_1, r.gene_2): abs(float(r.twinScore)) for r in red.itertuples() if pd.notna(r.twinScore)}
        auprc = auprc_from_scored_pairs(tw, true_edges, possible_edges)
        f1_rank = f1_topk(tw, true_edges, possible_edges)
        print(f"  [{mode:9s}] is_final_directed_edge CALLS: {len(called)} called, TP={tp} FP={fp} FN={fn} "
              f"precision={precision:.3f} recall={recall:.3f} F1(hard-call)={f1:.3f}  ||  "
              f"AUPRC(rank)={auprc:.3f} F1(topk-rank)={f1_rank:.3f} coverage={len(tw)}/{len(possible_edges)}")
    print()
