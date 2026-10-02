from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import warnings, time, sys, json
warnings.filterwarnings("ignore")
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/package")
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/synthetic_network_analysis")
from twinfer.inference.infer import infer_with_twinfer
from benchmarks.network_benchmarks.score.score_multistate_benchmark import true_edges_from_topo, auprc_from_scored_pairs, load_twinfer_scores
from benchmarks.network_benchmarks.score.score_original_benchmark_twinscore import f1_topk

INPUT_DATA = f'{TWINFER_PROJECT_ROOT}/input_data'
CASES = {
    "HSC_multistate": f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/HSC/simulate/multistate_2000steps/df_rows_0_0_0_0_0_0_0_0_0_0_0_04092026_004008_ncells_6000_HSC_multistate_rep_0_2e5177df.csv',
    "HSC_seeded": f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/HSC/simulate/multistate_2000steps_seeded/df_rows_0_0_0_0_0_0_0_0_0_0_0_04092026_003703_ncells_6000_HSC_seeded_rep_0_cdfd31ed.csv',
}
SIM_TIME = {"HSC_multistate": 2000, "HSC_seeded": 800}

topo = f"{INPUT_DATA}/real_world_networks/HSC.txt"
true_edges, possible_edges = true_edges_from_topo(topo)
print(f"{len(true_edges)} true / {len(possible_edges)} possible directed edges\n")

def load_twinscore(results):
    red = results["ranked_edges"]
    import pandas as pd, numpy as np
    return {(r.gene_1, r.gene_2): abs(float(r.twinScore)) for r in red.itertuples() if pd.notna(r.twinScore)}

def load_unfiltered(results):
    import pandas as pd, numpy as np
    mat = results["direction"]["unfiltered_matrix"]
    scores = {}
    for g1 in mat.index:
        for g2 in mat.columns:
            if g1 == g2: continue
            v = mat.loc[g1, g2]
            if pd.notna(v):
                scores[(g1,g2)] = abs(float(v))
    return scores

for label, path in CASES.items():
    base_config = {
        "n_cells": 6000, "simulation_time_before_division": SIM_TIME[label],
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
        t0 = time.time()
        results = infer_with_twinfer(
            path, merge_to_multiple_states=False, base_config=base_config, t1=1, t2=20,
            match_sim_details=False, check_for_steady_state=False, seed=101010, n_cores=15,
            ranked_list=True, **kwargs,
        )
        elapsed = time.time() - t0
        tw = load_twinscore(results)
        unf = load_unfiltered(results)
        auprc_tw = auprc_from_scored_pairs(tw, true_edges, possible_edges)
        auprc_unf = auprc_from_scored_pairs(unf, true_edges, possible_edges)
        f1_tw = f1_topk(tw, true_edges, possible_edges)
        print(f"  [{mode:9s}] elapsed={elapsed:6.1f}s  twinScore coverage={len(tw)}/{len(possible_edges)} "
              f"({len(tw)/len(possible_edges):.1%})  AUPRC(twinScore)={auprc_tw:.4f}  F1(twinScore)={f1_tw:.4f}  "
              f"AUPRC(unfiltered_matrix)={auprc_unf:.4f}  coverage(unfiltered)={len(unf)}/{len(possible_edges)}")
    print()
