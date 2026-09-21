"""t1=10 variant of signed_scoring_analysis.py -- glob-path change only, reuses run()/
score_one_signed()/competitor_signed_summary() unchanged. Part 6 follow-up."""
import pandas as pd

import signed_scoring_analysis as S

ROOT = S.ROOT
HERE = S.HERE


def main():
    S.run("network_sweep_final_t1_10",
          f"{ROOT}/analysis_data/network_sweep_final/twinfer_inference_allpairs_t1_10",
          lambda d: d.get("ground_truth_matrix"))

    S.run("mixed_network_sweep_t1_10",
          f"{ROOT}/analysis_data/mixed_network_sweep/twinfer_inference_allpairs_t1_10",
          lambda d: f"{ROOT}/input_data/mixed_network_sweep/{d.get('dataset_id')}.txt",
          beeline_csv=f"{ROOT}/analysis_data/mixed_network_sweep/beeline_analysis_output_t1_10.csv")

    TOPO_MAP = {"GSD": "GSD.txt", "HSC": "HSC.txt", "VSC": "VSC.txt", "mCAD": "mCAD.txt",
                "B_cell_activation": "B_cell.txt", "Circadian_cycle": "circadian.txt",
                "EMT": "EMT.txt", "Pluripotent": "Pluripotent.txt"}
    S.run("real_networks_t1_10",
          f"{ROOT}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs_t1_10",
          lambda d: f"{ROOT}/input_data/real_world_networks/{TOPO_MAP.get(d.get('sim_type'), '')}")


if __name__ == "__main__":
    main()
