"""
Scores the 150x7 BEELINE rankedEdges.csv results against ground truth,
producing beeline_analysis_output.csv for the 20260824 rerun.

Reuses score_beeline_sweep_folder and its dependencies verbatim from
benchmark_network_sweep.ipynb (extracted, not retyped). Our BEELINE directory
structure (inputs/network_sweep_final_20260824, beeline_inference/<dataset>/
simrep{N}_twin_paired/<algo>/rankedEdges.csv) matches the notebook's own
expected layout exactly, since it was built with the same converter
conventions -- so score_beeline_sweep_folder needs no adaptation.
"""
import sys
sys.path.insert(0, "/tmp")

from pathlib import Path
import beeline_scoring_extract as bse

BEELINE_OUTPUT_ROOT = Path("/home/gzu5140/TwINFER_KA/analysis_data/synthetic_network_benchmark_20260824/beeline_inference")
OUT_CSV = "/home/gzu5140/TwINFER_KA/analysis_data/synthetic_network_benchmark_20260824/beeline_analysis_output.csv"

if __name__ == "__main__":
    df = bse.score_beeline_sweep_folder(BEELINE_OUTPUT_ROOT, verbose=False)
    df.to_csv(OUT_CSV, index=False)
    print(f"Scored {len(df)} (dataset, run, algorithm) rows -> {OUT_CSV}")
    print(df.groupby(["dataset_id", "algorithm"]).size().unstack())
