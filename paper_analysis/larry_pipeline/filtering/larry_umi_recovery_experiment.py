#!/usr/bin/env python3
"""Experiment: recover low-read-count LARRY barcode observations that the main
pipeline's N_READS>=10 filter drops, by Hamming-matching them against the
11,986 already-clustered barcodes (fixed anchors) instead of re-clustering
everything from scratch (551,223 distinct raw barcodes at N_READS>=1 makes a
full pairwise re-cluster ~152 billion comparisons -- infeasible; matching
candidates against the existing anchor set is ~6 billion, tractable when
vectorized with numpy instead of done pairwise in Python).

Discard-before-correct (current pipeline: drop weak reads, then Hamming-merge
the survivors) throws away real UMI evidence that Hamming correction was
designed to recover: a barcode seen on only 1-2 reads is often a sequencing-
error variant of an already well-supported barcode, not independent noise.
This script does correct-before-discard for those specific dropped
observations, without redoing the full O(n^2) clustering.

Outputs, alongside the existing processed/ directory:
  processed/umi_table_recovered.csv  -- original umi_table.csv rows + recovered rows
  processed/recovery_summary.json
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, str(Path(__file__).parent))
from paper_analysis.larry_pipeline.build_matrix.build_larry_matrix import (  # noqa: E402
    DATASET_DIR, OUT_DIR, FASTQ_PATH, LSK_LIBRARIES, N_HAMMING,
    load_real_barcode_map,
)

BARCODE_LEN = 29  # the vectorized path only handles this length; PDF/LARRY_sorting_and_filtering.py's
                   # own validity check requires it, and it's the overwhelming majority of reads.


def run_count_pass_unfiltered():
    print("[recovery] streaming count pass over the LARRY fastq (awk, no read-depth floor)...", flush=True)
    awk_prog = r"""
{
  if (substr($0,1,1) == ">") {
    n = split(substr($0,2), h, ",")
    if (n == 3) { lib=h[1]; cell=h[2]; umi=h[3]; want_seq=1 } else { want_seq=0 }
    next
  }
  if (want_seq && $0 != "") {
    count[lib "\t" cell "\t" umi "\t" $0]++
    want_seq = 0
    next
  }
  want_seq = 0
}
END {
  for (k in count) print k "\t" count[k]
}
"""
    cmd = f"zcat {FASTQ_PATH} | gawk '{awk_prog}'"
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
    rows = [line.split("\t") for line in proc.stdout.splitlines() if line]
    df = pd.DataFrame(rows, columns=["library", "cell", "umi", "gfp_bc", "n_reads"])
    df["n_reads"] = df["n_reads"].astype(int)
    return df


def vectorized_hamming_match(candidates, anchors, threshold):
    """For each candidate barcode (length BARCODE_LEN), find the closest anchor
    within Hamming<=threshold (ties broken by the caller via anchor order/abundance).
    Returns a dict candidate -> matched_anchor for every candidate that has one.
    """
    anchor_arr = np.array([[ord(c) for c in a] for a in anchors], dtype=np.uint8)  # (A, L)
    match = {}
    chunk = 800
    for i in range(0, len(candidates), chunk):
        batch = candidates[i:i + chunk]
        if i % (chunk * 20) == 0:
            print(f"[recovery] matched {i}/{len(candidates)} candidates against {len(anchors)} anchors", flush=True)
        batch_arr = np.array([[ord(c) for c in b] for b in batch], dtype=np.uint8)  # (C, L)
        dist = (batch_arr[:, None, :] != anchor_arr[None, :, :]).sum(axis=2)  # (C, A)
        best_idx = dist.argmin(axis=1)
        best_dist = dist[np.arange(len(batch)), best_idx]
        for bc, idx, d in zip(batch, best_idx, best_dist):
            if d <= threshold:
                match[bc] = anchors[idx]
    return match


def main():
    reads = run_count_pass_unfiltered()
    print(f"[recovery] {len(reads)} raw (library,cell,umi,barcode) combos at N_READS>=1", flush=True)

    reads = reads[reads["library"].isin(LSK_LIBRARIES)].reset_index(drop=True)
    for lib in sorted(reads["library"].unique()):
        real_bc = load_real_barcode_map(lib)
        m = reads["library"] == lib
        reads.loc[m, "cell"] = reads.loc[m, "cell"].map(real_bc)
    print(f"[recovery] {len(reads)} combos after restricting to LSK libraries", flush=True)

    trans_map_df = pd.read_csv(OUT_DIR / "barcode_map_transitive.csv")
    existing_barcodes = set(trans_map_df["raw_barcode"])
    trans_map = trans_map_df.set_index("raw_barcode")["representative"].to_dict()
    print(f"[recovery] {len(existing_barcodes)} barcodes already clustered (fixed anchors)", flush=True)

    all_raw = reads["gfp_bc"].unique()
    candidates = [b for b in all_raw if b not in existing_barcodes]
    right_len = [b for b in candidates if len(b) == BARCODE_LEN]
    wrong_len = len(candidates) - len(right_len)
    print(f"[recovery] {len(candidates)} candidate barcodes not already clustered "
          f"({wrong_len} of non-{BARCODE_LEN} length, excluded from vectorized matching)", flush=True)

    abundance = reads.drop_duplicates(["library", "cell", "umi", "gfp_bc"]).groupby("gfp_bc").size()
    anchors_sorted = sorted(existing_barcodes, key=lambda b: -abundance.get(b, 0))  # ties -> highest-abundance anchor first
    anchors_29 = [a for a in anchors_sorted if len(a) == BARCODE_LEN]

    recovered_map = vectorized_hamming_match(right_len, anchors_29, N_HAMMING)
    print(f"[recovery] {len(recovered_map)} / {len(right_len)} candidates matched an existing "
          f"barcode within Hamming<={N_HAMMING}", flush=True)

    full_map = dict(trans_map)
    for bc, matched_anchor in recovered_map.items():
        full_map[bc] = trans_map[matched_anchor]

    reads_kept = reads[reads["gfp_bc"].isin(full_map)].copy()
    reads_kept["clone_bc"] = reads_kept["gfp_bc"].map(full_map)
    n_dropped_combos = len(reads) - len(reads_kept)
    print(f"[recovery] {len(reads_kept)}/{len(reads)} combos kept after recovery "
          f"({n_dropped_combos} combos have a barcode matching nothing, still dropped)", flush=True)

    umi_table_recovered = (
        reads_kept.drop_duplicates(["library", "cell", "umi", "clone_bc"])
        [["cell", "clone_bc", "library"]]
        .rename(columns={"cell": "cellID", "clone_bc": "barcode", "library": "sample"})
        .reset_index(drop=True)
    )
    umi_table_recovered.to_csv(OUT_DIR / "umi_table_recovered.csv", index=False)

    original = pd.read_csv(OUT_DIR / "umi_table.csv")
    summary = {
        "n_reads_threshold_original": 10,
        "n_combos_at_nreads_1_lsk_only": int(len(reads)),
        "n_candidate_barcodes_not_already_clustered": len(candidates),
        "n_candidates_wrong_length_excluded": wrong_len,
        "n_candidates_recovered_via_hamming_match": len(recovered_map),
        "n_combos_dropped_no_match": int(n_dropped_combos),
        "n_rows_umi_table_original": int(len(original)),
        "n_rows_umi_table_recovered": int(len(umi_table_recovered)),
    }
    (OUT_DIR / "recovery_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())
