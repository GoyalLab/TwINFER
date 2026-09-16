#!/usr/bin/env python3
"""Build the combined LARRY cell x gene matrix with lineage-barcode (clone) assignment.

Inputs (already produced by upstream inDrops quantify + LARRY Step 1):
  /scratch/gzu5140/ka_twinfer/larry_dataset/LSK_*.zip                (15 per-library count matrices)
  /scratch/gzu5140/ka_twinfer/larry_dataset/LARRY_sorted_and_filtered_barcodes.fastq.gz

Outputs:
  /scratch/gzu5140/ka_twinfer/larry_dataset/processed/larry_combined_counts.mtx
  /scratch/gzu5140/ka_twinfer/larry_dataset/processed/genes.txt
  /scratch/gzu5140/ka_twinfer/larry_dataset/processed/cells.txt
  /scratch/gzu5140/ka_twinfer/larry_dataset/processed/obs_metadata.csv (library per cell --
      no clone assignment here; that's singletCode's own decision, completed with
      singletcode_cross_sample_rescue.apply_cross_sample_rescue, run downstream on this
      script's umi_table.csv output by build_final_matrix_umi3.py)
  /scratch/gzu5140/ka_twinfer/larry_dataset/processed/run_summary.json
"""
import json
import pickle
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

DATASET_DIR = Path("/scratch/gzu5140/ka_twinfer/larry_dataset")
OUT_DIR = DATASET_DIR / "processed"
FASTQ_PATH = DATASET_DIR / "LARRY_sorted_and_filtered_barcodes.fastq.gz"

LSK_LIBRARIES = [
    "LSK_d2_1", "LSK_d2_2", "LSK_d2_3",
    "LSK_d4_1_1", "LSK_d4_1_2", "LSK_d4_1_3",
    "LSK_d4_2_1", "LSK_d4_2_2", "LSK_d4_2_3",
    "LSK_d6_1_1", "LSK_d6_1_2", "LSK_d6_1_3",
    "LSK_d6_2_1", "LSK_d6_2_2", "LSK_d6_2_3",
]

N_READS = 10
N_UMIS = 3
N_HAMMING = 3


def load_real_barcode_map(lib: str) -> dict:
    # abundant_barcodes.pickle maps the REAL ACGT-ACGT gel barcode to the
    # inDrops-internal short "bcXXXX" hash used everywhere else (matrix row
    # names, fastq cell field). The hash is assigned independently per
    # library, so it is not a stable cell identifier across libraries -- the
    # real barcode is. Confirmed 1:1 (0 collisions) for all 15 libraries.
    with zipfile.ZipFile(DATASET_DIR / f"{lib}.zip") as zf:
        with zf.open("abundant_barcodes.pickle") as f:
            d = pickle.load(f)
    return {bc: acgt for acgt, (bc, cnt) in d.items()}


def load_library_sparse(lib: str):
    # Decompress via unzip|zcat (native tools) rather than Python's zipfile/gzip
    # modules, and build the sparse matrix by streaming one row at a time
    # instead of materializing a dense 11k-25k-cell pandas DataFrame: this
    # environment's session memcg killed an earlier pandas-dtype-dict attempt
    # at ~2.7GB RSS on a single library. Counts are ~2.8% dense, so per-library
    # nonzeros are small (~8M) even though the dense shape is not.
    cmd = f"unzip -p {DATASET_DIR}/{lib}.zip {lib}.counts.tsv.gz | zcat"
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, text=True)
    header = proc.stdout.readline().rstrip("\n").split("\t")
    genes = header[1:]
    rows, cols, data, barcodes = [], [], [], []
    for i, line in enumerate(proc.stdout):
        parts = line.rstrip("\n").split("\t")
        barcodes.append(parts[0])
        vals = np.array(parts[1:], dtype=np.int32)
        nz = np.flatnonzero(vals)
        rows.append(np.full(nz.shape, i, dtype=np.int32))
        cols.append(nz.astype(np.int32))
        data.append(vals[nz])
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"decompression of {lib} failed with exit code {ret}")
    rows = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int32)
    cols = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int32)
    data = np.concatenate(data) if data else np.zeros(0, dtype=np.int32)
    mat = sp.csr_matrix((data, (rows, cols)), shape=(len(barcodes), len(genes)))
    return mat, genes, barcodes


def build_combined_matrix():
    gene_header = None
    blocks = []
    cell_ids = []
    library_col = []
    per_library_counts = {}
    for lib in LSK_LIBRARIES:
        print(f"[matrix] loading {lib}", flush=True)
        mat, genes, barcodes = load_library_sparse(lib)
        if gene_header is None:
            gene_header = genes
        elif genes != gene_header:
            raise RuntimeError(f"{lib}: gene header does not match {LSK_LIBRARIES[0]}'s")
        blocks.append(mat)
        real_bc = load_real_barcode_map(lib)
        barcodes = [real_bc[bc] for bc in barcodes]  # bcXXXX hash -> real ACGT-ACGT barcode
        cell_ids.extend(f"{lib}:{bc}" for bc in barcodes)
        library_col.extend([lib] * len(barcodes))
        per_library_counts[lib] = len(barcodes)
        print(f"[matrix] {lib}: {len(barcodes)} cells, {mat.nnz} nonzeros", flush=True)
    X = sp.vstack(blocks, format="csr")
    obs = pd.DataFrame({"library": library_col}, index=pd.Index(cell_ids, name="cell_id"))
    return X, gene_header, obs, per_library_counts


# --- LARRY Step 2: barcode/clone extraction, exact port of clonal_annotation.ipynb ---
# Cell counting (notebook cell 4+5) is done in awk in one streaming pass instead of a
# Python dict-of-tuples loop: the fastq is ~1.8GB gzip'd and a native Python loop over
# every read would be far slower and hold the full unfiltered table in memory. awk's
# associative array does the identical group-count-and-threshold operation natively.
COUNT_AWK = r"""
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
  for (k in count) if (count[k] >= NREADS) print k "\t" count[k]
}
"""


def run_count_pass():
    print("[step2] streaming count pass over the LARRY fastq (awk)...", flush=True)
    cmd = f"zcat {FASTQ_PATH} | gawk -v NREADS={N_READS} '{COUNT_AWK}'"
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
    rows = [line.split("\t") for line in proc.stdout.splitlines() if line]
    df = pd.DataFrame(rows, columns=["library", "cell", "umi", "gfp_bc", "n_reads"])
    df["n_reads"] = df["n_reads"].astype(int)
    return df


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def collapse_barcodes_transitive(all_bcs, abundance):
    # Connected components of the full Hamming<=N_HAMMING graph (union-find),
    # each component collapsing to its highest-total-UMI-count member.
    n = len(all_bcs)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        if i > 0 and i % 500 == 0:
            print(f"[step2] transitive: compared {i}/{n} distinct barcodes", flush=True)
        bc1 = all_bcs[i]
        for j in range(i + 1, n):
            if hamming(bc1, all_bcs[j]) <= N_HAMMING:
                union(i, j)

    components = {}
    for i in range(n):
        components.setdefault(find(i), []).append(i)

    bc_map = {}
    for members in components.values():
        rep_idx = max(members, key=lambda idx: (abundance.get(all_bcs[idx], 0), all_bcs[idx]))
        rep = all_bcs[rep_idx]
        for idx in members:
            bc_map[all_bcs[idx]] = rep
    return bc_map


def extract_clones():
    #Count the number of reads per (library,cell,umi,barcode) combo, keep those with >= N_READS
    counts_filtered = run_count_pass()
    print(f"[step2] {len(counts_filtered)} (library,cell,umi,barcode) combos pass N_READS>={N_READS}", flush=True)

    lsk_mask = counts_filtered["library"].isin(LSK_LIBRARIES)
    n_lk_dropped = int((~lsk_mask).sum())
    lk_libraries_seen = sorted(set(counts_filtered.loc[~lsk_mask, "library"]))
    print(f"[step2] dropping {n_lk_dropped} combos from non-LSK libraries: {lk_libraries_seen}", flush=True)
    counts_filtered = counts_filtered.loc[lsk_mask].reset_index(drop=True)

    # relabel the fastq's per-library bcXXXX hash to the real ACGT-ACGT barcode,
    # same mapping and reasoning as build_combined_matrix's cell IDs, so this
    # table joins directly onto obs (keyed on real barcodes) with no separate
    # relabeling pass needed afterwards.
    for lib in sorted(counts_filtered["library"].unique()):
        real_bc = load_real_barcode_map(lib)
        lib_mask = counts_filtered["library"] == lib
        counts_filtered.loc[lib_mask, "cell"] = counts_filtered.loc[lib_mask, "cell"].map(real_bc)
    n_unmapped = int(counts_filtered["cell"].isna().sum())
    if n_unmapped:
        raise RuntimeError(
            f"{n_unmapped} fastq (library,cell) combos have a bcXXXX hash absent from that "
            "library's abundant_barcodes.pickle -- LARRY_sorting_and_filtering.py's own "
            "validity check is supposed to guarantee membership, so this means that "
            "assumption doesn't hold here and needs investigating before trusting the join."
        )

    # Persisted *before* any collapsing, so the preprocessing notebook can
    # compare collapsing methods (or try others) without re-running the
    # expensive awk pass over the raw fastq.
    dedup_reads = counts_filtered.drop_duplicates(["library", "cell", "umi", "gfp_bc"])
    dedup_reads.to_csv(OUT_DIR / "reads_precollapse.csv", index=False)
    print(f"[step2] wrote {len(dedup_reads)}-row pre-collapse read table to "
          f"{OUT_DIR / 'reads_precollapse.csv'}", flush=True)

    all_gfp_bcs = sorted(counts_filtered["gfp_bc"].unique())
    abundance = dedup_reads.groupby("gfp_bc").size().to_dict()  # total UMIs per raw barcode

    print(f"[step2] collapsing {len(all_gfp_bcs)} distinct barcodes at Hamming<={N_HAMMING} (transitive)", flush=True)
    bc_map_transitive = collapse_barcodes_transitive(all_gfp_bcs, abundance)
    pd.DataFrame(
        {"raw_barcode": list(bc_map_transitive.keys()), "representative": list(bc_map_transitive.values())}
    ).to_csv(OUT_DIR / "barcode_map_transitive.csv", index=False)
    print(f"[step2] transitive: {len(set(bc_map_transitive.values()))} distinct clones remain", flush=True)

    bc_map = bc_map_transitive
    n_collapsed = sum(1 for k, v in bc_map.items() if k != v)
    print(f"[step2] collapsed {n_collapsed} barcodes into existing ones "
          f"({len(set(bc_map.values()))} distinct clones remain)", flush=True)
    counts_filtered["clone_bc"] = counts_filtered["gfp_bc"].map(bc_map)

    # One row per UMI, post-Hamming-collapse but *before* any UMI-count
    # thresholding of our own: this is the exact input shape singletCode's
    # get_singlets() expects (repeated cellID/barcode/sample rows = UMI
    # support), letting it apply its own UMI cutoff and singlet/multiplet call. 
    umi_table = (
        counts_filtered.drop_duplicates(["library", "cell", "umi", "clone_bc"])
        [["cell", "clone_bc", "library"]]
        .rename(columns={"cell": "cellID", "clone_bc": "barcode", "library": "sample"})
        .reset_index(drop=True)
    )
    umi_table.to_csv(OUT_DIR / "umi_table.csv", index=False)
    print(f"[step2] wrote {len(umi_table)}-row per-UMI table for singletCode to "
          f"{OUT_DIR / 'umi_table.csv'}", flush=True)

    # Removed: an independent fixed->=N_UMIS clone call used to be computed here
    # (obs_metadata.csv's larry_clone_umi3_reference_only). It served its purpose early
    # on as a from-scratch validation quantity (it's what produced the first 0.9989 ARI
    # check against h5ad's clone_id, before singletCode + criterion-4 rescue existed to
    # compare against directly), but singletCode is now the actual arbiter of clone
    # assignment for every cell that reaches the final matrix, and keeping a second,
    # independently-thresholded clone call around risked being mistaken for the real
    # one. See singletcode_cross_sample_rescue.py for the pipeline's actual clone
    # assignment logic.
    #
    # umi_support = (
    #     counts_filtered.drop_duplicates(["library", "cell", "umi", "clone_bc"])
    #     .groupby(["library", "cell", "clone_bc"])
    #     .size()
    #     .reset_index(name="n_umis")
    # )
    # called = umi_support[umi_support["n_umis"] >= N_UMIS]
    # clone_calls = (
    #     called.groupby(["library", "cell"])["clone_bc"]
    #     .apply(lambda s: "".join(sorted(s)))
    #     .reset_index()
    #     .rename(columns={"clone_bc": "larry_clone_umi3_reference_only"})
    # )

    return None, {
        "n_reads_threshold": N_READS,
        "n_hamming_threshold": N_HAMMING,
        "n_lk_combos_dropped": n_lk_dropped,
        "lk_libraries_seen": lk_libraries_seen,
        "n_distinct_barcodes_before_collapse": len(all_gfp_bcs),
        "n_distinct_clones_greedy": len(set(bc_map_greedy.values())),
        "n_distinct_clones_transitive": len(set(bc_map_transitive.values())),
        "barcode_collapse_method": "transitive_highest_umi",  # feeds umi_table.csv, the
        # real pipeline input for singletCode
        "n_distinct_clones_after_collapse": len(set(bc_map.values())),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X, gene_header, obs, per_library_counts = build_combined_matrix()
    # The pipeline's actual clone assignment comes from singletCode +
    # apply_cross_sample_rescue() (singletcode_cross_sample_rescue.py), run downstream
    # on this script's umi_table.csv output by build_final_matrix_umi3.py -- not from
    # here. extract_clones() no longer computes a competing clone call of its own (see
    # its comment); its second return value is just the barcode-collapse diagnostics.
    _, step2_summary = extract_clones()

    assert X.shape[0] == len(obs)
    assert obs.index.is_unique
    assert X.shape[1] == len(gene_header)

    print(f"[save] writing outputs to {OUT_DIR}", flush=True)
    sio.mmwrite(OUT_DIR / "larry_combined_counts.mtx", X)
    (OUT_DIR / "genes.txt").write_text("\n".join(gene_header) + "\n")
    (OUT_DIR / "cells.txt").write_text("\n".join(obs.index) + "\n")
    obs.to_csv(OUT_DIR / "obs_metadata.csv")

    summary = {
        "total_cells": int(X.shape[0]),
        "total_genes": int(X.shape[1]),
        "per_library_cell_counts": per_library_counts,
        **step2_summary,
    }
    (OUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())
