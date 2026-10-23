#!/usr/bin/env python3
#==================================================================================================
# stepThree (POOLED) -- starcode error-correction across ALL samples jointly
#
# WHAT CHANGED vs the original stepThree.py
#   1. POOLED clustering. All samples' barcodes are concatenated into ONE input per
#      barcode-length, starcode is run ONCE on the pooled set, and every read across every
#      sample is assigned the SAME global centroid. This guarantees a shared barcode
#      vocabulary so a clone present in >1 sample gets an identical string in each -- which
#      is required for a correct cross-sample overlap / lineage comparison. (Per-sample
#      clustering lets the representative sequence drift between samples and silently
#      undercounts overlap on an exact-string join.)
#   2. starcode is now RUN INTERNALLY (via subprocess). The original consumed
#      externally-generated starcode outputs; that cannot work for pooling because the
#      cluster indices must line up with the pooled read order. Running it here keeps the
#      indices aligned by construction.
#   3. BC50 now has d2, d4, d6 AND d8 (a distance sweep). BC40/BC30 columns are preserved.
#
# Requires starcode built once:
#     git clone https://github.com/gui11aume/starcode && make -C starcode
#
# Input : <stepTwo_base>/<sample>/stepTwoCellIDUMIBarcodes.txt  (for each sample)
# Output: <output_base>/<sample>/stepThreeStarcodeShavedReads.txt  (per sample, split back out)
#         <output_base>/pooled/...                                  (pooled intermediates)
#         <output_base>/pooled/stepThreeStarcodeShavedReads_POOLED.txt (all samples, one table)
#
# Usage:
#   python stepThree_pooled.py <stepTwo_base> <output_base> <starcode_bin> <sample1> [sample2 ...]
#
# Example:
#   python stepThree_pooled.py \
#       /projects/.../stepTwo \
#       /projects/.../stepThree \
#       /projects/.../starcode/starcode \
#       T0 T10 T15
#==================================================================================================

import os
import sys
import csv
import subprocess
from collections import defaultdict

import numpy as np
import pandas as pd

csv.field_size_limit(100000000)

THREADS = 8

# ---- CLI -----------------------------------------------------------------------------------------
if len(sys.argv) < 5:
    sys.exit("Usage: python stepThree_pooled.py <stepTwo_base> <output_base> "
             "<starcode_bin> <sample1> [sample2 ...]")

stepTwoBase = sys.argv[1]
outputBase  = sys.argv[2]
starcodeBin = sys.argv[3]
samples     = sys.argv[4:]

if not (os.path.isfile(starcodeBin) and os.access(starcodeBin, os.X_OK)):
    sys.exit("Error: starcode binary not found or not executable: " + starcodeBin)

# ---- What to compute: (column name, truncation length, starcode distance) ------------------------
# BC50 gets the full distance sweep d2/d4/d6/d8; BC40 and BC30 preserved from the original.
configs = [
    ("BC50StarcodeD2", 50, 2),
    ("BC50StarcodeD4", 50, 4),
    ("BC50StarcodeD6", 50, 6),
    ("BC50StarcodeD8", 50, 8),
    ("BC40StarcodeD8", 40, 8),
    ("BC30StarcodeD8", 30, 8),
    ("BC30StarcodeD6", 30, 6),
]

col_order = ["cellID", "UMI", "BC",
             "BC50StarcodeD2", "BC50StarcodeD4", "BC50StarcodeD6", "BC50StarcodeD8",
             "BC40StarcodeD8", "BC30StarcodeD8", "BC30StarcodeD6"]


def run_starcode(infile, outfile, dist, threads=THREADS):
    """Run starcode once; --seq-id emits, per cluster, the 1-based input line indices."""
    cmd = [starcodeBin, "-d", str(dist), "-t", str(threads),
           "-s", "--seq-id", "-i", infile, "-o", outfile]
    print("  starcode:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def apply_clusters(outfile, arr):
    """Overwrite each read's barcode with its cluster centroid, using starcode --seq-id indices."""
    with open(outfile, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            centroid = row[0]                                   # cluster representative
            idx = np.array(row[2].split(","), dtype=np.intp)   # 1-based read indices
            idx -= 1
            arr[idx] = centroid


# ---- 1. Load every sample and POOL into one table (fixed row order) ------------------------------
frames = []
for s in samples:
    fp = os.path.join(stepTwoBase, s, "stepTwoCellIDUMIBarcodes.txt")
    if not os.path.exists(fp):
        sys.exit("Error: input not found for sample '%s': %s" % (s, fp))
    d = pd.read_csv(fp, sep="\t", usecols=["cellID", "UMI", "BC"], dtype=str)
    d["sample"] = s
    frames.append(d)
    print("Loaded %-8s %d reads" % (s, len(d)))

pooled = pd.concat(frames, ignore_index=True)
n = len(pooled)
print("Pooled total: %d reads across %d samples" % (n, len(samples)))

pooledDir = os.path.join(outputBase, "pooled")
os.makedirs(pooledDir, exist_ok=True)

bc = pooled["BC"].values

# ---- 2. Group configs by truncation length so each input file is written/clustered once ----------
by_len = defaultdict(list)
for col, L, D in configs:
    by_len[L].append((col, D))

result_cols = {}
for L, items in by_len.items():
    # Write one pooled input file of L-bp barcodes, one line per read, in pooled order.
    trunc = np.array([s[:L] for s in bc], dtype=object)
    infile = os.path.join(pooledDir, "pooled_bc%d.txt" % L)
    with open(infile, "w") as f:
        f.write("\n".join(trunc.tolist()))
        f.write("\n")
    print("Wrote pooled %d-bp input (%d lines): %s" % (L, n, infile))

    # Run starcode once per requested distance on this same input.
    for col, D in items:
        outfile = os.path.join(pooledDir, "pooled_bc%d_d%d.seqid.txt" % (L, D))
        run_starcode(infile, outfile, D)
        arr = trunc.copy()          # default = raw truncated barcode
        apply_clusters(outfile, arr)
        result_cols[col] = arr
        print("  -> %s done" % col)

# ---- 3. Attach corrected columns to the pooled table ---------------------------------------------
for col, _, _ in configs:
    pooled[col] = result_cols[col]

# ---- SUMMARY: how much starcode collapsed barcodes, per column, per sample + pooled --------------
summary_rows = []
for col, L, D in configs:
    raw_col = "BC"  # raw barcodes truncated to L for the "before" count
    # pooled (all samples)
    n_before_pooled = pooled[raw_col].str[:L].nunique()
    n_after_pooled  = pooled[col].nunique()
    summary_rows.append({
        "column": col, "length": L, "distance": D, "scope": "POOLED",
        "unique_before": n_before_pooled,
        "unique_after":  n_after_pooled,
        "collapsed":     n_before_pooled - n_after_pooled,
        "pct_reduction": round(100 * (n_before_pooled - n_after_pooled) / n_before_pooled, 2)
                          if n_before_pooled else 0.0,
    })
    # per sample
    for s in samples:
        sub = pooled[pooled["sample"] == s]
        n_before = sub[raw_col].str[:L].nunique()
        n_after  = sub[col].nunique()
        summary_rows.append({
            "column": col, "length": L, "distance": D, "scope": s,
            "unique_before": n_before,
            "unique_after":  n_after,
            "collapsed":     n_before - n_after,
            "pct_reduction": round(100 * (n_before - n_after) / n_before, 2) if n_before else 0.0,
        })

summary_df = pd.DataFrame(summary_rows,
    columns=["column", "length", "distance", "scope",
             "unique_before", "unique_after", "collapsed", "pct_reduction"])
summary_out = os.path.join(pooledDir, "stepThree_starcode_summary.txt")
summary_df.to_csv(summary_out, sep="\t", index=False)
print("\n=== STARCODE COLLAPSE SUMMARY ===")
print(summary_df.to_string(index=False))
print("Wrote summary: " + summary_out)

# ---- 4. Write the pooled table (all samples together) --------------------------------------------
pooled_out = os.path.join(pooledDir, "stepThreeStarcodeShavedReads_POOLED.txt")
pooled[["sample"] + col_order].to_csv(pooled_out, sep="\t", index=False)
print("Wrote pooled table: " + pooled_out)

# ---- 5. Split back out per sample (identical filename/format to the original per-sample output) ---
for s in samples:
    sub = pooled[pooled["sample"] == s]
    outdir = os.path.join(outputBase, s)
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, "stepThreeStarcodeShavedReads.txt")
    sub[col_order].to_csv(outfile, sep="\t", index=False)
    print("Wrote %-8s %d reads -> %s" % (s, len(sub), outfile))

print("Done.")