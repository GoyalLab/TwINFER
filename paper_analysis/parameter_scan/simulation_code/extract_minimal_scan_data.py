"""Extract the minimal per-simulation data needed to re-run the TwINFER parameter-scan analysis.

FOR WHOEVER STILL HAS THE RAW PARAMETER-SCAN SIMULATIONS.

Run this over the folder holding the twin tables (the `df_row_*.csv` files, one per parameter set).
It writes one small gzipped CSV per input file, keeping only the rows and columns the analysis
actually reads. Measured on a real twin table: 23.2 MB -> 0.23 MB gzipped, so a 25,000-set topology
goes from ~580 GB to ~5.8 GB, and all four topologies from ~2.3 TB to ~23 GB.

WHAT IS KEPT, AND WHY EACH IS LOAD-BEARING
    clone_id        splits clones 1:1:2 into the t1, t2 and cross-sample groups
    replicate       cross-sample pairs take replicate 1 at t1 and replicate 2 at t2
    time_step       selects t1 and t2 -- THE COLUMN THAT MAKES UN-MERGING POSSIBLE
    gene_1_mRNA     every correlation: existence, twin, random-pair, directional
    gene_2_mRNA
    gene_1_protein  the Brunner-Munzel test deciding which pairs count as two-state
    gene_2_protein
Rows are restricted to time_step in {t1, t2}; everything else is discarded.

`cell_id` is NOT needed -- the analysis never references it. Neither are the
`simulation_before_division_df_*.csv` files: nothing reads them, and they are ~90% of the volume.

WHAT MUST NOT BE DONE
Do not pool or average across time points before sending. The point of this request is to recompute
statistics WITHOUT merging t1 and t2, which is only possible while every row still carries its own
time_step. Anything pre-combined is unusable for it.

Keep the original filenames. The `df_row_<r1>_<r2>_..._<TYPE>_*.csv` pattern identifies which
parameter set each file came from; renaming loses that link irrecoverably.

Usage
    # 1. CHECK FIRST. Reads one file, reports whether the format is what we expect, writes nothing.
    #    Takes seconds. If it complains, please send us the output rather than working around it.
    python3 extract_minimal_scan_data.py /path/to/your/A_B --check

    # 2. Then the real pass, one topology at a time.
    python3 extract_minimal_scan_data.py /path/to/your/A_B

The output directory defaults to a folder named after the input, NEXT TO THIS SCRIPT -- so if the
script sits in the shared drop-off folder, running it with just the input path puts the extract
straight where it needs to end up, with no path to mistype. Pass a second argument to override.

Requires only python3 + pandas. Safe to interrupt and re-run: files already written are skipped.
"""
import argparse
import glob
import os
import sys

import pandas as pd

KEEP = ["clone_id", "replicate", "time_step",
        "gene_1_mRNA", "gene_2_mRNA", "gene_1_protein", "gene_2_protein"]


# Longest first: "A_B" is a substring of nothing here, but "A_to_B"/"A_and_B"/"A_rep_B" all end in
# "_B", so a naive scan would mislabel them. Matching the longer names first avoids that.
TOPOLOGIES = ["A_and_B", "A_rep_B", "A_to_B", "A_B"]


def topology_from_filenames(files):
    """Read the topology out of the filenames, which embed it as
    df_row_<r1>_<r2>_<stamp>_ncells_<n>_<TYPE>_<id>.csv. The input DIRECTORY name is not trusted --
    it could be anything. Returns None if the files disagree or none matches, so the caller can stop
    rather than scatter output into a wrongly-named folder."""
    seen = set()
    for f in files:
        base = os.path.basename(f)
        for t in TOPOLOGIES:
            if f"_{t}_" in base:
                seen.add(t)
                break
    if len(seen) == 1:
        return seen.pop()
    return None


def find_inputs(input_dir):
    """Twin tables, plain or already gzipped. The before-division dumps are named
    simulation_before_division_df_*, so requiring a leading df_ already excludes them."""
    out = []
    for pat in ("df_*.csv", "df_*.csv.gz"):
        out += glob.glob(os.path.join(input_dir, pat))
    return sorted(out)


def describe_dir(input_dir):
    """On failure, say what IS there. A bare 'no files found' is useless to whoever ran it."""
    try:
        entries = sorted(os.listdir(input_dir))
    except OSError as e:
        return f"    cannot list {input_dir}: {e}"
    if not entries:
        return "    the directory is empty"
    shown = "\n".join(f"    {e}" for e in entries[:10])
    more = f"\n    ... and {len(entries) - 10:,} more" if len(entries) > 10 else ""
    return f"    {len(entries):,} entries, first few:\n{shown}{more}"


def check(input_dir, timepoints):
    """Inspect one file and report compatibility. Writes nothing. Returns a shell exit code."""
    print(f"CHECK: {input_dir}")
    files = find_inputs(input_dir)
    if not files:
        print("  NO twin tables matching df_*.csv or df_*.csv.gz")
        print(describe_dir(input_dir))
        print("\n  -> the files may be named differently here. Send us this output and we will adjust.")
        return 1
    print(f"  found {len(files):,} candidate twin tables")
    f = files[0]
    print(f"  inspecting {os.path.basename(f)}")
    try:
        d = pd.read_csv(f)
    except Exception as e:
        print(f"  UNREADABLE: {e}")
        return 1
    print(f"  shape: {d.shape[0]:,} rows x {d.shape[1]} columns")
    print(f"  columns: {list(d.columns)}")

    missing = [c for c in KEEP if c not in d.columns]
    if missing:
        print(f"  MISSING required columns: {missing}")
        print("  -> the column names may differ in your version. Send us this output.")
        return 1
    print("  all 7 required columns present")

    tps = sorted(d["time_step"].unique())
    print(f"  time_step values: {tps[:8]}{'...' if len(tps) > 8 else ''} ({len(tps)} distinct)")
    bad = False
    for t in timepoints:
        n = int((d["time_step"] == t).sum())
        flag = "   <- EMPTY, please tell us" if n == 0 else ""
        bad = bad or n == 0
        print(f"    rows at time_step={t}: {n:,}{flag}")
    if bad:
        return 1

    sub = d[d["time_step"].isin(timepoints)]
    print(f"  clones: {sub['clone_id'].nunique():,}   replicates: {sorted(sub['replicate'].unique())}")
    print("\n  COMPATIBLE -- safe to run the full pass.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir")
    ap.add_argument("output_dir", nargs="?", default=None,
                    help="default: a folder named after the input, next to this script")
    ap.add_argument("--timepoints", nargs="+", type=int, default=[1, 20],
                    help="time_step values to keep (default: 1 20, matching the shipped analysis)")
    ap.add_argument("--check", action="store_true",
                    help="inspect one file and report compatibility; write nothing")
    a = ap.parse_args()

    if a.check:
        sys.exit(check(a.input_dir, a.timepoints))

    files = find_inputs(a.input_dir)
    if not files:
        print(f"no twin tables matching df_*.csv or df_*.csv.gz in {a.input_dir}")
        print(describe_dir(a.input_dir))
        sys.exit("\nplease re-run with --check and send us the output")

    # Destination defaults to <this script's folder>/<TOPOLOGY>, with the topology read from the
    # FILENAMES rather than the input directory's name -- that folder could be called anything
    # (A_B_sims, positive_A_B, run3) and naming the output after it would scatter extracts into
    # folders nobody is expecting. Must come AFTER find_inputs, since it reads those filenames.
    if a.output_dir is None:
        here = os.path.dirname(os.path.abspath(__file__))
        topo = topology_from_filenames(files)
        if topo is None:
            sys.exit(f"could not read a single consistent topology from the filenames.\n"
                     f"  first file: {os.path.basename(files[0])}\n"
                     f"  expected one of {', '.join(TOPOLOGIES)} embedded as _<TYPE>_ in the name\n"
                     f"Stopping rather than guessing. Either the names differ here -- in which case "
                     f"please send us one filename -- or pass the destination explicitly:\n"
                     f"    python3 {os.path.basename(__file__)} '{a.input_dir}' '{here}/A_B'")
        a.output_dir = os.path.join(here, topo)
        print(f"topology '{topo}' read from the filenames -> {a.output_dir}")

    os.makedirs(a.output_dir, exist_ok=True)
    print(f"{len(files):,} twin tables -> {a.output_dir}  (keeping time_step in {a.timepoints})")

    done = skipped = 0
    tot_in = tot_out = 0
    for i, f in enumerate(files, 1):
        base = os.path.basename(f)
        if base.endswith(".gz"):
            base = base[:-3]
        dest = os.path.join(a.output_dir, base + ".gz")
        if os.path.exists(dest):
            skipped += 1
            continue
        try:
            d = pd.read_csv(f)
        except Exception as e:
            print(f"  UNREADABLE {base}: {e}")
            continue

        missing = [c for c in KEEP if c not in d.columns]
        if missing:
            sys.exit(f"\n{base} is missing required columns {missing}\n"
                     f"columns present: {list(d.columns)}\n"
                     f"Stopping rather than writing partial data. Please run --check and send us "
                     f"the output.")

        sub = d[d["time_step"].isin(a.timepoints)][KEEP]
        if sub.empty:
            print(f"  WARNING {base}: no rows at time_step {a.timepoints}; "
                  f"present: {sorted(d['time_step'].unique())[:6]}...")
        sub.to_csv(dest, index=False, compression="gzip")

        tot_in += os.path.getsize(f)
        tot_out += os.path.getsize(dest)
        done += 1
        if i % 500 == 0 or i == len(files):
            print(f"  {i:,}/{len(files):,}   written {done:,}   already present {skipped:,}")

    if done:
        print(f"\n{done:,} written, {skipped:,} already present")
        print(f"  {tot_in / 1e9:.2f} GB in -> {tot_out / 1e9:.3f} GB out  "
              f"({tot_in / max(tot_out, 1):.0f}x smaller)")
    else:
        print(f"\nnothing to do: {skipped:,} already present")


if __name__ == "__main__":
    main()
