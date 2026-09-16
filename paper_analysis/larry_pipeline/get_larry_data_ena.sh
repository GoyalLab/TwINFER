#!/usr/bin/env bash
#
# download_ena_fastqs.sh
#
# Downloads FASTQ files from the ENA mirror for a list of SRA run accessions,
# skipping the SRA toolkit's prefetch/fasterq-dump conversion step entirely.
#
# Usage:
#   ./download_ena_fastqs.sh SRR_Acc_List.txt [output_dir] [max_parallel]
#
# Arguments:
#   SRR_Acc_List.txt   Text file with one SRR/ERR/DRR accession per line
#                       (this is exactly what SRA Run Selector's
#                       "Accession List" button downloads)
#   output_dir          Where to save FASTQs (default: ./fastq)
#   max_parallel         Number of simultaneous downloads (default: 8)
#
# Requires: wget or curl, standard bash

set -uo pipefail

ACC_LIST="${1:?Usage: $0 SRR_Acc_List.txt [output_dir] [max_parallel]}"
OUTDIR="${2:-fastq}"
MAX_PARALLEL="${3:-8}"
LOGFILE="${OUTDIR}/download.log"
MAX_RETRIES=3

mkdir -p "$OUTDIR"
: > "$LOGFILE"

if [[ ! -f "$ACC_LIST" ]]; then
    echo "ERROR: accession list '$ACC_LIST' not found" >&2
    exit 1
fi

# --- Compute the ENA fastq directory path for a given accession -----------
# ENA lays out FASTQs as:
#   vol1/fastq/<first 6 chars>/<extra subdir>/<full accession>/
# The "extra subdir" depends on accession length:
#   len 9  (e.g. SRR123456)   -> no extra subdir
#   len 10 (e.g. SRR1234567)  -> 00 + last 1 digit
#   len 11 (e.g. SRR12345678) -> 0  + last 2 digits
#   len 12 (e.g. SRR123456789)-> last 3 digits
ena_dir_for_acc() {
    local acc="$1"
    local prefix="${acc:0:6}"
    local len=${#acc}
    local extra=""
    case "$len" in
        9)  extra="" ;;
        10) extra="00${acc: -1}/" ;;
        11) extra="0${acc: -2}/" ;;
        12) extra="${acc: -3}/" ;;
        *)  echo "WARN: unexpected accession length for $acc ($len chars)" >&2
            extra="" ;;
    esac
    echo "https://ftp.sra.ebi.ac.uk/vol1/fastq/${prefix}/${extra}${acc}"
}

# --- Download one run (handles single-end, paired-end, and inDrops-style
#     3-file runs by probing for _1/_2/_3 then falling back to the plain
#     single-file layout) --------------------------------------------------
download_run() {
    local acc="$1"
    local base_url
    base_url="$(ena_dir_for_acc "$acc")"

    local found_any=0

    for suffix in "_1" "_2" "_3"; do
        local fname="${acc}${suffix}.fastq.gz"
        local url="${base_url}/${fname}"
        local out="${OUTDIR}/${fname}"

        if [[ -s "$out" ]]; then
            echo "[$acc] $fname already exists, skipping" >> "$LOGFILE"
            found_any=1
            continue
        fi

        local attempt=0
        local ok=0
        while (( attempt < MAX_RETRIES )); do
            if wget -q --show-progress -O "$out" "$url" 2>>"$LOGFILE"; then
                if [[ -s "$out" ]]; then
                    ok=1
                    found_any=1
                    echo "[$acc] downloaded $fname" >> "$LOGFILE"
                    break
                fi
            fi
            rm -f "$out"
            attempt=$((attempt + 1))
        done

        if [[ $ok -eq 0 ]]; then
            echo "[$acc] $fname not found or failed after $MAX_RETRIES tries" >> "$LOGFILE"
        fi
    done

    # Fall back to single unsplit file if no _1/_2/_3 files were found
    if [[ $found_any -eq 0 ]]; then
        local fname="${acc}.fastq.gz"
        local url="${base_url}/${fname}"
        local out="${OUTDIR}/${fname}"

        local attempt=0
        local ok=0
        while (( attempt < MAX_RETRIES )); do
            if wget -q --show-progress -O "$out" "$url" 2>>"$LOGFILE"; then
                if [[ -s "$out" ]]; then
                    ok=1
                    echo "[$acc] downloaded $fname (single-file run)" >> "$LOGFILE"
                    break
                fi
            fi
            rm -f "$out"
            attempt=$((attempt + 1))
        done

        if [[ $ok -eq 0 ]]; then
            echo "[$acc] ERROR: no FASTQ files could be downloaded" | tee -a "$LOGFILE" >&2
        fi
    fi
}
export -f download_run ena_dir_for_acc
export OUTDIR LOGFILE MAX_RETRIES

echo "Downloading FASTQs for $(wc -l < "$ACC_LIST") runs into '$OUTDIR' (parallel=$MAX_PARALLEL)"
echo "Log: $LOGFILE"

# Run with bounded parallelism
cat "$ACC_LIST" | grep -v '^\s*$' | xargs -P "$MAX_PARALLEL" -I{} bash -c 'download_run "$@"' _ {}

echo "Done. Check $LOGFILE for any accessions that failed."
