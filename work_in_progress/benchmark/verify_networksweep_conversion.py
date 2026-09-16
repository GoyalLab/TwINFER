"""
Independently verify convert_network_sweep_to_beeline.py's output against the
raw TwINFER simulation files and ground-truth topology matrices.

This does NOT import or reuse any function from convert_network_sweep_to_beeline.py --
every check here is re-derived from scratch directly against the raw source files, so
a bug shared between the conversion script and this verifier can't hide from both.

Three checks, run across every (dataset, sim_rep, scheme) combination that actually
exists under inputs/network_sweep/ (not just a single spot-check sample, since these
are small 6-gene datasets and full coverage is cheap):

  A. Per-cell value fidelity: for every column in a converted ExpressionData.csv,
     parse its cell_id/time_step from the label, look up that exact
     (cell_id, time_step) row in the RAW simulation CSV, and assert the gene_*_mRNA
     values match exactly. Also checks PseudoTime.csv's value for that cell equals
     the time_step encoded in its own label.
  B. Structural completeness: spread should contain every raw cell_id exactly once,
     spread across all timepoints with no duplicates/omissions; twin_paired should
     contain exactly 12000 cells split 6000/6000 between only the two designed
     timepoints (t1=1, t2=20), with no other pseudotime value appearing.
  C. Ground-truth edge list: independently re-derive Gene1/Gene2/Type from each
     grn_n6_*.txt matrix (row=regulator, column=target -- confirmed against
     gillespie_script.py's own indexing) and diff against the actual
     GroundTruthNetwork.csv already on disk.

Run: python3 verify_networksweep_conversion.py
"""

import re
import sys
from pathlib import Path

import pandas as pd

SIMULATION_DATA = Path("/home/gzu5140/Keerthana_b1042/TwINFER/simulation_data/network_sweep")
TOPOLOGY_DATA = Path("/home/gzu5140/Keerthana_b1042/TwINFER/input_data/network_sweep")
BEELINE_INPUT = Path("/home/gzu5140/Keerthana_b1042/TwINFER/code/Beeline/inputs/network_sweep")

SIM_FILENAME_RE = re.compile(r"^df_(grn_n6_.+?_rep\d+)_rep(\d+)_\d+_\d+_ncells_\d+_.*\.csv$")
CELL_LABEL_RE = re.compile(r"^cell(\d+)_t(\d+)$")

T1, T2 = 1, 20

failures = []
checks_run = 0


def fail(msg):
    failures.append(msg)
    print(f"  FAIL: {msg}")


# ---------------------------------------------------------------------------
# Check C: ground-truth edge list, independently re-derived
# ---------------------------------------------------------------------------
print("=== Check C: ground-truth edge list re-derivation ===")
for topo_path in sorted(TOPOLOGY_DATA.glob("grn_n6_*.txt")):
    dataset_id = topo_path.stem
    gt_path = BEELINE_INPUT / dataset_id / "GroundTruthNetwork.csv"
    if not gt_path.exists():
        continue
    checks_run += 1

    matrix = pd.read_csv(topo_path, header=None).to_numpy()
    n = matrix.shape[0]
    gene_names = [f"gene_{i + 1}" for i in range(n)]

    expected_edges = set()
    for i in range(n):
        for j in range(n):
            if matrix[i, j] == 0:
                continue
            expected_edges.add((gene_names[i], gene_names[j], "+" if matrix[i, j] > 0 else "-"))

    actual = pd.read_csv(gt_path)
    actual_edges = set(zip(actual["Gene1"], actual["Gene2"], actual["Type"]))

    if actual_edges != expected_edges:
        fail(f"{dataset_id}: GroundTruthNetwork.csv mismatch. "
             f"Missing: {expected_edges - actual_edges}, Extra: {actual_edges - expected_edges}")

print(f"Checked {checks_run} ground-truth files.")

# ---------------------------------------------------------------------------
# Checks A & B: per-cell fidelity and structural completeness
# ---------------------------------------------------------------------------
print("\n=== Checks A & B: per-cell fidelity + structural completeness ===")
sim_files = [f for f in sorted(SIMULATION_DATA.glob("df_grn_n6_*_ncells_*.csv"))
             if not f.name.startswith("simulation_before_division")]

for sim_path in sim_files:
    m = SIM_FILENAME_RE.match(sim_path.name)
    if not m:
        continue
    dataset_id, sim_rep = m.group(1), m.group(2)

    gt_path = BEELINE_INPUT / dataset_id / "GroundTruthNetwork.csv"
    if not gt_path.exists():
        continue
    gt_for_genes = pd.read_csv(gt_path)
    n_genes = len(set(gt_for_genes["Gene1"]) | set(gt_for_genes["Gene2"]))
    mrna_cols = [f"gene_{i + 1}_mRNA" for i in range(n_genes)]
    raw = pd.read_csv(sim_path, usecols=["cell_id", "time_step"] + mrna_cols)
    raw_indexed = raw.set_index(["cell_id", "time_step"])

    for scheme in ["spread", "twin_paired"]:
        run_dir = BEELINE_INPUT / dataset_id / f"simrep{sim_rep}_{scheme}"
        expr_path = run_dir / "ExpressionData.csv"
        pt_path = run_dir / "PseudoTime.csv"
        if not expr_path.exists() or not pt_path.exists():
            continue
        checks_run += 1

        expr = pd.read_csv(expr_path, index_col=0)
        pt = pd.read_csv(pt_path, index_col=0)

        cell_ids_seen = []
        time_steps_seen = []

        for cell_label in expr.columns:
            cm = CELL_LABEL_RE.match(cell_label)
            if not cm:
                fail(f"{dataset_id}/simrep{sim_rep}_{scheme}: unparseable cell label '{cell_label}'")
                continue
            cid, ts = int(cm.group(1)), int(cm.group(2))
            cell_ids_seen.append(cid)
            time_steps_seen.append(ts)

            # Check A: gene values match the raw simulation row exactly
            try:
                raw_row = raw_indexed.loc[(cid, ts)]
            except KeyError:
                fail(f"{dataset_id}/simrep{sim_rep}_{scheme}/{cell_label}: "
                     f"(cell_id={cid}, time_step={ts}) not found in raw simulation file")
                continue

            expected_vals = raw_row[mrna_cols].values
            actual_vals = expr[cell_label].loc[[f"gene_{i+1}" for i in range(n_genes)]].values
            if not (expected_vals == actual_vals).all():
                fail(f"{dataset_id}/simrep{sim_rep}_{scheme}/{cell_label}: "
                     f"expression mismatch (raw={expected_vals.tolist()}, converted={actual_vals.tolist()})")

            # Check A continued: PseudoTime.csv value matches the label's own time_step
            pt_val = pt.loc[cell_label, "PseudoTime1"]
            if int(pt_val) != ts:
                fail(f"{dataset_id}/simrep{sim_rep}_{scheme}/{cell_label}: "
                     f"PseudoTime.csv says {pt_val}, label says t{ts}")

        # Check B: structural completeness per scheme
        if scheme == "spread":
            if sorted(cell_ids_seen) != sorted(raw["cell_id"].unique()):
                fail(f"{dataset_id}/simrep{sim_rep}_spread: cell_id set doesn't match "
                     f"raw file's 12000 unique cells (got {len(cell_ids_seen)} cells, "
                     f"{len(set(cell_ids_seen))} unique)")
            if len(cell_ids_seen) != len(set(cell_ids_seen)):
                fail(f"{dataset_id}/simrep{sim_rep}_spread: duplicate cell_ids in ExpressionData.csv")
        elif scheme == "twin_paired":
            if len(cell_ids_seen) != 12000:
                fail(f"{dataset_id}/simrep{sim_rep}_twin_paired: expected 12000 cells, got {len(cell_ids_seen)}")
            bad_times = set(time_steps_seen) - {T1, T2}
            if bad_times:
                fail(f"{dataset_id}/simrep{sim_rep}_twin_paired: unexpected pseudotime "
                     f"value(s) {bad_times}, expected only {{{T1}, {T2}}}")
            n_t1 = time_steps_seen.count(T1)
            n_t2 = time_steps_seen.count(T2)
            if n_t1 != 6000 or n_t2 != 6000:
                fail(f"{dataset_id}/simrep{sim_rep}_twin_paired: expected 6000/6000 "
                     f"split at t{T1}/t{T2}, got {n_t1}/{n_t2}")

print(f"Checked {checks_run} (dataset, sim_rep, scheme) combinations "
      f"(including {len(sim_files) * 2} expression/pseudotime file pairs).")

print(f"\n{'=' * 60}")
if failures:
    print(f"FAILED: {len(failures)} issue(s) found across {checks_run} checks.")
    sys.exit(1)
else:
    print(f"ALL CHECKS PASSED across {checks_run} (dataset, sim_rep, scheme) combinations.")
