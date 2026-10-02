"""2026-09-17: Generate BEELINE 'twin_paired' scheme inputs at t1=10,t2=20 (instead of t1=1,t2=20)
for all 3 simulated benchmarks. The 'spread' scheme is timepoint-agnostic (uses every available
timepoint 0..48 regardless of t1/t2) so it's NOT regenerated -- only twin_paired changes.

Writes to NEW, separate BEELINE input roots so the existing t1=1 results are untouched:
  code/Beeline/inputs/network_sweep_final_t1_10/<net>/simrep<k>_twin_paired/
  code/Beeline/inputs/e13_pos100_t1_10/<label>/simrep<k>_twin_paired/
  code/Beeline/inputs/mixed_network_sweep_t1_10/<net>/simrep<k>_twin_paired/
  code/Beeline/inputs/real_networks_t1_10/<net>/simrep<rep>_twin_paired/

Reuses each family's existing raw-simulation task discovery (from the corresponding
infer_*_allpairs.py script, already validated this session) plus twinfer_to_boolode_format.py's
shared build_twin_paired()/convert_ground_truth()/load_simulation() -- same functions the t1=1
inputs were built from, just called with t1=10.

    python build_t1_10_beeline_inputs.py [--family network_sweep_final|e13_pos100|mixed_network_sweep|real_networks]
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import argparse
import sys
from pathlib import Path

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/home/gzu5140/TwINFER_KA/code/TwINFER/work_in_progress/benchmark")
from benchmarks.network_benchmarks.to_beeline.twinfer_to_boolode_format import build_twin_paired, convert_ground_truth, load_simulation

PROJECT_ROOT = Path(f'{TWINFER_PROJECT_ROOT}')
BEELINE_ROOT = PROJECT_ROOT / "code" / "Beeline" / "inputs"
T1, T2 = 10, 20


def do_network_sweep_final():
    from benchmarks.network_benchmarks.infer import infer_network_sweep_final_allpairs as m
    out_root = BEELINE_ROOT / "network_sweep_final_t1_10"
    tasks = m.discover_tasks()
    print(f"[network_sweep_final] {len(tasks)} sim files")
    for net, sim_rep, path in tasks:
        dataset_dir = out_root / net
        run_dir = dataset_dir / f"simrep{sim_rep}_twin_paired"
        if (run_dir / "ExpressionData.csv").exists():
            continue
        n_genes = m.N_GENES
        if not (dataset_dir / "GroundTruthNetwork.csv").exists():
            convert_ground_truth(Path(m.TOPOLOGY_DIR) / f"{net}.txt", dataset_dir / "GroundTruthNetwork.csv")
        gene_names = [f"gene_{i+1}" for i in range(n_genes)]
        df = load_simulation(Path(path), n_genes)
        build_twin_paired(df, gene_names, run_dir, t1=T1, t2=T2)
        del df
        print(f"  built {net} simrep{sim_rep}")


def do_e13_pos100():
    from benchmarks.network_benchmarks.infer import infer_e13_pos100_allpairs as m
    out_root = BEELINE_ROOT / "e13_pos100_t1_10"
    tasks = m.discover_tasks()
    print(f"[e13_pos100] {len(tasks)} sim files")
    for label, sim_rep, path in tasks:
        dataset_dir = out_root / label
        run_dir = dataset_dir / f"simrep{sim_rep}_twin_paired"
        if (run_dir / "ExpressionData.csv").exists():
            continue
        n_genes = m.N_GENES
        if not (dataset_dir / "GroundTruthNetwork.csv").exists():
            convert_ground_truth(Path(m.LABEL_TO_GT[label]), dataset_dir / "GroundTruthNetwork.csv")
        gene_names = [f"gene_{i+1}" for i in range(n_genes)]
        df = load_simulation(Path(path), n_genes)
        build_twin_paired(df, gene_names, run_dir, t1=T1, t2=T2)
        del df
        print(f"  built {label} simrep{sim_rep}")


def do_mixed_network_sweep():
    from benchmarks.network_benchmarks.to_beeline import mixed_to_beeline_format as m
    out_root = BEELINE_ROOT / "mixed_network_sweep_t1_10"
    tasks = m.discover()
    print(f"[mixed_network_sweep] {len(tasks)} sim files")
    for net, sim_rep, path in tasks:
        dataset_dir = out_root / net
        run_dir = dataset_dir / f"simrep{sim_rep}_twin_paired"
        if (run_dir / "ExpressionData.csv").exists():
            continue
        if not (dataset_dir / "GroundTruthNetwork.csv").exists():
            n_genes = convert_ground_truth(m.TOPOLOGY_DIR / f"{net}.txt", dataset_dir / "GroundTruthNetwork.csv")
        else:
            import pandas as pd
            n_genes = pd.read_csv(m.TOPOLOGY_DIR / f"{net}.txt", header=None).shape[0]
        gene_names = [f"gene_{i+1}" for i in range(n_genes)]
        df = load_simulation(path, n_genes)
        build_twin_paired(df, gene_names, run_dir, t1=T1, t2=T2)
        del df
        print(f"  built {net} simrep{sim_rep}")


def do_real_networks():
    from benchmarks.network_benchmarks.infer import infer_real_network_allpairs as m
    out_root = BEELINE_ROOT / "real_networks_t1_10"
    tasks = m.build_tasks()
    print(f"[real_networks] {len(tasks)} replicate tasks")
    gt_done = set()
    for path, net, rep_id, base_config in tasks:
        cfg = m.NETWORKS[net]
        dataset_dir = out_root / net
        run_dir = dataset_dir / f"simrep{rep_id}_twin_paired"
        if (run_dir / "ExpressionData.csv").exists():
            continue
        n_genes = cfg["n_genes"]
        if net not in gt_done:
            convert_ground_truth(PROJECT_ROOT / "input_data" / "real_world_networks" / cfg["topology"],
                                  dataset_dir / "GroundTruthNetwork.csv")
            gt_done.add(net)
        gene_names = [f"gene_{i+1}" for i in range(n_genes)]
        df = load_simulation(Path(path), n_genes)
        build_twin_paired(df, gene_names, run_dir, t1=T1, t2=T2)
        del df
        print(f"  built {net} simrep{rep_id}")


FAMILIES = {
    "network_sweep_final": do_network_sweep_final,
    "e13_pos100": do_e13_pos100,
    "mixed_network_sweep": do_mixed_network_sweep,
    "real_networks": do_real_networks,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=list(FAMILIES), default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    fams = [args.family] if args.family else list(FAMILIES)
    for fam in fams:
        FAMILIES[fam]()
