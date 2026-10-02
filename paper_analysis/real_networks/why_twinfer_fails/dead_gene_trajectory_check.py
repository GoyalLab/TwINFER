from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_a2a3302f_2026-09-28_why_twinfer_fails; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import glob, os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

R = f'{TWINFER_PROJECT_ROOT}'
RW = f"{R}/input_data/real_world_networks"
OLD = f"{R}/simulation_data/twinfer_format"

def old_genes(net):
    return [l.strip() for l in open(f"{OLD}/{net}/gene_order.txt") if l.strip()]

NETS = {
    "mCAD": dict(genes=["Fgf8","Emx2","Pax6","Coup","Sp8"],
                 legacy_dir=f"{R}/simulation_data/real_data", legacy_tag="mCAD"),
    "GSD": dict(genes=old_genes("GSD"),
                legacy_dir=f"{R}/simulation_data/real_data", legacy_tag="GSD"),
    "HSC": dict(genes=old_genes("HSC"),
                legacy_dir=f"{R}/simulation_data/real_data", legacy_tag="HSC_balanced"),
    "VSC": dict(genes=old_genes("VSC"),
                legacy_dir=f"{R}/simulation_data/real_data", legacy_tag="VSC"),
    "B_cell_activation": dict(genes=["Ikaros","PU_1","Flk2","IL_7R","GATA_1","E2A","EBF","C_EBPa","PAX5","Notch_1"],
                 new_dir=f"{R}/analysis_data/paper_analysis/B_cell_activation/simulate/20260812_153622"),
    "EMT": dict(genes=[f"gene_{i+1}" for i in range(17)],
                new_dir=f"{R}/analysis_data/paper_analysis/EMT/simulate/20260825_224653"),
    "Pluripotent": dict(genes=[f"gene_{i+1}" for i in range(36)],
                new_dir=f"{R}/analysis_data/paper_analysis/Pluripotent/simulate/20260825_224511"),
}
CV_THRESH = 0.15

def cv_per_timepoint(path, gene_cols, n_checkpoints=12):
    # Only peek at time_step's range first (cheap), then read just the rows at a coarse,
    # evenly-spaced subset of timepoints -- some of these files have thousands of timepoints
    # and tens of millions of rows (e.g. GSD's pre-division file: 6000 steps, 36M rows, 9GB),
    # so computing CV at every single step is infeasible.
    tstep_col = pd.read_csv(path, usecols=["time_step"])["time_step"]
    tmax = tstep_col.max()
    checkpoints = sorted(set(int(round(tmax * f)) for f in np.linspace(0, 1, n_checkpoints)))
    out = {}
    # read in chunks, keep only rows whose time_step is a checkpoint (avoids loading the whole file)
    wanted = set(checkpoints)
    reader = pd.read_csv(path, usecols=["cell_id", "time_step"] + gene_cols, chunksize=500_000)
    collected = {c: [] for c in checkpoints}
    for chunk in reader:
        hit = chunk[chunk.time_step.isin(wanted)]
        if len(hit):
            for t, g in hit.groupby("time_step"):
                collected[t].append(g)
        if all(len(v) for v in collected.values()) and len(collected) == len(checkpoints):
            # keep reading anyway since a given checkpoint's rows can span multiple chunks;
            # but cap total chunks scanned for very long files
            pass
    for t in checkpoints:
        if collected[t]:
            g = pd.concat(collected[t]).drop_duplicates("cell_id")
            out[t] = {c: (g[c].std() / (g[c].mean() + 1e-9)) for c in gene_cols}
    return out

for net, cfg in NETS.items():
    print(f"START {net}", flush=True)
    genes = cfg["genes"]
    native = genes[0].startswith("gene_")
    col = (lambda g: g) if native else (lambda g: f"gene_{genes.index(g)+1}")
    gene_cols = [f"{col(g)}_protein" for g in genes]

    if "legacy_dir" in cfg:
        files = sorted(glob.glob(f"{cfg['legacy_dir']}/df_rows_*{cfg['legacy_tag']}_*.csv"))
        files = [f for f in files if "simulation_before_division" not in f]
        before_files = sorted(glob.glob(f"{cfg['legacy_dir']}/simulation_before_division_df_rows_*{cfg['legacy_tag']}_*.csv"))
    else:
        files = sorted(glob.glob(f"{cfg['new_dir']}/df_rows_*.csv"))
        files = [f for f in files if "simulation_before_division" not in f]
        before_files = sorted(glob.glob(f"{cfg['new_dir']}/simulation_before_division_df_rows_*.csv"))

    # final-timepoint CV per replicate (up to 5 reps for speed)
    final_cv = {}
    for f in files[:5]:
        tp = cv_per_timepoint(f, gene_cols)
        tmax = max(tp.keys())
        final_cv[os.path.basename(f)] = tp[tmax]

    # identify genes dead in the FIRST replicate at final timepoint
    dead_genes = [g for g in genes if final_cv[os.path.basename(files[0])][f"{col(g)}_protein"] < CV_THRESH]
    if not dead_genes:
        print(f"{net}: no dead genes at final tp in rep0 -- skipping"); continue

    print(f"got files, computing final_cv...", flush=True)
    print(f"\n=== {net}: {len(dead_genes)}/{len(genes)} dead genes in rep0 (final tp): {dead_genes} ===")

    # 1) were they ever varying pre-division (trunk trajectory)?
    if before_files:
        bf = before_files[0]
        tp = cv_per_timepoint(bf, gene_cols)
        print(f"  Pre-division trunk trajectory ({os.path.basename(bf)}), CV over time for dead genes:")
        for g in dead_genes:
            c = col(g)
            series = [tp[t][f"{c}_protein"] for t in sorted(tp.keys())]
            print(f"    {g:10s}: CV(t)= " + " ".join(f"{v:.2f}" for v in series) +
                  f"   [ever>{CV_THRESH}? {'YES' if max(series) > CV_THRESH else 'no'}]")
    else:
        print("  (no pre-division file found)")

    # 2) is it the SAME genes across replicates, or does it vary?
    print(f"  Final-timepoint CV across {len(final_cv)} replicates:")
    for g in dead_genes:
        c = col(g)
        row = [final_cv[f][f"{c}_protein"] for f in final_cv]
        n_dead = sum(v < CV_THRESH for v in row)
        print(f"    {g:10s}: CV per rep = " + " ".join(f"{v:.2f}" for v in row) +
              f"   [{n_dead}/{len(row)} reps dead]")
