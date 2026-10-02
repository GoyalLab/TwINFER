#!/usr/bin/env python
"""Convert the BoolODE twin sims of B_cell_activation / EMT_real / Pluripotent_real
(simulation_data/boolode_sims_replicates/<net>/replicate_<r>/twin_final_states.csv)
into the input format infer_with_twinfer() and the TwinScore_supplement runner expect
(same layout as the Gillespie df_rows files): one row per (twin cell, saved post-branch time),
columns clone_id, cell_id, time_step, replicate, gene_<i>_mRNA.

Genes are renamed gene_1..gene_N in the row/column order of input_data/real_world_networks/<topology>.txt
so the same interaction matrix the Gillespie runs use lines up with the columns.
replicate=1 <-> twin_A, replicate=2 <-> twin_B; cell_id = clone_id + 6000*(replicate-1).
Pre-branch (trunk) rows are dropped: twins are not distinguishable before the branch.

Output: analysis_data/boolode_sims_real_networks/twinfer_format/<net>/replicate_<r>_simulation.csv (+ gene_map.json)
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
import os
from pathlib import Path

import pandas as pd

ROOT = f'{TWINFER_PROJECT_ROOT}'
SIM = f"{ROOT}/simulation_data/boolode_sims_replicates"
OUT = f"{ROOT}/analysis_data/boolode_sims_real_networks/twinfer_format"
N_CELLS = 6000
OLD = f"{ROOT}/simulation_data/twinfer_format"  # rule-derived interaction_matrix.txt + gene_order.txt for the 4 curated nets


def _old_genes(net):
    return [l.strip() for l in open(f"{OLD}/{net}/gene_order.txt") if l.strip()]

# gene order == row/col order of the topology matrix (see input_data/real_world_networks/network_labels.txt)
NETS = {
    "B_cell_activation": ["Ikaros", "PU_1", "Flk2", "IL_7R", "GATA_1", "E2A", "EBF", "C_EBPa", "PAX5", "Notch_1"],
    "EMT_real": ["Cdh1", "Cldn7", "Foxc2", "Grhl2", "Gsc", "Klf8", "Np63a", "Ovol2", "Snai1", "Snai2", "Tcf3",
                 "Tgfbeta", "Twist1", "Twist2", "Vim", "Zeb1", "Zeb2"],
    "Pluripotent_real": ["ARID3B", "CEBPZ", "ETV4", "FOXH1", "HIC2", "HMGB3", "JARID2", "LIN28B", "MIS18BP1", "MYCN",
                         "NANOG", "POU5F1", "POU5F1B", "PRDM14", "REST", "SALL2", "SALL4", "SMARCC1", "SOX2",
                         "TEAD2", "TERF1", "TGIF1", "WDHD1", "ZBTB12", "ZBTB39", "ZNF281", "ZNF286A", "ZNF286B",
                         "ZNF322", "ZNF398", "ZNF462", "ZNF730", "ZNF90", "ZNF92", "ZSCAN10", "ZSCAN2"],
    # curated BoolODE nets: BoolODE column order == gene_order.txt; matrix = OLD/<net>/interaction_matrix.txt
    **{n: _old_genes(n) for n in ("GSD", "HSC", "mCAD", "VSC")},
}


def n_reps(net):
    return len([d for d in os.listdir(f"{SIM}/{net}") if d.startswith("replicate_")
                and os.path.exists(f"{SIM}/{net}/{d}/twin_final_states.csv")])


def convert(net, genes, rep):
    d = pd.read_csv(f"{SIM}/{net}/replicate_{rep}/twin_final_states.csv")
    branch_tp = int(d.loc[d.branch_tp != -1, "branch_tp"].iloc[0])
    d = d[d.source.isin(["twin_A", "twin_B"]) & (d.step >= branch_tp)]
    frames = []
    for src, rep_id in (("twin_A", 1), ("twin_B", 2)):
        x = d[d.source == src]
        f = pd.DataFrame({
            "clone_id": x.pair_id.astype(int).to_numpy(),
            "cell_id": x.pair_id.astype(int).to_numpy() + N_CELLS * (rep_id - 1),
            "time_step": x.step.astype(int).to_numpy(),
            "replicate": rep_id,
        })
        for i, g in enumerate(genes):
            f[f"gene_{i + 1}_mRNA"] = x[g].to_numpy()
        frames.append(f)
    out = pd.concat(frames, ignore_index=True).sort_values(["time_step", "cell_id"]).reset_index(drop=True)
    return out, branch_tp


def main():
    for net, genes in NETS.items():
        odir = Path(OUT, net)
        odir.mkdir(parents=True, exist_ok=True)
        json.dump({f"gene_{i + 1}": g for i, g in enumerate(genes)}, open(odir / "gene_map.json", "w"), indent=1)
        for rep in range(n_reps(net)):
            out, b = convert(net, genes, rep)
            out.to_csv(odir / f"replicate_{rep}_simulation.csv", index=False)
            if rep == 0:
                print(f"{net}: branch_tp={b}, time_steps={sorted(out.time_step.unique())}, rows={len(out)}, "
                      f"clones={out.clone_id.nunique()}")
        print(f"  wrote {n_reps(net)} replicates -> {odir}")


if __name__ == "__main__":
    main()
