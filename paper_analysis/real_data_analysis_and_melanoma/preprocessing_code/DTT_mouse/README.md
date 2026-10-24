# DTT mouse embryo (E13.5, DNA Typewriter; Yu et al.): GEO matrices + lineage tree -> sister pairs and counts

**Raw data**: GEO GSE341627 (`raw_data/DTT_GEO_GSE341627`) and the authors' repository github.com/shendurelab/dtt-mouse (`raw_data/DTT_authors_repo_tree`:
dated tree, cell metadata). The cell lists `wanted_cells.csv` and `wanted_cells_with_cousins.csv` are in `input_folder/`.

There are no clone ids: twins are **sister pairs** of the dated lineage tree = an internal node with exactly two sequenced leaf children, parent
dated E12.5 or later, both cells in the backbone tree (cells with >= 7 recorder tapes). These are inferred from the recorder edits, not observed divisions.

**Steps** (launcher `run_rebuild_dtt.sbatch`, which runs `run_dtt_rebuild.sh`)
1. `01_sister_pairs.py`: tree -> `sister_pairs_strict.csv` (32,146 pairs), `cousin_cells.csv`, `all_cherries_age_edits.csv`.
2. `04_extract_counts.py <GSM>` (one run per GEO matrix): raw counts of the cells of interest -> `counts_wanted/<GSM>.npz`.
3. `05_expressed_gene_list.py`: analysis matrix `counts_samecell_analysis.npz` (32,588 twin cells + 52,130 singleton cells) and the genes detected in >= 5 %.
`diff_dtt.py` compares a rerun with the reference files.
