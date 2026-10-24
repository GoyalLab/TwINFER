# Installation

Requires Python ≥ 3.10.

```bash
git clone https://github.com/GoyalLab/TwINFER.git
cd TwINFER
conda create -n twinfer python=3.12
conda activate twinfer
pip install -e package/          # installs twinfer and its dependencies
pip install jupyterlab anndata   # only to run the tutorials
```

The example real dataset (`real_data/LSK_d2_d4_d6.h5ad`, 238 MB) is stored with [Git LFS](https://git-lfs.com). If it is only a small text file after cloning, run `git lfs install` once and then `git lfs pull` in the repository.

Check the installation:

```bash
python -c "import twinfer; print(twinfer.__file__)"
```

## Quick start

```python
import pandas as pd
from twinfer.inference.covariance_decomposition_z_scores import infer_with_twinfer
from twinfer.plotting.network_plots import network_from_scenarios, plot_grn

# Twin pairs: one row per cell and time point, columns clone_id, replicate (1 or 2 = which sister), time_step, one column per gene
table = pd.read_csv("simulation_example_output_data/df_rows_0_0_24102026_140638_ncells_6000_A_to_B_04cf9c46.csv")
genes = ["gene_1_mRNA", "gene_2_mRNA"]

result = infer_with_twinfer(table, genes, time_points=[20])          # z_rho, z_rho_s, z_rho_Delta, inference_scenario per gene pair
result = infer_with_twinfer(table, genes, time_points=[1, 20])       # adds the direction (z_dag_xy, z_dag_yx, delta, p_delta)

# Real data: one row per cell with clone_id and normalised expression (no replicate column); clones of any size
# result = infer_with_twinfer(cells, genes)                         # or regulators=[...], targets=[...]

# Network: scenarios 2-3 solid, scenario 4 dashed
plot_kwargs, edges = network_from_scenarios(result)
plot_grn(**plot_kwargs)
```

For count data, normalise before calling (for example log1p of counts per 10,000).
