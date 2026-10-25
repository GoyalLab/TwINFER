# twinfer.inference

| Module | Contents |
|---|---|
| `covariance_decomposition_z_scores.py` | **The TwINFER inference.** `infer_with_twinfer` scores gene pairs from twin pairs (one or two time points) or from clones of any size (real data): the z-scores `z_rho` (expression correlation), `z_rho_s` (twin-shared part) and `z_rho_Delta` (twin-difference correlation), the inference scenario (1 nothing, 2 regulation, 3 regulation + heterogeneity, 4 heterogeneity), and, with two time points, the direction from the cross-time correlation of sisters. `classify_inference_scenario` assigns the scenario from any three z-scores; with many pairs, z_rho and z_rho_Delta are tested with Benjamini-Hochberg (`q_values_of_z_rho`, `q_values_of_z_rho_Delta`). The module docstring gives every formula. |
| `ranked_edges.py` | `ranked_edge_lists` ranks the gene pairs of an `infer_with_twinfer` result by \|z_rho_s\| (shared state) and by \|z_rho_Delta\| (regulation), among the pairs whose z_rho is significant; one table per score with a `rank` column. With two time points it adds a list ranked by the cross-time correlation (`z_dag`, one row per direction of `pair_directions`: in scenario 2 each direction on its own, in scenario 3 by `delta` / `p_delta`). |
| `network_naming.py` | Converts networks with real gene names to the `gene_1, gene_2, ...` names of the simulator and back. |

Example (see the [tutorials](../../../tutorials/)):

```python
from twinfer.inference.covariance_decomposition_z_scores import infer_with_twinfer

result = infer_with_twinfer(table, ["gene_1_mRNA", "gene_2_mRNA"], time_points=[20])     # twin pairs at 20 h
result = infer_with_twinfer(cells, genes, regulators=tfs, targets=genes)                # real data: one row per cell with clone_id
```
