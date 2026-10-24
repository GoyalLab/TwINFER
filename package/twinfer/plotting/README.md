# twinfer.plotting

| Module | Contents |
|---|---|
| `network_plots.py` | `plot_grn` draws a signed, weighted, directed network (`matrix[i, j]` = effect of node i on node j; positive = arrow, negative = bar; reciprocal edges curved). Per-edge line styles (`edge_styles`) and edges without direction (`undirected_edges`) are optional. `network_from_scenarios` turns the output of `infer_with_twinfer` into the inputs of `plot_grn`: scenarios 2-3 solid, scenario 4 dashed without direction, direction from the cross-time correlation when two time points were scored. |
| `assets/fonts/` | Arial font files used by the figure scripts. |

Example:

```python
from twinfer.plotting.network_plots import network_from_scenarios, plot_grn

plot_kwargs, edges = network_from_scenarios(result)     # result of infer_with_twinfer
fig, ax = plot_grn(**plot_kwargs)
```
