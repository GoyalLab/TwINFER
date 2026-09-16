# Enter paths below and run this cell to plot whatever network you inferred.
#
# inferred_network_path:
#   Leave blank to use this notebook's own `correlation_matrices` variable
#   (i.e. whatever you just inferred above). Or set it to a path to a saved
#   TwINFER output, e.g. ".../replicate_0/correlation_matrices.pkl".
#
# ground_truth_path:
#   Optional. Leave blank to only show the inferred network. Set it to a
#   path to a ground-truth adjacency matrix (e.g. ".../cycle_6_node.txt")
#   to plot it side by side with the inferred network for comparison.

inferred_network_path = ""
ground_truth_path = ""

if inferred_network_path.strip():
    _inferred = inferred_network_path.strip()
else:
    try:
        _inferred = correlation_matrices
    except NameError:
        raise RuntimeError(
            "No inferred_network_path given and no `correlation_matrices` "
            "found in this notebook. Either run the inference cells above, "
            "or set inferred_network_path to a .pkl/.json file."
        )

_ground_truth = ground_truth_path.strip() or None

fig = visualize_grn(_inferred, _ground_truth)
