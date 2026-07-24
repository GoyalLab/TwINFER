import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize, LinearSegmentedColormap
from adjustText import adjust_text


__version__ = "2026-07-16-one-reciprocal-edge-curved-v6"


# ============================================================
# 1. Colormap
# ============================================================

def make_reds_blues_colormap(vmin=-0.05, vmax=0.18):
    """
    Negative weights: red
    Zero: white
    Positive weights: blue
    """
    vmin = float(vmin)
    vmax = float(vmax)

    if np.isclose(vmin, vmax):
        if np.isclose(vmin, 0.0):
            vmin = -1.0
            vmax = 1.0
        elif vmin > 0:
            vmin = 0.0
        else:
            vmax = 0.0

    zero_position = np.clip(
        (0.0 - vmin) / (vmax - vmin),
        0.0,
        1.0,
    )

    n_total = 256
    n_reds = int(round(zero_position * n_total))
    n_blues = n_total - n_reds

    denominator = max(
        abs(vmin),
        abs(vmax),
        1e-12,
    )

    red_intensity = abs(vmin) / denominator
    blue_intensity = abs(vmax) / denominator

    low_clip = 0.50
    color_parts = []

    if n_reds > 0:
        reds = plt.cm.Reds(
            np.linspace(
                low_clip
                + (1.0 - low_clip) * red_intensity,
                low_clip,
                n_reds,
            )
        )
        color_parts.append(reds)

    if n_blues > 0:
        blues = plt.cm.Blues(
            np.linspace(
                low_clip,
                low_clip
                + (1.0 - low_clip) * blue_intensity,
                n_blues,
            )
        )
        color_parts.append(blues)

    colors = np.vstack(color_parts)

    return LinearSegmentedColormap.from_list(
        "RedsBlues",
        colors,
    )


# ============================================================
# 2. Angular helpers
# ============================================================

def _wrap_angle(angle):
    """Map an angle to [-pi, pi]."""
    return np.arctan2(
        np.sin(angle),
        np.cos(angle),
    )


def _circular_angle_distance(angle_a, angle_b):
    """Smallest absolute angular difference."""
    return abs(
        _wrap_angle(angle_a - angle_b)
    )


def _circular_mean(angles):
    """Circular mean of a sequence of angles."""
    angles = np.asarray(
        angles,
        dtype=float,
    )

    return np.arctan2(
        np.mean(np.sin(angles)),
        np.mean(np.cos(angles)),
    )


def _centered_slots(count):
    """
    Examples
    --------
    1 -> [0]
    2 -> [-0.5, +0.5]
    3 -> [-1, 0, +1]
    """
    if count <= 0:
        return np.empty(
            0,
            dtype=float,
        )

    return (
        np.arange(count, dtype=float)
        - (count - 1) / 2.0
    )


# ============================================================
# 3. Group crowded terminals
# ============================================================

def _group_terminals(
    terminals,
    angle_tolerance,
):
    """
    Group terminals whose natural contact directions are close.

    terminals:
        [((source, target), angle), ...]
    """
    if not terminals:
        return []

    proximity_graph = nx.Graph()
    proximity_graph.add_nodes_from(
        range(len(terminals))
    )

    for first in range(len(terminals)):
        for second in range(
            first + 1,
            len(terminals),
        ):
            angle_difference = (
                _circular_angle_distance(
                    terminals[first][1],
                    terminals[second][1],
                )
            )

            if (
                angle_difference
                <= float(angle_tolerance)
            ):
                proximity_graph.add_edge(
                    first,
                    second,
                )

    groups = []

    for component in nx.connected_components(
        proximity_graph
    ):
        groups.append(
            [
                terminals[index]
                for index in component
            ]
        )

    return groups


def _assign_desired_contact_angles(
    terminals,
    angle_tolerance,
    terminal_spread,
):
    """
    Assign evenly spaced contact angles to crowded terminals.
    """
    desired_angles = {}
    crowded_edges = set()

    groups = _group_terminals(
        terminals,
        angle_tolerance=angle_tolerance,
    )

    for group in groups:
        if len(group) == 1:
            edge, natural_angle = group[0]

            desired_angles[edge] = float(
                natural_angle
            )
            continue

        natural_angles = np.array(
            [
                natural_angle
                for _, natural_angle in group
            ],
            dtype=float,
        )

        mean_angle = _circular_mean(
            natural_angles
        )

        ordered_group = sorted(
            group,
            key=lambda item: _wrap_angle(
                item[1] - mean_angle
            ),
        )

        slots = _centered_slots(
            len(ordered_group)
        )

        for (
            edge,
            natural_angle,
        ), slot in zip(
            ordered_group,
            slots,
        ):
            desired_angle = (
                mean_angle
                + slot * float(terminal_spread)
            )

            desired_angles[edge] = float(
                _wrap_angle(desired_angle)
            )

            crowded_edges.add(edge)

    return desired_angles, crowded_edges


# ============================================================
# 4. Straight-line terminal shifts
# ============================================================

def _build_edge_shift_fractions(
    graph,
    positions,
    terminal_angle_tolerance,
    terminal_spread,
    reciprocal_separation,
    max_shift_fraction,
):
    """
    Compute a perpendicular whole-line shift for each edge.

    The shift is represented as a fraction of node radius.
    """
    source_desired_angles = {}
    target_desired_angles = {}

    source_crowded = set()
    target_crowded = set()

    # --------------------------------------------------------
    # Assign desired source and target contact angles
    # --------------------------------------------------------
    for node in graph.nodes():
        node_position = np.asarray(
            positions[node],
            dtype=float,
        )

        outgoing_terminals = []

        for _, target in graph.out_edges(node):
            target_position = np.asarray(
                positions[target],
                dtype=float,
            )

            vector = (
                target_position
                - node_position
            )

            natural_source_angle = np.arctan2(
                vector[1],
                vector[0],
            )

            outgoing_terminals.append(
                (
                    (node, target),
                    natural_source_angle,
                )
            )

        (
            node_source_angles,
            node_source_crowded,
        ) = _assign_desired_contact_angles(
            outgoing_terminals,
            angle_tolerance=(
                terminal_angle_tolerance
            ),
            terminal_spread=terminal_spread,
        )

        source_desired_angles.update(
            node_source_angles
        )

        source_crowded.update(
            node_source_crowded
        )

        incoming_terminals = []

        for source, _ in graph.in_edges(node):
            source_position = np.asarray(
                positions[source],
                dtype=float,
            )

            vector = (
                source_position
                - node_position
            )

            natural_target_angle = np.arctan2(
                vector[1],
                vector[0],
            )

            incoming_terminals.append(
                (
                    (source, node),
                    natural_target_angle,
                )
            )

        (
            node_target_angles,
            node_target_crowded,
        ) = _assign_desired_contact_angles(
            incoming_terminals,
            angle_tolerance=(
                terminal_angle_tolerance
            ),
            terminal_spread=terminal_spread,
        )

        target_desired_angles.update(
            node_target_angles
        )

        target_crowded.update(
            node_target_crowded
        )

    # --------------------------------------------------------
    # Convert contact-angle requests into line shifts
    # --------------------------------------------------------
    shift_fractions = {}

    for source, target in graph.edges():
        edge = (
            source,
            target,
        )

        source_position = np.asarray(
            positions[source],
            dtype=float,
        )

        target_position = np.asarray(
            positions[target],
            dtype=float,
        )

        direction = (
            target_position
            - source_position
        )

        natural_source_angle = np.arctan2(
            direction[1],
            direction[0],
        )

        natural_target_angle = _wrap_angle(
            natural_source_angle + np.pi
        )

        source_shift = 0.0
        target_shift = 0.0

        if edge in source_crowded:
            desired_source_angle = (
                source_desired_angles[edge]
            )

            source_delta = _wrap_angle(
                desired_source_angle
                - natural_source_angle
            )

            source_shift = np.sin(
                source_delta
            )

        if edge in target_crowded:
            desired_target_angle = (
                target_desired_angles[edge]
            )

            target_delta = _wrap_angle(
                desired_target_angle
                - natural_target_angle
            )

            # Target orientation is reversed relative to source.
            target_shift = -np.sin(
                target_delta
            )

        # Target terminal receives priority because arrowheads
        # and repression bars are drawn at the target.
        if edge in target_crowded:
            shift_fraction = target_shift

            if (
                edge in source_crowded
                and not np.isclose(
                    source_shift,
                    0.0,
                )
                and not np.isclose(
                    target_shift,
                    0.0,
                )
                and np.sign(source_shift)
                == np.sign(target_shift)
            ):
                shift_fraction += (
                    0.15 * source_shift
                )

        elif edge in source_crowded:
            shift_fraction = source_shift

        else:
            shift_fraction = 0.0

        shift_fractions[edge] = float(
            shift_fraction
        )

    # --------------------------------------------------------
    # Reciprocal pairs still receive slight endpoint separation.
    #
    # This is necessary because curving only changes the middle of
    # the edge; without endpoint separation, arrowheads/bars could
    # still occupy exactly the same point.
    # --------------------------------------------------------
    processed_pairs = set()

    for source, target in graph.edges():
        if not graph.has_edge(
            target,
            source,
        ):
            continue

        pair_key = frozenset(
            (source, target)
        )

        if pair_key in processed_pairs:
            continue

        processed_pairs.add(
            pair_key
        )

        forward_edge = (
            source,
            target,
        )

        reverse_edge = (
            target,
            source,
        )

        forward_shift = shift_fractions.get(
            forward_edge,
            0.0,
        )

        reverse_shift = shift_fractions.get(
            reverse_edge,
            0.0,
        )

        # Reverse edges have opposite perpendicular vectors.
        # Therefore physical separation is proportional to:
        #
        #     abs(forward_shift + reverse_shift)
        current_separation = abs(
            forward_shift + reverse_shift
        )

        required_separation = float(
            reciprocal_separation
        )

        if (
            current_separation
            < required_separation
        ):
            pair_sum = (
                forward_shift
                + reverse_shift
            )

            direction_sign = (
                1.0
                if pair_sum >= 0.0
                else -1.0
            )

            additional_shift = (
                required_separation
                - current_separation
            ) / 2.0

            forward_shift += (
                direction_sign
                * additional_shift
            )

            reverse_shift += (
                direction_sign
                * additional_shift
            )

        shift_fractions[forward_edge] = float(
            np.clip(
                forward_shift,
                -max_shift_fraction,
                max_shift_fraction,
            )
        )

        shift_fractions[reverse_edge] = float(
            np.clip(
                reverse_shift,
                -max_shift_fraction,
                max_shift_fraction,
            )
        )

    for edge in shift_fractions:
        shift_fractions[edge] = float(
            np.clip(
                shift_fractions[edge],
                -max_shift_fraction,
                max_shift_fraction,
            )
        )

    return shift_fractions


# ============================================================
# 5. Radial staggering of crowded target symbols
# ============================================================

def _build_terminal_depth_slots(
    graph,
    positions,
    terminal_angle_tolerance,
):
    """
    Give crowded incoming symbols different radial distances.

    This helps when large arrowheads or repression bars are located
    close together on the same side of a node.
    """
    depth_slots = {}

    for target in graph.nodes():
        target_position = np.asarray(
            positions[target],
            dtype=float,
        )

        incoming_terminals = []

        for source, _ in graph.in_edges(target):
            source_position = np.asarray(
                positions[source],
                dtype=float,
            )

            vector = (
                source_position
                - target_position
            )

            angle = np.arctan2(
                vector[1],
                vector[0],
            )

            incoming_terminals.append(
                (
                    (source, target),
                    angle,
                )
            )

        groups = _group_terminals(
            incoming_terminals,
            angle_tolerance=(
                terminal_angle_tolerance
            ),
        )

        for group in groups:
            if len(group) == 1:
                edge, angle = group[0]
                depth_slots[edge] = 0.0
                continue

            group_angles = np.array(
                [
                    angle
                    for _, angle in group
                ],
                dtype=float,
            )

            mean_angle = _circular_mean(
                group_angles
            )

            # The terminal nearest the center of the sector stays
            # closest to the node. Others move progressively back.
            ordered_group = sorted(
                group,
                key=lambda item: abs(
                    _wrap_angle(
                        item[1] - mean_angle
                    )
                ),
            )

            for depth_level, (
                edge,
                angle,
            ) in enumerate(ordered_group):
                depth_slots[edge] = float(
                    depth_level
                )

    return depth_slots


# ============================================================
# 6. Shifted straight-line geometry
# ============================================================

def _shifted_circle_contacts(
    source_center,
    target_center,
    node_radius,
    shift_fraction,
):
    """
    Shift the entire source-target line and intersect it with both
    node circles.
    """
    source_center = np.asarray(
        source_center,
        dtype=float,
    )

    target_center = np.asarray(
        target_center,
        dtype=float,
    )

    center_difference = (
        target_center
        - source_center
    )

    center_distance = float(
        np.linalg.norm(
            center_difference
        )
    )

    if center_distance <= 1e-12:
        raise ValueError(
            "Source and target positions must differ."
        )

    unit = (
        center_difference
        / center_distance
    )

    perpendicular = np.array(
        [
            -unit[1],
            unit[0],
        ]
    )

    shift_fraction = float(
        np.clip(
            shift_fraction,
            -0.999,
            0.999,
        )
    )

    shift_distance = (
        shift_fraction
        * float(node_radius)
    )

    radial_fraction = np.sqrt(
        max(
            1.0
            - shift_fraction ** 2,
            0.0,
        )
    )

    radial_distance = (
        radial_fraction
        * float(node_radius)
    )

    start = (
        source_center
        + perpendicular * shift_distance
        + unit * radial_distance
    )

    end = (
        target_center
        + perpendicular * shift_distance
        - unit * radial_distance
    )

    return (
        start,
        end,
        unit,
        perpendicular,
    )


# ============================================================
# 7. Curve tangent
# ============================================================

def _arc3_end_tangent(
    start,
    end,
    radius,
):
    """
    Approximate the tangent of Matplotlib's arc3 curve at the end.

    This is used to orient a curved repression bar perpendicular
    to the curve rather than perpendicular to the center-to-center
    line.
    """
    start = np.asarray(
        start,
        dtype=float,
    )

    end = np.asarray(
        end,
        dtype=float,
    )

    difference = (
        end - start
    )

    control_point = (
        0.5 * (start + end)
        + float(radius)
        * np.array(
            [
                difference[1],
                -difference[0],
            ]
        )
    )

    tangent = (
        end - control_point
    )

    tangent_norm = float(
        np.linalg.norm(tangent)
    )

    if tangent_norm <= 1e-12:
        tangent = difference

        tangent_norm = float(
            np.linalg.norm(tangent)
        )

    if tangent_norm <= 1e-12:
        return np.array(
            [1.0, 0.0]
        )

    return (
        tangent / tangent_norm
    )


# ============================================================
# 8. Main plotting function
# ============================================================

def plot_grn(
    matrix,
    node_labels=None,
    figsize=None,
    node_color="#1a1a1a",
    font_size=None,
    label_font_size=None,
    font_color="black",
    title=None,
    ax=None,
    fixed_pos=None,
    show_colorbar=True,
    vmin=None,
    vmax=None,
    *,
    terminal_spread=0.38,
    terminal_angle_tolerance=1.10,
    reciprocal_separation=0.30,
    reciprocal_curve=0.18,
    terminal_depth_step=0.35,
    max_shift_fraction=0.92,
    edge_halo=True,
):
    """
    Plot a signed, weighted directed GRN.

    Matrix convention
    -----------------
    matrix[i, j] represents:

        node i -> node j

    Positive value:
        activation arrow

    Negative value:
        repression bar

    Reciprocal-pair rule
    --------------------
    If both u -> v and v -> u exist:

        lower-index source -> higher-index target:
            straight

        higher-index source -> lower-index target:
            curved

    Parameters
    ----------
    terminal_spread
        Angular spacing between neighboring crowded terminals.

    terminal_angle_tolerance
        Natural terminal directions within this angular distance are
        treated as crowded.

    reciprocal_separation
        Small endpoint separation retained for reciprocal edges.

    reciprocal_curve
        Magnitude of the curve used for one edge in each reciprocal
        pair.

    terminal_depth_step
        Radial staggering of crowded target symbols, in node-radius
        units.
    """
    matrix = np.asarray(
        matrix,
        dtype=float,
    )

    if matrix.ndim != 2:
        raise ValueError(
            f"Matrix is not 2D: shape={matrix.shape}"
        )

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"Matrix is not square: shape={matrix.shape}"
        )

    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            "Matrix contains NaN or infinite values."
        )

    n_nodes = int(
        matrix.shape[0]
    )

    if n_nodes == 0:
        raise ValueError(
            "Matrix must contain at least one node."
        )

    if figsize is None:
        side_length = max(
            6.0,
            min(
                20.0,
                np.sqrt(n_nodes) * 2.5,
            ),
        )

        figsize = (
            side_length,
            side_length,
        )

    node_radius = max(
        0.012,
        min(
            0.05,
            0.15 / np.sqrt(n_nodes),
        ),
    )

    if font_size is None:
        font_size = max(
            6,
            min(
                12,
                int(
                    35
                    / np.log1p(n_nodes)
                ),
            ),
        )

    if label_font_size is None:
        label_font_size = max(
            font_size + 2,
            int(
                round(
                    font_size * 1.30
                )
            ),
        )

    if node_labels is None:
        node_labels = [
            f"$g_{{{node + 1}}}$"
            for node in range(n_nodes)
        ]

    if len(node_labels) != n_nodes:
        raise ValueError(
            "node_labels must contain one label per node."
        )

    # --------------------------------------------------------
    # Build graph
    # --------------------------------------------------------
    graph = nx.DiGraph()
    graph.add_nodes_from(
        range(n_nodes)
    )

    for source in range(n_nodes):
        for target in range(n_nodes):
            if (
                source != target
                and matrix[source, target] != 0
            ):
                graph.add_edge(
                    source,
                    target,
                    weight=float(
                        matrix[source, target]
                    ),
                )

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------
    if fixed_pos is not None:
        missing_nodes = [
            node
            for node in graph.nodes()
            if node not in fixed_pos
        ]

        if missing_nodes:
            raise ValueError(
                "fixed_pos is missing nodes: "
                f"{missing_nodes}"
            )

        positions = {
            node: np.asarray(
                fixed_pos[node],
                dtype=float,
            )
            for node in graph.nodes()
        }

    else:
        # Clockwise circular fallback.
        angles = np.linspace(
            3.0 * np.pi / 4.0,
            3.0 * np.pi / 4.0
            - 2.0 * np.pi,
            n_nodes,
            endpoint=False,
        )

        positions = {
            node: np.array(
                [
                    0.5
                    + 0.36 * np.cos(angle),
                    0.5
                    + 0.36 * np.sin(angle),
                ]
            )
            for node, angle in enumerate(
                angles
            )
        }

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(
            figsize=figsize
        )
        own_figure = True
    else:
        fig = ax.get_figure()
        own_figure = False

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    if title:
        ax.set_title(
            title,
            fontsize=font_size + 2,
            fontweight="bold",
            pad=10,
        )

    figure_scale = (
        min(figsize) / 8.0
    )

    head_scale = figure_scale

    # --------------------------------------------------------
    # Edge colors
    # --------------------------------------------------------
    edge_weights = np.array(
        [
            data["weight"]
            for _, _, data
            in graph.edges(data=True)
        ],
        dtype=float,
    )

    if edge_weights.size == 0:
        minimum_weight = (
            -1.0
            if vmin is None
            else float(vmin)
        )

        maximum_weight = (
            1.0
            if vmax is None
            else float(vmax)
        )

    else:
        minimum_weight = (
            min(
                0.0,
                float(edge_weights.min()),
            )
            if vmin is None
            else float(vmin)
        )

        maximum_weight = (
            max(
                0.0,
                float(edge_weights.max()),
            )
            if vmax is None
            else float(vmax)
        )

    if np.isclose(
        minimum_weight,
        maximum_weight,
    ):
        if np.isclose(
            minimum_weight,
            0.0,
        ):
            minimum_weight = -1.0
            maximum_weight = 1.0

        elif minimum_weight > 0:
            minimum_weight = 0.0

        else:
            maximum_weight = 0.0

    colormap = make_reds_blues_colormap(
        minimum_weight,
        maximum_weight,
    )

    color_norm = Normalize(
        vmin=minimum_weight,
        vmax=maximum_weight,
    )

    # --------------------------------------------------------
    # Terminal positions
    # --------------------------------------------------------
    edge_shift_fractions = (
        _build_edge_shift_fractions(
            graph=graph,
            positions=positions,
            terminal_angle_tolerance=(
                terminal_angle_tolerance
            ),
            terminal_spread=terminal_spread,
            reciprocal_separation=(
                reciprocal_separation
            ),
            max_shift_fraction=(
                max_shift_fraction
            ),
        )
    )

    terminal_depth_slots = (
        _build_terminal_depth_slots(
            graph=graph,
            positions=positions,
            terminal_angle_tolerance=(
                terminal_angle_tolerance
            ),
        )
    )

    line_width = (
        2.0 * figure_scale
    )

    terminal_gap = (
        node_radius * 0.40
    )

    # --------------------------------------------------------
    # Draw edges
    # --------------------------------------------------------
    for source, target, data in graph.edges(
        data=True
    ):
        edge = (
            source,
            target,
        )

        weight = float(
            data["weight"]
        )

        repression = (
            weight < 0
        )

        color = colormap(
            color_norm(weight)
        )

        shift_fraction = (
            edge_shift_fractions.get(
                edge,
                0.0,
            )
        )

        (
            start,
            end,
            unit,
            perpendicular,
        ) = _shifted_circle_contacts(
            source_center=positions[source],
            target_center=positions[target],
            node_radius=node_radius,
            shift_fraction=shift_fraction,
        )

        terminal_depth_level = (
            terminal_depth_slots.get(
                edge,
                0.0,
            )
        )

        terminal_depth = (
            terminal_depth_level
            * float(terminal_depth_step)
            * node_radius
        )

        # ----------------------------------------------------
        # Reciprocal curve rule
        # ----------------------------------------------------
        is_reciprocal = graph.has_edge(
            target,
            source,
        )

        curve_this_edge = (
            is_reciprocal
            and source > target
        )

        if curve_this_edge:
            # Bow the curved edge farther toward its current shifted
            # side rather than bending it back toward the straight edge.
            if np.isclose(
                shift_fraction,
                0.0,
            ):
                curve_sign = 1.0
            else:
                # For Matplotlib arc3, positive rad bows toward
                # -perpendicular, hence the negative sign.
                curve_sign = -np.sign(
                    shift_fraction
                )

            curve_radius = (
                curve_sign
                * abs(float(reciprocal_curve))
            )

        else:
            curve_radius = 0.0

        connection_style = (
            f"arc3,rad={curve_radius}"
        )

        # ----------------------------------------------------
        # Repression
        # ----------------------------------------------------
        if repression:
            bar_end = (
                end
                - unit
                * (
                    terminal_gap
                    + terminal_depth
                )
            )

            if curve_this_edge:
                if edge_halo:
                    ax.annotate(
                        "",
                        xy=bar_end,
                        xytext=start,
                        arrowprops=dict(
                            arrowstyle="-",
                            color="white",
                            lw=line_width + 2.4,
                            connectionstyle=(
                                connection_style
                            ),
                            shrinkA=0,
                            shrinkB=0,
                        ),
                        zorder=1.3,
                    )

                ax.annotate(
                    "",
                    xy=bar_end,
                    xytext=start,
                    arrowprops=dict(
                        arrowstyle="-",
                        color=color,
                        lw=line_width,
                        connectionstyle=(
                            connection_style
                        ),
                        shrinkA=0,
                        shrinkB=0,
                    ),
                    zorder=2.0,
                )

                terminal_tangent = (
                    _arc3_end_tangent(
                        start,
                        bar_end,
                        curve_radius,
                    )
                )

                bar_perpendicular = np.array(
                    [
                        -terminal_tangent[1],
                        terminal_tangent[0],
                    ]
                )

            else:
                if edge_halo:
                    ax.plot(
                        [
                            start[0],
                            bar_end[0],
                        ],
                        [
                            start[1],
                            bar_end[1],
                        ],
                        color="white",
                        lw=line_width + 2.4,
                        solid_capstyle="round",
                        zorder=1.3,
                    )

                ax.plot(
                    [
                        start[0],
                        bar_end[0],
                    ],
                    [
                        start[1],
                        bar_end[1],
                    ],
                    color=color,
                    lw=line_width,
                    solid_capstyle="round",
                    zorder=2.0,
                )

                bar_perpendicular = (
                    perpendicular
                )

            # Original larger repression bar.
            bar_half_length = (
                node_radius
                * 0.50
                * head_scale
            )

            point_1 = (
                bar_end
                - bar_perpendicular
                * bar_half_length
            )

            point_2 = (
                bar_end
                + bar_perpendicular
                * bar_half_length
            )

            if edge_halo:
                ax.plot(
                    [
                        point_1[0],
                        point_2[0],
                    ],
                    [
                        point_1[1],
                        point_2[1],
                    ],
                    color="white",
                    lw=(
                        line_width * 1.50
                        + 2.4
                    ),
                    solid_capstyle="butt",
                    zorder=2.2,
                )

            ax.plot(
                [
                    point_1[0],
                    point_2[0],
                ],
                [
                    point_1[1],
                    point_2[1],
                ],
                color=color,
                lw=line_width * 1.50,
                solid_capstyle="butt",
                zorder=2.6,
            )

        # ----------------------------------------------------
        # Activation
        # ----------------------------------------------------
        else:
            # Original larger arrowhead.
            head_length = (
                node_radius
                * 1.00
                * head_scale
            )

            head_width = (
                node_radius
                * 0.70
                * head_scale
            )

            arrow_style = (
                f"-|>,"
                f"head_length="
                f"{head_length / node_radius:.2f},"
                f"head_width="
                f"{head_width / node_radius:.2f}"
            )

            arrow_end = (
                end
                - unit * terminal_depth
            )

            arrow_start = (
                start
                + unit
                * 1.50
                * terminal_gap
            )

            if edge_halo:
                ax.annotate(
                    "",
                    xy=arrow_end,
                    xytext=arrow_start,
                    arrowprops=dict(
                        arrowstyle=arrow_style,
                        color="white",
                        lw=line_width + 2.4,
                        connectionstyle=(
                            connection_style
                        ),
                        shrinkA=0,
                        shrinkB=0,
                    ),
                    zorder=1.3,
                )

            ax.annotate(
                "",
                xy=arrow_end,
                xytext=arrow_start,
                arrowprops=dict(
                    arrowstyle=arrow_style,
                    color=color,
                    lw=line_width,
                    connectionstyle=(
                        connection_style
                    ),
                    shrinkA=0,
                    shrinkB=0,
                ),
                zorder=2.0,
            )

    # --------------------------------------------------------
    # Draw nodes
    # --------------------------------------------------------
    for node in graph.nodes():
        ax.add_patch(
            plt.Circle(
                positions[node],
                node_radius * 1.06,
                facecolor="white",
                edgecolor="none",
                zorder=3.5,
            )
        )

        ax.add_patch(
            plt.Circle(
                positions[node],
                node_radius,
                facecolor=node_color,
                edgecolor="white",
                linewidth=1.3,
                zorder=4.0,
            )
        )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------
    centroid = np.mean(
        [
            positions[node]
            for node in graph.nodes()
        ],
        axis=0,
    )

    label_texts = []

    for node in graph.nodes():
        x, y = positions[node]

        outward = (
            np.array([x, y])
            - centroid
        )

        outward_norm = float(
            np.linalg.norm(outward)
        )

        if outward_norm > 1e-6:
            label_direction = (
                outward / outward_norm
            )
        else:
            label_direction = np.array(
                [0.0, 1.0]
            )

        label_x = (
            x
            + label_direction[0]
            * node_radius
            * 2.0
        )

        label_y = (
            y
            + label_direction[1]
            * node_radius
            * 2.0
        )

        label_text = ax.text(
            label_x,
            label_y,
            node_labels[node],
            ha="center",
            va="center",
            fontsize=label_font_size,
            color=font_color,
            fontweight="bold",
            zorder=6,
        )

        label_texts.append(
            label_text
        )

    if fixed_pos is None:
        node_x_values = [
            positions[node][0]
            for node in graph.nodes()
        ]

        node_y_values = [
            positions[node][1]
            for node in graph.nodes()
        ]

        adjust_text(
            label_texts,
            x=node_x_values,
            y=node_y_values,
            ax=ax,
            force_points=(0.8, 0.8),
            force_text=(0.2, 0.2),
            ensure_inside_axes=True,
            arrowprops=dict(
                arrowstyle="-",
                color="#bdbdc4",
                lw=0.4,
                alpha=0.4,
            ),
        )

    # --------------------------------------------------------
    # Colorbar
    # --------------------------------------------------------
    if show_colorbar:
        scalar_mappable = (
            plt.cm.ScalarMappable(
                cmap=colormap,
                norm=color_norm,
            )
        )

        scalar_mappable.set_array([])

        fig.colorbar(
            scalar_mappable,
            ax=ax,
            orientation="vertical",
            fraction=0.03,
            pad=0.02,
            label="Edge Weight",
        )

    if own_figure:
        fig.tight_layout()

    return fig, ax