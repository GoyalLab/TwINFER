
#%%
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from community import community_louvain
from matplotlib.colors import Normalize, LinearSegmentedColormap
from adjustText import adjust_text
#%%
def make_reds_blues_colormap(vmin=-0.05, vmax=0.18):
    """Custom red–white–blue colormap with pure white at 0, asymmetric."""
    zero_position = max(0, (0 - vmin) / (vmax - vmin))
    
    n_total = 256
    n_reds = int(zero_position * n_total)
    n_blues = n_total - n_reds

    red_intensity  = abs(vmin) / max(abs(vmin), abs(vmax))
    blue_intensity = abs(vmax) / max(abs(vmin), abs(vmax))

    # Sample from 0.45 → avoids the near-white low end of each colormap
    LOW_CLIP = 0.35

    reds  = plt.cm.Reds (np.linspace(LOW_CLIP + (1 - LOW_CLIP) * red_intensity,  LOW_CLIP, n_reds))
    blues = plt.cm.Blues(np.linspace(LOW_CLIP, LOW_CLIP + (1 - LOW_CLIP) * blue_intensity, n_blues))

    colors = np.vstack((reds, blues))
    return LinearSegmentedColormap.from_list('RedsBlues', colors)
#%%
def plot_grn(
    matrix,
    node_labels=None,
    figsize=None,   
    node_color="#1a1a1a",
    font_size=None,
    font_color="black",
    title=None,
    ax=None,
):
    """
    Plot a signed, weighted GRN from an asymmetric matrix with conditional
    curving and size-dependent layout auto-scaling.
    """
    matrix = np.array(matrix)
    n = matrix.shape[0]

    # --- 1. Dynamic Auto-Scaling Factors based on Network Size (n) ---
    if figsize is None:
        # Dynamically scale frame between 6x6 and 20x20 inches depending on n
        side_length = max(6, min(20, int(np.sqrt(n) * 2.5)))
        figsize = (side_length, side_length)

    # Scale base geometric dimensions
    node_r = max(0.012, min(0.05, 0.15 / np.sqrt(n))) if n > 0 else 0.025
    
    if font_size is None:
        font_size = max(6, min(12, int(35 / np.log1p(n))))

    if node_labels is None:
        node_labels = [f"$g_{{{i+1}}}$" for i in range(n)]

    # --- Build graph ---
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if i != j and matrix[i, j] != 0:
                G.add_edge(i, j, weight=matrix[i, j])

    # --- Scale edge weights: map abs values to [0, 1] ---
    weights = np.array([abs(d["weight"]) for _, _, d in G.edges(data=True)])
    if len(weights) > 0 and weights.max() > weights.min():
        w_min, w_max = weights.min(), weights.max()
        for u, v, d in G.edges(data=True):
            d["scaled"] = (abs(d["weight"]) - w_min) / (w_max - w_min)
    else:
        for u, v, d in G.edges(data=True):
            d["scaled"] = 1.0

    # Edge widths scale proportionally with frame sizes
    fig_scale = min(figsize) / 8.0
    # lw_min = 1 * fig_scale
    # lw_max = 4.0 * fig_scale
    head_scale = fig_scale  

    # --- Community detection ---
    G_un = nx.Graph()
    G_un.add_nodes_from(G.nodes())
    for u, v, d in G.edges(data=True):
        w = abs(d["weight"])
        if G_un.has_edge(u, v):
            G_un[u][v]["weight"] = max(G_un[u][v]["weight"], w)
        else:
            G_un.add_edge(u, v, weight=w)
    if len(G_un.edges) > 0:
        try:
            partition = community_louvain.best_partition(G_un)
        except Exception:
            partition = {nd: 0 for nd in G.nodes()}
    else:
        partition = {nd: i for i, nd in enumerate(G.nodes())}

    communities = {}
    for nd, comm in partition.items():
        communities.setdefault(comm, []).append(nd)
    
    # --- Layout ---

    # Properly extract singletons as flat values
    singletons = [members[0] for members in communities.values() if len(members) == 1]
    real_comms  = {cid: members for cid, members in communities.items() if len(members) > 1}

    # Global scaling factor for community spacing
    intra_scale = 0.750

    # Build a meta-graph of real communities to space them out globally
    comm_ids = list(real_comms.keys())

    node_to_comm = {nd: cid for cid, members in real_comms.items() for nd in members}
    print(node_to_comm)
    meta = nx.Graph()
    meta.add_nodes_from(comm_ids)
    for u, v in G_un.edges():
        cu = node_to_comm.get(u)
        cv = node_to_comm.get(v)
        if cu is not None and cv is not None and cu != cv:
            if meta.has_edge(cu, cv):
                meta[cu][cv]["weight"] += 1
            else:
                meta.add_edge(cu, cv, weight=1)

    # Position distinct community clusters widely apart from each other
    if len(comm_ids) > 1:
        comm_centers = nx.spring_layout(meta, seed=42, k=3.5 / np.sqrt(len(comm_ids)))
    elif len(comm_ids) == 1:
        comm_centers = {comm_ids[0]: np.array([0.0, 0.0])}
    else:
        comm_centers = {}

    raw_pos = {}

    # Layout nodes with guaranteed minimum spacing inside each community
    for cid, members in real_comms.items():
        cx, cy = comm_centers[cid]
        subg = G_un.subgraph(members)
        
        if len(members) == 2:
            # Spaced out pair configuration
            sub = {members[0]: np.array([-1.2, 0.0]),
                   members[1]: np.array([ 1.2, 0.0])}
        elif len(members) <= 8:
            # FIX: Use a wide-radius circle (1.6) so nodes can NEVER overlap each other,
            # giving internal crossing lines plenty of clearance through the center
            angles = np.linspace(0, 2 * np.pi, len(members), endpoint=False)
            sub = {nd: 2.6 * np.array([np.cos(a), np.sin(a)]) for nd, a in zip(members, angles)}
        else:
            # Spring layout for large communities
            sub = nx.spring_layout(subg, seed=42, k=1 / np.sqrt(len(members)))
            
        # Place the nodes around their respective community centers
        for nd, p in sub.items():
            raw_pos[nd] = np.array([cx + p[0] * intra_scale,
                                    cy + p[1] * intra_scale])

    # --- Normalization Strategy ---
    pos = {}
    
    
    if raw_pos:
        all_xy = np.array(list(raw_pos.values()))
        mn, mx = all_xy.min(axis=0), all_xy.max(axis=0)
        span = mx - mn
        if singletons:
            max_rows_per_col = max(5, int(np.sqrt(len(singletons)) * 2))
            n_cols = int(np.ceil(len(singletons) / max_rows_per_col))
            x_gap = 0.9
            singleton_width = n_cols * 0.2 + 0.05  # total width needed
        else:
            n_cols = 0
            singleton_width = 0.1

        # Then use it in normalization
        print(f"Singleton width: {singleton_width:.3f}")
        margin_right = singleton_width
        margin_left = 0.075
        margin_y = 0.075

        canvas_x = 1 - margin_left - margin_right
        canvas_y = 1 - 2 * margin_y

        for nd in raw_pos:
            normalized = np.zeros(2)
            for dim in range(2):
                if span[dim] > 1e-6:
                    normalized[dim] = (raw_pos[nd][dim] - mn[dim]) / span[dim]
                else:
                    normalized[dim] = 0.5  # centre collapsed dimension

            pos[nd] = np.array([
                margin_left + normalized[0] * canvas_x,
                margin_y   + normalized[1] * canvas_y,
            ])

        start_x = 1 - n_cols * 0.1 - 0.02
        start_y = 0.9
    else:
        start_x = 0.1
        start_y = 0.90
    # Grid layout for unconnected nodes (singletons) on the right sidebar
    if singletons:
        max_rows_per_col = max(5, int(np.sqrt(len(singletons)) * 2))
        y_gap = 0.7 / max_rows_per_col  
        x_gap = 0.09                     
        
        for k, nd in enumerate(singletons):
            col_idx = k // max_rows_per_col
            row_idx = k % max_rows_per_col
            
            x_pos = start_x + (col_idx * x_gap)
            y_pos = start_y - (row_idx * y_gap)
            
            pos[nd] = np.array([x_pos, y_pos])


    # --- Figure Initialization ---
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=font_size + 2, fontweight="bold", pad=10)
    max_w = max(0, max((d["weight"]) for _, _, d in G.edges(data=True)))
    min_w = min(0, min((d["weight"]) for _, _, d in G.edges(data=True)))
    cmap = make_reds_blues_colormap(min_w, max_w)
    norm_color = Normalize(vmin=min_w, vmax=max_w)
    # --- Draw edges ---
    for u, v, d in G.edges(data=True):
        lw = 2.0 * fig_scale  # single fixed thickness
        repression = d["weight"] < 0
        
        color = cmap(norm_color(d["weight"]))

        pu, pv = np.array(pos[u]), np.array(pos[v])
        diff = pv - pu
        dist = np.linalg.norm(diff)
        if dist == 0:
            continue
        unit = diff / dist

        start = pu + unit * node_r
        end   = pv - unit * node_r

        # MODIFICATION: Curve ONLY if reciprocal paths coexist
        rad_val = "0.25" if G.has_edge(v, u) else ".10"
        c_style = f"arc3,rad={rad_val}"

                # MODIFICATION: Add a slight gap factor to pull the repression bar away from the node
        gap_factor = node_r * 0.4  

        if repression:
            # Shift the visual end of the line backward slightly
            bar_end = end - unit * gap_factor

            ax.annotate(
                "", xy=bar_end, xytext=start,
                arrowprops=dict(
                    arrowstyle="-",
                    color=color, lw=lw,
                    connectionstyle=c_style,
                ),
                zorder=2,
            )
            
            # Recalculate entry angle for the flat bar based on the curve geometry
            if G.has_edge(v, u):
                mid = (start + bar_end) / 2 + np.array([-unit[1], unit[0]]) * 0.0075
                target_diff = bar_end - mid
                unit_at_end = target_diff / np.linalg.norm(target_diff)
                perp = np.array([-unit_at_end[1], unit_at_end[0]])
            else:
                perp = np.array([-unit[1], unit[0]])
                
            bar_len = node_r * 0.5 * head_scale
            p1 = bar_end - perp * bar_len
            p2 = bar_end + perp * bar_len
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color=color, lw=lw * 1.5,
                    solid_capstyle="butt", zorder=3)

        else:
            hl = node_r * 1.0 * head_scale
            hw = node_r * 0.7 * head_scale
            ax.annotate(
                "", xy=end, xytext=start + unit * 1.5*gap_factor,
                arrowprops=dict(
                    arrowstyle=f"-|>,head_length={hl/node_r:.2f},head_width={hw/node_r:.2f}",
                    color=color, lw=lw,
                    connectionstyle=c_style,
                ),
                zorder=2,
            )

    # --- Draw nodes ---
    for nd in G.nodes():
        circle = plt.Circle(pos[nd], node_r, color=node_color, zorder=4)
        ax.add_patch(circle)

    # --- Labels (Vector-Driven Angular Offset Alignment) ---
    for nd in G.nodes():
        x, y = pos[nd]
        
        # 1. Accumulate all directional connection lines tied to this node
        vectors = []
        for neighbor in G.neighbors(nd):  # Outgoing edges
            vectors.append(np.array(pos[neighbor]) - np.array(pos[nd]))
        for source, _ in G.in_edges(nd):   # Incoming edges
            vectors.append(np.array(pos[source]) - np.array(pos[nd]))
            
        # 2. Compute the optimal text push direction
        if len(vectors) > 0:
            # Sum up all connected line trajectories
            summed_vector = np.sum(vectors, axis=0)
            norm = np.linalg.norm(summed_vector)
            
            if norm > 1e-4:
                # Invert the sum vector to push text into the emptiest available quadrant
                push_direction = -summed_vector / norm
            else:
                # If vectors perfectly cancel out, push top-right diagonally
                push_direction = np.array([1.0, 1.0]) / np.sqrt(2)
        else:
            # Singletons have no edges; push top-right cleanly
            push_direction = np.array([1.0, 1.0]) / np.sqrt(2)

        # 3. Calculate safe label boundaries outside the node's radius
        # The text base sits 1.35x node radii away from the center to prevent touching edges
        text_x = x + push_direction[0] * (node_r * 1.2)
        text_y = y + push_direction[1] * (node_r * 1.2)

        # 4. Dynamically anchor alignments to prevent long words from drifting back
        # If pushing left, align text right; if pushing down, align text top
        ha_align = "left" if push_direction[0] >= 0 else "right"
        va_align = "bottom" if push_direction[1] >= 0 else "top"

    # --- Labels ---
    texts = []
    for nd in G.nodes():
        x, y = pos[nd]
        
        # Start label already outside the node
        t = ax.text(
            x + 0.5*node_r,  # initial offset outside node
            y + 0.5*node_r,
            node_labels[nd],
            fontsize=font_size,
            color=font_color,
            fontweight="bold",
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.5)
        )
        texts.append(t)
    node_xs = [pos[nd][0] for nd in G.nodes()]
    node_ys = [pos[nd][1] for nd in G.nodes()]
    adjust_text(
        texts,
        x=node_xs,
        y=node_ys,
        ax=ax,
        expand=(1, 1),
        force_points=(0.8, 0.8),
        force_text=(0.1, 0.1),
        ensure_inside_axes=True,
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.5)
    )

    # --- Legend ---
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_color)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.03, pad=0.02, label="Edge Weight")

    fig.tight_layout()
    return fig, ax
#%%
if __name__ == "__main__":
    np.random.seed(0)
    n = 14
    labels = [f"$g_{{{i+1}}}$" for i in range(n)]

    gene_matrix = np.zeros((n, n))
    edges = [
        (0, 1, -0.8), (1, 0,  0.6), (0, 2,  0.7),
        (1, 3, -0.5), (4, 5, -0.9), (4, 6,  0.75),
        (6, 5,  0.6), (9, 10,-0.9), (10,11,-0.85),
        (11,13, 0.6), (11,12, 0.5), (12,13, 0.7),
        (13,12, 0.5),(6, 13,-0.4)
    ]
    for i, j, v in edges:
        gene_matrix[i, j] = v

    fig, ax = plot_grn(gene_matrix, node_labels=labels)
    fig.show()