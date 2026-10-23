"""Real-network topologies drawn in the same style as paper/networks_cycles.pdf.

Date: 2026-10-04
Data: input_data/real_world_networks/<net>.txt (canonical, directed, unsigned: edge i->j if M[i,j] != 0, i != j).
Layout: nodes on a regular circle (as in networks_cycles.pdf), white circles with dark outline.
        Activation = arrow head, repression = bar head (blue). Labels fixed at 8 pt (minimum font size).
        Every drawing unit is one point, so node size and font are identical across networks and each ring
        is sized from its gene count so that neighbouring circles never touch.
Labels: gene names from input_data/real_world_networks/network_labels.txt (GSD: alphabetical order, as in GSD.txt).
Output: final_plots/benchmark/real_network_simulations/real_networks_all.pdf (one page, all 7 networks)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.lines import Line2D

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['mathtext.fontset'] = "cm"
from matplotlib import font_manager as fm
ARIAL_DIR = "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/package/twinfer/plotting/assets/fonts"
for f in ["Arial.ttf", "Arial Bold.ttf", "Arial Italic.ttf", "Arial Bold Italic.ttf"]:
    fm.fontManager.addfont(os.path.join(ARIAL_DIR, f))
plt.rcParams['font.sans-serif'] = ["Arial"]

RW = "/gpfs/projects/b1255/hzhang/TwINFER_KA/input_data/real_world_networks"
OUT = "/gpfs/projects/b1255/hzhang/TwINFER_KA/final_plots/benchmark/real_network_simulations"
os.makedirs(OUT, exist_ok=True)

NAMES = {
    "mCAD": ["Pax6", "Coup", "Emx2", "Fgf8", "Sp8"],
    "VSC": ["Pax6", "Nkx2.2", "Dbx2", "Olig2", "Nkx6.1", "Irx3", "Dbx1", "Nkx6.2"],
    "HSC": ["cJun", "SCL", "Eklf", "Gata1", "PU.1", "CEBPa", "Gata2", "Gfi1", "EgrNab", "Fli1", "Fog"],
    "B_cell": ["Ikaros", "PU.1", "Flk2", "IL-7R", "GATA-1", "E2A", "EBF", "C/EBPa", "Pax-5", "Notch-1"],
    "GSD": sorted(["UGR", "CBX2", "PGD2", "DMRT1", "DHH", "DKK1", "AMH", "WNT4", "RSPO1", "FOXL2", "CTNNB1",
                   "GATA4", "WT1mKTS", "WT1pKTS", "NR5A1", "NR0B1", "SRY", "SOX9", "FGF9"]),
    "EMT": ["Cdh1", "Cldn7", "Foxc2", "Grhl2", "Gsc", "Klf8", "Np63a", "Ovol2", "Snai1", "Snai2", "Tcf3",
            "Tgfbeta", "Twist1", "Twist2", "Vim", "Zeb1", "Zeb2"],
    "Pluripotent": ["ARID3B", "CEBPZ", "ETV4", "FOXH1", "HIC2", "HMGB3", "JARID2", "LIN28B", "MIS18BP1", "MYCN",
                    "NANOG", "POU5F1", "POU5F1B", "PRDM14", "REST", "SALL2", "SALL4", "SMARCC1", "SOX2",
                    "TEAD2", "TERF1", "TGIF1", "WDHD1", "ZBTB12", "ZBTB39", "ZNF281", "ZNF286A", "ZNF286B",
                    "ZNF322", "ZNF398", "ZNF462", "ZNF730", "ZNF90", "ZNF92", "ZSCAN10", "ZSCAN2"],
}
TOPO = {"mCAD": "mCAD.txt", "VSC": "VSC.txt", "HSC": "HSC.txt", "B_cell": "B_cell.txt",
        "GSD": "GSD.txt", "EMT": "EMT.txt", "Pluripotent": "Pluripotent.txt"}

BLUE = "#08306b"   # activation (network_plots.py colormap, weight +1)
RED = "#67000d"    # repression (network_plots.py colormap, weight -1)
FONT = 8.0          # minimum font size used in the paper figures
CHAR_W = 0.56       # approx. average glyph width in units of font size (Arial)
GAP_PT = 3.0        # clearance between neighbouring circles
MARGIN_PT = 10.0    # white space around each ring
TITLE_IN = 0.32     # room above each panel for its title
PANEL_GAP_IN = 0.35


def node_diameter(net=None):
    """One circle diameter (points) for every network: the longest label over all networks at FONT, plus padding."""
    return max(max(len(x) for x in names) for names in NAMES.values()) * CHAR_W * FONT + 6


# Free space between neighbouring circles (points); the same for every network, so edges have room everywhere.
CLEAR_PT = 36.0


def ring_radius(net, diam):
    """Ring radius (points) that leaves CLEAR_PT between neighbouring circles; only the ring grows with gene count."""
    n = len(NAMES[net])
    return (diam + CLEAR_PT) / (2 * np.sin(np.pi / n))


def panel_side_pt(net):
    diam = node_diameter(net)
    return 2 * (ring_radius(net, diam) + diam / 2 + MARGIN_PT), diam


def _arc3_tangent(start, end, rad):
    """Unit tangent at the end of a matplotlib arc3 curve (used to orient curved repression bars)."""
    diff = end - start
    ctrl = 0.5 * (start + end) + rad * np.array([diff[1], -diff[0]])
    t = end - ctrl
    return t / np.linalg.norm(t)


def draw(ax, net, half_pt, diam):
    M = np.loadtxt(f"{RW}/{TOPO[net]}", delimiter=",")
    names = NAMES[net]
    n = len(names)
    assert M.shape[0] == n, (net, M.shape, n)
    R = ring_radius(net, diam)
    node_r = diam / 2
    th = np.pi / 2 - 2 * np.pi * np.arange(n) / n
    P = np.column_stack([np.cos(th), np.sin(th)]) * R
    # Edge rule as in twinfer/plotting/network_plots.py plot_grn: for a reciprocal pair, the edge from the
    # higher index to the lower index is curved; the other stays straight. Activation = arrowhead at the
    # target, repression = T-bar at the target. A white halo under each edge keeps crossings readable.
    for i in range(n):
        for j in range(n):
            if i == j or M[i, j] == 0:
                continue
            p0, p1 = P[i], P[j]
            d = p1 - p0
            u = d / np.hypot(*d)
            a = p0 + u * node_r
            b = p1 - u * node_r
            curved = M[j, i] != 0 and i > j
            rad = 0.18 if curved else 0.0
            cstyle = f"arc3,rad={rad}"
            lw = 2.0
            if M[i, j] > 0:
                for col, w in (("white", lw + 2.0), (BLUE, lw)):
                    ax.annotate("", xy=b, xytext=a, zorder=2 if col == BLUE else 1.5,
                                arrowprops=dict(arrowstyle="-|>,head_length=0.7,head_width=0.35", color=col,
                                                lw=w, connectionstyle=cstyle, shrinkA=0, shrinkB=0))
            else:
                bar_end = b - u * 2.0
                # repression colour (red) applies to the bar and its shaft

                tangent = u if not curved else _arc3_tangent(a, bar_end, rad)
                perp = np.array([-tangent[1], tangent[0]])
                for col, w in (("white", lw + 2.0), (RED, lw)):
                    ax.annotate("", xy=bar_end, xytext=a, zorder=2 if col == RED else 1.5,
                                arrowprops=dict(arrowstyle="-", color=col, lw=w, connectionstyle=cstyle,
                                                shrinkA=0, shrinkB=0))
                half = 3.0
                pts = np.array([bar_end - perp * half, bar_end + perp * half])
                ax.plot(pts[:, 0], pts[:, 1], color="white", lw=lw + 2.0, zorder=2.4, solid_capstyle="butt")
                ax.plot(pts[:, 0], pts[:, 1], color=RED, lw=lw * 1.5, zorder=2.6, solid_capstyle="butt")
    for i in range(n):
        ax.add_patch(Circle(P[i], node_r, facecolor="white", edgecolor="#333333", lw=1.0, zorder=3))
        ax.text(P[i][0], P[i][1], names[i], ha="center", va="center", fontsize=FONT, zorder=4, color="#222222")
    ax.set_xlim(-half_pt, half_pt)
    ax.set_ylim(-half_pt, half_pt)
    ax.set_aspect("equal")
    ax.axis("off")


LEGEND_IN = 1.6
PAGES = [
    [["EMT", "GSD"], ["Pluripotent"]],
    [["B_cell", "HSC"], ["VSC", "mCAD", "LEGEND"]],
]


def make_page(rows):
    """One page: each row is a list of panels; every panel keeps its own natural size (8 pt labels)."""
    sides = {k: panel_side_pt(k)[0] / 72 for k in NAMES}
    sides["LEGEND"] = LEGEND_IN
    row_w = [sum(sides[k] for k in row) + PANEL_GAP_IN * (len(row) - 1) for row in rows]
    row_h = [max(sides[k] for k in row) + TITLE_IN for row in rows]
    W = max(row_w) + 0.2
    H = sum(row_h) + PANEL_GAP_IN * (len(rows) - 1) + 0.2
    fig = plt.figure(figsize=(W, H))
    y_top = H - 0.1
    for row, rw, rh in zip(rows, row_w, row_h):
        x = (W - rw) / 2
        for key in row:
            s = sides[key]
            cy = y_top - TITLE_IN - (rh - TITLE_IN) / 2
            ax = fig.add_axes([x / W, (cy - s / 2) / H, s / W, s / H])
            if key == "LEGEND":
                handles = [Line2D([0], [0], color=BLUE, lw=1.2, marker=">", markersize=6, label="activation"),
                           Line2D([0], [0], color=RED, lw=1.2, marker="|", markersize=8, label="repression")]
                ax.legend(handles=handles, loc="center", frameon=False, fontsize=10)
                ax.axis("off")
            else:
                _, diam = panel_side_pt(key)
                draw(ax, key, s * 72 / 2, diam)
                ax.set_title(key.replace("_", " "), fontsize=10, pad=2)
            x += s + PANEL_GAP_IN
        y_top -= rh + PANEL_GAP_IN
    return fig


from matplotlib.backends.backend_pdf import PdfPages
with PdfPages(f"{OUT}/real_networks_all.pdf") as pdf:
    for rows in PAGES:
        fig = make_page(rows)
        pdf.savefig(fig, bbox_inches="tight", facecolor="white")
        plt.close(fig)
print("wrote", f"{OUT}/real_networks_all.pdf")
