# Rankings PDF in the layout of list.pdf (1 Oct 2026): for each dataset the top 100 of the listed pairs (Stage I pairs after the gene filter) ranked three ways,
#   unshared by |z_U|, shared by h (= sign(S) C / sigma_C, descending), co-expression by |rho|; "other ranks" = the pair's rank in the other two lists (U, C, rho).
# Plain tables only (no plots / shading). Input: analysis_data/<fmXX>/data/twinscore_pdf_candidates/pdf_procedure_listed_pairs.csv (base run, unnormalized).
# usage: python 23_make_rankings_pdf.py OUT.pdf
import sys, json, os
import pandas as pd, numpy as np
import matplotlib
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

R = "/gpfs/projects/b1255/hzhang/TwINFER_KA/"
fd = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
pdfmetrics.registerFont(TTFont("DV", os.path.join(fd, "DejaVuSans.ttf"))); pdfmetrics.registerFont(TTFont("DVB", os.path.join(fd, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DVI", os.path.join(fd, "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DVB", italic="DVI", boldItalic="DVB")

# dataset -> (folder, title, treatment text, cells/clones from obs_metadata)
DS = [
    ("FM01", "fm01", "Melanoma FM01, WM989 + 1 µM PLX4720",
     "WM989 melanoma cells treated with 1 µM PLX4720 (BRAF inhibitor); replicates A and B pooled. This is the system of Table 2 in list.pdf."),
    ("FM03", "fm03", "Melanoma FM03, DMSO pre-treated, then 1 µM vemurafenib",
     "WM989 A6-G3 melanoma cells, split A, DMSO pre-treated, then treated with 1 µM vemurafenib. One RNA replicate."),
    ("FM05", "fm05", "Melanoma FM05, 250 nM PLX",
     "WM989 melanoma cells treated with 250 nM PLX (low dose; from the GEO sample name FM05_A1/A2_250nMPLX, the sample description was not supplied). Replicates A1 and A2 pooled."),
    ("FM07", "fm07", "Melanoma FM07, 1 µM vemurafenib, continuous",
     "WM989 A6-G3 melanoma cells, split A, treated continuously with 1 µM vemurafenib. Replicates A and B pooled."),
]

def ranks(L):
    return {"U": L.z_U.abs().rank(ascending=False, method="first").astype(int),
            "C": L.h.rank(ascending=False, method="first").astype(int),
            "rho": L.rho.abs().rank(ascending=False, method="first").astype(int)}

def load(folder, label):
    L = pd.read_csv(R + f"analysis_data/{folder}/data/twinscore_pdf_candidates/pdf_procedure_listed_pairs.csv")
    S = json.load(open(R + f"analysis_data/{folder}/data/twinscore_pdf_candidates/summary.json"))
    o = pd.read_csv(R + f"finalized_data/{label}_data/qc_filtered/obs_metadata.csv", index_col=0)
    c = o.fatemap_clone_singletcode.fillna("").astype(str); c = c[c != ""]
    return L, S, len(o), c.value_counts()

st = {n: ParagraphStyle(n, **{"fontName": "DV", "fontSize": sz, "leading": sz * 1.25, **kw}) for n, sz, kw in
      [("body", 8, {}), ("title", 14, dict(alignment=1, fontName="DVB")), ("sub", 9, dict(alignment=1)), ("cap", 8.5, dict(alignment=1)),
       ("h", 9.5, dict(fontName="DVB", spaceBefore=6, spaceAfter=3)), ("small", 7, {})]}
cell = ParagraphStyle("cell", fontName="DV", fontSize=6.3, leading=7.4)

def line_table(data, colw, fs=7.2, hdr_rows=1, align_right_from=None):
    t = Table(data, colWidths=colw, repeatRows=hdr_rows)
    ts = [("FONT", (0, 0), (-1, -1), "DV", fs), ("FONT", (0, 0), (-1, hdr_rows - 1), "DVB", fs),
          ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black), ("LINEBELOW", (0, hdr_rows - 1), (-1, hdr_rows - 1), 0.5, colors.black),
          ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black), ("TOPPADDING", (0, 0), (-1, -1), 1.4), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.4)]
    t.setStyle(TableStyle(ts)); return t

def main(out):
    doc = SimpleDocTemplate(out, pagesize=landscape(letter), leftMargin=0.45 * inch, rightMargin=0.45 * inch, topMargin=0.5 * inch, bottomMargin=0.5 * inch,
                            title="Unshared, shared and co-expression rankings of TF–target pairs: FM01, FM03, FM05, FM07")
    E = [Paragraph("Unshared, shared and co-expression rankings of TF–target pairs", st["title"]),
         Paragraph("FateMap melanoma datasets FM01, FM03, FM05 and FM07, treatment conditions compared · 14 October 2026", st["sub"]), Spacer(1, 6)]
    E.append(Paragraph(
        "<b>How to read the tables.</b> Each table ranks the same listed pairs (Stage I pairs after the gene filter) three ways: <i>unshared</i>, by |z<sub>U</sub>|, the Stage III statistic of the unshared correlation "
        "U = S − C (TwinScore); <i>shared</i>, by h = sign(S)·C/σ<sub>C</sub> (descending), the Stage II statistic of the twin correlation C; <i>co-expression</i>, by |ρ| over single cells. "
        "“Other ranks” gives the pair’s rank in the other two lists (U = unshared, C = shared, ρ = co-expression), counted among the listed pairs of that dataset. "
        "The shared-driver brackets and direction calls of list.pdf are not computed here.", st["body"]))
    E.append(Spacer(1, 4))
    E.append(Paragraph(
        "<b>Same procedure for every dataset.</b> The pair universe is FM01’s: 176,512 TF × target pairs from 51 TFs with ChIP-Atlas binding (cell class “Epidermis”) within ±5 kb of the TSS. "
        "Genes and TFs must be detected in ≥ 5% of the scored cells, so pairs involving lowly detected genes are not scored in every dataset. Values are CP10k converted to clone-weighted mid-ranks (each cell weighted 1/clone size); "
        "no depth or cell-cycle normalization (unnormalized base run). Stage I BH q ≤ 0.05; the gene exclusion list is the rebuilt GO/Tirosh/histone list (FM01 reproduces list.pdf Table 2 at 99/100 pairs per list). "
        "QC: singletCode on the entire barcode sheet, mito ≤ 26%, 200–7,200 genes, singletCode’s automatic UMI cutoff.", st["body"]))
    E.append(Spacer(1, 6))

    # ---- summary table ----
    hdr = ["Dataset", "Treatment", "Cells", "Clones", "Twin clones", "Twin pairs", "Largest clone", "Listed pairs", "Inherited state", "Called (q_U ≤ 0.05)", "h across shared top 100"]
    rows = [hdr]; data = {}
    for lab, fol, title, txt in DS:
        L, S, ncell, sizes = load(fol, lab); data[lab] = L
        top = L.sort_values("h", ascending=False).h.head(100)
        rows.append([lab, Paragraph({"FM01": "1 µM PLX4720", "FM03": "DMSO pre-treat, then 1 µM vemurafenib", "FM05": "250 nM PLX", "FM07": "1 µM vemurafenib, continuous"}[lab], cell),
                     f"{ncell:,}", f"{len(sizes):,}", f"{int((sizes >= 2).sum()):,}", f"{int((sizes * (sizes - 1) / 2).sum()):,}", f"{int(sizes.max())}",
                     f"{len(L):,}", f"{int(L.inherited.sum()):,}", f"{int(L.called.sum()):,}", f"{top.iloc[0]:.2f} to {top.iloc[-1]:.2f}"])
    E.append(line_table(rows, [0.5 * inch, 1.9 * inch, 0.55 * inch, 0.6 * inch, 0.75 * inch, 0.75 * inch, 0.7 * inch, 0.8 * inch, 0.85 * inch, 1.05 * inch, 1.3 * inch], fs=6.8))
    E.append(Spacer(1, 6))
    E.append(Paragraph("<b>Treatment context.</b> FM01 and FM07 are 1 µM BRAF-inhibitor conditions (FM01 PLX4720, FM07 vemurafenib given continuously); FM03 received DMSO first and then 1 µM vemurafenib; FM05 is a 250 nM low-dose condition. "
                       "Drug duration and sampling time are not recorded here. FOSL1 is detected in 38.7% of cells in FM01 but 16.6%, 5.6% and 10.1% in FM03, FM05 and FM07, so rankings that depend on lowly detected genes differ in "
                       "power, not only in biology. Treat cross-dataset differences as descriptive; none has been normalized for depth.", st["body"]))

    # ---- one block of tables per dataset ----
    for lab, fol, title, txt in DS:
        L = data[lab]; rk = ranks(L)
        E.append(PageBreak())
        E.append(Paragraph(f"Table: {title}: top 100 of {len(L):,} listed pairs in each ranking.", st["cap"]))
        E.append(Paragraph(txt, st["sub"])); E.append(Spacer(1, 4))
        lists = {"unshared": L.reindex(L.z_U.abs().sort_values(ascending=False).index).head(100),
                 "shared": L.reindex(L.h.sort_values(ascending=False).index).head(100),
                 "coexpr": L.reindex(L.rho.abs().sort_values(ascending=False).index).head(100)}
        oth = {"unshared": ("C", "rho"), "shared": ("U", "rho"), "coexpr": ("U", "C")}
        sym = {"U": "U", "C": "C", "rho": "ρ"}
        def fmt(k, i):
            r = lists[k].iloc[i]; idx = lists[k].index[i]
            a, b = oth[k]
            return f"{r.TF}–{r.gene}", f"{sym[a]} {rk[a][idx]:,}, {sym[b]} {rk[b][idx]:,}"
        data_rows = [["#", "unshared, |z_U|", "", "shared, h", "", "co-expression, |ρ|", ""], ["", "pair", "other ranks", "pair", "other ranks", "pair", "other ranks"]]
        for i in range(100):
            row = [str(i + 1)]
            for k in ("unshared", "shared", "coexpr"):
                p, o = fmt(k, i); row += [p, o]
            data_rows.append(row)
        w = [0.3 * inch, 1.5 * inch, 1.35 * inch, 1.5 * inch, 1.35 * inch, 1.5 * inch, 1.35 * inch]
        t = Table(data_rows, colWidths=w, repeatRows=2)
        t.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "DV", 6.4), ("FONT", (0, 0), (-1, 1), "DVB", 6.8),
                               ("SPAN", (1, 0), (2, 0)), ("SPAN", (3, 0), (4, 0)), ("SPAN", (5, 0), (6, 0)),
                               ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black), ("LINEBELOW", (1, 0), (2, 0), 0.4, colors.black),
                               ("LINEBELOW", (3, 0), (4, 0), 0.4, colors.black), ("LINEBELOW", (5, 0), (6, 0), 0.4, colors.black),
                               ("LINEBELOW", (0, 1), (-1, 1), 0.5, colors.black), ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
                               ("ALIGN", (0, 0), (0, -1), "RIGHT"), ("TOPPADDING", (0, 0), (-1, -1), 0.9), ("BOTTOMPADDING", (0, 0), (-1, -1), 0.9)]))
        E.append(t)
    doc.build(E)
    print("wrote", out)

if __name__ == "__main__":
    main(sys.argv[1])
