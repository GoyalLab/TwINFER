from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
#!/usr/bin/env python3
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

DATA_DIR = f'{TWINFER_PROJECT_ROOT}/analysis_data/hPSC_20260927/data'
OUT_PDF = f"{DATA_DIR}/hpsc_endoderm_TABLE1.pdf"

df = pd.read_csv(f"{DATA_DIR}/hpsc_endoderm_TABLE1_correct.csv")

styles = getSampleStyleSheet()
doc = SimpleDocTemplate(OUT_PDF, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch,
                         leftMargin=0.5*inch, rightMargin=0.5*inch)

title = Paragraph(
    "hPSC_20260927 (endo_T0 &rarr; endo_T1) &mdash; TwinScore Stage III, ranked by |z|",
    styles["Title"])
subtitle = Paragraph(
    "z = U/sd(U), BH-corrected over Stage I pairs (q&le;0.05). "
    "z&gamma; = direction-stage z-score. Pearson rank = rho competitor's rank on the "
    "full 70,272-pair TF&rarr;target universe. Bold rows = ChIP-Atlas(+Perturb-seq) "
    "validated candidate edges.",
    styles["Normal"])

header = ["#", "TF", "gene", "direction", "z", "zγ", "Pearson rank", "true candidate"]
data = [header]
for _, r in df.iterrows():
    zg = f"{r.z_gamma:+.3f}" if pd.notna(r.z_gamma) else "—"
    d = str(r.direction) if pd.notna(r.direction) and str(r.direction).strip() else ""
    dsym = {"->": "→", "<-": "←", "sym.": "sym."}.get(d, d)
    data.append([
        str(int(r["#"])), r.TF, r.gene, dsym, f"{r.z:.3f}", zg,
        f"{int(r.pearson_rank):,}" if pd.notna(r.pearson_rank) else "—",
        "YES" if r.true_candidate else "",
    ])

table = Table(data, repeatRows=1,
              colWidths=[0.3*inch, 0.85*inch, 0.85*inch, 0.55*inch, 0.6*inch, 0.6*inch, 0.95*inch, 0.85*inch])
style_cmds = [
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2a3f5f")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8),
    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
]
for i, r in df.reset_index(drop=True).iterrows():
    if r.true_candidate:
        style_cmds.append(("BACKGROUND", (0, i + 1), (-1, i + 1), colors.HexColor("#d4edda")))
        style_cmds.append(("FONTNAME", (0, i + 1), (-1, i + 1), "Helvetica-Bold"))
table.setStyle(TableStyle(style_cmds))

doc.build([title, Spacer(1, 6), subtitle, Spacer(1, 12), table])
print(f"wrote {OUT_PDF}")
