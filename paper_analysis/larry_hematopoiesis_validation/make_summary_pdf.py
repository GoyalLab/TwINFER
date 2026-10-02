"""Build a summary PDF for today's LARRY formula-testing / yscher-comparison investigation:
what the new formula is, where its code and data live, and the key result tables."""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                 PageBreak, ListFlowable, ListItem)

# [2026-09-30 commented out: result file now in clean_data/, see REPOINT_LOG.tsv] OUT = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation/LARRY_formula_summary.pdf'
OUT = f'{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/larry_hematopoiesis_validation/LARRY_formula_summary.pdf'
PAGE_W = letter[0] - 1.3 * inch  # usable width after L/R margins below

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="Mono", fontName="Courier", fontSize=8, leading=10))
styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], spaceBefore=14))
styles.add(ParagraphStyle(name="H3", parent=styles["Heading3"], spaceBefore=10))
styles.add(ParagraphStyle(name="BodySmall", parent=styles["BodyText"], fontSize=9, leading=12))
styles.add(ParagraphStyle(name="CellHead", fontName="Helvetica-Bold", fontSize=7.5, leading=9, textColor=colors.white))
styles.add(ParagraphStyle(name="Cell", fontName="Helvetica", fontSize=7.5, leading=9))
styles.add(ParagraphStyle(name="CellMono", fontName="Courier", fontSize=6.8, leading=8.2))

story = []


def h1(t): story.append(Paragraph(t, styles["Title"]))
def h2(t): story.append(Paragraph(t, styles["H2"]))
def h3(t): story.append(Paragraph(t, styles["H3"]))
def p(t): story.append(Paragraph(t, styles["BodySmall"]))
def mono(t): story.append(Paragraph(t.replace("\n", "<br/>"), styles["Mono"]))
def sp(n=8): story.append(Spacer(1, n))


def _cell(v, is_header, mono_col):
    if isinstance(v, Paragraph):
        return v
    style = styles["CellHead"] if is_header else (styles["CellMono"] if mono_col else styles["Cell"])
    return Paragraph(str(v), style)


def make_table(header, rows, col_widths, mono_cols=(), font_size=7.5):
    """mono_cols: set of column indices to render in a small monospace font (for code/paths)."""
    assert sum(col_widths) <= PAGE_W + 1, f"table too wide: {sum(col_widths)} > {PAGE_W}"
    data = [[_cell(v, True, False) for v in header]]
    for row in rows:
        data.append([_cell(v, False, i in mono_cols) for i, v in enumerate(row)])
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)


# ============================================================ TITLE
h1("LARRY Formula Testing vs. yscher's Analysis — Summary")
p("Covers: today's new gated formula (with z_reg_gated) exactly as pasted, its results on our "
  "own 9 gene sets, its ablation/refinement on yscher's corrhigh panel, the best formula found, "
  "and its application to all 9 gene sets on both TF-restricted and all-gene x all-gene "
  "universes, compared against BEELINE competitors.")
sp(14)

# ============================================================ THE FORMULA
h2("1. The Formula")
h3("As pasted (starting point)")
mono("Score(x -> y) = max_t s(z_|rho|(t)) + s(z_gamma)\n"
     "  + I(max_t z_het < -2.326) * s(max_t z_het)\n"
     "  + I(max_t z_X > 2.326) * s(max_t z_X)      [z_X ignored per instruction -- undefined in codebase]\n"
     "  + I(max_t z_reg > 2.326) * s(max_t z_reg)")
p("z_reg = z_reg_gated (calculate_gated_regulation_statistic, the canonical package function). "
  "max_t is a genuine max over t1/t2 only for the |rho| term (the only one with true separate "
  "t1/t2 values in this pipeline); z_het/z_reg_gated are already single combined per-pair "
  "statistics, so their max_t wrapper is identity here. I(.) is a hard 0/1 indicator (boolean "
  "cast to float), not a soft gate.")

sp(10)
h3("2. Results of THIS formula (exactly as pasted, z_X dropped) on our own 9 gene sets")
make_table(
    ["gene_set", "AUPRC", "x random", "best competitor", "comp. x random"],
    [
        ["variability_high", "0.246", "1.38x", "ppcor (0.284)", "1.59x"],
        ["variability_mid", "0.131", "1.05x", "genie3 (0.137)", "1.10x"],
        ["variability_low", "0.136", "0.95x", "genie3 (0.199)", "1.40x"],
        ["detection_high", "0.414", "2.37x", "pidc (0.202)", "1.16x"],
        ["detection_mid", "0.114", "0.84x", "grnboost2 (0.199)", "1.47x"],
        ["detection_low", "0.191", "1.30x", "rho (0.214)", "1.46x"],
        ["correlation_high", "0.307", "1.48x", "rho (0.342)", "1.64x"],
        ["correlation_mid", "0.195", "1.12x", "genie3 (0.243)", "1.40x"],
        ["correlation_low", "0.184", "1.66x", "ppcor (0.161)", "1.45x"],
        ["MEAN", "0.213", "1.35x", "—", "1.41x"],
    ],
    col_widths=[130, 55, 60, 110, 90],
)
p("Beats the best-per-gene-set competitor on 2/9 (detection_high, correlation_low) by both raw "
  "AUPRC and ratio-to-random; mean ratio (1.35x) trails the best-competitor mean (1.41x). "
  "This is the formula's result with NO refinement yet — see Section 4 for what changed and why.")

story.append(PageBreak())

h3("Best formula found (after ablation/refinement on yscher's corrhigh panel, Section 5)")
mono("Score(x -> y) = s(z_abs_rho_t1) + s(z_abs_rho_t2) - s(|z_abs_rho_change|)\n"
     "  - |z_div| * I(z_het < -2.326)                 [divergence_penalty]\n"
     "  - z_het * I(z_d_het > 0)                       [heterogeneity_penalty]\n"
     "  + I(|s(z_gamma)| >= 1) * s(z_gamma)            [GATED, full value -- replaces\n"
     "                                                   original 1/2*sign(gamma)*I(...)]\n"
     "  + s(z_reg_gated)                                [addition kept from the pasted formula]")
p("Differences from the pasted formula above: the plain max(s(|rho_t1|),s(|rho_t2|)) term is "
  "expanded into shipped <i>calculate_twin_score</i>'s full A+B+C+divergence_penalty+"
  "heterogeneity_penalty structure (found to be more informative than a bare max — see Section "
  "5's ablation); the z_het gate/term is folded into divergence_penalty/heterogeneity_penalty "
  "(the same underlying z_het and a gate, just packaged as in the shipped formula); the gamma "
  "term keeps a gate but passes the full continuous value instead of collapsing to a sign; "
  "z_reg_gated is kept as an additive term, unchanged in spirit from the pasted version. pi (a "
  "rank-invariant constant in the shipped formula) is dropped throughout.")

sp(10)
h3("How each design choice was tested")
story.append(ListFlowable([
    ListItem(Paragraph("z_X: dropped entirely — not defined anywhere in the TwINFER codebase "
                        "(confirmed via full-package grep).", styles["BodySmall"])),
    ListItem(Paragraph("z_reg = z_reg_gated confirmed via docstring and by finding its real "
                        "output in yscher's exports/zreg_gated/ and our own infer_results/ JSONs.",
                        styles["BodySmall"])),
    ListItem(Paragraph("z_reg_gated is symmetric (one value per unordered pair) — a real bug was "
                        "found and fixed where only one direction was looked up, silently zeroing "
                        "half the pairs.", styles["BodySmall"])),
    ListItem(Paragraph("Full leave-one-out and single-term-alone ablation run on shipped TwinScore "
                        "(on yscher's corrhigh panel): the |rho(t2)-rho(t1)| penalty term is the "
                        "single most load-bearing piece despite being anti-predictive alone (AUC "
                        "0.40) -- it corrects a regression-to-the-mean artifact in the raw |rho| "
                        "magnitude terms (verified: false positives in the raw-magnitude top-K "
                        "have systematically less stable rho across time; adding this term demotes "
                        "them by +46.8 average rank while promoting true positives by -28.0).",
                        styles["BodySmall"])),
    ListItem(Paragraph("Every other available z-score tested standalone (z_div, z_het, z_gamma, "
                        "rho_change, z_reg_gated) -- z_reg_gated (raw, continuous) was the "
                        "strongest individual candidate not already in the shipped formula.",
                        styles["BodySmall"])),
    ListItem(Paragraph("\"Add every term better than random\" was tried literally and made things "
                        "WORSE (rho_change is individually >1x but destructively correlated with "
                        "the existing |rho_t2-rho_t1| term) -- only z_reg_gated survived as a "
                        "genuine, orthogonal addition.", styles["BodySmall"])),
], bulletType="bullet", leftIndent=14))

story.append(PageBreak())

# ============================================================ CODE
h2("3. Code")
h3("Our scripts (this investigation)")
make_table(
    ["Script", "Purpose"],
    [
        ["test_new_gated_formula.py", "First test of the pasted formula (z_X dropped) on our 9 gene sets"],
        ["apply_best_formula_all9.py", "Best formula applied to yscher's 9 panels + his BEELINE competitor files"],
        ["apply_best_formula_ours.py", "Best formula on OUR 9 gene sets, TF-restricted universe, vs benchmark_results.csv"],
        ["apply_best_formula_ours_allpairs.py", "Same, but on the full all-gene x all-gene universe, vs our *_allgenes.csv competitor files"],
        ["make_summary_pdf.py", "This report"],
    ],
    col_widths=[190, 320],
    mono_cols={0},
)
p("All under: <font face='Courier' size=7>.../code/TwINFER/paper_analysis/"
  "larry_hematopoiesis_validation/</font>")

sp(10)
h3("Package function used directly (not reimplemented)")
p("<font face='Courier' size=7.5>correlation_functions.py</font> — "
  "<font face='Courier' size=7.5>calculate_twin_score</font> (shipped formula, docstring gives the "
  "canonical definition) and <font face='Courier' size=7.5>calculate_gated_regulation_statistic</font> "
  "(z_reg_gated).")

sp(10)
h3("yscher's scripts referenced/read/executed")
make_table(
    ["Script", "Role"],
    [
        ["helpers/agreed_full_tables.py", "Generates AGREED_FULL_TABLES.md (GMM-cut/best-cut/AUC per panel); executed directly on his data to reproduce his exact numbers (exact match: 2.38x / 2.49x / 0.679)"],
        ["helpers/ag_prec_calls.py", "Defines candidate scoring FORMS; not directly executed (too many dependent driver variants to find one runnable entry point)"],
        ["helpers/gate_pr_larry24.py", "Different panel family (band15); read for methodology, not run"],
        ["helpers/gstar.py + exports/calibration/*.json", "'newcal' threshold calibration (existence/direction/heterogeneity); read for the definition, not used in the final reproduction"],
        ["helpers/_collectri.py", "CollecTRI loader; same file/columns as ours. His own copy is not readable to us (permission denied) — used our copy instead (confirmed structurally identical)"],
        ["helpers/summary_normalization.py", "'scrubbed' / 'unscrubbed' LARRY matrix definitions"],
    ],
    col_widths=[190, 320],
    mono_cols={0},
)
p("All under: <font face='Courier' size=7>/gpfs/projects/b1255/yscher/Transcriptomic Distance/</font>")

story.append(PageBreak())

# ============================================================ DATA
h2("4. Data")
h3("Ours")
make_table(
    ["Path (relative to .../larry_hematopoiesis_validation/resources/)", "Contents"],
    [
        ["analytic_infer/{gene_set}/twin_score_inputs.csv", "rho_t1/t2, z_abs_rho_t1/t2/change, z_gamma, z_het, z_div, z_d_het per pair (ALL genes, not TF-restricted)"],
        ["infer_results/{gene_set}_t2_t4_allpairs_50core/z_scores_by_step.json", "Production z_reg_gated + z_het (~40-gene keyspace, superset of the TF-restricted panel)"],
        ["networks/{method}_{gene_set}_allgenes.csv", "rho/ppcor/pidc/genie3/grnboost2 on the UNRESTRICTED (all-gene) universe"],
        ["benchmark/benchmark_results.csv", "Same 5 competitors, TF-restricted universe, plus random-baseline columns"],
        ["gene_sets.json, gene_sets_detail.json", "TF-restricted panel definitions (criterion x level -> TF list)"],
        ["collectri_mouse.tsv", "Ground truth (source_genesymbol, target_genesymbol, ...)"],
    ],
    col_widths=[260, 250],
    mono_cols={0},
)

sp(10)
h3("yscher's (read directly, not copied)")
make_table(
    ["Path (relative to .../Transcriptomic Distance/)", "Contents"],
    [
        ["exports/twinscore_gated/larry_log1pPF_stable24_{tag}_{panel}_24/twin_scores.csv", "Full twinScore decomposition (z_abs_rho_t1/t2/change, divergence_penalty, heterogeneity_penalty, gamma_bonus, twinScore) on log1pPF/stable24/newcal/cloneunit data"],
        ["exports/twinscore_gated/.../frozen_components.npz", "Gene list (panel definition) for that run"],
        ["exports/zreg_gated/{bn}_zreg_gated_t1.csv", "z_reg_gated, symmetric, one row per unordered pair (stat == 'gated')"],
        ["exports/networks/{method}_{panel}_T4_unrestricted.csv", "His competitors, unrestricted (all-gene) universe"],
        ["exports/calibration/gstar_calibration.json, threshold_sets_new_analysis.json", "'newcal' threshold values: existence 0.0126 (vs. manuscript 0.021); direction 0.0421 (vs. 0.046); heterogeneity z* = -9.76 (vs. -10.0)"],
    ],
    col_widths=[260, 250],
    mono_cols={0},
)
p("tag/panel mapping used: perm4clean/{corrhigh,corrmid,corrlow} &lt;-&gt; our "
  "correlation_{high,mid,low}; p4a01/detect_q{90100,7590,5075}_tf10 &lt;-&gt; our "
  "detection_{high,mid,low}; p4a01/spread_q{67100,3367,0033}_tf10 &lt;-&gt; our "
  "variability_{high,mid,low}.")

story.append(PageBreak())

# ============================================================ RESULTS: ablation
h2("5. Results — Ablation on yscher's corrhigh panel (44 genes, 1892 pairs, 109 true edges)")
make_table(
    ["Variant", "AUPRC", "x random", "hits/calls", "AUC", "GMM-cut", "best-cut"],
    [
        ["shipped TwinScore (baseline)", "0.106", "1.83x", "17/109", "0.679", "1.62x", "2.49x"],
        ["+ z_reg_gated, original gamma_bonus", "0.112", "1.94x", "20/109", "0.680", "1.62x", "2.02x"],
        ["+ z_reg_gated, s(z_gamma) ungated", "0.109", "1.89x", "15/109", "0.681", "1.62x", "2.83x"],
        ["+ z_reg_gated, GATED*s(z_gamma) [BEST]", "0.113", "1.96x", "19/109", "0.684", "1.62x", "3.11x"],
    ],
    col_widths=[195, 48, 48, 55, 42, 52, 50],
)
p("Note: AUROC ('AUC') is not very informative here — the set is heavily imbalanced (5.76% "
  "positive rate), so it barely moves across formulas while AUPRC/precision swing much more. "
  "GMM-cut was found to be flat/uninformative across nearly every candidate tested (even a "
  "literal random-noise ranking scores ~1.30x under best-cut, confirmed from yscher's own "
  "reproducible code) — trust AUPRC and best-cut over GMM-cut and AUC.")

sp(10)
h3("What yscher's own report actually says (AGREED_FULL_TABLES.md, corrhigh panel, exact match "
   "reproduced above for TwinScore shipped)")
make_table(
    ["method", "GMM-cut", "best-cut", "AUC"],
    [
        ["TwinScore shipped", "2.38x", "2.49x", "0.679"],
        ["TwinScore new", "1.83x", "3.78x", "0.714"],
        ["TwinScore apriori", "1.83x", "4.17x", "0.718"],
        ["rho (co-expression)", "1.80x", "1.85x", "0.660"],
        ["PIDC", "1.60x", "1.91x", "0.664"],
        ["ppcor", "1.13x", "1.43x", "0.599"],
        ["GRNBoost2", "1.09x", "1.04x", "0.471"],
        ["GENIE3", "0.42x", "1.05x", "0.446"],
        ["RANDOM (5 draws, iid Gaussian noise)", "1.04x", "1.15x", "0.508"],
        ["CALL EVERYTHING", "1.00x", "1.00x", "0.500"],
    ],
    col_widths=[220, 60, 60, 45],
)
sp(6)
p("His MEAN-over-10-panels table (his full panel set: corrhigh/mid/low, panel12, "
  "detect_q5075/7590/90100_tf10, spread_q0033/3367/67100_tf10):")
make_table(
    ["method", "GMM-cut", "best-cut", "AUC"],
    [
        ["TwinScore shipped", "2.15x", "2.51x", "0.563"],
        ["TwinScore new", "1.58x", "2.65x", "0.571"],
        ["TwinScore apriori", "1.58x", "2.52x", "0.571"],
        ["rho (co-expression)", "1.94x", "1.87x", "0.563"],
        ["PIDC", "1.20x", "1.72x", "0.569"],
        ["ppcor", "1.34x", "1.56x", "0.556"],
        ["GRNBoost2", "0.92x", "1.11x", "0.473"],
        ["GENIE3", "0.85x", "1.18x", "0.437"],
        ["RANDOM", "0.99x", "1.30x", "0.495"],
        ["CALL EVERYTHING", "1.00x", "1.00x", "0.500"],
    ],
    col_widths=[220, 60, 60, 45],
)
p("\"TwinScore new\" and \"TwinScore apriori\" are his own further-refined variants (not "
  "reproduced or tested here — they use additional partial-correlation/drift/regulator-ness "
  "terms from ag_prec_calls.py's FORMS that were read but not executed, see Section 3). The "
  "RANDOM row is literal i.i.d. Gaussian noise scored the same way — its best-cut of "
  "1.15x-1.30x is exactly the evidence that best-cut is an inflated, hindsight-biased metric "
  "(Section 5's own note); GMM-cut correctly keeps RANDOM at ~1.0x.")

sp(10)
h2("6. Results — Best formula on all 9 gene sets")

h3("6a. On yscher's data/panels, vs. his BEELINE competitors (his unrestricted universe)")
make_table(
    ["gene_set", "n_genes", "n_true", "shipped", "NEW", "best competitor"],
    [
        ["correlation_high", "44", "109", "1.83x", "1.96x", "PIDC 1.56x"],
        ["correlation_mid", "68", "192", "2.20x", "2.33x", "PIDC 1.95x"],
        ["correlation_low", "63", "210", "1.30x", "1.27x", "ppcor 1.36x"],
        ["detection_high", "56", "96", "1.39x", "1.43x", "PIDC 1.18x"],
        ["detection_mid", "47", "84", "0.94x", "0.95x", "PIDC 1.05x"],
        ["detection_low", "51", "104", "2.01x", "2.16x", "ppcor 1.35x"],
        ["variability_high", "50", "83", "1.00x", "0.96x", "PIDC 1.54x"],
        ["variability_mid", "54", "115", "1.85x", "1.79x", "ppcor 1.20x"],
        ["variability_low", "56", "133", "1.14x", "1.18x", "PIDC 1.33x"],
        ["MEAN", "", "", "1.52x", "1.56x", "1.39x"],
    ],
    col_widths=[120, 55, 50, 55, 55, 95],
)
p("NEW beats best competitor on 5/9; beats shipped TwinScore on 5/9, ties 1, loses 3.")

sp(10)
h3("6b. On OUR data/gene sets, TF-restricted universe, vs benchmark_results.csv")
make_table(
    ["gene_set", "existdir", "shipped", "NEW", "hits", "best competitor"],
    [
        ["variability_high", "1.48x", "1.43x", "1.71x", "44/121", "ppcor 1.66x"],
        ["variability_mid", "0.97x", "0.91x", "1.08x", "10/75", "genie3 1.18x"],
        ["variability_low", "1.06x", "1.08x", "1.13x", "4/30", "genie3 1.61x"],
        ["detection_high", "2.63x", "1.83x", "2.45x", "2/8", "pidc 1.66x"],
        ["detection_mid", "1.03x", "1.25x", "1.14x", "6/44", "grnboost2 1.63x"],
        ["detection_low", "1.32x", "1.40x", "1.42x", "21/113", "rho 1.54x"],
        ["correlation_high", "1.65x", "1.34x", "1.66x", "32/83", "rho 1.73x"],
        ["correlation_mid", "1.32x", "1.29x", "1.36x", "17/97", "genie3 1.47x"],
        ["correlation_low", "1.64x", "1.08x", "1.34x", "5/43", "ppcor 1.63x"],
        ["MEAN", "1.46x", "1.29x", "1.48x", "", "1.57x"],
    ],
    col_widths=[110, 55, 55, 45, 55, 110],
)
p("NEW beats shipped on 8/9, beats existdir on 7/9, beats best competitor on only 2/9.")

story.append(PageBreak())

h3("6c. On OUR data, ALL-GENE x ALL-GENE universe (unrestricted, matching yscher's "
   "convention), vs our own *_allgenes.csv competitor networks")
make_table(
    ["gene_set", "n_pairs", "n_true", "existdir", "shipped", "NEW", "hits", "best competitor"],
    [
        ["variability_high", "2256", "125", "1.21x", "1.39x", "1.84x", "19/125", "ppcor 1.98x"],
        ["variability_mid", "1892", "75", "0.91x", "0.82x", "1.06x", "3/75", "genie3 1.83x"],
        ["variability_low", "756", "30", "0.87x", "0.95x", "1.01x", "0/30", "pidc 1.10x"],
        ["detection_high", "132", "8", "4.33x", "3.04x", "3.84x", "2/8", "genie3 2.54x"],
        ["detection_mid", "930", "44", "0.96x", "1.06x", "1.11x", "4/44", "pidc 2.71x"],
        ["detection_low", "2970", "114", "1.16x", "1.18x", "1.42x", "7/114", "rho 2.30x"],
        ["correlation_high", "930", "83", "1.55x", "1.38x", "1.58x", "10/83", "pidc 1.69x"],
        ["correlation_mid", "1560", "102", "1.75x", "1.67x", "1.72x", "12/102", "grnboost2 3.78x"],
        ["correlation_low", "870", "55", "1.82x", "1.06x", "1.56x", "8/55", "grnboost2 5.69x"],
        ["MEAN", "", "", "1.62x", "1.39x", "1.68x", "", "2.62x"],
    ],
    col_widths=[90, 42, 38, 40, 40, 38, 42, 88],
)
p("NEW beats best competitor on only 1/9 here. GENIE3/GRNBoost2 are NOT uniformly hurt by the "
  "unrestricted universe on our data (they are in fact the strongest competitors on several "
  "gene sets, up to 5.69x) — this directly contradicts an earlier hypothesis (that tree/"
  "regression methods are inherently punished by large unrestricted candidate pools, based on "
  "the GENIE3 n10-e10 failure found earlier this session on a different dataset). That "
  "mechanism does not generalize; it was specific to that topology's extreme sparsity "
  "(exactly one true regulator per target), not a general property of large candidate pools.")

sp(14)
h2("7. Open Question")
p("Competitors score much lower on yscher's data (mean best-competitor ~1.39x, unrestricted) "
  "than on ours (~1.57x TF-restricted, ~2.62x unrestricted). Ruled out: ground-truth convention "
  "(both directed), universe size (confirmed matching, both all-pairs), and the tree/regression-"
  "method-punished-by-large-pools hypothesis (falsified on our own data). Not yet confirmed: "
  "the remaining candidate is the normalization/QC pipeline itself (his log1pPF + stable24 "
  "joint-day classification + newcal calibration vs. our log1pCP10k pipeline) — this was not "
  "directly isolated and would need a controlled comparison to confirm.")

doc = SimpleDocTemplate(OUT, pagesize=letter, topMargin=0.6*inch, bottomMargin=0.6*inch,
                         leftMargin=0.65*inch, rightMargin=0.65*inch)
doc.build(story)
print("wrote", OUT)
