# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""TwinScore on the HIGH co-expression panel of every dataset, one row per dataset, against the best competitor
on the same gene list, computed on the cells of the window's first timepoint (competitor file name token per row). Score, filters, competitor admission and the row format are twinscore_model.row's."""
import os, sys
# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "helpers")
from paper_analysis.larry_hematopoiesis_validation.twinscore_modules.twinscore_model import row, TERMS
ROWS = [("LARRY", "day 2 - day 4", "larry_log1pPF_stable24_det5_p4a01_corrhigh_24", "_T2_"),
        ("LARRY", "day 4 - day 6", "larry_log1pPF_stable24_det5_p4a46_corrhigh_46", "_T4_"),   # high co-expression list built on day 4 and 6 cells
        ("michaels", "day 5 - day 10", "michaels_mic5a01_corrhigh_panel_lines_T1_T2_det5", "_5_"),
        ("hPSC", "T0 - T10", "net_t26r12a01_corrhigh_panel_lines_T0_T10_det5", "_T0_"),
        ("hPSC", "T10 - T15", "net_t26r12a01_corrhigh_panel_lines_T10_T15_det5", "_T10_"),
        ("CellTag", "day 2.5 - day 5", "barcode_overlap_ct5a01_corrhigh__log1pPF_stable24", "_2_")]
print("""
Filters, applied to every ordered pair; a pair failing either is CALLED NO and ranks last:
  existence   |z_rho|        > 2.576 at t1 OR t2
  regulation   z_reg_gated  > 2.326 at t1 OR t2   (signed)

TwinScore(x->y) = max_t s(z(|rho(t)|))                                 existence
                  + s(z_flux)                                           direction
                  - I(z_stable > 2.326) * s(z_stable)                   change
                  + I(max_t z_het(t) < -2.326) * s(max_t z_het(t))
                  - I(min|z_dagger| > 2.326) * s(min|z_dagger|)         inherited channel

  z_flux = reg(x) - reg(y),  reg(g) = mean_w z_rho_dagger(g->w) - mean_w z_rho_dagger(w->g)
  z_stable = z(|rho(t2) - rho(t1)|),  min|z_dagger| = min(|z_rho_dagger_x->y|, |z_rho_dagger_y->x|)
""")
print("TwinScore, high co-expression panel of each dataset")
print("%-9s %-16s %6s %6s %7s %6s %6s %7s %7s %-16s %s" % ("dataset", "window", "calls", "hits", "prec %", "rec %", "no", "AUROC", "AUPRC", "x base precision", "best competitor"))
for ds, win, BN, tok in ROWS:
    try: d = row(BN, comp_token=tok)
    except Exception as e: print("%-9s %-16s not scorable: %s" % (ds, win, str(e)[:80])); continue
    print("%-9s %-16s %6d %6.1f %7.2f %6.2f %6d %7.3f %7.3f %-16s %s" % (ds, win, d["calls"], d["hits"], d["prec"], 100 * d["hits"] / d["calls"], d["no"], d["auroc"], d["auprc"], "%.2fx" % d["x_base"], d["best"]))
    print("%-26s AUPRC/base %.3f" % ("", d["auprc_base"]))
