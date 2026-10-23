#!/usr/bin/env python3
"""Completes singletCode's own unfinished 4th singlet criterion (its TODO:
"Integrate multi-sample singlets into above - singlets_step4"):

  1. single barcode per cellID                          -- singletCode, active
  2. one barcode has significantly more UMIs (dominant)  -- singletCode, active
  3. same barcode combination recurs in OTHER cells,
     SAME sample                                         -- singletCode, active
  4. same barcode combination recurs in OTHER cells,
     ACROSS samples                                      -- singletCode computes
                                                             this but never applies it

singletCode's own count_doublets() already returns `good_data` with a "label" column
set by criteria 1-3. Any cell still labeled "Multiplet" at that point is exactly the
population criterion 4 should be tested against -- criteria 1-3 already rescued
everything they can, so re-running the SAME barcode-combination lookup singletCode
uses internally (generate_barcode_combo + extract_two_barcode_singlets), but on the
FULL cross-sample good_data instead of one sample at a time, finds combination
matches specifically ACROSS samples (same-sample matches were already rescued and
removed from the multiplet pool, so nothing here can double-count criterion 3).
"""
from singletCode.count_doublets_utils_copy import generate_barcode_combo, extract_two_barcode_singlets


def apply_cross_sample_rescue(good_data):
    """good_data: the DataFrame returned by singletCode.get_singlets() (cellID, barcode,
    sample, nUMI, label columns; label already set by criteria 1-3).

    Returns a NEW DataFrame (good_data is not mutated) with `label` updated in place
    for any cellID rescued by criterion 4, plus the list of rescued cellIDs for
    reporting/provenance.
    """
    good_data = good_data.copy()
    remaining_multiplets = good_data.loc[good_data["label"] == "Multiplet", "cellID"].unique().tolist()

    # same call singletCode makes internally per-sample (lines 97-100 of
    # count_doublets_utils_copy.py), but on the whole cross-sample table --
    # generate_barcode_combo looks up each cellID's rows via good_data["cellID"]==cellID,
    # which naturally spans every sample that cellID appears in.
    barcode_dict, barcode_count_dict = generate_barcode_combo(remaining_multiplets, good_data)
    rescued = extract_two_barcode_singlets(barcode_dict, barcode_count_dict)

    good_data.loc[good_data["cellID"].isin(rescued), "label"] = "Singlet"
    return good_data, rescued
