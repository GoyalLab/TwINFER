# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import pandas as pd

# fabricate a tiny raw_index_pos + masks covering every branch
raw_index_pos = {
    "libA:bc1": 0,  # will fail whitelist
    "libA:bc2": 1,  # will fail mito
    "libA:bc3": 2,  # no lineage barcode record
    "libA:bc4": 3,  # singletcode multiplet
    "libA:bc5": 4,  # fails gene floor
    "libA:bc6": 5,  # rule ii
    "libA:bc7": 6,  # rule iii
    "libA:bc8": 7,  # passes everything (shouldn't be a "gap" cell in real data, sanity check UNEXPLAINED branch)
}
target_keys = list(raw_index_pos.keys()) + ["libA:not_present_at_all"]

whitelist_mask = [False, True, True, True, True, True, True, True]
pass_mito      = [None,  False, True, True, True, True, True, True]
has_barcode_record = [None, None, False, True, True, True, True, True]
is_singlet     = [None, None, None, False, True, True, True, True]
pass_genes     = [None, None, None, None, False, True, True, True]

rule_ii_map = {"libA:bc6": True, "libA:bc7": False, "libA:bc8": False}
rule_iii_map = {"libA:bc6": False, "libA:bc7": True, "libA:bc8": False}

rows = []
for key in target_keys:
    if key not in raw_index_pos:
        stage = "not_in_raw_matrix"
    else:
        i = raw_index_pos[key]
        if not whitelist_mask[i]:
            stage = "fails_whitelist"
        elif not pass_mito[i]:
            stage = "fails_mito"
        elif not has_barcode_record[i]:
            stage = "no_lineage_barcode_record"
        elif not is_singlet[i]:
            stage = "singletcode_multiplet"
        elif not pass_genes[i]:
            stage = "fails_gene_floor"
        else:
            rii = rule_ii_map.get(key, False)
            riii = rule_iii_map.get(key, False)
            if rii and riii:
                stage = "rule_ii_and_iii"
            elif rii:
                stage = "rule_ii"
            elif riii:
                stage = "rule_iii"
            else:
                stage = "UNEXPLAINED_still_present"
    rows.append((key, stage))

df = pd.DataFrame(rows, columns=["key","stage"])
print(df)

expected = {
    "libA:bc1": "fails_whitelist",
    "libA:bc2": "fails_mito",
    "libA:bc3": "no_lineage_barcode_record",
    "libA:bc4": "singletcode_multiplet",
    "libA:bc5": "fails_gene_floor",
    "libA:bc6": "rule_ii",
    "libA:bc7": "rule_iii",
    "libA:bc8": "UNEXPLAINED_still_present",
    "libA:not_present_at_all": "not_in_raw_matrix",
}
for k, v in expected.items():
    got = df.set_index("key").loc[k, "stage"]
    assert got == v, f"{k}: expected {v}, got {got}"
print("ALL ASSERTIONS PASSED")
