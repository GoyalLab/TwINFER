from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_a2a3302f_2026-09-28_why_twinfer_fails; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import numpy as np
import pandas as pd

D = f'{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/real_networks/why_twinfer_fails/protein_halflife_test'
FILES = {
    "halflife45 (baseline)": f"{D}/df_halflife45_28092026_183300_ncells_2000_mCAD_halflife45_d5239553.csv",
    "halflife16 (test)": f"{D}/df_halflife16_28092026_183639_ncells_2000_mCAD_halflife16_9034d31b.csv",
}
GENES = ["gene_1", "gene_2", "gene_3", "gene_4", "gene_5"]
gene_cols = [f"{g}_protein" for g in GENES]
k_deg = {"halflife45 (baseline)": np.log(2) / 45, "halflife16 (test)": np.log(2) / 16}

for tag, f in FILES.items():
    df = pd.read_csv(f, usecols=["time_step", "clone_id", "replicate"] + gene_cols)
    steps = sorted(df.time_step.unique())
    t1 = steps[0]
    d1 = df[df.time_step == t1]
    A1 = d1[d1.replicate == 1].sort_values("clone_id").set_index("clone_id")
    B1 = d1[d1.replicate == 2].sort_values("clone_id").set_index("clone_id")
    common0 = A1.index.intersection(B1.index)

    rows = []
    for t2 in steps[1:]:
        dt = df[df.time_step == t2]
        At2 = dt[dt.replicate == 1].sort_values("clone_id").set_index("clone_id")
        Bt2 = dt[dt.replicate == 2].sort_values("clone_id").set_index("clone_id")
        common = common0.intersection(At2.index).intersection(Bt2.index)
        a1 = A1.loc[common]; a2 = At2.loc[common]; b2 = Bt2.loc[common]
        for c in gene_cols:
            if a1[c].std() < 1e-9 or a2[c].std() < 1e-9 or b2[c].std() < 1e-9:
                continue
            same_cell = np.corrcoef(a1[c], a2[c])[0, 1]
            sister = np.corrcoef(a1[c], b2[c])[0, 1]
            rows.append(dict(t2=t2, gene=c, same_cell=same_cell, sister=sister))
    out = pd.DataFrame(rows)
    out.to_csv(f"{D}/divergence_{tag.split()[0]}.csv", index=False)

    print(f"\n=== {tag} (predicted single-gene decay: exp(-ln2/{'45' if '45' in tag else '16'} * t)) ===")
    print("t    mean_same_cell   mean_sister   predicted_single_gene")
    for t in [1, 6, 12, 24, 36, 48]:
        row = out[out.t2 == t]
        if len(row):
            pred = np.exp(-k_deg[tag] * t)
            print(f"{t:3d}   {row.same_cell.mean():.3f}          {row.sister.mean():.3f}         {pred:.3f}")
