from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_a2a3302f_2026-09-28_why_twinfer_fails; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import os, re, glob, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

R = f'{TWINFER_PROJECT_ROOT}'
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] BOOLODE_RULES = f"{R}/code/BoolODE/data"
BOOLODE_RULES = f"{R}/clean_data/benchmarks/boolode/model_inputs"
# [2026-09-30 commented out: data now in clean_data/, see REPOINT_LOG.tsv] CUSTOM_RULES = f"{R}/code/BoolODE/twins/custom_networks"
CUSTOM_RULES = f"{R}/clean_data/benchmarks/boolode/twins/custom_networks"
OLD_FMT = f"{R}/simulation_data/twinfer_format"
RW = f"{R}/input_data/real_world_networks"
ALGOS = ["GENIE3", "GRNBOOST2", "PEARSON", "PIDC", "PPCOR", "SCODE", "SCSGL"]

NETS = {
    "mCAD": dict(genes=None, matrix=f"{OLD_FMT}/mCAD/interaction_matrix.txt", rule_file=f"{BOOLODE_RULES}/mCAD.txt",
                 beeline_dir=f"{R}/analysis_data/real_networks/beeline_inference/mCAD",
                 twinfer_glob=f"{R}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs/mCAD_rep_*_all_results.json"),
    "GSD": dict(genes=None, matrix=f"{OLD_FMT}/GSD/interaction_matrix.txt", rule_file=f"{BOOLODE_RULES}/GSD.txt",
                beeline_dir=f"{R}/analysis_data/real_networks/beeline_inference/GSD",
                twinfer_glob=f"{R}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs/GSD_rep_*_all_results.json"),
    "HSC": dict(genes=None, matrix=f"{OLD_FMT}/HSC/interaction_matrix.txt", rule_file=f"{BOOLODE_RULES}/HSC.txt",
                beeline_dir=f"{R}/analysis_data/real_networks/beeline_inference/HSC",
                twinfer_glob=f"{R}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs/HSC_rep_*_all_results.json"),
    "VSC": dict(genes=None, matrix=f"{OLD_FMT}/VSC/interaction_matrix.txt", rule_file=f"{BOOLODE_RULES}/VSC.txt",
                beeline_dir=f"{R}/analysis_data/real_networks/beeline_inference/VSC",
                twinfer_glob=f"{R}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs/VSC_rep_*_all_results.json"),
    "B_cell_activation": dict(genes=None, matrix=f"{RW}/B_cell.txt", rule_file=f"{CUSTOM_RULES}/B_cell_activation_rules.txt",
                 beeline_dir=f"{R}/analysis_data/paper_analysis/real_networks/beeline_inference/B_cell_activation",
                 twinfer_glob=f"{R}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs/B_cell_activation_rep_*_all_results.json"),
    "EMT": dict(genes=None, matrix=f"{RW}/EMT.txt", rule_file=f"{CUSTOM_RULES}/EMT_real_rules.txt",
                beeline_dir=f"{R}/analysis_data/paper_analysis/real_networks/beeline_inference/EMT",
                twinfer_glob=f"{R}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs/EMT_rep_*_all_results.json"),
    "Pluripotent": dict(genes=None, matrix=f"{RW}/Pluripotent.txt", rule_file=f"{CUSTOM_RULES}/Pluripotent_real_rules.txt",
                beeline_dir=f"{R}/analysis_data/paper_analysis/real_networks/beeline_inference/Pluripotent",
                twinfer_glob=f"{R}/analysis_data/paper_analysis/real_networks/twinfer_inference_allpairs/Pluripotent_rep_*_all_results.json"),
}

# gene name lists (order = matrix row/col order), reusing what was established earlier this session
GENE_LISTS = {
    "mCAD": ["Fgf8", "Emx2", "Pax6", "Coup", "Sp8"],
    "GSD": [l.strip() for l in open(f"{OLD_FMT}/GSD/gene_order.txt") if l.strip()],
    "HSC": [l.strip() for l in open(f"{OLD_FMT}/HSC/gene_order.txt") if l.strip()],
    "VSC": [l.strip() for l in open(f"{OLD_FMT}/VSC/gene_order.txt") if l.strip()],
    "B_cell_activation": ["Ikaros", "PU_1", "Flk2", "IL_7R", "GATA_1", "E2A", "EBF", "C_EBPa", "PAX5", "Notch_1"],
    "EMT": ["Cdh1", "Cldn7", "Foxc2", "Grhl2", "Gsc", "Klf8", "Np63a", "Ovol2", "Snai1", "Snai2", "Tcf3",
            "Tgfbeta", "Twist1", "Twist2", "Vim", "Zeb1", "Zeb2"],
    "Pluripotent": ["ARID3B", "CEBPZ", "ETV4", "FOXH1", "HIC2", "HMGB3", "JARID2", "LIN28B", "MIS18BP1", "MYCN",
                    "NANOG", "POU5F1", "POU5F1B", "PRDM14", "REST", "SALL2", "SALL4", "SMARCC1", "SOX2",
                    "TEAD2", "TERF1", "TGIF1", "WDHD1", "ZBTB12", "ZBTB39", "ZNF281", "ZNF286A", "ZNF286B",
                    "ZNF322", "ZNF398", "ZNF462", "ZNF730", "ZNF90", "ZNF92", "ZSCAN10", "ZSCAN2"],
}
for n in NETS: NETS[n]["genes"] = GENE_LISTS[n]


def parse_rule_file(path, genes):
    """Read a Gene\\tRule BoolODE-format file, return {gene: numpy-vectorizable rule string}."""
    df = pd.read_csv(path, sep="\t")
    rules = {}
    for _, row in df.iterrows():
        g, rule = row["Gene"], row["Rule"]
        expr = re.sub(r"\bnot\b", "~", rule)
        expr = re.sub(r"\band\b", "&", expr)
        expr = re.sub(r"\bor\b", "|", expr)
        rules[g] = expr
    for g in genes:
        if g not in rules:
            rules[g] = g  # no rule found -> self-persistence placeholder
    return rules


def enumerate_fixed_points(rules, genes, max_exact_n=20, n_sample=500_000, seed=0):
    """Fixed points of the synchronous/asynchronous-invariant Boolean map f: s -> s.
    Exact brute force for n<=max_exact_n (vectorized over all 2^n states); random-sample
    estimate otherwise (returns a lower-bound count, flagged via `exact=False`)."""
    n = len(genes)
    if n <= max_exact_n:
        N = 2 ** n
        bits = ((np.arange(N)[:, None] >> np.arange(n)[None, :]) & 1).astype(bool)  # (N, n)
        ns = {g: bits[:, i] for i, g in enumerate(genes)}
        next_bits = np.zeros_like(bits)
        for i, g in enumerate(genes):
            next_bits[:, i] = eval(rules[g], {}, ns)
        is_fp = np.all(next_bits == bits, axis=1)
        return int(is_fp.sum()), True, N
    else:
        rng = np.random.default_rng(seed)
        bits = rng.integers(0, 2, size=(n_sample, n)).astype(bool)
        ns = {g: bits[:, i] for i, g in enumerate(genes)}
        next_bits = np.zeros_like(bits)
        for i, g in enumerate(genes):
            next_bits[:, i] = eval(rules[g], {}, ns)
        is_fp = np.all(next_bits == bits, axis=1)
        return int(is_fp.sum()), False, n_sample


def reachable_from(rules, genes, start, n_runs=40, max_sweeps=200, seed=0):
    """Asynchronous simulation from a fixed start state, n_runs independent random update
    orders; returns (set of distinct FIXED POINTS actually converged to, n_converged, n_runs).
    Runs that never stop changing within max_sweeps (a genuine async limit cycle, e.g. EMT's
    Snai1 self-repression) are excluded from the reached set, not silently miscounted as fixed."""
    rng = np.random.default_rng(seed)
    reached = set()
    n_converged = 0
    for _ in range(n_runs):
        s = dict(zip(genes, start))
        order_base = list(genes)
        converged = False
        for _sweep in range(max_sweeps):
            order = order_base.copy(); rng.shuffle(order)
            changed = False
            for g in order:
                ns = {k: bool(v) for k, v in s.items()}
                newval = bool(eval(rules[g], {}, ns))
                if newval != s[g]:
                    s[g] = newval; changed = True
            if not changed:
                converged = True
                break
        if converged:
            n_converged += 1
            reached.add(tuple(int(s[g]) for g in genes))
    return reached, n_converged, n_runs


def build_ground_truth_rules(net, cfg):
    genes = cfg["genes"]
    return parse_rule_file(cfg["rule_file"], genes)


def _remap(df, genes):
    """rankedEdges.csv for B_cell_activation/EMT/Pluripotent uses native gene_1..gene_N ids;
    remap to biological names (matrix/rule-file order) when present."""
    native = {f"gene_{i+1}": g for i, g in enumerate(genes)}
    if df.Gene1.iloc[0] in native or (len(df) > 1 and df.Gene1.iloc[1] in native):
        df = df.copy()
        df["Gene1"] = df["Gene1"].map(lambda x: native.get(x, x))
        df["Gene2"] = df["Gene2"].map(lambda x: native.get(x, x))
    return df


def load_ppcor_signs(beeline_dir, genes):
    """PPCOR signed correlation for every ordered pair, as the fallback sign source (BEELINE's
    own convention: 'for every other algorithm, we assigned signs based on PPCOR')."""
    reps = sorted(glob.glob(f"{beeline_dir}/simrep*_twin_paired"))
    if not reps:
        reps = sorted(glob.glob(f"{beeline_dir}/simrep*_spread"))
    f = f"{reps[0]}/PPCOR/rankedEdges.csv"
    df = pd.read_csv(f, sep="\t")
    df = _remap(df, genes)
    df = df[df.Gene1 != df.Gene2]
    return {(r.Gene1, r.Gene2): np.sign(r.EdgeWeight) for r in df.itertuples()}


def load_ranked(beeline_dir, algo, genes, scheme="twin_paired"):
    reps = sorted(glob.glob(f"{beeline_dir}/simrep*_{scheme}"))
    if not reps:
        return None
    f = f"{reps[0]}/{algo}/rankedEdges.csv"
    if not os.path.exists(f):
        return None
    df = pd.read_csv(f, sep="\t")
    df = _remap(df, genes)
    df = df[df.Gene1 != df.Gene2].reset_index(drop=True)
    df["absw"] = df.EdgeWeight.abs()
    df = df.sort_values("absw", ascending=False).reset_index(drop=True)
    return df


def load_twinfer_ranked(glob_pat):
    files = sorted(glob.glob(glob_pat))
    if not files:
        return None
    d = json.load(open(files[0]))
    re_ = d["ranked_edges"]
    df = pd.DataFrame(re_["data"], columns=re_["columns"])
    gmap = {f"gene_{i+1}": g for i, g in enumerate(d["gene_names"])}
    # gene_names in this JSON are the network's own biological names already for GSD/HSC/mCAD/VSC
    # (matches NETS[genes]); for B_cell/EMT/Pluripotent the JSON uses gene_1..N native ids, remapped below.
    return df, gmap, d["gene_names"]


def build_inferred_rules(ranked_df, signs, genes, gt_matrix, gene_idx, ppcor_signs):
    """BEELINE's own edge-selection rule: walk down the ranking until every gene has >=1 incoming
    edge (self-loops excluded from ranking, added back from ground truth separately). `signs`:
    array aligned with ranked_df rows, or None to always fall back to PPCOR's sign for the pair."""
    activators, repressors = {g: [] for g in genes}, {g: [] for g in genes}
    covered = set()
    n_edges = 0
    for i, row in enumerate(ranked_df.itertuples()):
        src, tgt = getattr(row, "Gene1"), getattr(row, "Gene2")
        if src not in gene_idx or tgt not in gene_idx:
            continue
        if signs is not None:
            sign = np.sign(signs[i]) if signs[i] != 0 else ppcor_signs.get((src, tgt), 1.0)
        else:
            sign = ppcor_signs.get((src, tgt), 1.0)
        if sign >= 0:
            if src not in activators[tgt]: activators[tgt].append(src)
        else:
            if src not in repressors[tgt]: repressors[tgt].append(src)
        n_edges += 1
        covered.add(tgt)
        if len(covered) >= len(genes):
            break
    rules = {}
    for g in genes:
        act = activators[g]; rep = repressors[g]
        self_loop = gt_matrix[gene_idx[g], gene_idx[g]] != 0
        if self_loop and g not in act and g not in rep:
            act = act + [g]
        act_c = "(" + " | ".join(act) + ")" if act else ""
        rep_c = "~(" + " | ".join(rep) + ")" if rep else ""
        if act_c and rep_c: rules[g] = f"{act_c} & {rep_c}"
        elif act_c: rules[g] = act_c
        elif rep_c: rules[g] = rep_c
        else: rules[g] = g
    return rules, n_edges


def count_true_edges(gt_matrix):
    return int((gt_matrix != 0).sum())


results = []
for net, cfg in NETS.items():
    genes = cfg["genes"]; n = len(genes)
    gene_idx = {g: i for i, g in enumerate(genes)}
    M = np.loadtxt(cfg["matrix"], delimiter=",")
    n_true_edges = count_true_edges(M)
    gt_rules = build_ground_truth_rules(net, cfg)
    gt_fp, gt_exact, gt_space = enumerate_fixed_points(gt_rules, genes)
    all_off = tuple(0 for _ in genes)
    gt_reach_set, gt_nconv, gt_nruns = reachable_from(gt_rules, genes, all_off)
    gt_reach = len(gt_reach_set)
    print(f"{net}: n={n}, true_edges={n_true_edges}, GT fixed points={gt_fp} (exact={gt_exact}, space={gt_space}), "
          f"reachable from all-off={gt_reach} (converged {gt_nconv}/{gt_nruns} runs)", flush=True)

    ppcor_signs = load_ppcor_signs(cfg["beeline_dir"], genes)

    for algo in ALGOS:
        rdf = load_ranked(cfg["beeline_dir"], algo, genes)
        if rdf is None:
            continue
        signed = (rdf.EdgeWeight < 0).any()
        signs = rdf.EdgeWeight.to_numpy() if signed else None
        rules, n_edges = build_inferred_rules(rdf, signs, genes, M, gene_idx, ppcor_signs)
        fp, exact, space = enumerate_fixed_points(rules, genes)
        reach_set, nconv, nruns = reachable_from(rules, genes, all_off)
        reach = len(reach_set)
        results.append(dict(net=net, algorithm=algo, n_genes=n, n_true_edges=n_true_edges,
                             n_inferred_edges=n_edges, edge_ratio=n_edges / n_true_edges,
                             gt_fixed_points=gt_fp, inferred_fixed_points=fp,
                             fp_ratio=fp / max(gt_fp, 1), gt_exact=gt_exact,
                             gt_reachable=gt_reach, inferred_reachable=reach,
                             reach_ratio=reach / max(gt_reach, 1), inferred_conv_rate=nconv / nruns,
                             gt_conv_rate=gt_nconv / gt_nruns))
        print(f"  {algo}: edges={n_edges} (ratio {n_edges/n_true_edges:.2f}), fixed_points={fp} "
              f"(ratio {fp/max(gt_fp,1):.2f}), reachable={reach}/{gt_reach} (conv {nconv}/{nruns})", flush=True)

    # TwINFER
    tw = load_twinfer_ranked(cfg["twinfer_glob"])
    if tw is not None:
        df, gmap, jnames = tw
        if jnames[0].startswith("gene_"):
            df["gene_1"] = df["gene_1"].map(gmap); df["gene_2"] = df["gene_2"].map(gmap)
        rdf = df.rename(columns={"gene_1": "Gene1", "gene_2": "Gene2", "twinScore": "EdgeWeight"})
        rdf = _remap(rdf, genes)
        rdf = rdf[rdf.Gene1 != rdf.Gene2].reset_index(drop=True)
        rdf["absw"] = rdf.EdgeWeight.abs()
        rdf = rdf.sort_values("absw", ascending=False).reset_index(drop=True)
        signs = rdf["rho_t2"].to_numpy() if "rho_t2" in rdf.columns else None
        rules, n_edges = build_inferred_rules(rdf, signs, genes, M, gene_idx, ppcor_signs)
        fp, exact, space = enumerate_fixed_points(rules, genes)
        reach_set, nconv, nruns = reachable_from(rules, genes, all_off)
        reach = len(reach_set)
        results.append(dict(net=net, algorithm="TwINFER", n_genes=n, n_true_edges=n_true_edges,
                             n_inferred_edges=n_edges, edge_ratio=n_edges / n_true_edges,
                             gt_fixed_points=gt_fp, inferred_fixed_points=fp,
                             fp_ratio=fp / max(gt_fp, 1), gt_exact=gt_exact,
                             gt_reachable=gt_reach, inferred_reachable=reach,
                             reach_ratio=reach / max(gt_reach, 1), inferred_conv_rate=nconv / nruns,
                             gt_conv_rate=gt_nconv / gt_nruns))
        print(f"  TwINFER: edges={n_edges} (ratio {n_edges/n_true_edges:.2f}), fixed_points={fp} "
              f"(ratio {fp/max(gt_fp,1):.2f}), reachable={reach}/{gt_reach} (conv {nconv}/{nruns})", flush=True)

df = pd.DataFrame(results)
# [2026-10-01 commented out: wrote next to the script, i.e. into the code tree; results live in clean_data/] df.to_csv(f"{os.path.dirname(os.path.abspath(__file__))}/dynamics_fidelity_results.csv", index=False)
df.to_csv(f"{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/real_networks/why_twinfer_fails/dynamics_fidelity_results.csv", index=False)
print("\nSaved dynamics_fidelity_results.csv,", len(df), "rows")
