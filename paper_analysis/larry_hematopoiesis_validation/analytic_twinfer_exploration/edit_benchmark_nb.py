from twinfer.utils.paths import get_data_root as _res_gdr  # [2026-10-01 LARRY resources/ moved out of the code tree to analysis_data/paper_analysis/larry_hematopoiesis_validation/resources (user)]
RES_HERE = f"{_res_gdr()}/paper_analysis/larry_hematopoiesis_validation"
import zlib  # [2026-10-01 added]
from twinfer.utils.paths import get_repo_root as _twinfer_get_repo_root  # [2026-09-30 added]
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Adapt benchmark_methods.ipynb: add analytic_twinfer + perm_twinfer as methods, read the
current _allpairs (not _50core) infer dirs, skip any method whose file is missing."""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json
# [2026-09-30 commented out: pointed into the original code tree; now the clean_code copy] NB = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation/benchmark_methods.ipynb'
NB = f'{_twinfer_get_repo_root()}/paper_analysis/larry_hematopoiesis_validation/benchmark_methods.ipynb'
nb = json.load(open(NB))


def setcell(i, text):
    nb["cells"][i]["source"] = [l + "\n" for l in text.split("\n")][:-1] + [text.split("\n")[-1]] \
        if not text.endswith("\n") else [l + "\n" for l in text.split("\n")[:-1]] + [""]
    nb["cells"][i]["source"] = text.splitlines(keepends=True)
    if nb["cells"][i]["cell_type"] == "code":
        nb["cells"][i]["outputs"] = []
        nb["cells"][i]["execution_count"] = None


# ---- cell 1: method list --------------------------------------------------------------------
c1 = "".join(nb["cells"][1]["source"])
c1 = c1.replace('ALL_METHODS = ["twinfer"] + COMPETITOR_METHODS',
                'TWINFER_METHODS = ["analytic_twinfer", "perm_twinfer"]\n'
                'ALL_METHODS = TWINFER_METHODS + COMPETITOR_METHODS')
setcell(1, c1)

# ---- cell 5: loaders (skip-if-missing, current dir names, analytic loader) -------------------
setcell(5, '''def _align_scores(edge_score, universe):
    """edge_score: dict[(a,b)] -> float. Returns y_score in universe order, missing pairs filled
    with min(edge_score) (least confident -- the pair was never called, not tied for best)."""
    if not edge_score:
        return np.zeros(len(universe), dtype=float)
    floor = min(edge_score.values())
    return np.array([edge_score.get(pair, floor) for pair in universe], dtype=float)


def load_competitor(method, gene_set, universe):
    p = os.path.join(NETWORKS_DIR, f"{method}_{gene_set}.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    edge_score = {(r.TF, r.target): float(r.importance) for r in df.itertuples(index=False)}
    return _align_scores(edge_score, universe)


def _load_ranked_edges(path, universe, tfs):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    tf_set = set(tfs)
    df = df[df["gene_1"].isin(tf_set) & (df["gene_1"] != df["gene_2"])]
    edge_score = {(r.gene_1, r.gene_2): float(r.twinScore) for r in df.itertuples(index=False)}
    return _align_scores(edge_score, universe)


def load_analytic_twinfer(gene_set, universe, tfs):
    # run_analytic_twinfer.py -- shuffle-free z-scores (analytic_zscores.py), cp10k input
    # [2026-10-01 commented out: LARRY resources/ moved to analysis_data (user)] return _load_ranked_edges(os.path.join(HERE, "resources", "analytic_infer", gene_set,
    return _load_ranked_edges(os.path.join(RES_HERE, "resources", "analytic_infer", gene_set,
                                           "ranked_edges.csv"), universe, tfs)


def load_perm_twinfer(gene_set, universe, tfs):
    # 5000-shuffle ALL_PAIRS run (run_infer_correlation_high.py); may be absent for some sets
    return _load_ranked_edges(os.path.join(INFER_DIR, f"{gene_set}_t2_t4_allpairs",
                                           "ranked_edges.csv"), universe, tfs)


scores = {}
for gs in GENE_SETS:
    universe = universes[gs]["universe"]
    tfs = universes[gs]["tfs"]
    got = {}
    for name, loader in (("analytic_twinfer", load_analytic_twinfer),
                         ("perm_twinfer", load_perm_twinfer)):
        s = loader(gs, universe, tfs)
        if s is not None:
            got[name] = s
    for m in COMPETITOR_METHODS:
        s = load_competitor(m, gs, universe)
        if s is not None:
            got[m] = s
    scores[gs] = got
    print(f"{gs:<18} methods present: {list(got)}")''')

# ---- cell 9: iterate over methods actually present ------------------------------------------
setcell(9, '''import zlib  # [2026-10-01 added: the stable seed below uses it]
rows = []
for gs in GENE_SETS:
    y_true = universes[gs]["y_true"]
    for method, y_score in scores[gs].items():
        obs = observed_metrics(y_true, y_score)
        # [2026-10-01 commented out: hash() of strings is randomised per process (PYTHONHASHSEED) so this seed was not reproducible; zlib.crc32 of the repr is stable] rnd = random_baseline(y_true, y_score, seed=SEED + hash((gs, method)) % (2**31))
        rnd = random_baseline(y_true, y_score, seed=SEED + zlib.crc32(repr((gs, method)).encode()) % (2**31))
        rows.append(dict(
            gene_set=gs, method=method,
            n_universe=len(y_true), n_curated=int(y_true.sum()),
            density=float(y_true.mean()),
            early_precision=obs["early_precision"],            # top-k precision  (k = n_curated)
            f1=obs["f1"],                                      # top-k F1  (== early_precision at this k)
            epr=obs["epr"],                                    # top-k precision / density
            auprc=obs["auprc"],
            auprc_random=rnd["auprc_random"],
            auprc_ratio=obs["auprc"] / rnd["auprc_random"] if rnd["auprc_random"] > 0 else float("nan"),
            f1_random=rnd["f1_random"],
            f1_ratio=obs["f1"] / rnd["f1_random"] if rnd["f1_random"] > 0 else float("nan"),
        ))
    print(f"{gs}: done")

results = pd.DataFrame(rows)
results.head(len(ALL_METHODS))''')

# ---- cell 11: pivots over present methods --------------------------------------------------
setcell(11, '''out_path = os.path.join(OUT_DIR, "benchmark_results.csv")
results.to_csv(out_path, index=False)
print(f"wrote {out_path}  ({len(results)} rows)")

present = [m for m in ALL_METHODS if m in set(results.method)]
for metric in ("early_precision", "f1", "auprc", "auprc_ratio", "epr"):
    print(f"\\n=== {metric} ===")
    piv = results.pivot(index="gene_set", columns="method", values=metric).reindex(GENE_SETS)
    display(piv[[c for c in present if c in piv.columns]].round(3))

print("\\n=== mean across gene sets, per method ===")
summary = results.groupby("method")[["early_precision", "f1", "epr", "auprc", "auprc_ratio"]].mean()
display(summary.reindex(present).round(3))
summary.reindex(present).to_csv(os.path.join(OUT_DIR, "benchmark_summary_by_method.csv"))
print("wrote benchmark_summary_by_method.csv")''')

json.dump(nb, open(NB, "w"), indent=1)
json.load(open(NB))
print("OK: benchmark_methods.ipynb adapted (analytic_twinfer + perm_twinfer, skip-if-missing)")
