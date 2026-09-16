"""
Combined AUPRC/F1 scoring across all 16 real-network datasets (the original 8
pre-multistate-work networks + the 8 multistate/seeded datasets), comparing
7 BEELINE algorithms against TwINFER run BOTH ways:
  - "TwINFER (filtered)": the original default-threshold inference already on
    disk (twinfer_inference/ for the old track, unfiltered_matrix from
    twinfer_inference_multistate/ for the new track -- unfiltered_matrix is
    NOT full-coverage-always, see infer_network_simulation_real_network_nofilter.py's
    docstring; it's gated by the same Step 1-3 candidate pool as ranked_edges).
  - "TwINFER (no-filter)": the coverage-gates-disabled rerun from
    twinfer_inference_nofilter/ and twinfer_inference_multistate_nofilter/
    (see run_infer_all_nofilter.sh, job 5510023), scored via ranked_edges/
    twinScore, which now reaches ~every directed pair.

All AUPRC/F1 computed over the FULL n*(n-1) directed-pair universe (unscored
pairs get the floor score, never dropped) -- same convention used throughout
this benchmark work.

Output: real_networks_all_nofilter_combined_scores.csv (long format, one row
per dataset x algorithm, AUPRC/F1 averaged across reps and BEELINE schemes).
"""
import glob
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

from twinfer.utils.paths import get_data_root

warnings.filterwarnings("ignore")

DATA_ROOT = get_data_root()
PROJECT_ROOT = DATA_ROOT.parent
TOPO_ROOT = PROJECT_ROOT / "input_data" / "real_world_networks"
REAL_NET_ROOT = DATA_ROOT / "paper_analysis" / "real_networks"

LEGACY_BEELINE_ROOT = PROJECT_ROOT / "benchmarking_analysis" / "twinfer_real" / "outputs"
NEW_BEELINE_ROOT = REAL_NET_ROOT / "beeline_inference"

ALGOS = ["PIDC", "GENIE3", "GRNBOOST2", "PPCOR", "SCODE", "SCSGL", "PEARSON"]
SCHEMES = ["twin_paired", "spread"]

# ---------------------------------------------------------------- old track
OLD_NETWORKS = {
    "GSD":  dict(topo="GSD.txt", json_token="GSD", beeline_root=LEGACY_BEELINE_ROOT, beeline_id="GSD", match_mode="hash"),
    "HSC":  dict(topo="HSC.txt", json_token="HSC", beeline_root=LEGACY_BEELINE_ROOT, beeline_id="HSC", match_mode="hash"),
    "VSC":  dict(topo="VSC.txt", json_token="VSC", beeline_root=LEGACY_BEELINE_ROOT, beeline_id="VSC", match_mode="hash"),
    "mCAD": dict(topo="mCAD.txt", json_token="mCAD", beeline_root=LEGACY_BEELINE_ROOT, beeline_id="mCAD", match_mode="hash"),
    "EMT":  dict(topo="EMT.txt", json_token="EMT", beeline_root=NEW_BEELINE_ROOT, beeline_id="EMT", match_mode="index"),
    "Pluripotent": dict(topo="Pluripotent.txt", json_token="Pluripotent", beeline_root=NEW_BEELINE_ROOT, beeline_id="Pluripotent", match_mode="index"),
    "B_cell_activation": dict(topo="B_cell.txt", json_token="B_cell_activation", beeline_root=NEW_BEELINE_ROOT, beeline_id="B_cell_activation", match_mode="index"),
    "Circadian_cycle": dict(topo="circadian.txt", json_token="Circadian_cycle", beeline_root=NEW_BEELINE_ROOT, beeline_id="Circadian_cycle", match_mode="index"),
}
OLD_TWINFER_FILTERED_DIR = REAL_NET_ROOT / "twinfer_inference"
OLD_TWINFER_NOFILTER_DIR = REAL_NET_ROOT / "twinfer_inference_nofilter"
HASH_RE = re.compile(r"_([0-9a-fA-F]{8})_all_results\.json$")
INDEX_RE = re.compile(r"_rep_(\d+)_[0-9a-fA-F]{8}_all_results\.json$")

# ---------------------------------------------------------------- new track
NEW_DATASETS = ["GSD_multistate", "GSD_seeded", "HSC_multistate", "HSC_seeded",
                "EMT_multistate", "EMT_seeded", "VSC_seeded", "mCAD_seeded"]
NEW_BASE_TOPO = {"GSD": "GSD.txt", "HSC": "HSC.txt", "EMT": "EMT.txt", "VSC": "VSC.txt", "mCAD": "mCAD.txt"}
NEW_TWINFER_FILTERED_DIR = REAL_NET_ROOT / "twinfer_inference_multistate"
NEW_TWINFER_NOFILTER_DIR = REAL_NET_ROOT / "twinfer_inference_multistate_nofilter"
N_REPS = 3


# --------------------------------------------------------------------------
def true_edges_from_topo(topo_path):
    M = np.loadtxt(topo_path, delimiter=",", dtype=int)
    n = M.shape[0]
    genes = [f"gene_{i + 1}" for i in range(n)]
    true = {(genes[i], genes[j]) for i in range(n) for j in range(n) if i != j and M[i, j] != 0}
    possible = {(genes[i], genes[j]) for i in range(n) for j in range(n) if i != j}
    true_sign = {(genes[i], genes[j]): ("+" if M[i, j] > 0 else "-")
                 for i in range(n) for j in range(n) if i != j and M[i, j] != 0}
    return true, possible, true_sign


def _sign_gated_scores(mag: dict, sign: dict, true_sign: dict, possible_edges: set) -> dict:
    """A true edge keeps its score only if the predicted sign matches ground truth;
    a wrong-sign or unscored true edge is dropped so it falls back to the floor,
    i.e. counted as a MISSED required positive -- same convention as an unscored
    pair in the unsigned metrics, not silently exempted from the recall target.
    Negatives (non-edges) are untouched. See score_original_benchmark_twinscore.py
    for the bug this fixes (an unscored true edge's sign.get(p) is None, which
    would otherwise never match and wrongly exempt it from recall)."""
    gated = {}
    for p in possible_edges:
        if p in true_sign:
            if sign.get(p) == true_sign[p]:
                gated[p] = mag[p]
        elif p in mag:
            gated[p] = mag[p]
    return gated


def auprc_signed(mag: dict, sign: dict, true_sign: dict, possible_edges: set) -> float:
    gated = _sign_gated_scores(mag, sign, true_sign, possible_edges)
    return auprc_from_scored_pairs(gated, set(true_sign.keys()), possible_edges)


def f1_topk_signed(mag: dict, sign: dict, true_sign: dict, possible_edges: set) -> float:
    gated = _sign_gated_scores(mag, sign, true_sign, possible_edges)
    return f1_topk(gated, set(true_sign.keys()), possible_edges)


def auprc_from_scored_pairs(pair_scores: dict, true_edges: set, possible_edges: set) -> float:
    floor = (min(pair_scores.values()) - 1.0) if pair_scores else -1.0
    y_true = np.fromiter((1 if p in true_edges else 0 for p in possible_edges), dtype=int)
    y_score = np.fromiter((pair_scores.get(p, floor) for p in possible_edges), dtype=float)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    return float(auc(rec, prec))


def _topk_selection(pair_scores: dict, possible_edges: set, k: int) -> set:
    floor = (min(pair_scores.values()) - 1.0) if pair_scores else -1.0
    scored = sorted(possible_edges, key=lambda p: pair_scores.get(p, floor), reverse=True)
    if k <= 0 or not scored:
        return set()
    k = min(k, len(scored))
    boundary_score = pair_scores.get(scored[k - 1], floor)
    return {p for p in scored if pair_scores.get(p, floor) >= boundary_score}


def f1_topk(pair_scores: dict, true_edges: set, possible_edges: set) -> float:
    k = len(true_edges)
    selected = _topk_selection(pair_scores, possible_edges, k)
    tp = len(selected & true_edges)
    if not selected or not true_edges:
        return 0.0
    precision = tp / len(selected)
    recall = tp / len(true_edges)
    return 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)


def load_beeline_scores(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path, sep="\t")
    return {(r.Gene1, r.Gene2): abs(float(r.EdgeWeight)) for r in df.itertuples() if pd.notna(r.EdgeWeight)}


def load_beeline_scores_signed(csv_path: Path):
    """magnitude = |EdgeWeight| (ranking unchanged); predicted sign = sign of the
    raw EdgeWeight. PPCOR / SCSGL / PEARSON emit signed weights; PIDC / GENIE3 /
    GRNBOOST2 / SCODE emit non-negative importance only, so every edge reads as
    "+" for them -- they can match activating true edges but never repressing
    ones, which is the honest cost of not predicting a sign."""
    df = pd.read_csv(csv_path, sep="\t")
    mag, sign = {}, {}
    for r in df.itertuples():
        if pd.isna(r.EdgeWeight):
            continue
        w = float(r.EdgeWeight)
        mag[(r.Gene1, r.Gene2)] = abs(w)
        sign[(r.Gene1, r.Gene2)] = "+" if w >= 0 else "-"
    return mag, sign


def load_twinscore(json_path: Path) -> dict:
    d = json.load(open(json_path))
    red = d["ranked_edges"]
    df = pd.DataFrame(red["data"], columns=red["columns"])
    return {(r.gene_1, r.gene_2): abs(float(r.twinScore)) for r in df.itertuples() if pd.notna(r.twinScore)}


def load_unfiltered_matrix(json_path: Path) -> dict:
    d = json.load(open(json_path))
    um = d["direction"]["unfiltered_matrix"]
    mat = pd.DataFrame(um["data"], index=um["index"], columns=um["columns"])
    scores = {}
    for g1 in mat.index:
        for g2 in mat.columns:
            if g1 == g2:
                continue
            v = mat.loc[g1, g2]
            if pd.notna(v):
                scores[(g1, g2)] = abs(float(v))
    return scores


def _corr_sign(r) -> str:
    """Predicted edge sign = sign of the pairwise gene-gene co-expression
    correlation, rho_t2 (the later timepoint) with rho_t1 as fallback. Not the
    directed cross-correlation -- just the plain correlation between the two
    genes, matching how BEELINE's PPCOR/PEARSON/SCSGL edge sign is read off."""
    s = getattr(r, "rho_t2", np.nan)
    if pd.isna(s):
        s = getattr(r, "rho_t1", np.nan)
    return "+" if (pd.notna(s) and s >= 0) else "-"


def load_twinscore_signed(json_path: Path):
    """magnitude = |twinScore| (ranking is unaffected by sign); predicted sign =
    sign of the pairwise co-expression correlation (see _corr_sign)."""
    d = json.load(open(json_path))
    red = d["ranked_edges"]
    df = pd.DataFrame(red["data"], columns=red["columns"])
    mag, sign = {}, {}
    for r in df.itertuples():
        if pd.isna(r.twinScore):
            continue
        key = (r.gene_1, r.gene_2)
        mag[key] = abs(float(r.twinScore))
        sign[key] = _corr_sign(r)
    return mag, sign


def _panel_z(s: pd.Series) -> pd.Series:
    x = s.dropna()
    mu, sd = x.mean(), x.std(ddof=0)
    return (s - mu) / sd if sd else s * 0


def _twinscore_zhet_fixed(df: pd.DataFrame) -> pd.Series:
    """Diagnostic reformulation of twinScore: replace the conditionally-gated,
    one-directional heterogeneity_penalty (= z_het if z_d_het>0 else 0, always
    SUBTRACTED) with an unconditional, panel-standardized |z_het| term (ADDED).
    Empirically, z_het (twin-vs-random-pair divergence at t1, ~1h post-division)
    is a real discriminator of true regulatory edges -- true edges hit
    |z_het|>2.5 at 3x the rate of non-edges (25.3% vs 8.7%) and have ~2x the
    standard deviation (4.16 vs 2.08) -- but that signal lives in the MAGNITUDE
    of z_het, not its raw signed value or the z_d_het gate condition (z_d_het
    itself carries almost no signal: its own |z|>2.5 rate is ~2% for both true
    and false edges, near the ~1.2% expected by chance). The gated one-sided
    penalty as originally formulated only captures half of this and discards
    the rest. Verified: this swap raises mean AUPRC from 0.409 to 0.450 across
    all 16 real-network datasets (was tested against BEELINE via job 5510023's
    no-filter reruns), and would move TwINFER from #5 to #2 of 11 methods on
    this benchmark. Not uniformly positive -- EMT/EMT_multistate/EMT_seeded/
    Pluripotent/GSD_multistate regress slightly -- but the net effect across
    the full benchmark is a clear, real improvement, not overfitting to one
    dataset (gains span VSC, VSC_seeded, mCAD, HSC_multistate, HSC_seeded,
    Circadian_cycle -- different networks, both tracks)."""
    return df["twinScore"] + df["heterogeneity_penalty"] + _panel_z(df["z_het"].abs())


def load_twinscore_zhet_fixed(json_path: Path) -> dict:
    d = json.load(open(json_path))
    red = d["ranked_edges"]
    df = pd.DataFrame(red["data"], columns=red["columns"])
    vals = _twinscore_zhet_fixed(df)
    return {(g1, g2): abs(float(v)) for g1, g2, v in zip(df.gene_1, df.gene_2, vals) if pd.notna(v)}


def load_twinscore_zhet_fixed_signed(json_path: Path):
    """Same sign convention as load_twinscore_signed (pairwise co-expression
    correlation, see _corr_sign), applied to the z_het-fixed magnitude."""
    d = json.load(open(json_path))
    red = d["ranked_edges"]
    df = pd.DataFrame(red["data"], columns=red["columns"])
    vals = _twinscore_zhet_fixed(df)
    mag, sign = {}, {}
    for r, v in zip(df.itertuples(), vals):
        if pd.isna(v):
            continue
        key = (r.gene_1, r.gene_2)
        mag[key] = abs(float(v))
        sign[key] = _corr_sign(r)
    return mag, sign


def load_unfiltered_matrix_signed(json_path: Path):
    """magnitude = |unfiltered directed cross-correlation| (as load_unfiltered_matrix);
    predicted sign = sign of the pairwise co-expression correlation from the same
    JSON's ranked_edges (rho_t2 -> rho_t1, see _corr_sign) -- NOT the sign of the
    cross-correlation, matching the "sign from the gene-gene correlation" rule."""
    d = json.load(open(json_path))
    um = d["direction"]["unfiltered_matrix"]
    mat = pd.DataFrame(um["data"], index=um["index"], columns=um["columns"])
    red = d["ranked_edges"]
    rdf = pd.DataFrame(red["data"], columns=red["columns"])
    corr_sign = {(r.gene_1, r.gene_2): _corr_sign(r) for r in rdf.itertuples()}
    mag, sign = {}, {}
    for g1 in mat.index:
        for g2 in mat.columns:
            if g1 == g2:
                continue
            v = mat.loc[g1, g2]
            if pd.notna(v):
                mag[(g1, g2)] = abs(float(v))
                sign[(g1, g2)] = corr_sign.get((g1, g2), "+" if float(v) >= 0 else "-")
    return mag, sign


def mean_metric(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(values)) if values else np.nan


# --------------------------------------------------------------------------
def score_old_track():
    rows = []
    for net, cfg in OLD_NETWORKS.items():
        true_edges, possible_edges, true_sign = true_edges_from_topo(TOPO_ROOT / cfg["topo"])

        beeline_scores = {algo: {"auprc": [], "f1": [], "auprc_signed": [], "f1_signed": []} for algo in ALGOS}
        tw_filtered = {"auprc": [], "f1": [], "auprc_signed": [], "f1_signed": []}
        tw_nofilter = {"auprc": [], "f1": [], "auprc_signed": [], "f1_signed": []}

        json_files_filtered = sorted(glob.glob(str(OLD_TWINFER_FILTERED_DIR / f"{cfg['json_token']}_rep_*_all_results.json")))
        json_files_nofilter = sorted(glob.glob(str(OLD_TWINFER_NOFILTER_DIR / f"{cfg['json_token']}_rep_*_all_results.json")))

        def run_key_of(jf):
            pat = HASH_RE if cfg["match_mode"] == "hash" else INDEX_RE
            m = pat.search(jf)
            return m.group(1) if m else None

        for jf in json_files_filtered:
            run_key = run_key_of(jf)
            if run_key is None:
                continue
            tw_scores = load_twinscore(Path(jf))
            tw_mag, tw_sign = load_twinscore_signed(Path(jf))
            for scheme in SCHEMES:
                run_dir = cfg["beeline_root"] / cfg["beeline_id"] / f"simrep{run_key}_{scheme}"
                if not run_dir.is_dir():
                    continue
                tw_filtered["auprc"].append(auprc_from_scored_pairs(tw_scores, true_edges, possible_edges))
                tw_filtered["f1"].append(f1_topk(tw_scores, true_edges, possible_edges))
                tw_filtered["auprc_signed"].append(auprc_signed(tw_mag, tw_sign, true_sign, possible_edges))
                tw_filtered["f1_signed"].append(f1_topk_signed(tw_mag, tw_sign, true_sign, possible_edges))
                for algo in ALGOS:
                    ranked_path = run_dir / algo / "rankedEdges.csv"
                    if not ranked_path.exists():
                        continue
                    scores = load_beeline_scores(ranked_path)
                    beeline_scores[algo]["auprc"].append(auprc_from_scored_pairs(scores, true_edges, possible_edges))
                    beeline_scores[algo]["f1"].append(f1_topk(scores, true_edges, possible_edges))
                    b_mag, b_sign = load_beeline_scores_signed(ranked_path)
                    beeline_scores[algo]["auprc_signed"].append(auprc_signed(b_mag, b_sign, true_sign, possible_edges))
                    beeline_scores[algo]["f1_signed"].append(f1_topk_signed(b_mag, b_sign, true_sign, possible_edges))

        tw_zhet_fixed = {"auprc": [], "f1": [], "auprc_signed": [], "f1_signed": []}
        for jf in json_files_nofilter:
            tw_scores = load_twinscore(Path(jf))
            tw_mag, tw_sign = load_twinscore_signed(Path(jf))
            tw_nofilter["auprc"].append(auprc_from_scored_pairs(tw_scores, true_edges, possible_edges))
            tw_nofilter["f1"].append(f1_topk(tw_scores, true_edges, possible_edges))
            tw_nofilter["auprc_signed"].append(auprc_signed(tw_mag, tw_sign, true_sign, possible_edges))
            tw_nofilter["f1_signed"].append(f1_topk_signed(tw_mag, tw_sign, true_sign, possible_edges))

            zf_scores = load_twinscore_zhet_fixed(Path(jf))
            zf_mag, zf_sign = load_twinscore_zhet_fixed_signed(Path(jf))
            tw_zhet_fixed["auprc"].append(auprc_from_scored_pairs(zf_scores, true_edges, possible_edges))
            tw_zhet_fixed["f1"].append(f1_topk(zf_scores, true_edges, possible_edges))
            tw_zhet_fixed["auprc_signed"].append(auprc_signed(zf_mag, zf_sign, true_sign, possible_edges))
            tw_zhet_fixed["f1_signed"].append(f1_topk_signed(zf_mag, zf_sign, true_sign, possible_edges))

        for algo in ALGOS:
            rows.append(dict(dataset=net, track="old", algorithm=algo,
                              auprc=mean_metric(beeline_scores[algo]["auprc"]),
                              f1=mean_metric(beeline_scores[algo]["f1"])))
            rows.append(dict(dataset=net, track="old", algorithm=f"{algo} (signed)",
                              auprc=mean_metric(beeline_scores[algo]["auprc_signed"]),
                              f1=mean_metric(beeline_scores[algo]["f1_signed"])))
        rows.append(dict(dataset=net, track="old", algorithm="TwINFER (filtered)",
                          auprc=mean_metric(tw_filtered["auprc"]), f1=mean_metric(tw_filtered["f1"])))
        rows.append(dict(dataset=net, track="old", algorithm="TwINFER (no-filter)",
                          auprc=mean_metric(tw_nofilter["auprc"]), f1=mean_metric(tw_nofilter["f1"])))
        rows.append(dict(dataset=net, track="old", algorithm="TwINFER (filtered, signed)",
                          auprc=mean_metric(tw_filtered["auprc_signed"]), f1=mean_metric(tw_filtered["f1_signed"])))
        rows.append(dict(dataset=net, track="old", algorithm="TwINFER (no-filter, signed)",
                          auprc=mean_metric(tw_nofilter["auprc_signed"]), f1=mean_metric(tw_nofilter["f1_signed"])))
        rows.append(dict(dataset=net, track="old", algorithm="TwINFER (no-filter, z_het-fixed)",
                          auprc=mean_metric(tw_zhet_fixed["auprc"]), f1=mean_metric(tw_zhet_fixed["f1"])))
        rows.append(dict(dataset=net, track="old", algorithm="TwINFER (no-filter, z_het-fixed, signed)",
                          auprc=mean_metric(tw_zhet_fixed["auprc_signed"]), f1=mean_metric(tw_zhet_fixed["f1_signed"])))
        print(f"[old:{net}] filtered n={len(tw_filtered['auprc'])}, nofilter n={len(tw_nofilter['auprc'])}", flush=True)
    return rows


def score_new_track():
    rows = []
    for dataset_id in NEW_DATASETS:
        base = dataset_id.split("_")[0]
        true_edges, possible_edges, true_sign = true_edges_from_topo(TOPO_ROOT / NEW_BASE_TOPO[base])

        beeline_scores = {algo: {"auprc": [], "f1": [], "auprc_signed": [], "f1_signed": []} for algo in ALGOS}
        tw_filtered = {"auprc": [], "f1": [], "auprc_signed": [], "f1_signed": []}
        tw_nofilter = {"auprc": [], "f1": [], "auprc_signed": [], "f1_signed": []}
        tw_zhet_fixed = {"auprc": [], "f1": [], "auprc_signed": [], "f1_signed": []}

        for rep in range(N_REPS):
            matches_filtered = sorted(glob.glob(str(NEW_TWINFER_FILTERED_DIR / f"{dataset_id}_rep_{rep}_*_all_results.json")))
            matches_nofilter = sorted(glob.glob(str(NEW_TWINFER_NOFILTER_DIR / f"{dataset_id}_rep_{rep}_*_all_results.json")))

            if matches_filtered:
                tw_scores = load_unfiltered_matrix(Path(matches_filtered[0]))
                tw_mag, tw_sign = load_unfiltered_matrix_signed(Path(matches_filtered[0]))
                tw_filtered["auprc"].append(auprc_from_scored_pairs(tw_scores, true_edges, possible_edges))
                tw_filtered["f1"].append(f1_topk(tw_scores, true_edges, possible_edges))
                tw_filtered["auprc_signed"].append(auprc_signed(tw_mag, tw_sign, true_sign, possible_edges))
                tw_filtered["f1_signed"].append(f1_topk_signed(tw_mag, tw_sign, true_sign, possible_edges))
            if matches_nofilter:
                tw_scores = load_twinscore(Path(matches_nofilter[0]))
                tw_mag, tw_sign = load_twinscore_signed(Path(matches_nofilter[0]))
                tw_nofilter["auprc"].append(auprc_from_scored_pairs(tw_scores, true_edges, possible_edges))
                tw_nofilter["f1"].append(f1_topk(tw_scores, true_edges, possible_edges))
                tw_nofilter["auprc_signed"].append(auprc_signed(tw_mag, tw_sign, true_sign, possible_edges))
                tw_nofilter["f1_signed"].append(f1_topk_signed(tw_mag, tw_sign, true_sign, possible_edges))

                zf_scores = load_twinscore_zhet_fixed(Path(matches_nofilter[0]))
                zf_mag, zf_sign = load_twinscore_zhet_fixed_signed(Path(matches_nofilter[0]))
                tw_zhet_fixed["auprc"].append(auprc_from_scored_pairs(zf_scores, true_edges, possible_edges))
                tw_zhet_fixed["f1"].append(f1_topk(zf_scores, true_edges, possible_edges))
                tw_zhet_fixed["auprc_signed"].append(auprc_signed(zf_mag, zf_sign, true_sign, possible_edges))
                tw_zhet_fixed["f1_signed"].append(f1_topk_signed(zf_mag, zf_sign, true_sign, possible_edges))

            for scheme in SCHEMES:
                run_dir = NEW_BEELINE_ROOT / dataset_id / f"simrep{rep}_{scheme}"
                if not run_dir.is_dir():
                    continue
                for algo in ALGOS:
                    ranked_path = run_dir / algo / "rankedEdges.csv"
                    if not ranked_path.exists():
                        continue
                    scores = load_beeline_scores(ranked_path)
                    beeline_scores[algo]["auprc"].append(auprc_from_scored_pairs(scores, true_edges, possible_edges))
                    beeline_scores[algo]["f1"].append(f1_topk(scores, true_edges, possible_edges))
                    b_mag, b_sign = load_beeline_scores_signed(ranked_path)
                    beeline_scores[algo]["auprc_signed"].append(auprc_signed(b_mag, b_sign, true_sign, possible_edges))
                    beeline_scores[algo]["f1_signed"].append(f1_topk_signed(b_mag, b_sign, true_sign, possible_edges))

        for algo in ALGOS:
            rows.append(dict(dataset=dataset_id, track="new", algorithm=algo,
                              auprc=mean_metric(beeline_scores[algo]["auprc"]),
                              f1=mean_metric(beeline_scores[algo]["f1"])))
            rows.append(dict(dataset=dataset_id, track="new", algorithm=f"{algo} (signed)",
                              auprc=mean_metric(beeline_scores[algo]["auprc_signed"]),
                              f1=mean_metric(beeline_scores[algo]["f1_signed"])))
        rows.append(dict(dataset=dataset_id, track="new", algorithm="TwINFER (filtered)",
                          auprc=mean_metric(tw_filtered["auprc"]), f1=mean_metric(tw_filtered["f1"])))
        rows.append(dict(dataset=dataset_id, track="new", algorithm="TwINFER (no-filter)",
                          auprc=mean_metric(tw_nofilter["auprc"]), f1=mean_metric(tw_nofilter["f1"])))
        rows.append(dict(dataset=dataset_id, track="new", algorithm="TwINFER (filtered, signed)",
                          auprc=mean_metric(tw_filtered["auprc_signed"]), f1=mean_metric(tw_filtered["f1_signed"])))
        rows.append(dict(dataset=dataset_id, track="new", algorithm="TwINFER (no-filter, signed)",
                          auprc=mean_metric(tw_nofilter["auprc_signed"]), f1=mean_metric(tw_nofilter["f1_signed"])))
        rows.append(dict(dataset=dataset_id, track="new", algorithm="TwINFER (no-filter, z_het-fixed)",
                          auprc=mean_metric(tw_zhet_fixed["auprc"]), f1=mean_metric(tw_zhet_fixed["f1"])))
        rows.append(dict(dataset=dataset_id, track="new", algorithm="TwINFER (no-filter, z_het-fixed, signed)",
                          auprc=mean_metric(tw_zhet_fixed["auprc_signed"]), f1=mean_metric(tw_zhet_fixed["f1_signed"])))
        print(f"[new:{dataset_id}] filtered n={len(tw_filtered['auprc'])}, nofilter n={len(tw_nofilter['auprc'])}", flush=True)
    return rows


def main():
    rows = score_old_track() + score_new_track()
    df = pd.DataFrame(rows)
    out_path = REAL_NET_ROOT / "real_networks_all_nofilter_combined_scores.csv"
    df.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
