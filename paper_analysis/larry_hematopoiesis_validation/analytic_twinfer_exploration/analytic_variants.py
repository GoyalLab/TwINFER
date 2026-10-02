# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Shuffle-free TwINFER, two knobs: correlation weighting (clone vs unweighted) and TwinScore
composition (default vs yscher's tuned `twinscore_current.py`). Scored on yscher's exact corrhigh
panel by yscher's rule (genes x genes, all CollecTRI edges, prec@R / base, AUPRC / base).

Ranking-only, so analytic null SDs are irrelevant: TwinScore panel-standardizes every term (s()),
which removes any uniform per-term scale. Each s(z_type) is replaced by s() of the underlying
correlation quantity (monotone in z_type). The z_reg_gated gate is approximated by |rho_reg(lambda)|
percentile.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import json, sys, itertools
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/home/gzu5140/TwINFER_KA/code/TwINFER/package")
from twinfer.inference.correlation_functions import (
    assign_twin_id, calculate_pairwise_gene_gene_correlation_matrix,
    calculate_twin_random_correlations, _build_cross_time_twins, get_cross_correlations,
    weighted_spearman, split_twins,
)

H = f'{TWINFER_PROJECT_ROOT}/code/TwINFER/paper_analysis/larry_hematopoiesis_validation'
import os as _os
# [2026-10-01 commented out: pointed into an ephemeral Claude scratchpad that held a copy of yscher's panel json; the same file is now in clean_data/external_yscher]
# PANEL = json.load(open(f"/gpfs/home/gzu5140/.claude-tmp/claude-2000104/-gpfs-projects-b1255-hzhang-TwINFER-KA/8c755a86-0074-4569-970e-28157d9bb13a/scratchpad/yscher_{_os.environ.get('PANEL','corrhigh')}_panel.json"))
PANEL = json.load(open(f"{TWINFER_PROJECT_ROOT}/clean_data/external_yscher/Transcriptomic_Distance/exports/panels/tf_target_panel/p4a01_{_os.environ.get('PANEL','corrhigh')}_panel.json"))
_G0 = sorted(PANEL["panel"])
_SAN = {g: g.replace("-", "_") for g in _G0}
_INV = {v: k for k, v in _SAN.items()}
G = [_SAN[g] for g in _G0]
CT = pd.read_csv(f"{H}/resources/collectri_mouse.tsv", sep="\t")
CT = CT[CT.source_genesymbol != CT.target_genesymbol]
CE = {(_SAN.get(a, a), _SAN.get(b, b)) for a, b in zip(CT.source_genesymbol, CT.target_genesymbol)}

# rebuild the input for the 47 panel genes -- NORMALIZED: log1p(CP10k), totals over full transcriptome
import scipy.io as sio
X = sio.mmread(f"{TWINFER_PROJECT_ROOT}/finalized_data/LARRY_data/filtered/larry_qc_counts.mtx").tocsr()
allg = pd.Index(open(f'{TWINFER_PROJECT_ROOT}/finalized_data/LARRY_data/filtered/genes.txt').read().split())
obs = pd.read_csv(f'{TWINFER_PROJECT_ROOT}/finalized_data/LARRY_data/filtered/obs_metadata.csv', index_col=0)
gi = {g: i for i, g in enumerate(allg)}
day = obs["library"].astype(str).str.extract(r"_d(\d+)_", expand=False).astype(int)
_tot = np.asarray(X.sum(1)).ravel()
_norm = np.log1p(X[:, [gi[_INV.get(g, g)] for g in G]].toarray() / _tot[:, None] * 1e4)
df = pd.DataFrame(_norm, columns=[f"{g}_mRNA" for g in G], index=obs.index)
df.insert(0, "time_step", day.to_numpy()); df.insert(0, "cell_id", obs.index)
df.insert(0, "clone_id", obs["larry_clone_singletcode"].to_numpy()); df = df.reset_index(drop=True)

t1_raw = df[df.time_step == 2].reset_index(drop=True)
t2_raw = df[df.time_step == 4].reset_index(drop=True)
t1_tw = assign_twin_id(t1_raw).reset_index(drop=True)
t2_tw = assign_twin_id(t2_raw).reset_index(drop=True)
at1, at2 = _build_cross_time_twins(t1_raw, t2_raw)
ordered = [(a, b) for a in G for b in G if a != b]
gidx = {g: k for k, g in enumerate(G)}


def corr_set(use_clone, unit):
    r1 = calculate_pairwise_gene_gene_correlation_matrix(t1_raw, G, use_clone=use_clone)
    r2 = calculate_pairwise_gene_gene_correlation_matrix(t2_raw, G, use_clone=use_clone)
    tw1, _ = calculate_twin_random_correlations(t1_raw, t1_tw, G, random_state=0, unit=unit)
    tw2, _ = calculate_twin_random_correlations(t2_raw, t2_tw, G, random_state=0, unit=unit)
    rr1 = np.mean([calculate_twin_random_correlations(t1_raw, t1_tw, G, random_state=100 + i, unit=unit)[1].to_numpy() for i in range(20)], 0)
    rr2 = np.mean([calculate_twin_random_correlations(t2_raw, t2_tw, G, random_state=200 + i, unit=unit)[1].to_numpy() for i in range(20)], 0)
    xc = get_cross_correlations(at1, at2, gene_pairs=ordered + [(g, g) for g in G], unit=unit)
    # gated: rho_same / rho_cross on pooled 2n twin rows (unit-weighted)
    tw0a, tw1b = split_twins(t1_tw)
    Xa = tw0a[[f"{g}_mRNA" for g in G]].to_numpy(float); Xb = tw1b[[f"{g}_mRNA" for g in G]].to_numpy(float)
    Xp = np.vstack([Xa, Xb])
    if unit == "clone":
        # each twin weight 1/K_c, member row 1/(2K_c); approximate: clone weight split over its twins
        kc = tw0a.groupby("clone_id").transform("size").iloc[:, 0].to_numpy() if False else None
    w = np.ones(len(Xp))
    Rk = np.apply_along_axis(rankdata, 0, Xp); Rk = Rk - Rk.mean(0)
    S = np.sqrt((Rk ** 2).sum(0)); S[S < 1e-12] = 1
    rho_same = (Rk.T @ Rk) / np.outer(S, S)                     # (x_pooled, y_pooled) same order
    Yp_cross = np.vstack([Xb, Xa])                              # sibling-swapped Y
    RkY = np.apply_along_axis(rankdata, 0, Yp_cross); RkY = RkY - RkY.mean(0)
    SY = np.sqrt((RkY ** 2).sum(0)); SY[SY < 1e-12] = 1
    rho_cross = (Rk.T @ RkY) / np.outer(S, SY)
    return dict(r1=r1, r2=r2, tw1=tw1, tw2=tw2, rr1=pd.DataFrame(rr1, index=G, columns=G),
               rr2=pd.DataFrame(rr2, index=G, columns=G), xc=xc,
               rho_same=pd.DataFrame(rho_same, index=G, columns=G),
               rho_cross=pd.DataFrame(rho_cross, index=G, columns=G))


def s(v):
    v = np.asarray(v, float); f = np.isfinite(v)
    o = np.full(len(v), 0.0)
    if f.sum() > 1:
        o[f] = (v[f] - v[f].mean()) / max(v[f].std(ddof=1), 1e-12)
    return o


def build_rows(C):
    rows = []
    for a, b in ordered:
        rt1, rt2 = C["r1"].loc[a, b], C["r2"].loc[a, b]
        rd1 = C["tw1"].loc[a, b]
        drr = C["rr2"].loc[a, b] - C["rr1"].loc[a, b]
        rxy, ryx = C["xc"].loc[a, b], C["xc"].loc[b, a]
        het = rd1 - C["rr1"].loc[a, b]
        lam = min(1.0, abs(het) / (2.33 * 0.025))
        rreg = C["rho_same"].loc[a, b] - lam * C["rho_cross"].loc[a, b]
        rows.append(dict(a=a, b=b, rt1=rt1, rt2=rt2, rd1=rd1, drr=drr,
                         rxy=rxy, ryx=ryx, het=het, rreg=rreg))
    return pd.DataFrame(rows)


def score(smap, univ, y, base, R, label):
    fl = min(smap.values()); sv = np.array([smap.get(p, fl) for p in univ])
    idx = np.argsort(-sv, kind="stable"); au = average_precision_score(y, sv)
    out = f"  {label:34s}"
    for k in [10, 20, 50, R]:
        out += f"  P@{k if k != R else 'R'}={y[idx[:k]].sum() / k / base:.2f}x"
    out += f"   AUPRC/base={au / base:.2f}x"
    print(out)


univ = ordered
y = np.array([1 if p in CE else 0 for p in univ]); R = int(y.sum()); base = y.mean()
print(f"yscher corrhigh panel: {len(G)} genes, universe {len(univ)}, R={R}, base {100*base:.2f}%")
print("yscher published: TwinScore x_base 4.62, AUPRC/base 3.24\n")

for use_clone, unit in [(True, "clone"), (False, "twin")]:
    tag = "CLONE-weighted" if use_clone else "UNWEIGHTED"
    C = corr_set(use_clone, unit)
    D = build_rows(C)
    reg_g = D.groupby("a")["rreg"].apply(lambda v: np.nanmean(np.abs(v)))
    RG = {g: (reg_g.get(g, np.nan)) for g in G}
    rgv = np.array([v for v in RG.values() if np.isfinite(v)])
    RGs = {g: ((RG[g] - rgv.mean()) / max(rgv.std(ddof=1), 1e-12) if np.isfinite(RG[g]) else 0.0) for g in G}

    # DEFAULT-ish: s(|rt1|)+s(|rt2|)-s(|drho|) - |z_div|*I(z_het<-2.576) - z_het*I(...) ... simplified
    E1 = s(np.abs(D.rt1)); E2 = s(np.abs(D.rt2)); CHG = s(np.abs(D.rt2 - D.rt1))
    default_ts = E1 + E2 - CHG
    score({(r.a, r.b): default_ts[i] for i, r in enumerate(D.itertuples())}, univ, y, base, R,
          f"{tag} | default (s|rt1|+s|rt2|-s|drho|)")

    # TUNED (twinscore_current.py TS), s() of the underlying quantities
    XS = s(np.minimum(np.abs(D.rxy), np.abs(D.ryx)))
    GAM = s(np.abs(np.abs(D.rxy) - np.abs(D.ryx)))
    ZDD = s(D.rxy - D.ryx)
    REGt = np.array([RGs[r.a] - RGs[r.b] for r in D.itertuples()])
    TS = (E1 - CHG - s(np.abs(D.rd1)) - s(np.abs(D.drr)) - XS - GAM + 0.5 * ZDD + REGt)
    # gate: |rho_reg(lambda)| in top 60% (proxy for |z_reg_gated|>1.645); failing -> score = -E1
    gate = np.abs(D.rreg) > np.quantile(np.abs(D.rreg), 0.40)
    Sfinal = np.where(gate, TS, -E1)
    score({(r.a, r.b): TS[i] for i, r in enumerate(D.itertuples())}, univ, y, base, R,
          f"{tag} | tuned TS (no gate)")
    score({(r.a, r.b): Sfinal[i] for i, r in enumerate(D.itertuples())}, univ, y, base, R,
          f"{tag} | tuned TS + rho_reg gate")
    print()
