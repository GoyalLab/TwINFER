"""
Pulse-and-chase local linear response for the mCAD Gillespie network.

Idea: the sim is already at steady state; take that state, apply a small SUSTAINED
bump to one gene's mRNA production (k_prod_mRNA *= 1+EPS, Hill constants K held at
their WT calibration), run forward a short time from the identical state with
COMMON RANDOM NUMBERS (per-cell shared seed for the perturbed & control runs so
intrinsic SSA noise cancels cell-by-cell), and read the response of every gene.

  R[b, a](tau) = mean_cells( protein_b^pert(tau) - protein_b^ctrl(tau) ) / EPS

R is the local causal influence of a on b at the WT operating point -- an
infinitesimal-perturbation ground truth that (unlike a full knockout) does not
move the system to a different regime.

Outputs (this folder):
  mcad_pop0.npz              steady-state population (n_species x n_cells) + meta
  mcad_response_protein.csv  R[target, source] and SEM at each tau  (protein)
  mcad_response_mrna.csv     same for mRNA
  mcad_vs_twinscore.csv      per directed pair: |R|, topology edge?, sign, twinScore, z's
  mcad_pulse_chase.png       response heatmap (topology boxed) + R-vs-twinScore
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths (/home/gzu5140/TwINFER_KA, /gpfs/projects/b1255/hzhang/TwINFER_KA, old Keerthana_b1042 tree)]
import glob
import json
import os
import sys
import time

import numpy as np
import numba
from numba import prange

# [2026-09-30 commented out: modules are imported via dotted package paths (env.sh puts clean_code and clean_code/package on PYTHONPATH)]
# sys.path.insert(0, "/gpfs/projects/b1255/hzhang/TwINFER_KA/code/TwINFER/package")
from twinfer.simulation.gillespie_simulations import (
    read_input_matrix, generate_reaction_network_from_matrix,
    generate_initial_state_from_genes, assign_parameters_to_genes,
    resolve_all_k_add, add_interaction_terms, setup_gillespie_params_from_reactions,
)

# HERE = os.path.dirname(os.path.abspath(__file__))   [2026-09-30 replaced: this script used to sit in the data dir analysis_data/paper_analysis/real_networks/pulse_and_chase and read/write next to itself; that dir is now referenced explicitly]
HERE = f'{TWINFER_PROJECT_ROOT}/analysis_data/paper_analysis/real_networks/pulse_and_chase'
RNROOT = f'{TWINFER_PROJECT_ROOT}'
PARAM_CSV = f"{RNROOT}/input_data/network_sweep/parameters.csv"

# network -> (topology file, twinfer_inference token, optional gene labels)
NETS = {
    "mCAD":  ("mCAD.txt", "mCAD", {1: "Pax6", 2: "Coup", 3: "Emx2", 4: "Fgf8", 5: "Sp8"}),
    "VSC":   ("VSC.txt", "VSC", {1: "Pax6", 2: "Nkx2.2", 3: "Dbx2", 4: "Olig2", 5: "Nkx6.1",
                                 6: "Irx3", 7: "Dbx1", 8: "Nkx6.2"}),
    "HSC":   ("HSC.txt", "HSC", {1: "cJun", 2: "SCL", 3: "Eklf", 4: "Gata1", 5: "PU.1", 6: "CEBPa",
                                 7: "Gata2", 8: "Gfi1", 9: "EgrNab", 10: "Fli1", 11: "Fog"}),
    "GSD":   ("GSD.txt", "GSD", {}),
    "EMT":   ("EMT.txt", "EMT", {}),
    "B_cell_activation": ("B_cell.txt", "B_cell_activation", {}),
    "Pluripotent": ("Pluripotent.txt", "Pluripotent", {}),
}
NET = os.environ.get("NET", "mCAD")
_topo, _tok, _lab = NETS[NET]
TOPO = f"{RNROOT}/input_data/real_world_networks/{_topo}"
JSON_GLOB = (f"{RNROOT}/analysis_data/paper_analysis/real_networks/"
             f"twinfer_inference_nofilter/{_tok}_rep_*_all_results.json")
LABELS = {i: _lab.get(i, f"g{i}") for i in range(1, 60)}
os.makedirs(f"{HERE}/out", exist_ok=True)
OUT = f"{HERE}/out/{NET}"          # file prefix, not a directory

N_CELLS = int(os.environ.get("N_CELLS", "12000"))
T_BURN = int(os.environ.get("T_BURN", "300"))       # protein t1/2 = 45 h -> ~7 half-lives
EPS = float(os.environ.get("EPS", "0.10"))                     # +10 % on k_prod_mRNA
TAUS = np.array([float(x) for x in os.environ.get("TAUS", "3,12,48").split(",")])
SEED0 = 20260909
numba.set_num_threads(int(os.environ.get("N_CORES", "16")))


# ----------------------------------------------------------------- CRN SSA
@numba.njit(parallel=True, fastmath=True)
def ssa_record(update_prop, update_matrix, pop0_mat, sample_times, seeds):
    """Per-cell SSA from pop0_mat, recording state at sample_times.
    np.random.seed(seeds[cell]) at the top of each cell -> common random numbers
    across two calls that pass the same seeds array."""
    n_species, n_cells = pop0_mat.shape
    n_s = sample_times.shape[0]
    n_rxns = update_matrix.shape[0]
    out = np.empty((n_cells, n_s, n_species), dtype=np.int64)
    for cell in prange(n_cells):
        np.random.seed(seeds[cell])
        pop = pop0_mat[:, cell].copy()
        t = 0.0
        i_s = 0
        prop = np.zeros(n_rxns, dtype=np.float64)
        stuck = 0
        while i_s < n_s:
            update_prop(prop, pop, t)
            total = prop.sum()
            if total <= 0.0:
                stuck += 1
                if stuck > 100000:
                    for k in range(i_s, n_s):
                        out[cell, k, :] = pop
                    break
                # jump to next sample time
                while i_s < n_s and t >= sample_times[i_s]:
                    out[cell, i_s, :] = pop
                    i_s += 1
                t = sample_times[i_s] if i_s < n_s else t + 1.0
                continue
            stuck = 0
            t += np.random.exponential(1.0 / total)
            while i_s < n_s and t >= sample_times[i_s]:
                out[cell, i_s, :] = pop
                i_s += 1
            cum = np.cumsum(prop)
            r = np.searchsorted(cum, np.random.rand() * total)
            pop += update_matrix[r]
    return out


def build(param_dict_override=None):
    n_genes, M = read_input_matrix(TOPO)
    reactions_df, gene_list = generate_reaction_network_from_matrix(
        M, combinatorial_interaction_type="additive")
    init_states = generate_initial_state_from_genes(gene_list)
    pdict = assign_parameters_to_genes(PARAM_CSV, gene_list, [[0] * n_genes][0] if False else [0] * n_genes)
    pdict = resolve_all_k_add(pdict, M, gene_list, use_csv_k_add=True, verbose=False)
    _, full = add_interaction_terms(pdict, M, gene_list, n_matrix=np.full((n_genes, n_genes), 2.0))
    if param_dict_override:
        full = dict(full)
        full.update(param_dict_override)
    pop0, upd, prop, sidx = setup_gillespie_params_from_reactions(init_states, reactions_df, full)
    return n_genes, M, gene_list, full, pop0, upd, prop, sidx


def main():
    t0 = time.time()
    n_genes, M, gene_list, full, pop0, upd, prop, sidx = build()
    prot_ix = np.array([sidx[f"{g}_protein"] for g in gene_list])
    mrna_ix = np.array([sidx[f"{g}_mRNA"] for g in gene_list])
    print(f"{NET}: {n_genes} genes {gene_list}, {upd.shape[0]} reactions, "
          f"{len(sidx)} species", flush=True)

    # --- burn-in to steady state from empty init (cached) ---------------------
    cache = f"{OUT}_pop0.npz"
    if os.path.exists(cache) and os.environ.get("REBURN", "") != "1":
        z = np.load(cache)
        pop_ss = z["pop_ss"]
        if pop_ss.shape[1] != N_CELLS:
            raise SystemExit(f"cache has {pop_ss.shape[1]} cells != N_CELLS={N_CELLS}; set REBURN=1")
        print(f"loaded steady-state population from cache ({pop_ss.shape})", flush=True)
    else:
        seeds_burn = (SEED0 + np.arange(N_CELLS)).astype(np.int64)
        pop0_mat = np.tile(pop0[:, None], (1, N_CELLS)).astype(np.int64)
        ss = ssa_record(prop, upd, pop0_mat, np.array([float(T_BURN)]), seeds_burn)[:, 0, :]
        pop_ss = ss.T.copy()                                                          # (n_species, n_cells)
        np.savez(cache, pop_ss=pop_ss, gene_list=np.array(gene_list),
                 prot_ix=prot_ix, mrna_ix=mrna_ix, T_BURN=T_BURN, N_CELLS=N_CELLS)
    print(f"burn-in {T_BURN}h done ({time.time()-t0:.0f}s).  WT mean protein: "
          + ", ".join(f"{LABELS[i+1]}={pop_ss[prot_ix[i]].mean():.0f}" for i in range(n_genes)), flush=True)

    # --- control chase --------------------------------------------------------
    seeds = (SEED0 + 7919 + np.arange(N_CELLS)).astype(np.int64)
    ctrl = ssa_record(prop, upd, pop_ss, TAUS, seeds)                                     # (n_cells, n_tau, n_species)
    print(f"control chase done ({time.time()-t0:.0f}s)", flush=True)

    # --- perturbed chases: bump each gene's k_prod_mRNA ----------------------
    Rp = np.zeros((len(TAUS), n_genes, n_genes))   # [tau, target, source]
    Rp_sem = np.zeros_like(Rp)
    Rm = np.zeros_like(Rp); Rm_sem = np.zeros_like(Rp)
    for a in range(n_genes):
        key = f"{{k_prod_mRNA_{gene_list[a]}}}"
        _, _, _, full_p, _, _, prop_p, _ = build({key: full[key] * (1.0 + EPS)})
        pert = ssa_record(prop_p, upd, pop_ss, TAUS, seeds)
        dprot = pert[:, :, prot_ix] - ctrl[:, :, prot_ix]        # (n_cells, n_tau, n_genes)
        dmrna = pert[:, :, mrna_ix] - ctrl[:, :, mrna_ix]
        Rp[:, :, a] = dprot.mean(0) / EPS                        # (n_tau, n_genes=target)
        Rp_sem[:, :, a] = dprot.std(0) / np.sqrt(N_CELLS) / EPS
        Rm[:, :, a] = dmrna.mean(0) / EPS
        Rm_sem[:, :, a] = dmrna.std(0) / np.sqrt(N_CELLS) / EPS
        print(f"  perturbed {LABELS[a+1]:5s}  ({time.time()-t0:.0f}s)", flush=True)

    # --- save response tables ----------------------------------------------
    import pandas as pd
    def to_long(R, Rsem, kind):
        rows = []
        for ti, tau in enumerate(TAUS):
            for b in range(n_genes):
                for a in range(n_genes):
                    rows.append(dict(tau=tau, source=LABELS[a+1], target=LABELS[b+1],
                                     source_g=f"gene_{a+1}", target_g=f"gene_{b+1}",
                                     kind=kind, R=R[ti, b, a], R_sem=Rsem[ti, b, a],
                                     true_edge=int(M[a, b] != 0), edge_sign=int(np.sign(M[a, b])),
                                     self=int(a == b)))
        return pd.DataFrame(rows)
    dfp = to_long(Rp, Rp_sem, "protein"); dfp.to_csv(f"{OUT}_response_protein.csv", index=False)
    dfm = to_long(Rm, Rm_sem, "mrna");    dfm.to_csv(f"{OUT}_response_mrna.csv", index=False)

    # --- pull twinScore / z-scores for mCAD ------------------------------------
    ZC = ["z_abs_rho_t1", "z_abs_rho_t2", "z_het", "z_div", "z_d_het", "z_gamma", "twinScore"]
    frames = []
    for jf in sorted(glob.glob(JSON_GLOB)):
        d = json.load(open(jf))
        red = pd.DataFrame(d["ranked_edges"]["data"], columns=d["ranked_edges"]["columns"])
        frames.append(red[["gene_1", "gene_2"] + ZC])
    tw = pd.concat(frames).groupby(["gene_1", "gene_2"], as_index=False)[ZC].mean() if frames else None

    # steady-state response (last tau) merged with twinScore
    last = TAUS[-1]
    comp = dfp[(dfp.tau == last) & (dfp.self == 0)][["source_g", "target_g", "R", "R_sem",
                                                     "true_edge", "edge_sign"]].copy()
    comp["abs_R"] = comp.R.abs()
    comp["R_significant"] = (comp.R.abs() > 3 * comp.R_sem).astype(int)
    if tw is not None:
        comp = comp.merge(tw.rename(columns={"gene_1": "source_g", "gene_2": "target_g"}),
                          on=["source_g", "target_g"], how="left")
    comp.to_csv(f"{OUT}_vs_zscores.csv", index=False)

    # --- figure --------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    names = [LABELS[i+1] for i in range(n_genes)]
    fig, axes = plt.subplots(2, 3, figsize=(17, 11))
    panel_tis = sorted(set(np.linspace(0, len(TAUS) - 1, 3).round().astype(int)))
    while len(panel_tis) < 3:
        panel_tis.append(panel_tis[-1])
    for ax, ti in zip(axes[0], panel_tis):
        Rn = Rp[ti]
        v = np.nanpercentile(np.abs(Rn[~np.eye(n_genes, dtype=bool)]), 98) or 1
        im = ax.imshow(Rn, cmap="RdBu_r", vmin=-v, vmax=v)
        for i in range(n_genes):
            for j in range(n_genes):
                if M[j, i] != 0:   # source j -> target i  (row=target, col=source): edge is M[source,target]=M[j,i]
                    c = "#2ca02c" if M[j, i] > 0 else "#111"
                    ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=c, lw=2.5))
                ax.text(j, i, f"{Rn[i, j]:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(Rn[i, j]) > v * 0.6 else "0.2")
        ax.set_xticks(range(n_genes)); ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticks(range(n_genes)); ax.set_yticklabels(names)
        ax.set_xlabel("source (perturbed)"); ax.set_ylabel("target (response)")
        ax.set_title(f"protein response  R[b,a]  at $\\tau$={TAUS[ti]:.0f} h")
        fig.colorbar(im, ax=ax, fraction=0.046)

    # R vs twinScore scatter (steady-state tau)
    ax = axes[1, 0]
    if tw is not None and "twinScore" in comp:
        for lab, sub in comp.groupby("true_edge"):
            ax.scatter(sub.abs_R, sub.twinScore, s=45,
                       label=("true edge" if lab else "non-edge"),
                       color=("#2ca02c" if lab else "#9aa0a6"))
        ax.set_xlabel(f"|local response| |R| at $\\tau$={last:.0f} h"); ax.set_ylabel("twinScore (mean)")
        ax.set_xscale("symlog"); ax.legend(); ax.set_title("twinScore vs local response")

    # |R| true vs non-edge (steady-state)
    ax = axes[1, 1]
    for k, (lab, sub) in enumerate(comp.groupby("true_edge")):
        ax.scatter(np.full(len(sub), k) + np.random.uniform(-.1, .1, len(sub)), sub.abs_R,
                   color=("#2ca02c" if lab else "#9aa0a6"), s=40)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["non-edge", "true edge"])
    ax.set_yscale("symlog"); ax.set_ylabel(f"|R| at $\\tau$={last:.0f} h")
    ax.set_title("local response, true vs non-edge")

    # response magnitude vs tau, per true edge
    ax = axes[1, 2]
    for a in range(n_genes):
        for b in range(n_genes):
            if a == b:
                continue
            style = "-" if M[a, b] != 0 else ":"
            col = ("#2ca02c" if M[a, b] > 0 else "#d62728") if M[a, b] != 0 else "0.8"
            ax.plot(TAUS, np.abs(Rp[:, b, a]), style, color=col, lw=1.6 if M[a, b] else 0.8)
    ax.set_xlabel("chase time $\\tau$ (h)"); ax.set_ylabel("|R[b,a]| (protein)")
    ax.set_title("solid+coloured = true edge, dotted grey = non-edge")
    fig.suptitle(f"{NET} pulse-and-chase  ·  +{EPS*100:.0f}% k_prod_mRNA  ·  "
                 f"{N_CELLS} CRN-paired cells  ·  burn-in {T_BURN}h", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{OUT}_pulse_chase.png", dpi=130, bbox_inches="tight")

    # --- print summary -----------------------------------------------------
    print("\n=== steady-state (tau=%g) protein response R[target<-source], /EPS ===" % last)
    print("           " + "  ".join(f"{n:>8s}" for n in names) + "   (columns = perturbed source)")
    for i in range(n_genes):
        print(f"{names[i]:>10s} " + "  ".join(f"{Rp[-1, i, j]:8.1f}" for j in range(n_genes)))
    print("\n(true edges boxed in the figure; M[source,target]!=0)")
    print("\n=== |R| true vs non-edge (tau=%g) ===" % last)
    print(comp.groupby("true_edge").abs_R.agg(["mean", "median", "max", "count"]).round(1).to_string())
    if tw is not None and "twinScore" in comp:
        print("\nSpearman(|R|, twinScore) =",
              round(comp[["abs_R", "twinScore"]].corr("spearman").iloc[0, 1], 3))
        print("\nper-pair:\n", comp[["source_g", "target_g", "R", "R_sem", "R_significant",
                                     "true_edge", "edge_sign", "twinScore", "z_het"]].round(2).to_string(index=False))
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
