#!/usr/bin/env python
"""
Twin-cell similarity sweep for BoolODE, generalized across the four
BEELINE "Curated" Boolean models (GSD, HSC, mCAD, VSC).

For a set of "parent" trajectories, simulate one full-length trunk
trajectory (t=0 -> t=simulation_time) per parent, then, for each candidate
branch timepoint (expressed as a FRACTION of the total simulation time, so
it generalizes across networks with different simulation_time), spawn TWO
independent continuations ("twins") from the exact trunk state at that
timepoint to the end of the simulation.

At each branch time, compare the twins' final expression vectors
(Euclidean distance, Pearson correlation) and compare against a "random
cells" baseline built from pairs of *different* parents' final states
(i.e. cells that never shared any trunk -- the normal BoolODE regime).

Reuses BoolODE's own model-generation code and Euler-Maruyama integrator
(BoolODE/simulator.py) directly, so results are on identical footing with
a normal BoolODE run of the same model.

Usage:
    python twin_similarity_sweep.py --network GSD --n-pairs 6000 --workers 20
    python twin_similarity_sweep.py --network HSC --n-pairs 6000 --workers 20
    python twin_similarity_sweep.py --network mCAD --n-pairs 6000 --workers 20
    python twin_similarity_sweep.py --network VSC --n-pairs 6000 --workers 20

Outputs, per network, under twins/output/<NETWORK>/:
    twin_similarity_results.csv  -- long format, one row per (pair, branch_tp,
                                     comparison) with euclidean + pearson.
    twin_final_states.csv        -- raw per-gene final-state vectors for every
                                     trunk and every twin branch (lets the
                                     analysis notebook redo any per-gene /
                                     clustering analysis without resimulating).
"""
import os
import sys
import ast
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
import multiprocessing as mp
from importlib.machinery import SourceFileLoader

SCRIPT_DIR = Path(__file__).resolve().parent
# BOOLODE_DIR = str(SCRIPT_DIR.parent)   # [2026-09-30 replaced: in the original layout the script dir sat inside the full BoolODE install (code/BoolODE/twins); here it is vendored, so point at the full install (BoolODE/ package + data/)]
BOOLODE_DIR = os.environ.get("TWINFER_BOOLODE_PATH", str(SCRIPT_DIR.parent))
# Capture the caller's cwd BEFORE chdir'ing into BOOLODE_DIR below, so any
# relative user-supplied path (--output-dir, --model-file, etc.) can be
# resolved against where the user actually ran this from, not against
# BOOLODE_DIR. Getting this backwards is a real bug we hit once already
# (OUT_DIR silently resolving one directory too high) -- same footgun
# applies to every relative CLI path argument, not just that one.
ORIGINAL_CWD = Path.cwd()
sys.path.insert(0, BOOLODE_DIR)
os.chdir(BOOLODE_DIR)

from BoolODE import utils
from BoolODE.model_generator import GenerateModel
from BoolODE import simulator

# ---------------------------------------------------------------- settings
MODEL_DIR = "data"
INTEGRATION_STEP_SIZE = 0.01

# Matches BoolODE/config-files/beeline-inputs-boolean.yaml -- the same
# jobs BEELINE's own Curated example datasets were generated from.
NETWORKS = {
    'GSD':  dict(model_definition='GSD.txt',  model_ics='GSD_ics.txt',  simulation_time=8),
    'HSC':  dict(model_definition='HSC.txt',  model_ics='HSC_ics.txt',  simulation_time=8),
    'mCAD': dict(model_definition='mCAD.txt', model_ics='mCAD_ics.txt', simulation_time=5),
    'VSC':  dict(model_definition='VSC.txt',  model_ics=None,           simulation_time=5),
}

# Branch points as fractions of total simulation time, not raw step counts,
# so the same sweep design applies regardless of a network's simulation_time.
BRANCH_FRACTIONS = [0.5]

# Save every TRAJECTORY_SAVE_STRIDE-th timepoint of the trunk and each twin's
# post-branch continuation (plus the branch point and true final step, always
# included explicitly even if not a multiple of the stride) -- not just the
# final state. All saved steps use GLOBAL step indices (0..N-1 on the shared
# tspan grid), so trunk/twin_A/twin_B rows all sit on one common time axis
# and can be plotted/concatenated directly.
TRAJECTORY_SAVE_STRIDE = 100

DEFAULT_N_PAIRS = 6000

# Each pair gets a reserved, disjoint block of seeds: 1 for the trunk + 2 per
# branch fraction (twin A, twin B). Must stay comfortably above
# 1 + 2*len(BRANCH_FRACTIONS) so no two (pair, branch, twin) triples can ever
# collide -- this was a real bug in an earlier version, where seeds were
# derived from pair_id*1000 + branch_tp + small_offset, which collided
# between different pairs once branch_tp/offset ranges overlapped the pair
# spacing. A dedicated block per pair makes collision structurally
# impossible instead of merely unlikely.
SEED_BLOCK_SIZE = 100
assert SEED_BLOCK_SIZE > 1 + 2 * len(BRANCH_FRACTIONS), \
    "SEED_BLOCK_SIZE too small for the number of branch fractions"
SEED_BASE = 1_000_000

# Stride between independent replicate runs (--seed-offset N shifts the
# entire seed space by N * REPLICATE_SEED_STRIDE). Comfortably supports up
# to 1,000,000 pairs per replicate before the next replicate's seed range
# could ever overlap -- far beyond any realistic n_pairs here. Without this,
# rerunning the script with the same --n-pairs is fully deterministic (seeds
# only depend on pair_id), so naively looping it N times for "N replicates"
# would silently produce N identical copies, not independent samples.
REPLICATE_SEED_STRIDE = 100_000_000


def detect_available_cpus():
    """Number of CPUs actually usable by this process, not just present on
    the machine. Prefers, in order:
    1. SLURM_CPUS_PER_TASK -- authoritative when submitted as a SLURM job,
       since a shared node's total core count (cpu_count()) can badly
       overstate what this job was actually allocated.
    2. os.sched_getaffinity(0) -- the CPU affinity mask the OS/scheduler
       (cgroups, taskset, etc.) has actually granted this process. More
       accurate than cpu_count() on any shared/restricted node.
    3. os.cpu_count() -- last-resort fallback (e.g. on platforms without
       sched_getaffinity, such as macOS).
    """
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if slurm_cpus:
        try:
            return max(1, int(slurm_cpus))
        except ValueError:
            pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def build_settings(network, out_dir, simulation_time_override=None, custom_spec=None):
    """custom_spec, when given, is a dict with absolute Paths for
    'model_file' (required), 'ics_file' and 'species_type_file' (either may
    be None), and a required 'simulation_time' -- used for networks that
    aren't in the built-in NETWORKS registry (e.g. ones converted from an
    edge list via edges_to_rules.py). Otherwise falls back to the registry
    lookup, same as before."""
    if custom_spec is not None:
        model_path = custom_spec['model_file']
        ics_path = custom_spec.get('ics_file') or Path(MODEL_DIR, '')
        species_type_path = custom_spec.get('species_type_file') or Path(MODEL_DIR, '')
        simulation_time = simulation_time_override if simulation_time_override is not None else custom_spec['simulation_time']
    else:
        spec = NETWORKS[network]
        ics_name = spec['model_ics'] or ''
        model_path = Path(MODEL_DIR, spec['model_definition'])
        ics_path = Path(MODEL_DIR, ics_name)
        species_type_path = Path(MODEL_DIR, '')
        simulation_time = simulation_time_override if simulation_time_override is not None else spec['simulation_time']

    return {
        'name': network,
        'outprefix': Path(out_dir),
        'modelpath': model_path,
        'simulation_time': simulation_time,
        'icsPath': ics_path,
        'num_cells': 1,
        'sample_cells': False,
        'nClusters': 1,
        'doParallel': False,
        'identical_pars': False,
        'sample_pars': False,
        'sample_std': 0.1,
        'integration_step_size': INTEGRATION_STEP_SIZE,
        'parameter_inputs_path': Path(MODEL_DIR, ''),
        'parameter_set': Path(MODEL_DIR, ''),
        'interaction_strengths': Path(MODEL_DIR, ''),
        'species_type': species_type_path,
        'burnin': False,
        'writeProtein': False,
        'normalizeTrajectory': False,
        'add_dummy': False,
        'max_parents': 1,
        'modeltype': 'hill',
    }


def build_model_and_ics(network, out_dir, simulation_time_override=None, custom_spec=None):
    """Rebuild the ODE model + shared initial condition exactly as a normal
    BoolODE run would (deterministic since sample_pars=False)."""
    settings = build_settings(network, out_dir, simulation_time_override, custom_spec)
    os.makedirs(out_dir, exist_ok=True)

    parameterInputsDF = utils.checkValidInputPath(settings['parameter_inputs_path'])
    parameterSetDF = utils.checkValidInputPath(settings['parameter_set'])
    icsDF = utils.checkValidInputPath(settings['icsPath'])
    interactionStrengthDF = utils.checkValidInputPath(settings['interaction_strengths'])
    speciesTypeDF = utils.checkValidInputPath(settings['species_type'])

    mg = GenerateModel(settings, parameterInputsDF, parameterSetDF, interactionStrengthDF, speciesTypeDF)
    model = SourceFileLoader("model", mg.path_to_ode_model.as_posix()).load_module()

    varmapper = mg.varmapper
    revvarmapper = {v: k for k, v in varmapper.items()}
    rnaIndex = [i for i in range(len(varmapper)) if 'x_' in varmapper[i]]
    proteinIndex = [i for i in range(len(varmapper)) if 'p_' in varmapper[i]]

    # Build "ss" the same way Experiment() does (BoolODE/run_experiment.py:69-96)
    ss = np.zeros(len(varmapper))
    for i, k in varmapper.items():
        if 'x_' in k:
            ss[i] = 1.0
        elif 'p_' in k:
            if k.replace('p_', '') in mg.proteinlist:
                ss[i] = 20.

    if not icsDF.empty:
        icsspec = icsDF.loc[0]
        genes = ast.literal_eval(icsspec['Genes'])
        values = ast.literal_eval(icsspec['Values'])
        icsmap = {g: v for g, v in zip(genes, values)}
        for p in mg.proteinlist:
            ss[revvarmapper['p_' + p]] = icsmap.get(p, 0.01)
        for g in mg.genelist:
            ss[revvarmapper['x_' + g]] = icsmap.get(g, 0.01)

    y0 = simulator.getInitialCondition(
        ss, mg.ModelSpec, rnaIndex, proteinIndex,
        mg.genelist, mg.proteinlist, varmapper, revvarmapper
    )

    parNames = sorted(mg.ModelSpec['pars'].keys())
    pars = [mg.ModelSpec['pars'][k] for k in parNames]
    # RATE_PREFIXES are the only parameters with literal 1/time units in BoolODE's Hill model
    # (model_generator.py: dx/dt = m_*f(regulators) - l_x_*x, dp/dt = r_*x - l_p_*p). Scaling ALL
    # FOUR by one shared per-lineage factor is exact time-rescaling (same fixed points/trajectory
    # shape, different speed) -- unlike n_/k_/sigmaH_ (Hill coefficient/threshold, dimensionless
    # shape parameters) or alpha_/a_... (boolean rule-logic flags), which must NOT be scaled.
    RATE_PREFIXES = ('m_', 'l_x_', 'r_', 'l_p_')
    rate_mask = [any(name.startswith(p) for p in RATE_PREFIXES) for name in parNames]

    simulation_time = settings['simulation_time']
    tspan = np.linspace(0, simulation_time, int(simulation_time / INTEGRATION_STEP_SIZE))
    gid = [i for i in rnaIndex]
    genenames = [varmapper[i].replace('x_', '') for i in gid]

    return model.Model, y0, pars, tspan, gid, genenames, rate_mask


# ---------------------------------------------------------- worker process
# Shared, read-only simulation inputs are set once per worker via the Pool
# initializer instead of being re-pickled on every one of n_pairs tasks --
# matters once n_pairs is in the thousands.
_W = {}


def _init_worker(Model, y0, pars, tspan, gid, branch_tps, seed_offset, stagger_range=None,
                  rate_mask=None, rate_het_range=None, harvest_range=None, post_division_gap_frac=0.1):
    _W['Model'] = Model
    _W['y0'] = y0
    _W['pars'] = pars
    _W['tspan'] = tspan
    _W['gid'] = gid
    _W['branch_tps'] = branch_tps
    _W['seed_offset'] = seed_offset
    _W['stagger_range'] = stagger_range
    _W['rate_mask'] = rate_mask
    _W['rate_het_range'] = rate_het_range
    _W['harvest_range'] = harvest_range
    _W['post_division_gap_frac'] = post_division_gap_frac


def _stride_steps(start, end, stride):
    """Sorted global step indices in [start, end], at multiples of `stride`
    (as measured from 0, so trunk/twin rows share one common grid) plus
    start and end explicitly -- guarantees the branch point and true final
    step are always present even when they don't fall on the stride."""
    steps = set(range(0, end + 1, stride))
    steps = {s for s in steps if s >= start}
    steps.add(start)
    steps.add(end)
    return sorted(steps)


def run_one_pair(pair_id):
    """Simulate one full trunk trajectory and branch it at every branch_tp,
    saving every TRAJECTORY_SAVE_STRIDE-th timepoint (not just the final
    state) for the trunk and each twin's post-branch continuation.

    If stagger_range is set (--stagger-branch-range LOW HIGH), branch_tps is
    ignored and this pair instead gets its OWN single branch point, drawn
    uniformly at random (seeded off this pair's own seed_block, so reruns
    are deterministic) from fraction range [LOW, HIGH] of the total tspan --
    every pair still ends up measured at the SAME final time N-1, but each
    pair's post-division AGE (final_time - its own branch_tp) differs. This
    mimics how a real experiment actually harvests a population: everyone
    is captured at one calendar snapshot, but individual lineages are at
    different points since their last division -- unlike the fixed global
    branch fraction, which synchronizes every pair's division time and so
    never captures cells mid-transient unless that one shared fraction
    happens to land inside the informative window for every gene pair.
    If rate_het_range is set (--rate-heterogeneity-range LOW HIGH), this pair draws ONE random
    speed multiplier (uniform on [LOW, HIGH], seeded off this pair's own seed_block so reruns are
    deterministic) and applies it to every m_/l_x_/r_/l_p_ (transcription/mRNA-degradation/
    translation/protein-degradation) parameter -- the only parameters with literal 1/time units in
    BoolODE's Hill model. Scaling all four by one shared factor is exact time-rescaling: same
    fixed points/trajectory shape, different speed. The SAME multiplier is used for the trunk
    (pre-division) and BOTH twin branches, since it represents an inherited per-lineage
    differentiation rate, not independent per-cell noise -- both siblings of a pair should share
    their parent's speed. Different pairs get independent multipliers, so measuring everyone at
    one fixed calendar harvest time samples lineages at genuinely different effective
    developmental stages (fast lineages already mature, slow lineages still mid-transient) --
    the same kind of population-level stage diversity BEELINE's own per-cell random-timepoint
    sampling (BoolODE/post_processing.py:genSamples) achieves by a different mechanism.

    If harvest_range is set (--stagger-harvest-range LOW HIGH), this pair instead draws its OWN
    random HARVEST time (not just branch time) uniformly from [LOW, HIGH] of the total tspan --
    this is the mechanism BoolODE's own genSamples() actually uses to give BEELINE's competitor
    methods real signal (np.random.choice(range(1,maxtime), size=sample_size): one random
    ABSOLUTE time per independent cell, so the pooled population spans everything from early
    transient to fully committed). Division happens a short, fixed gap (post_division_gap_frac of
    total tspan) before that pair's own harvest time, clamped to >=0 -- so every pair is still a
    genuine sibling comparison, just captured at very different points along the maturation
    process. Both twins are only simulated over [branch_tp, harvest_tp], not to the global final
    step. Overrides stagger_range/branch_tps entirely when given.
    Returns (trunk_final, metric_rows, raw_state_rows) for this pair."""
    Model, y0, pars = _W['Model'], _W['y0'], _W['pars']
    tspan, gid, branch_tps = _W['tspan'], _W['gid'], _W['branch_tps']
    seed_offset = _W['seed_offset']
    stagger_range = _W.get('stagger_range')
    rate_mask = _W.get('rate_mask')
    rate_het_range = _W.get('rate_het_range')
    harvest_range = _W.get('harvest_range')
    gap_frac = _W.get('post_division_gap_frac', 0.1)
    N = len(tspan)
    end_step = N - 1  # default: everyone measured at the shared global final step

    seed_block = SEED_BASE + seed_offset * REPLICATE_SEED_STRIDE + pair_id * SEED_BLOCK_SIZE
    trunk_seed = seed_block + 0

    if harvest_range is not None:
        lo, hi = harvest_range
        rng = np.random.RandomState(trunk_seed + 2)  # own stream, doesn't collide with stagger/rate-het draws
        harvest_frac = rng.uniform(lo, hi)
        end_step = int(round(harvest_frac * N))
        gap_steps = int(round(gap_frac * N))
        branch_tps = [max(0, end_step - gap_steps)]
    elif stagger_range is not None:
        lo, hi = stagger_range
        rng = np.random.RandomState(trunk_seed)  # deterministic per (pair_id, seed_offset)
        frac = rng.uniform(lo, hi)
        branch_tps = [int(round(frac * N))]

    if rate_het_range is not None:
        lo, hi = rate_het_range
        # Separate rng stream (offset +1) from the branch-stagger draw above, so enabling both
        # features at once doesn't correlate the two random draws through a shared seed.
        rng = np.random.RandomState(trunk_seed + 1)
        speed = rng.uniform(lo, hi)
        pars = [p * speed if m else p for p, m in zip(pars, rate_mask)]

    Y_trunk = simulator.eulersde(Model, simulator.noise, y0, tspan, pars, seed=trunk_seed)
    # trunk_final always at the GLOBAL final step (N-1), not this pair's own end_step -- it only
    # feeds the "random unrelated cells" baseline, which should stay on one common footing across
    # pairs regardless of any per-pair harvest staggering.
    trunk_final = Y_trunk[N - 1, gid]  # valid final row (see simulator.eulersde row semantics)

    metric_rows = []
    state_rows = []
    for step in _stride_steps(0, N - 1, TRAJECTORY_SAVE_STRIDE):
        vec = Y_trunk[step, gid]
        state_rows.append({'pair_id': pair_id, 'branch_tp': -1, 'source': 'trunk',
                           'step': step, 't': float(tspan[step]),
                           **{f'gene_{i}': v for i, v in enumerate(vec)}})

    for branch_idx, b in enumerate(branch_tps):
        y_branch = Y_trunk[b, :]
        # NOTE: eulersde hardcodes currtime starting at 0 but uses
        # tspan[-1] as the absolute maxtime target -- passing it an
        # absolute-time slice like tspan[b:] (which starts near t=b*h,
        # not 0) desyncs its internal step count from the pre-allocated
        # noise array and throws an IndexError. The model is fully
        # time-autonomous (no explicit t-dependence in Model/noise), so
        # a relative time axis starting at 0 is an exact substitute.
        # Runs to end_step (this pair's OWN harvest point when --stagger-harvest-range is given,
        # else the shared global final step N-1).
        L = end_step - b + 1
        remaining_tspan = np.linspace(0, tspan[end_step] - tspan[b], L)
        # Global steps b..end_step that should be saved for this branch, at
        # multiples of the stride (same grid as the trunk) plus b and end_step
        # explicitly. Converted to a LOCAL index into Y_branch (0..L-1) by
        # subtracting b, since Y_branch's own clock restarts at 0 (see the
        # relative-time note above).
        global_save_steps = _stride_steps(b, end_step, TRAJECTORY_SAVE_STRIDE)

        twin_states = {}
        for twin_offset, twin_label in enumerate(('A', 'B')):
            seed = seed_block + 1 + branch_idx * 2 + twin_offset
            Y_branch = simulator.eulersde(Model, simulator.noise, y_branch, remaining_tspan, pars, seed=seed)
            final_state = Y_branch[L - 1, gid]
            twin_states[twin_label] = final_state
            for global_step in global_save_steps:
                local_step = global_step - b
                vec = Y_branch[local_step, gid]
                state_rows.append({'pair_id': pair_id, 'branch_tp': b, 'source': f'twin_{twin_label}',
                                   'step': global_step, 't': float(tspan[global_step]),
                                   **{f'gene_{i}': v for i, v in enumerate(vec)}})

        vecA, vecB = twin_states['A'], twin_states['B']
        euclid = float(np.linalg.norm(vecA - vecB))
        try:
            pear = float(pearsonr(vecA, vecB)[0])
        except Exception:
            pear = np.nan
        metric_rows.append({'pair_id': pair_id, 'branch_tp': b, 'comparison': 'twin',
                            'euclidean': euclid, 'pearson': pear})

    return trunk_final, metric_rows, state_rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--network', required=True,
                        help='Which network to run. Either one of the built-in Curated '
                             f'models ({", ".join(sorted(NETWORKS.keys()))}), or an '
                             'arbitrary label for a custom network -- in the latter case '
                             '--model-file and --simulation-time are required, since a '
                             'custom network has no registry entry to fall back on. '
                             'Used as-is for the output subdirectory name either way.')
    parser.add_argument('--model-file', type=str, default=None,
                        help='Path to a custom Gene/Rule Boolean model file (e.g. from '
                             'edges_to_rules.py), for networks not in the built-in '
                             'NETWORKS registry. Presence of this flag is what selects '
                             'the custom-network path; --network becomes just a label.')
    parser.add_argument('--ics-file', type=str, default=None,
                        help='Optional Genes/Values initial-condition file for a custom '
                             'network (only meaningful together with --model-file).')
    parser.add_argument('--species-type-file', type=str, default=None,
                        help='Optional Node/Type file marking nodes gene vs protein for '
                             'a custom network (only meaningful together with --model-file).')
    parser.add_argument('--n-pairs', type=int, default=DEFAULT_N_PAIRS,
                        help=f'Number of replicate parent trunks (default {DEFAULT_N_PAIRS})')
    parser.add_argument('--workers', type=int, default=None,
                        help='Override the auto-detected worker count (default: auto)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Base output directory (default: twins/output/<NETWORK>, '
                             'relative to this script\'s own location). If given, results '
                             'are written to <output-dir>/<NETWORK>/.')
    parser.add_argument('--simulation-time', type=float, default=None,
                        help='Override this network\'s default simulation_time '
                             '(e.g. to test whether it has reached a stable attractor '
                             'yet). Branch points stay at the same FRACTIONS of the '
                             'total time, so they rescale automatically. REQUIRED when '
                             '--model-file is given (custom networks have no default).')
    parser.add_argument('--stagger-branch-range', type=float, nargs=2, default=None,
                        metavar=('LOW_FRAC', 'HIGH_FRAC'),
                        help='Instead of every pair dividing at the same global BRANCH_FRACTIONS '
                             'point, give each pair its OWN division time drawn independently '
                             'and uniformly at random from [LOW_FRAC, HIGH_FRAC] of the total '
                             'simulation time. Every pair is still measured at the same final '
                             'time -- only how long ago each pair divided differs -- so the '
                             'population is naturally staggered across the transient the way a '
                             'real (division-asynchronous) experimental harvest would be, '
                             'instead of everyone being sampled at one synchronized stage. '
                             'Overrides BRANCH_FRACTIONS entirely when given.')
    parser.add_argument('--stagger-harvest-range', type=float, nargs=2, default=None,
                        metavar=('LOW_FRAC', 'HIGH_FRAC'),
                        help='Give each pair its own random HARVEST time (not just branch time), '
                             'drawn uniformly from [LOW_FRAC, HIGH_FRAC] of total simulation time '
                             '-- this is the actual mechanism BoolODE\'s own genSamples() uses to '
                             'give BEELINE competitor methods real signal (one random absolute '
                             'time per independent cell, pooled population spans early-transient '
                             'to fully-committed). Division happens --post-division-gap-frac '
                             'before that harvest time (clamped to >=0). Overrides '
                             '--stagger-branch-range entirely when given.')
    parser.add_argument('--post-division-gap-frac', type=float, default=0.1,
                        help='Fraction of total simulation time between a pair\'s division and its '
                             'own harvest time, only used with --stagger-harvest-range (default 0.1).')
    parser.add_argument('--rate-heterogeneity-range', type=float, nargs=2, default=None,
                        metavar=('LOW_MULT', 'HIGH_MULT'),
                        help='Give each PAIR (shared by both twins, not independent per cell) its '
                             'own differentiation-speed multiplier, drawn uniformly at random from '
                             '[LOW_MULT, HIGH_MULT], applied to every m_/l_x_/r_/l_p_ rate '
                             'parameter (exact time-rescaling -- same fixed points, different '
                             'speed). Lets one fixed calendar harvest time sample lineages at '
                             'genuinely different developmental stages, instead of everyone '
                             'having identical kinetics.')
    parser.add_argument('--seed-offset', type=int, default=0,
                        help='Shifts the entire seed space by seed_offset * %d. '
                             'REQUIRED to differ between independent replicate runs '
                             'of the same --network/--n-pairs -- seeds otherwise '
                             'depend only on pair_id, so rerunning with the same '
                             'settings and --seed-offset 0 reproduces the exact '
                             'same simulations, not a fresh replicate.' % REPLICATE_SEED_STRIDE)
    args = parser.parse_args()

    if args.model_file is None and args.network not in NETWORKS:
        parser.error(f"--network {args.network!r} is not a built-in network "
                     f"({', '.join(sorted(NETWORKS.keys()))}) -- pass --model-file "
                     f"to run a custom network under that label.")
    if args.model_file is not None and args.simulation_time is None:
        parser.error("--simulation-time is required when --model-file is given "
                     "(custom networks have no default simulation_time to fall back on).")
    return args


def main():
    args = parse_args()
    network = args.network
    n_pairs = args.n_pairs

    custom_spec = None
    if args.model_file is not None:
        # Resolve against ORIGINAL_CWD, not the script's own post-chdir cwd
        # -- same reasoning as --output-dir below.
        custom_spec = {
            'model_file': Path(ORIGINAL_CWD, args.model_file).resolve(),
            'ics_file': Path(ORIGINAL_CWD, args.ics_file).resolve() if args.ics_file else None,
            'species_type_file': Path(ORIGINAL_CWD, args.species_type_file).resolve() if args.species_type_file else None,
            'simulation_time': args.simulation_time,
        }

    # Always resolve to an absolute path -- SCRIPT_DIR is already absolute
    # (Path(__file__).resolve()), and resolving a user-supplied --output-dir
    # against ORIGINAL_CWD (captured before this script's own os.chdir into
    # BOOLODE_DIR) makes it absolute the way the user actually meant, not
    # relative to BOOLODE_DIR. Downstream code (workers, GenerateModel's
    # file writes, the analysis notebook) all rely on this being
    # unambiguous regardless of the working directory the script happened
    # to be launched from.
    if args.output_dir is not None:
        out_dir = str(Path(ORIGINAL_CWD, args.output_dir).resolve() / network)
    else:
        out_dir = str(SCRIPT_DIR / "output" / network)
    results_csv = os.path.join(out_dir, "twin_similarity_results.csv")
    states_csv = os.path.join(out_dir, "twin_final_states.csv")

    if args.workers is not None:
        n_workers = max(1, args.workers)
    else:
        n_workers = detect_available_cpus()
    n_workers = min(n_workers, n_pairs)

    print(f"Output directory (absolute): {out_dir}")
    if custom_spec is not None:
        print(f"Custom network: model_file={custom_spec['model_file']}, "
              f"ics_file={custom_spec['ics_file']}, "
              f"species_type_file={custom_spec['species_type_file']}, "
              f"simulation_time={custom_spec['simulation_time']}")
    elif args.simulation_time is not None:
        print(f"simulation_time overridden: {NETWORKS[network]['simulation_time']} -> {args.simulation_time}")
    print(f"Building {network} model and shared initial condition...")
    Model, y0, pars, tspan, gid, genenames, rate_mask = build_model_and_ics(
        network, out_dir, args.simulation_time, custom_spec)
    N = len(tspan)
    branch_tps = [int(round(f * N)) for f in BRANCH_FRACTIONS]
    print(f"Model has {len(genenames)} genes: {genenames}")
    print(f"tspan has {N} timepoints (t=0..{tspan[-1]})")
    if args.stagger_branch_range is not None:
        lo, hi = args.stagger_branch_range
        print(f"Staggered branch times: each pair divides independently at a random fraction "
              f"in [{lo}, {hi}] of total time (steps [{int(round(lo*N))}, {int(round(hi*N))}]), "
              f"all measured at the common final step {N - 1}")
    else:
        print(f"Branch timepoints: {branch_tps} (fractions {BRANCH_FRACTIONS})")
    print(f"Running {n_pairs} parent trunks on {n_workers} workers "
          f"(auto-detected from {'SLURM_CPUS_PER_TASK' if os.environ.get('SLURM_CPUS_PER_TASK') else 'sched_getaffinity'})...")
    if args.seed_offset:
        print(f"seed_offset={args.seed_offset} (independent replicate, not overlapping seed_offset=0's seeds)")
    if args.rate_heterogeneity_range is not None:
        lo, hi = args.rate_heterogeneity_range
        print(f"Rate heterogeneity: each pair gets its own speed multiplier in [{lo}, {hi}] "
              f"(shared by both twins), applied to m_/l_x_/r_/l_p_ parameters")
    if args.stagger_harvest_range is not None:
        lo, hi = args.stagger_harvest_range
        print(f"Staggered HARVEST times: each pair measured at its own random time in [{lo}, {hi}] "
              f"of total time, dividing {args.post_division_gap_frac} of total time beforehand "
              f"(clamped to >=0)")

    with mp.Pool(n_workers, initializer=_init_worker,
                initargs=(Model, y0, pars, tspan, gid, branch_tps, args.seed_offset,
                          tuple(args.stagger_branch_range) if args.stagger_branch_range else None,
                          rate_mask,
                          tuple(args.rate_heterogeneity_range) if args.rate_heterogeneity_range else None,
                          tuple(args.stagger_harvest_range) if args.stagger_harvest_range else None,
                          args.post_division_gap_frac)) as pool:
        results = pool.map(run_one_pair, range(n_pairs))

    trunk_finals = [r[0] for r in results]
    all_metric_rows = []
    all_state_rows = []
    for _, metric_rows, state_rows in results:
        all_metric_rows.extend(metric_rows)
        all_state_rows.extend(state_rows)

    # "Random cells" baseline: pair up final states of DIFFERENT, unrelated
    # parents (they only ever shared y0, never any trunk noise) -- this is
    # exactly the relationship between any two ordinary cells in a normal
    # BoolODE run.
    for i in range(n_pairs):
        j = (i + 1) % n_pairs
        vecA, vecB = trunk_finals[i], trunk_finals[j]
        euclid = float(np.linalg.norm(vecA - vecB))
        try:
            pear = float(pearsonr(vecA, vecB)[0])
        except Exception:
            pear = np.nan
        all_metric_rows.append({'pair_id': i, 'branch_tp': -1, 'comparison': 'random',
                                'euclidean': euclid, 'pearson': pear})

    os.makedirs(out_dir, exist_ok=True)

    metrics_df = pd.DataFrame(all_metric_rows)
    metrics_df.to_csv(results_csv, index=False)
    print(f"Wrote {len(metrics_df)} rows to {results_csv}")

    states_df = pd.DataFrame(all_state_rows)
    gene_cols = [c for c in states_df.columns if c.startswith('gene_')]
    states_df = states_df.rename(columns={f'gene_{i}': name for i, name in enumerate(genenames)})
    states_df.to_csv(states_csv, index=False)
    print(f"Wrote {len(states_df)} rows to {states_csv} ({len(genenames)} gene columns)")

    summary = metrics_df.groupby(['comparison', 'branch_tp'])[['euclidean', 'pearson']].agg(['mean', 'std'])
    print(summary)


if __name__ == '__main__':
    main()
