#!/usr/bin/env python
"""
Convert a signed edge list (Source/Target/Type or Gene1/Gene2/Type, comma-
or whitespace-separated) into a BoolODE Boolean rule file, using the
standard default convention when no finer regulatory logic is known:

    target = (activator_1 or activator_2 or ...) and not (repressor_1 or repressor_2 or ...)

This is the same convention BoolODE's own bundled example uses
(data/randNet.txt -> data/randBool.txt). It assumes independence between
regulators -- any known AND-requirement between specific regulators (like
GSD's "NR5A1 and WT1mKTS and CBX2") has to be hand-edited into the output
afterward, since that information isn't recoverable from a signed edge list.

Also writes a species_type.txt file (Node/Type, tab-separated) marking any
node matching --protein-pattern as 'protein' (single-variable, no
transcription/translation cascade -- see BoolODE/model_generator.py's
readBooleanRules()) and everything else as 'gene'. Every node must be
listed for BoolODE to build correctly -- listing only a subset means
unlisted nodes get no variable spec at all, not a default.

Usage:
    python edges_to_rules.py network.csv --protein-pattern '^miR' \
        --rules-out network_rules.txt --species-out network_species.txt
"""
import re
import argparse
import pandas as pd
from collections import defaultdict


def sanitize_name(name):
    """BoolODE evaluates Boolean rules by exec()'ing them as literal Python
    code (model_generator.py: `exec('booleval = ' + row['Rule'], ...)` and
    similar), so every gene name becomes a Python identifier. Names with
    characters like '-' break this silently in a confusing way -- e.g.
    'OCT4-FOXD3 = 0' parses as the *subtraction* 'OCT4 - FOXD3', not an
    assignment, raising 'SyntaxError: can't assign to operator' deep inside
    BoolODE with no indication the actual problem is the gene name (hit
    this for real with Pluripotency52N's complex names like 'OCT4-FOXD3',
    'Mad-Max', 'MYC-SP1'). Replace anything that isn't [A-Za-z0-9_] with
    '_', and prefix with '_' if that would leave a leading digit (also an
    invalid Python identifier start)."""
    sanitized = re.sub(r'[^A-Za-z0-9_]', '_', name)
    if sanitized[:1].isdigit():
        sanitized = '_' + sanitized
    return sanitized


def sniff_and_read(path):
    with open(path) as f:
        first_line = f.readline()
    sep = ',' if ',' in first_line else r'\s+'
    df = pd.read_csv(path, sep=sep, engine='python')
    cols = {c.lower(): c for c in df.columns}
    src_col = cols.get('gene1') or cols.get('source')
    tgt_col = cols.get('gene2') or cols.get('target')
    type_col = cols.get('type')
    if not (src_col and tgt_col and type_col):
        raise ValueError(f"Couldn't find Source/Target/Type-like columns in {list(df.columns)}")
    df = df.rename(columns={src_col: 'Source', tgt_col: 'Target', type_col: 'Type'})
    df = df[['Source', 'Target', 'Type']].copy()

    renamed = {}
    for col in ('Source', 'Target'):
        df[col] = df[col].astype(str).str.strip()
        for original in df[col].unique():
            safe = sanitize_name(original)
            if safe != original:
                renamed[original] = safe
        df[col] = df[col].map(lambda n: renamed.get(n, n))

    if renamed:
        print(f"Sanitized {len(renamed)} node name(s) to valid Python identifiers "
              f"(BoolODE exec()'s rules as literal Python code):")
        for original, safe in sorted(renamed.items()):
            print(f"  {original!r} -> {safe!r}")

    return df


def build_rules(df):
    activators = defaultdict(list)
    repressors = defaultdict(list)
    targets_order = []
    seen_targets = set()
    sources = set()

    for _, row in df.iterrows():
        src = str(row['Source']).strip()
        tgt = str(row['Target']).strip()
        sign = str(row['Type']).strip()
        sources.add(src)
        if tgt not in seen_targets:
            seen_targets.add(tgt)
            targets_order.append(tgt)
        if sign == '+':
            if src not in activators[tgt]:
                activators[tgt].append(src)
        elif sign == '-':
            if src not in repressors[tgt]:
                repressors[tgt].append(src)
        else:
            raise ValueError(f"Unrecognized edge type {sign!r} for {src}->{tgt}")

    all_nodes = sources | seen_targets
    source_only = sources - seen_targets  # never a target -- BoolODE will
                                          # auto-add a self-activation rule
                                          # for these (readBooleanRules()),
                                          # which may or may not be what you
                                          # actually intend.
    return activators, repressors, targets_order, all_nodes, source_only


def write_rule_file(activators, repressors, targets_order, source_only, path):
    """source_only nodes (never a target, only ever a regulator of something
    else) get an explicit self-activation row written directly here, rather
    than relying on BoolODE's own auto-fill for unrresolved regulators
    (model_generator.py readBooleanRules(): a `for n in self.withoutRules:
    ... self.withoutRules.remove(n)` loop that mutates the list it's
    iterating over, which silently skips a node whenever there are 2+ such
    nodes -- a real bug in BoolODE, reproduced and confirmed while building
    this converter's output for a network with 3 unresolved miRNA nodes).
    Writing the rows explicitly here sidesteps that path entirely."""
    with open(path, 'w') as out:
        out.write('Gene\tRule\n')
        for tgt in targets_order:
            act = activators.get(tgt, [])
            rep = repressors.get(tgt, [])
            act_clause = '(' + ' or '.join(act) + ')' if act else ''
            rep_clause = 'not (' + ' or '.join(rep) + ')' if rep else ''
            if act_clause and rep_clause:
                rule = f'{act_clause} and {rep_clause}'
            elif act_clause:
                rule = act_clause
            elif rep_clause:
                rule = rep_clause
            else:
                rule = tgt  # shouldn't happen -- every target has >=1 edge
            out.write(f'{tgt}\t{rule}\n')
        for node in sorted(source_only):
            out.write(f'{node}\t{node}\n')


def write_species_type_file(all_nodes, protein_pattern, path):
    regex = re.compile(protein_pattern) if protein_pattern else None
    with open(path, 'w') as out:
        out.write('Node\tType\n')
        for node in sorted(all_nodes):
            t = 'protein' if (regex and regex.search(node)) else 'gene'
            out.write(f'{node}\t{t}\n')


def parse_merge_spec(spec):
    """Parse 'NEW_NAME:OLD1,OLD2,OLD3' into (new_name, {OLD1, OLD2, OLD3})."""
    new_name, _, old_names = spec.partition(':')
    if not old_names:
        raise ValueError(f"--merge spec {spec!r} must be 'NEW_NAME:OLD1,OLD2,...'")
    return new_name.strip(), {n.strip() for n in old_names.split(',') if n.strip()}


def apply_merges(df, merge_specs):
    """Rename every occurrence (as Source or Target) of each group of old
    node names to one new representative name, in both the edge list and
    the caller-visible node identity. Regulator lists are naturally
    deduplicated downstream (build_rules() already skips repeated
    src->tgt entries), so e.g. 5 near-redundant regulators that all get
    merged into one collapse to a single edge automatically -- no special
    handling needed here beyond the renaming itself.

    Use case: a set of regulators that are functionally redundant/
    co-regulated (e.g. the miR-200 family in EMT circuits) can be
    represented as one node instead of many, which also reduces the
    per-gene regulator count driving BoolODE's 2^regulators combinatorial
    model-generation cost -- this is the same simplification BoolODE's own
    bundled data/EMT.txt example makes with a single 'miR200' node.
    """
    rename = {}
    for spec in merge_specs:
        new_name, old_names = parse_merge_spec(spec)
        for old in old_names:
            rename[old] = new_name
    if not rename:
        return df
    df = df.copy()
    df['Source'] = df['Source'].map(lambda n: rename.get(n, n))
    df['Target'] = df['Target'].map(lambda n: rename.get(n, n))
    print(f"Merged {len(rename)} node(s) into {len(set(rename.values()))} "
          f"representative node(s):")
    for spec in merge_specs:
        new_name, old_names = parse_merge_spec(spec)
        print(f"  {sorted(old_names)} -> {new_name!r}")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('edge_csv', help='Path to the signed edge list')
    parser.add_argument('--rules-out', required=True, help='Output path for the Gene/Rule file')
    parser.add_argument('--species-out', default=None,
                        help='Output path for the Node/Type species_type file '
                             '(omit to skip -- everything defaults to gene type)')
    parser.add_argument('--protein-pattern', default=None,
                        help="Regex; any node matching this is marked 'protein' "
                             "in the species_type file (e.g. '^miR' for miRNAs)")
    parser.add_argument('--merge', action='append', default=[],
                        help="Collapse a group of functionally-redundant nodes into one "
                             "representative node, format 'NEW_NAME:OLD1,OLD2,OLD3'. "
                             "Repeatable for multiple independent groups. Reduces both "
                             "the target's regulator count and BoolODE's 2^regulators "
                             "model-generation cost.")
    args = parser.parse_args()

    df = sniff_and_read(args.edge_csv)
    df = apply_merges(df, args.merge)
    activators, repressors, targets_order, all_nodes, source_only = build_rules(df)

    write_rule_file(activators, repressors, targets_order, source_only, args.rules_out)
    print(f"Wrote {len(targets_order)} regulated genes + {len(source_only)} "
          f"self-activation placeholder(s) to {args.rules_out}")

    if args.species_out:
        write_species_type_file(all_nodes, args.protein_pattern, args.species_out)
        n_protein = sum(1 for n in all_nodes if args.protein_pattern and re.search(args.protein_pattern, n))
        print(f"Wrote species types for {len(all_nodes)} nodes to {args.species_out} "
              f"({n_protein} marked 'protein')")

    print(f"\nTotal nodes: {len(all_nodes)}.")
    if source_only:
        print(f"\n{len(source_only)} node(s) are ONLY ever a source, never a target "
              f"in the edge list -- given an explicit self-activation rule "
              f"(Gene = Gene) directly in the output file, since they have no "
              f"real regulatory logic in your data. Verify this is intended, "
              f"not a sign the edge list is incomplete:")
        for n in sorted(source_only):
            print(f"  - {n}")


if __name__ == '__main__':
    main()
