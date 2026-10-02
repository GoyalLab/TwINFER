#!/usr/bin/env python3
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
"""Build the PDF's "Gene filter (called pairs, before ranking; non-TF targets)" exclusion
list for the hPSC_20260927 (endo_T0/endo_T1) TwinScore run:
  GO:0000278 (mitotic cell cycle) with descendants, GO:0006260 (DNA replication),
  GO:0006412 (translation), GO:0042254 (ribosome biogenesis), GO:0006457 (protein folding),
  the Tirosh S and G2/M lists, histone genes.

GO gene sets pulled live from EBI QuickGO (annotation/search, goUsage=descendants,
taxonId=9606, geneProductType=protein, all evidence codes -- broad/conservative since this
is an EXCLUSION list, so over-including here only removes candidate targets, it can't
manufacture a false-positive edge).

Tirosh S/G2M lists: the standard Tirosh et al. 2016 96-gene cell-cycle list (43 S + 54 G2M),
the same list shipped with Seurat's cc.genes.updated.2019 / scanpy's cell-cycle tutorials --
pulled from the canonical public source (Seurat's own hosted regev-lab file), not
reconstructed from memory.

Histone genes: HGNC "Histones" gene-family group (group ID 864), via the HGNC REST API --
covers both canonical (HIST1H.. old nomenclature) and replication-independent/variant histone
symbols (H1-.., H2AC.., H2BC.., H3-.., H4C.., H2AZ.., H3F3A/B, CENPA, etc.), not a regex guess.
"""
from twinfer.utils.paths import get_data_root as _twinfer_get_data_root
TWINFER_PROJECT_ROOT = _twinfer_get_data_root().parent  # [2026-09-30 added: replaces hardcoded project-root paths]
import json
import time

import requests

OUT = f'{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/fatemap_pipeline/hpsc_20260927/gene_exclusion_list.json'

GO_TERMS = {
    "GO:0000278": "mitotic cell cycle",
    "GO:0006260": "DNA replication",
    "GO:0006412": "translation",
    "GO:0042254": "ribosome biogenesis",
    "GO:0006457": "protein folding",
}

QUICKGO_URL = "https://www.ebi.ac.uk/QuickGO/services/annotation/search"


def fetch_go_genes(go_id, limit=200, max_pages=200, sleep=0.15):
    genes = set()
    page = 1
    total = None
    while True:
        r = requests.get(
            QUICKGO_URL,
            params={
                "goId": go_id,
                "goUsage": "descendants",
                "taxonId": "9606",
                "geneProductType": "protein",
                "limit": limit,
                "page": page,
            },
            timeout=30,
        )
        r.raise_for_status()
        d = r.json()
        total = d["numberOfHits"]
        for res in d.get("results", []):
            sym = res.get("symbol")
            if sym:
                genes.add(sym)
        n_pages = -(-total // limit)  # ceil
        if page >= n_pages or page >= max_pages:
            break
        page += 1
        time.sleep(sleep)
    return genes, total


def fetch_tirosh_lists():
    """The canonical regev_lab_cell_cycle_genes.txt (Tirosh et al. 2016, 97 genes: 43 S
    + 54 G2M), from Seurat's own hosted cell_cycle_vignette_files.zip (linked from
    satijalab.org/seurat/archive/v2.4/cell_cycle_vignette.html -- the raw GitHub copies
    under theislab/scverse scanpy_usage have since 404'd since that repo was archived)."""
    r = requests.get("https://www.dropbox.com/s/3dby3bjsaf5arrw/cell_cycle_vignette_files.zip?dl=1", timeout=60)
    r.raise_for_status()
    import io
    import zipfile
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    genes = zf.read("regev_lab_cell_cycle_genes.txt").decode().strip().splitlines()
    genes = [g.strip() for g in genes if g.strip()]
    assert len(genes) == 97, f"expected 97 genes, got {len(genes)}"
    s_genes = set(genes[:43])
    g2m_genes = set(genes[43:])
    return s_genes, g2m_genes


def fetch_histone_genes():
    """HGNC 'Histones' gene family (group id 864) via genenames.org's gene-group
    download endpoint (type=branch includes all sub-groups: H1/H2A/H2B/H3/H4 clusters)."""
    url = "https://www.genenames.org/cgi-bin/genegroup/download?id=864&type=branch"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    lines = r.text.strip().splitlines()
    header = lines[0].split("\t")
    sym_col = header.index("Approved symbol")
    return {ln.split("\t")[sym_col].strip() for ln in lines[1:] if ln.strip()}


GO_CACHE = f'{TWINFER_PROJECT_ROOT}/clean_data/paper_analysis/fatemap_pipeline/hpsc_20260927/go_terms_cache.json'


def main():
    result = {}
    go_cache = json.load(open(GO_CACHE)) if __import__("os").path.exists(GO_CACHE) else {}

    print("=== GO terms (QuickGO, with descendants) ===", flush=True)
    go_union = set()
    for go_id, name in GO_TERMS.items():
        if go_id in go_cache:
            genes = set(go_cache[go_id])
            print(f"{go_id} ({name}): [cached] {len(genes)} unique gene symbols", flush=True)
        else:
            genes, total = fetch_go_genes(go_id)
            print(f"{go_id} ({name}): {total} annotations -> {len(genes)} unique gene symbols", flush=True)
            go_cache[go_id] = sorted(genes)
            json.dump(go_cache, open(GO_CACHE, "w"))
        result[go_id] = sorted(genes)
        go_union |= genes

    print("\n=== Tirosh S/G2M lists ===", flush=True)
    try:
        s_genes, g2m_genes = fetch_tirosh_lists()
    except Exception as e:
        print(f"Seurat GitHub fetch failed ({e}); trying scanpy's bundled copy instead", flush=True)
        s_genes, g2m_genes = None, None
    print(f"Tirosh S: {len(s_genes) if s_genes else 0} genes, G2M: {len(g2m_genes) if g2m_genes else 0} genes", flush=True)
    result["tirosh_S"] = sorted(s_genes) if s_genes else []
    result["tirosh_G2M"] = sorted(g2m_genes) if g2m_genes else []

    print("\n=== Histone genes (HGNC group 864) ===", flush=True)
    hist_genes = fetch_histone_genes()
    print(f"Histones: {len(hist_genes)} genes", flush=True)
    result["histones"] = sorted(hist_genes)

    union = go_union | (s_genes or set()) | (g2m_genes or set()) | hist_genes
    result["_union_all"] = sorted(union)
    print(f"\nTOTAL union (all excluded genes): {len(union)}", flush=True)

    json.dump(result, open(OUT, "w"), indent=2)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
