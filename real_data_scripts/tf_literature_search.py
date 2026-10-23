#!/usr/bin/env python3
"""Systematic literature check of a TF's candidate targets: PubMed (title/abstract) query  (TF OR TF-name) AND (gene OR aliases)  for every gene,
abstracts fetched, sentences that mention both the TF and the gene extracted and tagged with regulatory verbs; Europe PMC full-text co-mention count as an extra column.
Aliases from NCBI Gene via MyGene.info (official symbol + up to 6 aliases of >= 3 characters; ambiguous short aliases dropped).
usage: tf_literature_search.py <TF> <name1,name2> <genes.txt or comma list> <out_dir>
e.g.   tf_literature_search.py EOMES eomesodermin genes.txt literature/"""
import sys, re, time, json, os, urllib.parse, urllib.request, xml.etree.ElementTree as ET
import pandas as pd
TF, TFNAMES, GENES, OUT = sys.argv[1], sys.argv[2].split(","), sys.argv[3], sys.argv[4]
genes = [g.strip() for g in (open(GENES).read().split() if os.path.exists(GENES) else GENES.split(",")) if g.strip()]
os.makedirs(OUT, exist_ok=True)
def get(url, tries=4):
    for t in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "twinfer-lit-check"}), timeout=60) as r: return r.read().decode("utf-8", "replace")
        except Exception as e:
            time.sleep(1.5 * (t + 1))
    return ""
def aliases(g):
    j = json.loads(get("https://mygene.info/v3/query?q=symbol:%s&species=human&fields=symbol,alias,name&size=1" % urllib.parse.quote(g)) or "{}")
    h = (j.get("hits") or [{}])[0]; a = h.get("alias", []); a = [a] if isinstance(a, str) else a
    return [x for x in a if len(x) >= 3 and re.match(r"^[A-Za-z0-9\-]+$", x)][:6], h.get("name", "")
tfq = " OR ".join(f'"{n}"[tiab]' for n in [TF] + TFNAMES)
rows, pm = [], {}
for i, g in enumerate(genes):
    al, name = aliases(g); names = [g] + al; gq = " OR ".join(f'"{n}"[tiab]' for n in names)
    q = f"({tfq}) AND ({gq})"
    j = json.loads(get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmode=json&retmax=200&term=" + urllib.parse.quote(q)) or "{}")
    ids = j.get("esearchresult", {}).get("idlist", []); n = int(j.get("esearchresult", {}).get("count", 0))
    ep = json.loads(get("https://www.ebi.ac.uk/europepmc/webservices/rest/search?format=json&pageSize=1&query=" + urllib.parse.quote(f'{TF} AND "{g}"')) or "{}").get("hitCount", "")
    rows.append(dict(gene=g, gene_name=name, aliases=";".join(al), pubmed_hits=n, europepmc_fulltext_cooccur=ep, pmids=";".join(ids)))
    pm[g] = (ids, names); print(f"{i+1}/{len(genes)} {g}: PubMed {n} hits (aliases {al}) | EuropePMC {ep}", flush=True); time.sleep(0.35)
pd.DataFrame(rows).to_csv(f"{OUT}/{TF}_pubmed_hits_per_gene.csv", index=False)
allids = sorted({p for ids, _ in pm.values() for p in ids}); recs = {}
for k in range(0, len(allids), 150):
    x = get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&retmode=xml&id=" + ",".join(allids[k:k + 150])); time.sleep(0.4)
    try: root = ET.fromstring(x)
    except Exception: continue
    for a in root.iter("PubmedArticle"):
        pid = a.findtext(".//PMID"); title = "".join(a.find(".//ArticleTitle").itertext()) if a.find(".//ArticleTitle") is not None else ""
        ab = " ".join("".join(t.itertext()) for t in a.findall(".//AbstractText")); yr = a.findtext(".//PubDate/Year") or (a.findtext(".//PubDate/MedlineDate") or "")[:4]
        recs[pid] = (title, ab, yr)
verbs = re.compile(r"\b(bind|bound|binds|target|regulat|activat|induc|repress|upstream|downstream|direct|chip|knockdown|knockout|deficien|overexpress|loss of|promot|enhancer|transactivat|drives|control)", re.I)
sent_rows = []
for g, (ids, names) in pm.items():
    gre = re.compile(r"(?<![A-Za-z0-9])(" + "|".join(re.escape(n) for n in names) + r")(?![A-Za-z0-9])", re.I); tre = re.compile(r"(eomes|eomesodermin)", re.I)
    for pid in ids:
        if pid not in recs: continue
        title, ab, yr = recs[pid]
        for s in re.split(r"(?<=[.!?])\s+", title + ". " + ab):
            if tre.search(s) and gre.search(s): sent_rows.append(dict(gene=g, pmid=pid, year=yr, title=title[:140], sentence=s.strip()[:420], regulatory_verb=bool(verbs.search(s))))
S = pd.DataFrame(sent_rows); S.to_csv(f"{OUT}/{TF}_cooccurrence_sentences.csv", index=False)
print(f"\n{len(genes)} genes | genes with >=1 PubMed hit: {(pd.DataFrame(rows).pubmed_hits>0).sum()} | papers fetched {len(recs)} | co-occurrence sentences {len(S)} in {S.gene.nunique() if len(S) else 0} genes")
