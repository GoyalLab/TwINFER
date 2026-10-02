# [copied 2026-09-30 from /gpfs/projects/b1255/yscher/Transcriptomic Distance/exports/_probe_tmp/tf_wide/machinery.py (author/owner: yscher/laj2116), md5 bba9dd359c9d2ae02ddc0d4571082ad2; unmodified below this header]
"""Curated cellular-machinery gene families (removed from sources and targets, both datasets), by gene-symbol family (case-insensitive, mouse and human):
mitochondrial (mt-, OXPHOS complexes, mito ribosome, import, mtDNA), cytosolic ribosome and ribosome biogenesis / translation, chaperones,
proteasome / ubiquitin, spindle / kinetochore / mitosis, DNA replication / repair / histones, splicing and RNA processing, general transcription
machinery (Pol II, TFIID/TFIIH, Mediator), nuclear transport, core glycolysis."""
import re
PATTERNS = [
 r"^mt-", r"^nduf[abcsv]", r"^cox[4-8]", r"^cox1[0-9]", r"^atp5", r"^uqcr", r"^sdh[abcd]", r"^cyc1$", r"^cycs$", r"^mrpl", r"^mrps", r"^timm", r"^tomm", r"^atad3", r"^tfb[12]m$", r"^tfam$", r"^polrmt$", r"^slc25a", r"^chchd", r"^ndufaf", r"^coa[3-7]$", r"^cox1[1-9]$", r"^phb2?$", r"^bola[123]$", r"^emc[0-9]", r"^yipf[0-9]", r"^sec6[123]", r"^ssr[1-4]$", r"^srp[0-9]", r"^vdac", r"^immt$", r"^opa1$", r"^mfn[12]$",
 r"^rpl", r"^rps", r"^rplp", r"^eif[1-6]", r"^eef[12]", r"^eef1", r"^etf1$", r"^pabpc", r"^nop[0-9]", r"^nol[0-9c]", r"^ncl$", r"^npm[13]$", r"^fbl$", r"^dkc1$", r"^gnl[23]", r"^rrp[0-9]", r"^utp[0-9]", r"^pa2g4$", r"^lyar$", r"^rrs1$", r"^nolc1$", r"^gar1$", r"^nhp2", r"^nop56$", r"^nop58$", r"^bop1$", r"^pes1$", r"^wdr43$", r"^wdr75$", r"^ddx2[17]$", r"^ddx5[16]$", r"^dhx", r"^rsl1d1$", r"^ttf1$", r"^ebna1bp2$", r"^naca$", r"^btf3$", r"^ybx[13]$",
 r"^hsp", r"^hspa", r"^hspd1$", r"^hspe1$", r"^cct[1-8]", r"^tcp1$", r"^dnaj", r"^ppia$", r"^ppib$", r"^pdia", r"^calr$", r"^canx$",
 r"^psm[abcdeg]", r"^uba[1-7]", r"^ube2", r"^ubb$", r"^ubc$", r"^uba52$", r"^rps27a$", r"^sumo",
 r"^haus", r"^cenp", r"^kif", r"^ncap[dgh]", r"^smc[1-6]", r"^nuf2$", r"^ndc80$", r"^spc2[45]$", r"^bub[13]", r"^mad2l", r"^aurk", r"^plk[14]$", r"^ttk$", r"^cdc20$", r"^cdca", r"^kntc1$", r"^zwint$", r"^ska[123]$", r"^mis18", r"^tubb", r"^tuba1", r"^tubg", r"^stmn1$", r"^top2a$", r"^mki67$", r"^hmgb2$", r"^anln$", r"^prc1$", r"^ect2$", r"^ccn[abe][12]$", r"^cdk1$", r"^cdkn3$", r"^pttg1$", r"^birc5$", r"^tpx2$", r"^nusap1$", r"^ube2c$", r"^cks[12]", r"^h2af", r"^hist", r"^h1f", r"^h2a", r"^h2b", r"^h3f", r"^h4c",
 r"^e2f[1-8]$", r"^tfdp[12]$", r"^mybl2$", r"^foxm1$", r"^cdt1$", r"^cdc6$", r"^gmnn$", r"^lin9$", r"^lin54$", r"^mcm[2-8]", r"^pcna$", r"^rfc[1-5]", r"^pol[ade][1-4]?$", r"^pole[1-4]?$", r"^rrm[12]$", r"^tyms$", r"^fen1$", r"^lig1$", r"^dut$", r"^gins", r"^cdc4[5]$", r"^cdc[67]$", r"^orc[1-6]", r"^chaf1", r"^dhfr$", r"^tk1$", r"^dtl$", r"^uhrf1$", r"^hells$", r"^rad51", r"^brca[12]$", r"^msh[26]$", r"^prim[12]$", r"^clspn$", r"^atad2$", r"^kiaa0101$", r"^pclaf$",
 r"^sf3[ab]", r"^snrp", r"^srsf", r"^hnrnp", r"^lsm[1-8]", r"^prpf", r"^u2af", r"^sfpq$", r"^ddx39", r"^cpsf", r"^cstf", r"^nono$", r"^tra2", r"^rbm[0-9]",
 r"^polr[123]", r"^gtf2", r"^gtf3", r"^taf[0-9]", r"^med[0-9]", r"^ccnh$", r"^cdk7$", r"^supt[0-9]", r"^ncbp",
 r"^nup[0-9]", r"^kpn[ab]", r"^xpo[1-7]", r"^ran$", r"^ranbp", r"^ipo[0-9]", r"^tnpo",
 r"^gapdh", r"^pgk1$", r"^eno1$", r"^ldha$", r"^pkm$", r"^aldoa$", r"^tpi1$", r"^pgam1$", r"^actb$", r"^actg1$", r"^tmsb", r"^pfn1$", r"^cfl1$", r"^ptma$", r"^set$", r"^ftl$", r"^fth1$",
]
RX = re.compile("|".join(PATTERNS), re.I)
def is_machinery(g): return bool(RX.search(str(g)))
