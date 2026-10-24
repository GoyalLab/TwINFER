# PeTrace (mouse embryos, E7.5-E10.0): the authors' files

**Raw data**: Zenodo record 23088914, files `E<day>-R<replicate>.h5td` (counts, embedding and lineage tree per embryo). Figure 5 uses E9.5 (R1-R3 together).
No preprocessing: the files are used as published. `chunk_dl.sh` downloads a file in resumable byte ranges; `assemble_verify.sh` joins the parts and checks
size and md5 against Zenodo.

Twins are **sister cells** of the lineage tree (leaf cells under the same parent node); the `clone` column of the files is a whole tree and is not used.
