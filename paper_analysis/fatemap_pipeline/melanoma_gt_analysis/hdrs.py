# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
import subprocess, collections, json
MEL=["melanoma","sk-mel","skmel","a375","501mel","colo829","mewo","wm989","wm","malme"]
out={}
for t in ["MITF","SOX10","JUN","JUND","FOSL1","FOSL2","TEAD1","TEAD2","TEAD3","TEAD4"]:
    line=subprocess.run(f'curl -s -m 200 "https://chip-atlas.dbcls.jp/data/hg38/target/{t}.5.tsv" | head -1',shell=True,capture_output=True,text=True).stdout.rstrip("\n")
    h=line.split("\t")[1:]; cols=[c for c in h if "|" in c and "|Average" not in c]
    cells=collections.Counter(c.split("|",1)[1] for c in cols); out[t]=dict(cells)
    mel=collections.Counter(c.split("|",1)[1] for c in cols if any(k in c.lower() for k in MEL))
    print(f"{t}: {len(cols)} experiments, {len(cells)} cell types | matched melanoma keywords: {dict(mel)}",flush=True)
json.dump(out,open("chip_cells.json","w"))
