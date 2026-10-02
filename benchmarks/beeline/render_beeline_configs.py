"""Render the Beeline config templates (config-files/*.yaml) into concrete YAMLs.

Beeline reads its YAML directly with no variable expansion, so absolute paths must be filled in before a run.
Templates carry three placeholders: ${TWINFER_DATA_ROOT} (analysis_data), ${TWINFER_PROJECT_ROOT} (TwINFER_KA) and ${TWINFER_BEELINE_PATH} (full Beeline install).
Settings (no CLI needed; edit here or set the env vars from env.sh):
"""
import os, re, sys
from pathlib import Path

TEMPLATE_DIR = Path(__file__).parent / "config-files"
# Where the rendered configs go. Beeline's BLRunner.py is run from the full install and reads config-files/ there.
OUT_DIR = None                  # None -> $TWINFER_BEELINE_PATH/config-files
ONLY = None                     # None -> all templates, or a list of file names, e.g. ["config_real_networks.yaml"]

def _values():
    from twinfer.utils.paths import get_data_root
    data = os.environ.get("TWINFER_DATA_ROOT") or str(get_data_root())
    bee = os.environ.get("TWINFER_BEELINE_PATH")
    if not bee:
        raise SystemExit("TWINFER_BEELINE_PATH is not set (source clean_code/env.sh)")
    proj = os.environ.get("TWINFER_PROJECT_ROOT") or str(Path(data).parent)   # [2026-10-01 added]
    return {"TWINFER_DATA_ROOT": data.rstrip("/"), "TWINFER_PROJECT_ROOT": proj.rstrip("/"), "TWINFER_BEELINE_PATH": bee.rstrip("/")}

def render_text(text, values):
    out = []
    for ln in text.split("\n"):
        if ln.lstrip().startswith("# [2026-10-01 replaced by template placeholder"):
            continue   # bookkeeping comment, not part of the rendered config
        out.append(ln)
    text = "\n".join(out)
    for k, v in values.items():
        text = text.replace("${%s}" % k, v)
    left = re.findall(r"\$\{[^}]*\}", text)
    if left:
        raise ValueError(f"unresolved placeholders: {sorted(set(left))}")
    return text

def main(out_dir=None):
    values = _values()
    out_dir = Path(out_dir or OUT_DIR or Path(values["TWINFER_BEELINE_PATH"]) / "config-files")
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(TEMPLATE_DIR.glob("*.yaml")):
        if ONLY and f.name not in ONLY:
            continue
        (out_dir / f.name).write_text(render_text(f.read_text(), values))
        n += 1
    print(f"rendered {n} configs -> {out_dir}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)   # optional: output dir (use a scratch dir to test; the default is the live Beeline install)
