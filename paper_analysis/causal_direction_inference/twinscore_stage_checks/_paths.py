"""Output location for the TwinScore stage checks (data lives in analysis_data, never in the repo)."""
import os
from pathlib import Path
_root = os.environ.get("TWINFER_DATA_ROOT") or str(Path(__file__).resolve().parents[5] / "analysis_data")
DATA = Path(_root) / "paper_analysis" / "causal_direction_inference" / "figure2_1000_zscores"
