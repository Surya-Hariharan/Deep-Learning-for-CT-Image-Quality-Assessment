"""Pytest bootstrap: put src/ on sys.path so tests can ``import ct_iqa...``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
