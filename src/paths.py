from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_ROOT = PROJECT_ROOT / "dataset" / "Segmentation"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoints" / "best_model.pth"


def setup_src_path() -> None:
    """Ensure `src/` is on sys.path for imports when running scripts directly."""
    src_path = str(SRC_DIR)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
