"""
Project entry point.

Usage:
    python train.py                   # train default model (unet)
    python train.py --model unet
    python train.py --model hrdecoder

This script only sets up the Python path and delegates to src/model/train.py.
All training logic lives in src/model/train.py.
"""
import sys
from pathlib import Path

# Ensure src/ is on sys.path so all internal imports resolve correctly.
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model.train import main  # noqa: E402

if __name__ == "__main__":
    main()
