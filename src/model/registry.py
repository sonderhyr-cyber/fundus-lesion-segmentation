"""
Model registry.

Maps --model <name> to a factory function that returns an nn.Module.

Usage:
    from model.registry import build_model
    model = build_model("m2mrf", num_classes=5)

Adding a new model:
    1. Implement it under src/models/<name>/ (or src/model/<name>.py).
    2. Add an entry to _REGISTRY below.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import torch.nn as nn

# Ensure src/ is always importable regardless of cwd
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _build_unet(num_classes: int) -> nn.Module:
    from model.unet import UNet
    return UNet(in_channels=3, num_classes=num_classes)


def _build_hrdecoder(num_classes: int) -> nn.Module:
    from models.hrdecoder import HRDecoder
    return HRDecoder(num_classes=num_classes, pretrained=True)


def _build_m2mrf(num_classes: int) -> nn.Module:
    from models.m2mrf import M2MRFNet
    return M2MRFNet(num_classes=num_classes, variant='A')


# ── registry ─────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, Callable[[int], nn.Module]] = {
    "unet":      _build_unet,
    "hrdecoder": _build_hrdecoder,
    "m2mrf":     _build_m2mrf,
}


def build_model(name: str, num_classes: int = 5) -> nn.Module:
    """
    Instantiate a model by registry name.

    Args:
        name:        one of 'unet', 'hrdecoder', 'm2mrf'.
        num_classes: number of output segmentation classes.

    Returns:
        Uninitialised nn.Module (not yet moved to device).
    """
    name = name.lower()
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](num_classes)


def available_models() -> list[str]:
    return sorted(_REGISTRY.keys())
