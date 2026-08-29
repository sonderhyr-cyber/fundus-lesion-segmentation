"""M2MRF feature reassembly blocks reusable by other segmentation models."""
from __future__ import annotations

import torch
import torch.nn as nn

from .m2mrf_module import M2MRF


class M2MRFCascadeUpsampler(nn.Module):
    """Upsample one feature branch by repeated learnable 2x M2MRF steps."""

    def __init__(
        self,
        channels: int,
        steps: int,
        patch_size: int,
        encode_channels_rate: int,
        fc_channels_rate: int,
    ) -> None:
        super().__init__()
        self.steps = nn.Sequential(*[
            M2MRF(
                scale_factor=2.0,
                in_channels=channels,
                out_channels=channels,
                patch_size=patch_size,
                encode_channels_rate=encode_channels_rate,
                fc_channels_rate=fc_channels_rate,
            )
            for _ in range(steps)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.steps(x)
