"""
M2MRFNet: top-level segmentation model.

Combines HRNet_M2MRF backbone with FCNDecoder.

Input  : [B, 3, H, W]   (H=W=512 in this project)
Output : [B, num_classes, H, W]   (bilinear-upsampled to match input)

Forward signature is identical to UNet so it slots directly into the existing
training loop without any modification.

Tensor shape flow (variant A, W18-Small, 5 classes, 512×512 input):
  Input          [B,   3, 512, 512]
  stem           [B,  64, 128, 128]
  layer1         [B, 256, 128, 128]
  stage4 out:
    branch 0     [B,  18, 128, 128]
    branch 1     [B,  36,  64,  64]
    branch 2     [B,  72,  32,  32]
    branch 3     [B, 144,  16,  16]
  FCNDecoder     [B,   5, 128, 128]  (cat=270 ch → 270 ch → 5 ch)
  upsample       [B,   5, 512, 512]
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import (
    HRNet_M2MRF,
    HRNet_M2MRF_A,
    HRNet_M2MRF_B,
    HRNet_M2MRF_C,
    HRNet_M2MRF_D,
)
from .decoder import FCNDecoder


_VARIANTS: dict[str, type[HRNet_M2MRF]] = {
    'base': HRNet_M2MRF,
    'A':    HRNet_M2MRF_A,
    'B':    HRNet_M2MRF_B,
    'C':    HRNet_M2MRF_C,
    'D':    HRNet_M2MRF_D,
}


class M2MRFNet(nn.Module):
    """
    M2MRF segmentation model (backbone + FCN decoder).

    Args:
        num_classes:               output classes incl. background (default 5).
        variant:                   M2MRF variant 'A'/'B'/'C'/'D'/'base' (default 'A').
        stage2_channels:           HRNet-W channel widths for stage2 (default W18: (18,36)).
        stage3_channels:           stage3 channel widths (default (18,36,72)).
        stage4_channels:           stage4 channel widths (default (18,36,72,144)).
        m2mrf_patch_size:          (patch_down, patch_up) — patch size for the M2MRF module.
        m2mrf_encode_channels_rate: channel compression ratio inside M2MRF.
        m2mrf_fc_channels_rate:    bottleneck ratio for the 1-D FC layers.
        decoder_mid_channels:      intermediate channels in FCNDecoder (None → sum of stage4).
        dropout:                   spatial dropout in the decoder.
    """

    def __init__(
        self,
        num_classes: int = 5,
        variant: str = 'A',
        stage2_channels: tuple[int, ...] = (18, 36),
        stage3_channels: tuple[int, ...] = (18, 36, 72),
        stage4_channels: tuple[int, ...] = (18, 36, 72, 144),
        m2mrf_patch_size: tuple[int, int] = (8, 8),
        m2mrf_encode_channels_rate: int = 4,
        m2mrf_fc_channels_rate: int = 64,
        decoder_mid_channels: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if variant not in _VARIANTS:
            raise ValueError(
                f"Unknown M2MRF variant '{variant}'. "
                f"Choose from {list(_VARIANTS.keys())}."
            )

        backbone_cls = _VARIANTS[variant]
        self.backbone = backbone_cls(
            stage2_channels=stage2_channels,
            stage3_channels=stage3_channels,
            stage4_channels=stage4_channels,
            m2mrf_patch_size=m2mrf_patch_size,
            m2mrf_encode_channels_rate=m2mrf_encode_channels_rate,
            m2mrf_fc_channels_rate=m2mrf_fc_channels_rate,
        )

        self.decoder = FCNDecoder(
            in_channels_list=list(stage4_channels),
            num_classes=num_classes,
            mid_channels=decoder_mid_channels,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]
        features = self.backbone(x)          # list of 4 tensors
        logits   = self.decoder(features)    # [B, C, H/4, W/4]
        if logits.shape[2:] != input_size:
            logits = F.interpolate(
                logits, size=input_size,
                mode='bilinear', align_corners=False,
            )
        return logits
