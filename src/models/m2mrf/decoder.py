"""
FCN-style multi-scale decoder for the M2MRF backbone.

Upsamples all HRNet branch features to the highest resolution, concatenates,
and applies two 3×3 convolutions followed by a 1×1 classification layer.

This mirrors the FCNHead used in the original M2MRF configs
(mmseg/models/decode_heads/fcn_head.py with num_convs=2).

Tensor shape flow (W18-Small, 5 classes):
    features  : [[B,18,H/4,W/4], [B,36,H/8,W/8], [B,72,H/16,W/16], [B,144,H/32,W/32]]
    upsample  : all → [B, C_i, H/4, W/4]
    cat       : [B, 270, H/4, W/4]                    (18+36+72+144 = 270)
    conv×2    : [B, 270, H/4, W/4]
    cls       : [B,   5, H/4, W/4]
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNDecoder(nn.Module):
    """
    Fully-Convolutional decoder head for multi-scale HRNet features.

    Args:
        in_channels_list: channel widths of each branch (e.g. [18,36,72,144]).
        num_classes:      number of segmentation classes (including background).
        mid_channels:     width of the two intermediate conv layers.
                          Defaults to sum(in_channels_list) to match paper behaviour.
        dropout:          spatial dropout rate before the classification conv.
    """

    def __init__(
        self,
        in_channels_list: list[int],
        num_classes: int,
        mid_channels: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        total_in = sum(in_channels_list)
        if mid_channels is None:
            mid_channels = total_in   # matches original FCNHead default

        self.fuse = nn.Sequential(
            nn.Conv2d(total_in, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.cls = nn.Conv2d(mid_channels, num_classes, 1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: list of 4 tensors from HRNet stage4,
                      highest-res first (branch 0 = stride-4).

        Returns:
            logits [B, num_classes, H/4, W/4] — caller upsamples to input size.
        """
        target = features[0].shape[2:]
        aligned = [features[0]] + [
            F.interpolate(f, size=target, mode='bilinear', align_corners=False)
            for f in features[1:]
        ]
        x = torch.cat(aligned, dim=1)
        x = self.fuse(x)
        return self.cls(x)
