"""
FCNHead decoder for HRDecoder.

Reproduces the FCNHead config from the paper:
    type='FCNHead'
    in_channels=720   (sum of HRNet-W48 branch channels)
    channels=64
    kernel_size=7
    num_convs=1
    compress=True     (1×1 conv before main kernel)
    concat_input=False
    dropout_ratio=-1  (disabled)

Architecture:
    compress : Conv2d(720, 64, 1) → BN → ReLU
    conv     : Conv2d( 64, 64, 7, padding=3) → BN → ReLU
    cls_seg  : Conv2d( 64, num_classes, 1)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FCNHead(nn.Module):
    """
    FCN segmentation head with dimension compression.

    Input:  (B, in_channels, H, W)
    Output: (B, num_classes, H, W)   — logits, NOT upsampled to original size.
    """

    def __init__(
        self,
        in_channels: int = 720,
        channels: int = 64,
        num_classes: int = 5,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()

        # compress=True: 1×1 bottleneck before the main kernel
        self.compress = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # num_convs=1: single conv with the specified kernel
        self.conv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # 1×1 classification conv
        self.cls_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.compress(x)
        x = self.conv(x)
        return self.cls_seg(x)
