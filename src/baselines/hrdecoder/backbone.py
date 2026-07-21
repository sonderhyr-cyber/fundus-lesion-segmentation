"""
HRNet-W48 backbone wrapper for HRDecoder.

Outputs 4 parallel branch feature maps from stage4 of HRNet-W48:
  branch[0]: (B,  48, H/4,  W/4)
  branch[1]: (B,  96, H/8,  W/8)
  branch[2]: (B, 192, H/16, W/16)
  branch[3]: (B, 384, H/32, W/32)

Backbone is loaded from timm ('hrnet_w48').
Pretrained weights are from the ImageNet-trained MSRA model.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn


class HRNetW48(nn.Module):
    """
    HRNet-W48 backbone that returns the 4 parallel branch outputs of stage4.

    Channel widths (paper notation W = 48):
        branch 0 : 1 × W  =  48 ch  (stride 4)
        branch 1 : 2 × W  =  96 ch  (stride 8)
        branch 2 : 4 × W  = 192 ch  (stride 16)
        branch 3 : 8 × W  = 384 ch  (stride 32)

    Total concatenated channels (after transform_inputs): 48+96+192+384 = 720.
    """

    CHANNELS: list[int] = [48, 96, 192, 384]

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self._hrnet = timm.create_model("hrnet_w48", pretrained=pretrained)
        # Remove classification head — we only need up to stage4.
        del self._hrnet.incre_modules
        del self._hrnet.downsamp_modules
        del self._hrnet.final_layer
        del self._hrnet.global_pool
        del self._hrnet.head_drop
        del self._hrnet.classifier

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W)

        Returns:
            List of 4 feature tensors from HRNet stage4 parallel branches.
        """
        x = self._hrnet.act1(self._hrnet.bn1(self._hrnet.conv1(x)))
        x = self._hrnet.act2(self._hrnet.bn2(self._hrnet.conv2(x)))
        x = self._hrnet.layer1(x)

        # transition1: 1 branch → 2 branches
        xl = [t(x) for t in self._hrnet.transition1]
        yl = self._hrnet.stage2(xl)

        # transition2: 2 branches → 3 branches
        xl = [
            t(yl[-1]) if i >= len(yl) else yl[i]
            for i, t in enumerate(self._hrnet.transition2)
        ]
        yl = self._hrnet.stage3(xl)

        # transition3: 3 branches → 4 branches
        xl = [
            t(yl[-1]) if i >= len(yl) else yl[i]
            for i, t in enumerate(self._hrnet.transition3)
        ]
        yl = self._hrnet.stage4(xl)

        # yl: [f0(B,48,H/4), f1(B,96,H/8), f2(B,192,H/16), f3(B,384,H/32)]
        return yl
