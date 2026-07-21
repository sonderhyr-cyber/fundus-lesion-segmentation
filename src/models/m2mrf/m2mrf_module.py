"""
M2MRF: Many-to-Many Reassembly of Features (core patch-based module).

Paper : "Automated Lesion Segmentation in Fundus Images with
         Many-to-Many Reassembly of Features"
Source: https://github.com/CVIU-CSU/M2MRF-Lesion-Segmentation

Pure-PyTorch port — no mmcv dependency.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class M2MRF_Module(nn.Module):
    """
    Patch-level feature reassembly via 1-D FC bottleneck.

    Divides the feature map into non-overlapping patches of size `size × size`,
    maps each patch token to a bottleneck with `sample_fc`, then expands back to
    `size*scale_factor × size*scale_factor` tokens with `sample_fc1`, and folds
    them into the output feature map.

    Args:
        scale_factor:     spatial scaling ratio (e.g. 2 for 2× upsample, 0.5 for
                          half-resolution downsample).
        encode_channels:  number of feature channels fed to this module.
        fc_channels:      bottleneck width for the 1-D FC layers.
        size:             patch spatial size (integer, e.g. 8).
        groups:           groups for Conv1d (default 1).
    """

    def __init__(
        self,
        scale_factor: float,
        encode_channels: int,
        fc_channels: int,
        size: int,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.size = size
        self.groups = groups

        self.unfold_params = dict(
            kernel_size=size, dilation=1, padding=0, stride=size
        )
        self.fold_kernel = int(size * scale_factor)
        self.fold_stride = int(size * scale_factor)
        out_patch_len = int(size * size * scale_factor * scale_factor * encode_channels)

        self.sample_fc = nn.Conv1d(
            size * size * encode_channels, fc_channels,
            kernel_size=1, groups=groups,
        )
        self.sample_fc1 = nn.Conv1d(
            fc_channels, out_patch_len,
            kernel_size=1, groups=groups,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        # [n, c*size^2, n_patches]
        x = nn.Unfold(**self.unfold_params)(x)
        x = x.view(n, c * self.size * self.size, -1)
        x = self.sample_fc(x)   # bottleneck
        x = self.sample_fc1(x)  # expand
        out_h = int(h * self.scale_factor)
        out_w = int(w * self.scale_factor)
        x = nn.Fold(
            (out_h, out_w),
            kernel_size=self.fold_kernel,
            dilation=1,
            padding=0,
            stride=self.fold_stride,
        )(x)
        return x


class M2MRF(nn.Module):
    """
    Many-to-Many Reassembly of Features.

    Wraps M2MRF_Module with a channel encode/decode step so it can be used as
    a drop-in replacement for bilinear upsample or strided convolution.

    Tensor shape flow (example: 2× upsample, in_ch=48, out_ch=48, patch=8):
        Input  [B, 48, H,   W  ]
        encode [B,  12, H,   W  ]   (48/4 = 12 encode_channels)
        patch  [B,  12, 2H, 2W ]   (M2MRF_Module, scale_factor=2)
        decode [B, 48, 2H, 2W ]
        crop   [B, 48, 2H, 2W ]   (remove padding artefacts)

    Args:
        scale_factor:             >1 → upsample; <1 → downsample.
        in_channels:              input feature channels.
        out_channels:             output feature channels.
        patch_size:               spatial patch size (default 8).
        encode_channels_rate:     compress ratio for the encode conv (default 4).
        fc_channels_rate:         bottleneck ratio for the 1-D FC (default 64).
        groups:                   Conv1d groups (default 1).
    """

    def __init__(
        self,
        scale_factor: float,
        in_channels: int,
        out_channels: int,
        patch_size: int = 8,
        encode_channels_rate: int = 4,
        fc_channels_rate: int = 64,
        version: int = 0,          # kept for API compat, unused
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.patch_size = patch_size

        encode_channels = max(1, int(in_channels / encode_channels_rate))
        fc_channels = max(
            1,
            int(patch_size * patch_size * encode_channels / fc_channels_rate),
        )

        self.encode = nn.Conv2d(in_channels, encode_channels, kernel_size=1)
        self.sample = M2MRF_Module(
            scale_factor=scale_factor,
            encode_channels=encode_channels,
            fc_channels=fc_channels,
            size=patch_size,
            groups=groups,
        )
        self.decode = nn.Conv2d(encode_channels, out_channels, kernel_size=1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _pad(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """Pad spatial dims to be divisible by patch_size; return target out size."""
        b, c, h, w = x.shape
        ph = h + (self.patch_size - h % self.patch_size) % self.patch_size
        pw = w + (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, [0, pw - w, 0, ph - h])
        out_h = max(int(h * self.scale_factor), 1)
        out_w = max(int(w * self.scale_factor), 1)
        return x, (out_h, out_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, out_shape = self._pad(x)
        x = self.encode(x)
        x = self.sample(x)
        x = self.decode(x)
        return x[:, :, : out_shape[0], : out_shape[1]]
