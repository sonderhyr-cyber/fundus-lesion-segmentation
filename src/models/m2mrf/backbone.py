"""
HRNet backbone with optional M2MRF feature-reassembly modules.

Pure-PyTorch port of mmseg/models/backbones/hrnet_m2mrf.py from
https://github.com/CVIU-CSU/M2MRF-Lesion-Segmentation

No mmcv / mmdet dependency.  Residual blocks (BasicBlock, Bottleneck) are
reimplemented inline.

Tensor shape flow (default W18-Small, input [B,3,512,512]):
  stem           [B, 64,  128, 128]   (2 × stride-2 conv)
  layer1         [B, 256, 128, 128]   (4 × Bottleneck)
  stage2 branch0 [B,  18, 128, 128]
  stage2 branch1 [B,  36,  64,  64]
  stage3 branch2 [B,  72,  32,  32]
  stage4 branch3 [B, 144,  16,  16]
  → returns list of 4 tensors (the 4 parallel branches of stage4)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .m2mrf_module import M2MRF


# ---------------------------------------------------------------------------
# Residual blocks
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    """2-conv residual block (expansion=1)."""

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class Bottleneck(nn.Module):
    """3-conv bottleneck block (expansion=4)."""

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * self.expansion)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


def _make_layer(
    block: type,
    inplanes: int,
    planes: int,
    num_blocks: int,
    stride: int = 1,
) -> nn.Sequential:
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion, 1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )
    layers: list[nn.Module] = [block(inplanes, planes, stride=stride, downsample=downsample)]
    inplanes = planes * block.expansion
    for _ in range(1, num_blocks):
        layers.append(block(inplanes, planes))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# HRModule: one multi-branch stage with M2MRF-aware fuse layers
# ---------------------------------------------------------------------------

class HRModule(nn.Module):
    """
    One HRNet parallel-branch module.

    When any of m2mrf_onestep_up / m2mrf_cascade_up is True, bilinear upsample
    is replaced by learnable M2MRF.  Similarly for downsample directions.
    """

    def __init__(
        self,
        num_branches: int,
        block: type,
        num_blocks: list[int],
        in_channels: list[int],
        num_channels: list[int],
        multiscale_output: bool = True,
        # M2MRF config
        m2mrf_patch_size: tuple[int, int] = (8, 8),
        m2mrf_encode_channels_rate: int = 4,
        m2mrf_fc_channels_rate: int = 64,
        m2mrf_groups: int = 1,
        m2mrf_cascade_down: bool = False,
        m2mrf_onestep_down: bool = False,
        m2mrf_cascade_up: bool = False,
        m2mrf_onestep_up: bool = False,
    ) -> None:
        super().__init__()
        assert num_branches == len(num_blocks) == len(num_channels) == len(in_channels)

        self.num_branches = num_branches
        self.multiscale_output = multiscale_output
        self.in_channels = list(in_channels)  # mutable; updated by _make_branches

        # M2MRF params
        self._ps    = m2mrf_patch_size
        self._ecr   = m2mrf_encode_channels_rate
        self._fcr   = m2mrf_fc_channels_rate
        self._grp   = m2mrf_groups
        self._cd    = m2mrf_cascade_down
        self._od    = m2mrf_onestep_down
        self._cu    = m2mrf_cascade_up
        self._ou    = m2mrf_onestep_up

        self.branches    = self._make_branches(block, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu        = nn.ReLU(inplace=False)

    # ── branch construction ───────────────────────────────────────────────────

    def _make_branches(
        self,
        block: type,
        num_blocks: list[int],
        num_channels: list[int],
    ) -> nn.ModuleList:
        branches = []
        for i in range(self.num_branches):
            downsample = None
            if self.in_channels[i] != num_channels[i] * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.in_channels[i],
                              num_channels[i] * block.expansion,
                              1, bias=False),
                    nn.BatchNorm2d(num_channels[i] * block.expansion),
                )
            layers: list[nn.Module] = [
                block(self.in_channels[i], num_channels[i], downsample=downsample)
            ]
            self.in_channels[i] = num_channels[i] * block.expansion
            for _ in range(1, num_blocks[i]):
                layers.append(block(self.in_channels[i], num_channels[i]))
            branches.append(nn.Sequential(*layers))
        return nn.ModuleList(branches)

    # ── fuse-layer construction ───────────────────────────────────────────────

    def _m2mrf(self, scale: float, ic: int, oc: int, is_up: bool) -> M2MRF:
        return M2MRF(
            scale_factor=scale,
            in_channels=ic,
            out_channels=oc,
            patch_size=self._ps[1] if is_up else self._ps[0],
            encode_channels_rate=self._ecr,
            fc_channels_rate=self._fcr,
            groups=self._grp,
        )

    def _make_fuse_layers(self) -> nn.ModuleList | None:
        if self.num_branches == 1:
            return None

        ic = self.in_channels
        nb = self.num_branches
        n_out = nb if self.multiscale_output else 1

        fuse_layers = []
        for i in range(n_out):
            row = []
            for j in range(nb):
                if j > i:
                    # j is lower-res → upsample to i
                    if self._ou:
                        row.append(nn.Sequential(
                            nn.Conv2d(ic[j], ic[i], 1, bias=False),
                            nn.BatchNorm2d(ic[i]),
                            self._m2mrf(2 ** (j - i), ic[i], ic[i], is_up=True),
                        ))
                    elif self._cu:
                        layers: list[nn.Module] = [
                            nn.Conv2d(ic[j], ic[i], 1, bias=False),
                            nn.BatchNorm2d(ic[i]),
                        ]
                        for _ in range(j - i):
                            layers.append(self._m2mrf(2, ic[i], ic[i], is_up=True))
                        row.append(nn.Sequential(*layers))
                    else:
                        row.append(nn.Sequential(
                            nn.Conv2d(ic[j], ic[i], 1, bias=False),
                            nn.BatchNorm2d(ic[i]),
                            nn.Upsample(scale_factor=2 ** (j - i),
                                        mode='bilinear', align_corners=False),
                        ))

                elif j == i:
                    row.append(None)  # type: ignore[arg-type]

                else:
                    # j is higher-res → downsample to i
                    if self._od:
                        row.append(nn.Sequential(
                            self._m2mrf(0.5 ** (i - j), ic[j], ic[i], is_up=False),
                            nn.BatchNorm2d(ic[i]),
                        ))
                    elif self._cd:
                        layers = []
                        for k in range(i - j):
                            last = (k == i - j - 1)
                            out_c = ic[i] if last else ic[j]
                            layers.append(nn.Sequential(
                                self._m2mrf(0.5, ic[j], out_c, is_up=False),
                                nn.BatchNorm2d(out_c),
                                *([] if last else [nn.ReLU(inplace=False)]),
                            ))
                        row.append(nn.Sequential(*layers))
                    else:
                        layers = []
                        for k in range(i - j):
                            last = (k == i - j - 1)
                            out_c = ic[i] if last else ic[j]
                            layers.append(nn.Sequential(
                                nn.Conv2d(ic[j], out_c, 3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(out_c),
                                *([] if last else [nn.ReLU(inplace=False)]),
                            ))
                        row.append(nn.Sequential(*layers))

            fuse_layers.append(nn.ModuleList(row))

        return nn.ModuleList(fuse_layers)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        # Apply each branch independently
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        # Cross-branch fusion
        x_fuse: list[torch.Tensor] = []
        for i in range(len(self.fuse_layers)):  # type: ignore[arg-type]
            y: torch.Tensor = x[i]              # identity (j == i)
            for j in range(self.num_branches):
                if j == i:
                    continue
                fused = self.fuse_layers[i][j](x[j])  # type: ignore[index]
                if j > i:
                    # Ensure exact spatial match after upsample/M2MRF
                    fused = F.interpolate(
                        fused, size=x[i].shape[2:],
                        mode='bilinear', align_corners=False,
                    )
                y = y + fused
            x_fuse.append(self.relu(y))

        return x_fuse


# ---------------------------------------------------------------------------
# HRNet_M2MRF backbone
# ---------------------------------------------------------------------------

class HRNet_M2MRF(nn.Module):
    """
    HRNet backbone with learnable M2MRF feature reassembly.

    Default width: W18-Small (trains from scratch; ~7 M parameters).
    The variant flags (onestep_down/up, cascade_down/up) mirror the A/B/C/D
    variants from the original paper.

    Output: list of 4 feature maps from stage4 parallel branches.
        branch[0]: [B, s4_ch[0], H/4,  W/4 ]   e.g. [B,  18, 128, 128]
        branch[1]: [B, s4_ch[1], H/8,  W/8 ]   e.g. [B,  36,  64,  64]
        branch[2]: [B, s4_ch[2], H/16, W/16]   e.g. [B,  72,  32,  32]
        branch[3]: [B, s4_ch[3], H/32, W/32]   e.g. [B, 144,  16,  16]
    """

    def __init__(
        self,
        in_channels: int = 3,
        stage2_channels: tuple[int, ...] = (18, 36),
        stage3_channels: tuple[int, ...] = (18, 36, 72),
        stage4_channels: tuple[int, ...] = (18, 36, 72, 144),
        # M2MRF hyper-params
        m2mrf_patch_size: tuple[int, int] = (8, 8),
        m2mrf_encode_channels_rate: int = 4,
        m2mrf_fc_channels_rate: int = 64,
        m2mrf_groups: int = 1,
        # Variant flags — per-stage tuples of 3 bools (stage2, stage3, stage4)
        m2mrf_cascade_down_list: tuple[bool, ...] = (False, False, False),
        m2mrf_onestep_down_list: tuple[bool, ...] = (False, False, False),
        m2mrf_cascade_up_list:   tuple[bool, ...] = (False, False, False),
        m2mrf_onestep_up_list:   tuple[bool, ...] = (False, False, False),
    ) -> None:
        super().__init__()

        self._m2mrf_kw = dict(
            m2mrf_patch_size=m2mrf_patch_size,
            m2mrf_encode_channels_rate=m2mrf_encode_channels_rate,
            m2mrf_fc_channels_rate=m2mrf_fc_channels_rate,
            m2mrf_groups=m2mrf_groups,
        )
        self._cdl = m2mrf_cascade_down_list
        self._odl = m2mrf_onestep_down_list
        self._cul = m2mrf_cascade_up_list
        self._oul = m2mrf_onestep_up_list
        self._ps  = m2mrf_patch_size
        self._ecr = m2mrf_encode_channels_rate
        self._fcr = m2mrf_fc_channels_rate
        self._grp = m2mrf_groups

        # ── Stem: stride-4 ────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )

        # ── Stage 1: 4 Bottleneck blocks 64→256 ───────────────────────────
        self.layer1 = _make_layer(Bottleneck, 64, 64, num_blocks=4)
        stage1_out = 64 * Bottleneck.expansion  # 256

        # ── Stage 2 ───────────────────────────────────────────────────────
        s2 = list(stage2_channels)
        self.transition1 = self._make_transition([stage1_out], s2, stage_idx=0)
        self.stage2, s2 = self._make_stage(
            in_ch=s2, num_modules=1, block=BasicBlock,
            num_blocks=[4] * len(s2), num_channels=s2, stage_idx=0,
        )

        # ── Stage 3 ───────────────────────────────────────────────────────
        s3 = list(stage3_channels)
        self.transition2 = self._make_transition(s2, s3, stage_idx=1)
        self.stage3, s3 = self._make_stage(
            in_ch=s3, num_modules=4, block=BasicBlock,
            num_blocks=[4] * len(s3), num_channels=s3, stage_idx=1,
        )

        # ── Stage 4 ───────────────────────────────────────────────────────
        s4 = list(stage4_channels)
        self.transition3 = self._make_transition(s3, s4, stage_idx=2)
        self.stage4, s4 = self._make_stage(
            in_ch=s4, num_modules=3, block=BasicBlock,
            num_blocks=[4] * len(s4), num_channels=s4, stage_idx=2,
        )

        self.out_channels: list[int] = s4

    # ── helpers ───────────────────────────────────────────────────────────────

    def _make_transition(
        self,
        pre: list[int],
        cur: list[int],
        stage_idx: int,
    ) -> nn.ModuleList:
        """Build transition layers between two consecutive stages."""
        use_m2mrf = self._cdl[stage_idx] or self._odl[stage_idx]
        layers: list[nn.Module | None] = []
        for i, ch_cur in enumerate(cur):
            if i < len(pre):
                if ch_cur == pre[i]:
                    layers.append(None)
                else:
                    layers.append(nn.Sequential(
                        nn.Conv2d(pre[i], ch_cur, 3, padding=1, bias=False),
                        nn.BatchNorm2d(ch_cur),
                        nn.ReLU(inplace=True),
                    ))
            else:
                # new lower-resolution branch
                ch_in = pre[-1]
                if use_m2mrf:
                    layers.append(nn.Sequential(
                        M2MRF(0.5, ch_in, ch_cur,
                              patch_size=self._ps[0],
                              encode_channels_rate=self._ecr,
                              fc_channels_rate=self._fcr,
                              groups=self._grp),
                        nn.BatchNorm2d(ch_cur),
                        nn.ReLU(inplace=True),
                    ))
                else:
                    layers.append(nn.Sequential(
                        nn.Conv2d(ch_in, ch_cur, 3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(ch_cur),
                        nn.ReLU(inplace=True),
                    ))
        return nn.ModuleList(layers)

    def _make_stage(
        self,
        in_ch: list[int],
        num_modules: int,
        block: type,
        num_blocks: list[int],
        num_channels: list[int],
        stage_idx: int,
    ) -> tuple[nn.Sequential, list[int]]:
        modules = []
        cur_in = list(in_ch)
        for _ in range(num_modules):
            mod = HRModule(
                num_branches=len(cur_in),
                block=block,
                num_blocks=num_blocks,
                in_channels=list(cur_in),   # pass copy; HRModule mutates it
                num_channels=num_channels,
                multiscale_output=True,
                **self._m2mrf_kw,
                m2mrf_cascade_down=self._cdl[stage_idx],
                m2mrf_onestep_down=self._odl[stage_idx],
                m2mrf_cascade_up=self._cul[stage_idx],
                m2mrf_onestep_up=self._oul[stage_idx],
            )
            cur_in = mod.in_channels   # updated in-place by HRModule
            modules.append(mod)
        return nn.Sequential(*modules), cur_in

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        x = self.layer1(x)

        # stage 2
        x_list = [t(x) if t is not None else x for t in self.transition1]
        x_list = self.stage2(x_list)

        # stage 3 — new branches sourced from lowest-res output (y[-1])
        y = x_list
        x_list = [
            t(y[-1]) if t is not None else y[i]
            for i, t in enumerate(self.transition2)
        ]
        x_list = self.stage3(x_list)

        # stage 4
        y = x_list
        x_list = [
            t(y[-1]) if t is not None else y[i]
            for i, t in enumerate(self.transition3)
        ]
        x_list = self.stage4(x_list)

        return x_list   # list of 4 branch tensors


# ---------------------------------------------------------------------------
# Paper variants A / B / C / D
# ---------------------------------------------------------------------------

class HRNet_M2MRF_A(HRNet_M2MRF):
    """Variant A: one-step down + one-step up (all stages)."""
    def __init__(self, **kw: object) -> None:
        super().__init__(
            m2mrf_onestep_down_list=(True, True, True),
            m2mrf_onestep_up_list=(True, True, True),
            **kw,  # type: ignore[arg-type]
        )


class HRNet_M2MRF_B(HRNet_M2MRF):
    """Variant B: one-step down + cascade up."""
    def __init__(self, **kw: object) -> None:
        super().__init__(
            m2mrf_onestep_down_list=(True, True, True),
            m2mrf_cascade_up_list=(True, True, True),
            **kw,  # type: ignore[arg-type]
        )


class HRNet_M2MRF_C(HRNet_M2MRF):
    """Variant C: cascade down + one-step up."""
    def __init__(self, **kw: object) -> None:
        super().__init__(
            m2mrf_cascade_down_list=(True, True, True),
            m2mrf_onestep_up_list=(True, True, True),
            **kw,  # type: ignore[arg-type]
        )


class HRNet_M2MRF_D(HRNet_M2MRF):
    """Variant D: cascade down + cascade up."""
    def __init__(self, **kw: object) -> None:
        super().__init__(
            m2mrf_cascade_down_list=(True, True, True),
            m2mrf_cascade_up_list=(True, True, True),
            **kw,  # type: ignore[arg-type]
        )
