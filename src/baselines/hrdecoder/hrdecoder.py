"""
HRDecoder: High-Resolution Decoder Network for Fundus Image Lesion Segmentation.
MICCAI 2024 — CVIU-CSU/HRDecoder.

This is a pure-PyTorch port of the original MMsegmentation implementation.
The training strategy and all architectural decisions are faithful to the paper.

Key training strategy
---------------------
1. Resize input to hr_scale (no-op when input is already hr_scale).
2. Run backbone → multi-scale features → transform (concat after upsample).
3. Compute LR logit via FCNHead.
4. HR auxiliary training:
     Draw `crop_num` random sub-regions (same scale for all crops per step).
     Crop the feature map at each region, upsample back to full feature size,
     run through FCNHead → HR logit.
     Compute aux loss on the cropped ground-truth sub-regions (weight 0.1).
5. Sliding-window fuse: divide the full feature map into hr_scale tiles,
     average each tile's prediction with the LR prediction at that location.
6. Fuse logit → main loss.

For 512×512 input with hr_scale=(512,512):
  • Step 1 and 5 are no-ops (single tile = full image).
  • Step 4 is active: sub-crops with scale_ratio ∈ (0.75, 1.25) are drawn;
    crops smaller than the full image provide local auxiliary supervision.

Forward signature
-----------------
  training : model(x, gt=masks)  →  (fuse_logit, aux_loss_scalar)
  inference: model(x)            →  fuse_logit
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import HRNetW48
from .decoder import FCNHead


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _get_crop_bbox(
    img_h: int,
    img_w: int,
    crop_h: int,
    crop_w: int,
    divisible: int = 8,
) -> tuple[int, int, int, int]:
    """
    Sample a random (y1, y2, x1, x2) crop bbox inside the image.
    Crop dimensions are exactly (crop_h, crop_w); position is random.
    """
    crop_h = min(crop_h, img_h)
    crop_w = min(crop_w, img_w)
    margin_h = img_h - crop_h
    margin_w = img_w - crop_w
    n_steps_h = max(margin_h // divisible, 0)
    n_steps_w = max(margin_w // divisible, 0)
    offset_h = np.random.randint(0, n_steps_h + 1) * divisible if n_steps_h > 0 else 0
    offset_w = np.random.randint(0, n_steps_w + 1) * divisible if n_steps_w > 0 else 0
    y1 = min(offset_h, img_h - crop_h)
    y2 = y1 + crop_h
    x1 = min(offset_w, img_w - crop_w)
    x2 = x1 + crop_w
    return y1, y2, x1, x2


def _crop(t: torch.Tensor, y1: int, y2: int, x1: int, x2: int) -> torch.Tensor:
    if t.dim() == 4:
        return t[:, :, y1:y2, x1:x2]
    if t.dim() == 3:
        return t[:, y1:y2, x1:x2]
    if t.dim() == 2:
        return t[y1:y2, x1:x2]
    raise ValueError(f"Unsupported tensor dim: {t.dim()}")


def _resizecrop_feat(
    feat: torch.Tensor,
    ori_h: int,
    ori_w: int,
    y1: int,
    y2: int,
    x1: int,
    x2: int,
) -> torch.Tensor:
    """
    Crop the region of `feat` that corresponds to the image bbox (y1,y2,x1,x2),
    then upsample the cropped feature back to feat's original spatial size.

    This mirrors `_resizecrop_feats` in the original HRDecoder code.
    """
    feat_h, feat_w = feat.shape[2], feat.shape[3]
    scale_h = feat_h / ori_h
    scale_w = feat_w / ori_w

    fy1 = int(y1 * scale_h)
    fy2 = max(int(y2 * scale_h), fy1 + 1)
    fx1 = int(x1 * scale_w)
    fx2 = max(int(x2 * scale_w), fx1 + 1)

    # Clamp to valid range
    fy2 = min(fy2, feat_h)
    fx2 = min(fx2, feat_w)

    cropped = feat[:, :, fy1:fy2, fx1:fx2]
    return F.interpolate(cropped, (feat_h, feat_w), mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# HRDecoder
# ---------------------------------------------------------------------------

class HRDecoder(nn.Module):
    """
    HRDecoder: HRNet-W48 backbone + FCNHead decoder + HR crop auxiliary training.

    Input:  (B, 3, H, W)   — expected H=W=512 in this project
    Output: (B, num_classes, H, W)

    During training, pass `gt` (ground-truth mask, LongTensor [B, H, W]):
        fuse_logit, aux_loss = model(images, gt=masks)
        total_loss = criterion(fuse_logit, masks) + aux_loss

    During evaluation (model.eval() or gt=None):
        logits = model(images)
    """

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        # HR settings (matching the paper config)
        hr_scale: tuple[int, int] = (512, 512),
        scale_ratio: tuple[float, float] = (0.75, 1.25),
        hr_loss_weight: float = 0.1,
        crop_num: int = 2,
        divisible: int = 8,
        criterion: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.hr_scale = hr_scale
        self.scale_ratio = scale_ratio
        self.hr_loss_weight = hr_loss_weight
        self.crop_num = crop_num
        self.divisible = divisible
        self.criterion = criterion  # used for HR auxiliary loss

        self.backbone = HRNetW48(pretrained=pretrained)
        in_ch = sum(self.backbone.CHANNELS)  # 720
        self.head = FCNHead(
            in_channels=in_ch,
            channels=64,
            num_classes=num_classes,
            kernel_size=7,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _transform_inputs(self, feats: list[torch.Tensor]) -> torch.Tensor:
        """
        Upsample all branch features to branch[0]'s resolution, then concat.
        Mirrors `_transform_inputs` in the original (bilinear upsample + cat).

        Output: (B, 720, H/4, W/4)
        """
        target_h, target_w = feats[0].shape[2], feats[0].shape[3]
        upsampled = [
            F.interpolate(f, (target_h, target_w), mode="bilinear", align_corners=False)
            for f in feats
        ]
        return torch.cat(upsampled, dim=1)

    def _get_random_crop_size(self) -> tuple[int, int]:
        """Draw one random scale in [scale_ratio[0], scale_ratio[1]] × hr_scale."""
        min_r, max_r = self.scale_ratio
        ratio = np.random.uniform(min_r, max_r)
        h = int(self.hr_scale[0] * ratio // self.divisible * self.divisible)
        w = int(self.hr_scale[1] * ratio // self.divisible * self.divisible)
        return max(h, self.divisible), max(w, self.divisible)

    def _slide_fuse(
        self,
        lr_feat: torch.Tensor,
        up_lr_logit: torch.Tensor,
        ori_h: int,
        ori_w: int,
    ) -> torch.Tensor:
        """
        Sliding-window HR inference over the feature map.

        For each tile in (ori_h, ori_w) of size hr_scale:
          1. Crop + upsample the corresponding region of lr_feat.
          2. Run through FCNHead.
          3. Upsample to tile size.
          4. Average with the LR logit at that location.

        For 512×512 input with hr_scale=(512,512): one tile = full image → no-op.
        """
        h_crop, w_crop = self.hr_scale
        h_stride, w_stride = h_crop, w_crop

        h_grids = max(ori_h - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(ori_w - w_crop + w_stride - 1, 0) // w_stride + 1

        fuse_logit = up_lr_logit.clone()

        for hi in range(h_grids):
            for wi in range(w_grids):
                y1 = hi * h_stride
                x1 = wi * w_stride
                y2 = min(y1 + h_crop, ori_h)
                x2 = min(x1 + w_crop, ori_w)
                # Adjust top-left so tile is exactly hr_scale (clamped at image edge)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                tile_feat = _resizecrop_feat(lr_feat, ori_h, ori_w, y1, y2, x1, x2)
                tile_logit = self.head(tile_feat)
                tile_logit_up = F.interpolate(
                    tile_logit,
                    (y2 - y1, x2 - x1),
                    mode="bilinear",
                    align_corners=False,
                )
                fuse_logit[:, :, y1:y2, x1:x2] = (
                    tile_logit_up + up_lr_logit[:, :, y1:y2, x1:x2]
                ) / 2.0

        return fuse_logit

    def _hr_aux_loss(
        self,
        lr_feat: torch.Tensor,
        gt: torch.Tensor,
        ori_h: int,
        ori_w: int,
    ) -> torch.Tensor:
        """
        Compute HR auxiliary loss.

        Draws `crop_num` random sub-regions at the same random scale, crops the
        feature map at each, upsamples to full feature size, runs FCNHead, then
        computes loss against the corresponding sub-region of `gt`.

        Returns: scalar loss tensor (pre-multiplied by hr_loss_weight).
        """
        crop_h, crop_w = self._get_random_crop_size()

        hr_feats: list[torch.Tensor] = []
        hr_gts: list[torch.Tensor] = []

        for _ in range(self.crop_num):
            y1, y2, x1, x2 = _get_crop_bbox(ori_h, ori_w, crop_h, crop_w, self.divisible)
            hr_feats.append(_resizecrop_feat(lr_feat, ori_h, ori_w, y1, y2, x1, x2))
            hr_gts.append(_crop(gt, y1, y2, x1, x2))

        # Actual crop dimensions after clamping to image bounds
        actual_h = min(crop_h, ori_h)
        actual_w = min(crop_w, ori_w)

        # Batch all crops → single forward pass through FCNHead
        hr_feat_cat = torch.cat(hr_feats, dim=0)          # (B*crop_num, 720, H/4, W/4)
        hr_logit = self.head(hr_feat_cat)                  # (B*crop_num, C, H/4, W/4)

        # Upsample logit to the actual gt crop size
        hr_logit_up = F.interpolate(
            hr_logit, (actual_h, actual_w), mode="bilinear", align_corners=False
        )
        hr_gt_cat = torch.cat(hr_gts, dim=0)               # (B*crop_num, actual_h, actual_w)

        return self.hr_loss_weight * self.criterion(hr_logit_up, hr_gt_cat)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        gt: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:  (B, 3, H, W) — input image tensor, normalised to [0, 1].
            gt: (B, H, W)    — integer class mask; only used in training mode.

        Returns:
            Inference: (B, num_classes, H, W) logits.
            Training:  ((B, num_classes, H, W) logits,  scalar aux_loss).
        """
        ori_h, ori_w = x.shape[2], x.shape[3]

        # -- 1. Resize to hr_scale (no-op when input == hr_scale) --
        lr_x = F.interpolate(x, self.hr_scale, mode="bilinear", align_corners=False)

        # -- 2. Backbone + feature transform --
        multi_feats = self.backbone(lr_x)                   # list of 4 branch tensors
        lr_feat = self._transform_inputs(multi_feats)       # (B, 720, H/4, W/4)

        # -- 3. LR decode --
        lr_logit = self.head(lr_feat)                       # (B, C, H/4, W/4)
        up_lr_logit = F.interpolate(
            lr_logit, (ori_h, ori_w), mode="bilinear", align_corners=False
        )

        # -- 4. Sliding-window fuse --
        fuse_logit = self._slide_fuse(lr_feat, up_lr_logit, ori_h, ori_w)

        # -- Inference path --
        if not self.training or gt is None:
            return fuse_logit

        # -- 5. HR auxiliary loss (training only) --
        assert self.criterion is not None, (
            "HRDecoder requires `criterion` at __init__ for HR auxiliary loss. "
            "Pass criterion=<loss_fn> when constructing the model."
        )
        aux_loss = self._hr_aux_loss(lr_feat, gt, ori_h, ori_w)

        return fuse_logit, aux_loss
