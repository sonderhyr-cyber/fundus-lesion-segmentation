from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for heavily imbalanced segmentation.
    Abraham & Khan, IEEE ISBI 2019 (arXiv:1810.07842).

    For each foreground class c:
        TP = sum(pred_c * target_c)
        FP = sum(pred_c * (1 - target_c))
        FN = sum((1 - pred_c) * target_c)
        TI_c = (TP + smooth) / (TP + α·FP + β·FN + smooth)
        FTL_c = (1 - TI_c)^γ

    Final loss = mean over foreground classes (background always excluded).

    α=0.3, β=0.7 → penalises missed lesions (FN) more than false alarms (FP).
    γ=0.75       → moderate focus on hard (foreground) examples.
    """

    def __init__(
        self,
        num_classes: int,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 0.75,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.alpha  = alpha
        self.beta   = beta
        self.gamma  = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits:  [B, C, H, W]
        # targets: [B, H, W]  integer class indices
        probs = F.softmax(logits, dim=1)

        targets_onehot = (
            F.one_hot(targets, self.num_classes)  # [B, H, W, C]
            .permute(0, 3, 1, 2)                  # [B, C, H, W]
            .float()
        )

        losses: list[torch.Tensor] = []
        for c in range(1, self.num_classes):      # skip background (class 0)
            pred_c   = probs[:, c].reshape(-1)
            target_c = targets_onehot[:, c].reshape(-1)

            tp = (pred_c * target_c).sum()
            fp = (pred_c * (1.0 - target_c)).sum()
            fn = ((1.0 - pred_c) * target_c).sum()

            ti = (tp + self.smooth) / (
                tp + self.alpha * fp + self.beta * fn + self.smooth
            )
            losses.append((1.0 - ti) ** self.gamma)

        return torch.stack(losses).mean()


class DiceLoss(nn.Module):
    """
    Soft Dice Loss over foreground classes only (background excluded).

    For each foreground class c:
        Dice_c = (2 * sum(pred_c * target_c) + smooth) /
                 (sum(pred_c) + sum(target_c) + smooth)
        Loss_c = 1 - Dice_c

    Final loss = mean over all foreground classes.
    """

    def __init__(self, num_classes: int, smooth: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits:  [B, C, H, W]
        # targets: [B, H, W]  (integer class indices)
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets → [B, C, H, W]
        targets_onehot = (
            F.one_hot(targets, self.num_classes)   # [B, H, W, C]
            .permute(0, 3, 1, 2)                   # [B, C, H, W]
            .float()
        )

        dice_losses: list[torch.Tensor] = []
        for c in range(1, self.num_classes):       # skip background (class 0)
            pred_c = probs[:, c].reshape(-1)
            target_c = targets_onehot[:, c].reshape(-1)
            intersection = (pred_c * target_c).sum()
            denom = pred_c.sum() + target_c.sum()
            dice_c = (2.0 * intersection + self.smooth) / (denom + self.smooth)
            dice_losses.append(1.0 - dice_c)

        return torch.stack(dice_losses).mean()


class DiceCELoss(nn.Module):
    """
    Combined Dice + Weighted CrossEntropy loss.

        Loss = dice_weight * DiceLoss + ce_weight * CrossEntropyLoss

    Dice optimises foreground overlap directly (class-imbalance-robust).
    Weighted CE provides stable per-pixel gradients across all classes.
    """

    def __init__(
        self,
        num_classes: int,
        class_weights: torch.Tensor | None = None,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.dice = DiceLoss(num_classes)
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.dice_weight * self.dice(logits, targets)
            + self.ce_weight * self.ce(logits, targets)
        )
