from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """Soft Dice loss averaged over foreground classes (background excluded)."""

    def __init__(self, num_classes: int, smooth: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits:  [B, C, H, W]
        # targets: [B, H, W] integer class indices
        probs = F.softmax(logits, dim=1)
        targets_onehot = (
            F.one_hot(targets, self.num_classes)  # [B, H, W, C]
            .permute(0, 3, 1, 2)                  # [B, C, H, W]
            .float()
        )
        losses: list[torch.Tensor] = []
        for c in range(1, self.num_classes):       # skip background (class 0)
            p = probs[:, c].reshape(-1)
            t = targets_onehot[:, c].reshape(-1)
            intersection = (p * t).sum()
            dice_c = (2.0 * intersection + self.smooth) / (
                p.sum() + t.sum() + self.smooth
            )
            losses.append(1.0 - dice_c)
        return torch.stack(losses).mean()


class CombinedDiceCELoss(nn.Module):
    """
    50/50 Dice + CrossEntropy loss.

    Dice handles foreground class imbalance.
    CrossEntropy provides stable per-pixel gradients including background.
    """

    def __init__(
        self,
        num_classes: int,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.dice = SoftDiceLoss(num_classes)
        self.ce = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.dice_weight * self.dice(logits, targets)
            + self.ce_weight * self.ce(logits, targets)
        )
