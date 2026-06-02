from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """
    Soft Dice loss over foreground classes.

    class_weights: length (num_classes - 1), one weight per foreground class
                   (order: MA, HE, EX, SE for a 5-class problem).
                   None → equal weights.
    """

    def __init__(
        self,
        num_classes: int,
        smooth: float = 1.0,
        class_weights: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        if class_weights is not None:
            assert len(class_weights) == num_classes - 1, (
                f"class_weights must have length {num_classes - 1}"
            )
            w = torch.tensor(class_weights, dtype=torch.float32)
            # Normalise so the weights sum to (num_classes - 1), keeping the
            # overall loss magnitude comparable to the unweighted version.
            self.register_buffer("weights", w / w.mean())
        else:
            self.register_buffer("weights", torch.ones(num_classes - 1))

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
        for i, c in enumerate(range(1, self.num_classes)):  # skip background
            p = probs[:, c].reshape(-1)
            t = targets_onehot[:, c].reshape(-1)
            intersection = (p * t).sum()
            dice_c = (2.0 * intersection + self.smooth) / (
                p.sum() + t.sum() + self.smooth
            )
            losses.append(self.weights[i] * (1.0 - dice_c))
        return torch.stack(losses).mean()


class CombinedDiceCELoss(nn.Module):
    """
    Weighted Dice + Weighted CrossEntropy loss.

    dice_class_weights : per-foreground-class weights for the Dice term
                         (length = num_classes - 1).  None → equal.
    ce_class_weights   : per-class weights for CrossEntropyLoss
                         (length = num_classes, includes background).  None → equal.
    """

    def __init__(
        self,
        num_classes: int,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        dice_class_weights: list[float] | None = None,
        ce_class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.dice = SoftDiceLoss(num_classes, class_weights=dice_class_weights)
        self.ce = nn.CrossEntropyLoss(weight=ce_class_weights)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.dice_weight * self.dice(logits, targets)
            + self.ce_weight * self.ce(logits, targets)
        )
