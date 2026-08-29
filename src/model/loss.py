from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropySegLoss(nn.Module):
    """Cross entropy wrapper with the same ``(logits, targets)`` interface."""

    def __init__(self, class_weights: torch.Tensor | None = None) -> None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, targets)


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


class CompositeLoss(nn.Module):
    """Weighted sum of named segmentation losses.

    Keeping the components in a ``ModuleDict`` makes every sub-loss visible in
    checkpoints and ensures tensor buffers such as CE class weights move with
    ``criterion.to(device)``.
    """

    def __init__(
        self,
        components: dict[str, nn.Module],
        weights: dict[str, float],
        normalize_weights: bool = True,
    ) -> None:
        super().__init__()
        if not components:
            raise ValueError("CompositeLoss needs at least one component")
        if set(components) != set(weights):
            raise ValueError("CompositeLoss component and weight names must match")
        if any(weight < 0 for weight in weights.values()):
            raise ValueError("CompositeLoss weights must be non-negative")

        total = float(sum(weights.values()))
        if total <= 0:
            raise ValueError("At least one CompositeLoss weight must be positive")

        self.components = nn.ModuleDict(components)
        divisor = total if normalize_weights else 1.0
        self.weights = {name: float(weight) / divisor for name, weight in weights.items()}

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        losses = [
            self.weights[name] * component(logits, targets)
            for name, component in self.components.items()
        ]
        return torch.stack(losses).sum()


def build_loss(loss_cfg: dict | None, num_classes: int) -> nn.Module:
    """Build a loss from YAML-compatible configuration.

    Supported names are ``focal_tversky``, ``dice``, ``cross_entropy``,
    ``dice_ce`` and ``composite``.  Composite configuration uses a list under
    ``components``; each item has its own ``name``, ``weight`` and parameters.
    """
    cfg = dict(loss_cfg or {})
    name = str(cfg.pop("name", "focal_tversky")).lower()

    if name == "focal_tversky":
        return FocalTverskyLoss(
            num_classes=num_classes,
            alpha=float(cfg.get("alpha", 0.3)),
            beta=float(cfg.get("beta", 0.7)),
            gamma=float(cfg.get("gamma", 0.75)),
            smooth=float(cfg.get("smooth", 1.0)),
        )
    if name == "dice":
        return DiceLoss(num_classes=num_classes, smooth=float(cfg.get("smooth", 1.0)))
    if name == "cross_entropy":
        raw_weights = cfg.get("class_weights")
        class_weights = None if raw_weights is None else torch.tensor(raw_weights, dtype=torch.float32)
        if class_weights is not None and class_weights.numel() != num_classes:
            raise ValueError(
                f"cross_entropy class_weights needs {num_classes} values, "
                f"got {class_weights.numel()}"
            )
        return CrossEntropySegLoss(class_weights)
    if name == "dice_ce":
        raw_weights = cfg.get("class_weights")
        class_weights = None if raw_weights is None else torch.tensor(raw_weights, dtype=torch.float32)
        return DiceCELoss(
            num_classes=num_classes,
            class_weights=class_weights,
            dice_weight=float(cfg.get("dice_weight", 0.5)),
            ce_weight=float(cfg.get("ce_weight", 0.5)),
        )
    if name == "composite":
        raw_components = cfg.get("components", [])
        components: dict[str, nn.Module] = {}
        weights: dict[str, float] = {}
        for index, raw_component in enumerate(raw_components):
            component_cfg = dict(raw_component)
            weight = float(component_cfg.pop("weight", 1.0))
            component_name = str(component_cfg.get("name", "")).lower()
            key = f"{index}_{component_name}"
            components[key] = build_loss(component_cfg, num_classes)
            weights[key] = weight
        return CompositeLoss(
            components,
            weights,
            normalize_weights=bool(cfg.get("normalize_weights", True)),
        )

    raise ValueError(
        f"Unknown loss '{name}'. Supported: focal_tversky, dice, "
        "cross_entropy, dice_ce, composite."
    )
