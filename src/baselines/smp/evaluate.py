from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_SRC_DIR = Path(__file__).resolve().parent.parent.parent  # src/
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from paths import OUTPUT_DIR
from dataset import CLASS_NAMES, NUM_CLASSES, FundusSegmentationDataset
from baselines.smp.model import build_smp_unet, normalize_imagenet

BATCH_SIZE = 4
SMP_DIR = OUTPUT_DIR / "smp"
BEST_MODEL_PATH = SMP_DIR / "checkpoints" / "best_model.pth"
RESULTS_PATH = SMP_DIR / "evaluation_results.txt"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    model = build_smp_unet(num_classes=NUM_CLASSES).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def compute_metrics_from_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict[str, float | dict[int, float]]:
    """
    Compute Dice and IoU per class over an entire DataLoader.

    Returns a dict with keys:
      mean_dice, mean_iou (averaged over foreground classes)
      dice_per_class, iou_per_class (dicts keyed by class id)
    """
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    model.eval()
    for images, masks in loader:
        images = normalize_imagenet(images.to(device))
        masks = masks.to(device)

        preds = model(images).argmax(dim=1)  # [B, H, W]

        valid = (masks >= 0) & (masks < num_classes)
        flat_pred = preds[valid].cpu().numpy()
        flat_mask = masks[valid].cpu().numpy()
        indices = num_classes * flat_mask + flat_pred
        confusion += np.bincount(indices, minlength=num_classes**2).reshape(
            num_classes, num_classes
        )

    dice_per_class: dict[int, float] = {}
    iou_per_class: dict[int, float] = {}

    for c in range(num_classes):
        tp = int(confusion[c, c])
        fp = int(confusion[:, c].sum()) - tp
        fn = int(confusion[c, :].sum()) - tp
        denom_dice = 2 * tp + fp + fn
        denom_iou = tp + fp + fn
        dice_per_class[c] = (2 * tp / denom_dice) if denom_dice > 0 else 0.0
        iou_per_class[c] = (tp / denom_iou) if denom_iou > 0 else 0.0

    fg_dice = [dice_per_class[c] for c in range(1, num_classes)]
    fg_iou = [iou_per_class[c] for c in range(1, num_classes)]

    return {
        "mean_dice": float(np.mean(fg_dice)) if fg_dice else 0.0,
        "mean_iou": float(np.mean(fg_iou)) if fg_iou else 0.0,
        "dice_per_class": dice_per_class,
        "iou_per_class": iou_per_class,
    }


def print_and_save_metrics(
    split: str,
    metrics: dict,
    results_path: Path,
) -> None:
    lines = [
        f"=== {split} ===",
        f"Mean Dice (foreground): {metrics['mean_dice']:.4f}",
        f"Mean IoU  (foreground): {metrics['mean_iou']:.4f}",
        "",
        f"  {'Class':<12}  {'Dice':>6}  {'IoU':>6}",
        f"  {'-'*12}  {'------':>6}  {'------':>6}",
    ]
    for c in range(NUM_CLASSES):
        name = CLASS_NAMES[c]
        dice = metrics["dice_per_class"][c]
        iou = metrics["iou_per_class"][c]
        lines.append(f"  {name:<12}  {dice:>6.4f}  {iou:>6.4f}")
    lines.append("")

    text = "\n".join(lines)
    print(text)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def main() -> None:
    device = get_device()
    model = load_model(BEST_MODEL_PATH, device)
    print(f"Loaded checkpoint: {BEST_MODEL_PATH}")

    if RESULTS_PATH.exists():
        RESULTS_PATH.unlink()

    for split in ("valid", "test"):
        dataset = FundusSegmentationDataset(split=split)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        metrics = compute_metrics_from_loader(model, loader, device, NUM_CLASSES)
        print_and_save_metrics(split, metrics, RESULTS_PATH)

    print(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
