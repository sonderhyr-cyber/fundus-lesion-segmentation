from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paths import CHECKPOINT_PATH, OUTPUT_DIR
from dataset import CLASS_NAMES, NUM_CLASSES, FundusSegmentationDataset
from model.unet import UNet

BATCH_SIZE = 4
RESULTS_PATH = OUTPUT_DIR / "evaluation_results.txt"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path: Path, device: torch.device) -> UNet:
    model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def compute_confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> np.ndarray:
    mask = (target >= 0) & (target < num_classes)
    pred = pred[mask]
    target = target[mask]
    indices = num_classes * target + pred
    return np.bincount(indices.cpu().numpy(), minlength=num_classes ** 2).reshape(num_classes, num_classes)


def compute_metrics(confusion: np.ndarray) -> dict[str, float | dict[int, float]]:
    total = confusion.sum()
    pixel_acc = float(np.trace(confusion) / total) if total > 0 else 0.0

    iou_per_class: dict[int, float] = {}
    for cls_id in range(NUM_CLASSES):
        tp = confusion[cls_id, cls_id]
        fp = confusion[:, cls_id].sum() - tp
        fn = confusion[cls_id, :].sum() - tp
        denom = tp + fp + fn
        iou_per_class[cls_id] = float(tp / denom) if denom > 0 else 0.0

    valid_ious = [iou for cls_id, iou in iou_per_class.items() if cls_id > 0]
    mean_iou = float(np.mean(valid_ious)) if valid_ious else 0.0

    return {
        "pixel_accuracy": pixel_acc,
        "mean_iou": mean_iou,
        "iou_per_class": iou_per_class,
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float | dict[int, float]]:
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        for pred, target in zip(preds, masks):
            confusion += compute_confusion_matrix(pred, target, NUM_CLASSES)

    return compute_metrics(confusion)


def save_results(split: str, metrics: dict[str, float | dict[int, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(f"Split: {split}\n")
        f.write(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}\n")
        f.write(f"Mean IoU (foreground): {metrics['mean_iou']:.4f}\n")
        for cls_id, iou in metrics["iou_per_class"].items():
            f.write(f"  IoU {CLASS_NAMES[cls_id]}: {iou:.4f}\n")
        f.write("\n")


def main() -> None:
    device = get_device()
    model = load_model(CHECKPOINT_PATH, device)

    if RESULTS_PATH.exists():
        RESULTS_PATH.unlink()

    for split in ("valid", "test"):
        dataset = FundusSegmentationDataset(split=split)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        metrics = evaluate(model, loader, device)
        save_results(split, metrics, RESULTS_PATH)

        print(f"=== {split} ===")
        print(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
        print(f"Mean IoU (foreground): {metrics['mean_iou']:.4f}")
        for cls_id, iou in metrics["iou_per_class"].items():
            print(f"  IoU {CLASS_NAMES[cls_id]}: {iou:.4f}")
        print()

    print(f"Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
