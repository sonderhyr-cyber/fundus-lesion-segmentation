from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_SRC_DIR = Path(__file__).resolve().parent.parent.parent  # src/
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from paths import OUTPUT_DIR
from dataset import NUM_CLASSES, ForegroundPatchDataset, FundusSegmentationDataset
from baselines.smp.model import build_smp_unet, normalize_imagenet
from baselines.smp.losses import CombinedDiceCELoss
from baselines.smp.evaluate import compute_metrics_from_loader

# ── Hyperparameters ──────────────────────────────────────────────────────────
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# ForegroundPatchDataset config — matches the existing UNet pipeline exactly,
# so the only variable between the two runs is the model architecture.
EPOCH_SIZE = 1500
FG_RATIO = 0.8

# Stop early if val Dice does not improve for this many consecutive epochs.
EARLY_STOP_PATIENCE = 15

# ── Class weights (derived from run-1 per-class Dice) ────────────────────────
# Dice weights: foreground only — order: MA, HE, EX, SE
# Higher weight → model pays more attention to that class.
# Inverse-Dice from run 1: MA=0.068, HE=0.174, EX=0.356, SE=0.118
# Soft inverse (square-root scaling to avoid over-penalising hard classes):
#   sqrt(1/Dice) → MA≈3.84, HE≈2.40, EX≈1.68, SE≈2.91  → normalised below
DICE_CLASS_WEIGHTS = [2.0, 1.2, 0.8, 1.5]   # MA, HE, EX, SE

# CE weights: all 5 classes (background kept low — already at 0.9963 Dice)
CE_CLASS_WEIGHTS = [0.5, 4.0, 2.0, 1.0, 3.0]  # bg, MA, HE, EX, SE

# ── Output paths ─────────────────────────────────────────────────────────────
SMP_DIR = OUTPUT_DIR / "smp"
CHECKPOINT_DIR = SMP_DIR / "checkpoints"
LOGS_DIR = SMP_DIR / "logs"
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"
TRAIN_LOG_PATH = LOGS_DIR / "training_log.csv"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_samples = 0

    for images, masks in loader:
        images = normalize_imagenet(images.to(device))
        masks = masks.to(device)

        optimizer.zero_grad()
        loss = criterion(model(images), masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        n_samples += images.size(0)

    return total_loss / max(n_samples, 1)


@torch.no_grad()
def validate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n_samples = 0

    for images, masks in loader:
        images = normalize_imagenet(images.to(device))
        masks = masks.to(device)
        loss = criterion(model(images), masks)
        total_loss += loss.item() * images.size(0)
        n_samples += images.size(0)

    return total_loss / max(n_samples, 1)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    path: Path,
    epoch: int,
    val_dice: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_dice": val_dice,
        },
        path,
    )


def append_log_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    device = get_device()
    print(f"Device: {device}")

    train_dataset = ForegroundPatchDataset(
        split="train",
        epoch_size=EPOCH_SIZE,
        fg_ratio=FG_RATIO,
    )
    val_dataset = FundusSegmentationDataset(split="valid")

    # Use num_workers > 0 only on CUDA; MPS/CPU can have fork issues with the cache.
    num_workers = 4 if device.type == "cuda" else 0
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = build_smp_unet(num_classes=NUM_CLASSES).to(device)
    ce_weights = torch.tensor(CE_CLASS_WEIGHTS, dtype=torch.float32).to(device)
    criterion = CombinedDiceCELoss(
        num_classes=NUM_CLASSES,
        dice_class_weights=DICE_CLASS_WEIGHTS,
        ce_class_weights=ce_weights,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # mode="max": scheduler reduces LR when val Dice stops increasing.
    # patience=7: wait 7 epochs before halving LR (run-1 used 3 — too aggressive).
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=7
    )

    best_val_dice = 0.0
    epochs_no_improve = 0

    print(f"Starting SMP UNet (ResNet34/ImageNet) training for {EPOCHS} epochs")
    print(f"Checkpoints → {BEST_MODEL_PATH}")
    print(f"Log         → {TRAIN_LOG_PATH}\n")

    for epoch in range(1, EPOCHS + 1):
        # Re-sample the patch index each epoch (same as the existing UNet pipeline).
        train_dataset._build_epoch_index()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_loss(model, val_loader, criterion, device)

        metrics = compute_metrics_from_loader(model, val_loader, device, NUM_CLASSES)
        val_dice = float(metrics["mean_dice"])
        val_miou = float(metrics["mean_iou"])

        scheduler.step(val_dice)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:03d}/{EPOCHS}]  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"dice={val_dice:.4f}  mIoU={val_miou:.4f}  "
            f"lr={current_lr:.2e}"
        )

        append_log_row(
            TRAIN_LOG_PATH,
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_dice": f"{val_dice:.6f}",
                "val_miou": f"{val_miou:.6f}",
                "lr": f"{current_lr:.2e}",
            },
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, scheduler, BEST_MODEL_PATH, epoch, val_dice)
            print(f"  ✓ Best model saved  (val_dice={val_dice:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    print(f"\nDone. Best val Dice: {best_val_dice:.4f}")
    print(f"Checkpoint : {BEST_MODEL_PATH}")
    print(f"Training log: {TRAIN_LOG_PATH}")


if __name__ == "__main__":
    main()
