from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paths import CHECKPOINT_PATH
from dataset import NUM_CLASSES, ForegroundPatchDataset, FundusSegmentationDataset
from model.unet import UNet
from model.loss import FocalTverskyLoss

BATCH_SIZE    = 4
EPOCHS        = 100
LEARNING_RATE = 3e-4

# Patches sampled per epoch from the foreground-biased patch dataset.
# 80 % of patches are centred on a foreground polygon (native resolution crop).
EPOCH_SIZE = 1500
FG_RATIO   = 0.8

# Early stopping: halt training if val loss does not improve for this many epochs.
EARLY_STOP_PATIENCE = 20

# Set to True to ignore an existing checkpoint and train from scratch.
# Needed when the saved model was trained with a different strategy.
RESET_TRAINING = True


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

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)
        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    path: Path,
    epoch: int,
    val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> tuple[int, float]:
    """Load checkpoint and return (start_epoch, best_val_loss)."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint["val_loss"]
    print(f"Resumed from epoch {checkpoint['epoch']}, best val loss: {best_val_loss:.4f}")
    return start_epoch, best_val_loss


def main() -> None:
    device = get_device()
    print(f"Device: {device}")

    train_dataset = ForegroundPatchDataset(
        split="train",
        epoch_size=EPOCH_SIZE,
        fg_ratio=FG_RATIO,
        transform=None,
    )
    val_dataset = FundusSegmentationDataset(split="valid")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(device)
    criterion = FocalTverskyLoss(num_classes=NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Resume from checkpoint unless RESET_TRAINING is set.
    if not RESET_TRAINING and CHECKPOINT_PATH.exists():
        start_epoch, best_val_loss = load_checkpoint(
            CHECKPOINT_PATH, model, optimizer, scheduler, device
        )
    else:
        if RESET_TRAINING and CHECKPOINT_PATH.exists():
            print("RESET_TRAINING=True — ignoring existing checkpoint, starting fresh.")
        start_epoch = 1
        best_val_loss = float("inf")

    if start_epoch > EPOCHS:
        print(f"Already trained for {EPOCHS} epochs. Done.")
        return

    epochs_no_improve = 0

    for epoch in range(start_epoch, EPOCHS + 1):
        # Re-sample the patch index each epoch so the model sees different
        # crops every epoch rather than the same EPOCH_SIZE patches repeatedly.
        train_dataset._build_epoch_index()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch}/{EPOCHS}] train loss: {train_loss:.4f}  val loss: {val_loss:.4f}  lr: {current_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, scheduler, CHECKPOINT_PATH, epoch, val_loss)
            print(f"  Saved best model to {CHECKPOINT_PATH}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve}/{EARLY_STOP_PATIENCE} epochs.")
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    print(f"Training finished. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
