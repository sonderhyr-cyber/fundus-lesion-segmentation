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
from dataset import NUM_CLASSES, FundusSegmentationDataset, GeometricTransform
from model.unet import UNet
from model.loss import DiceCELoss

BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 3e-4

# Geometric-only augmentation (flip + rotate90), numpy-based, no extra deps.
# Colour jitter excluded — retinal image colour carries diagnostic meaning.
TRAIN_TRANSFORM = GeometricTransform(p=0.5)

# Median-frequency class weights for the CE component of DiceCELoss.
# freq[c] = pixel_count[c] / total_pixels  (from outputs/dataset_class_stats.csv)
# weight[c] = median(freq) / freq[c]
CLASS_WEIGHTS = torch.tensor([0.0027, 7.37, 0.53, 1.00, 2.80])


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


def save_checkpoint(model: nn.Module, path: Path, epoch: int, val_loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_loss": val_loss,
        },
        path,
    )


def main() -> None:
    device = get_device()
    print(f"Device: {device}")

    train_dataset = FundusSegmentationDataset(split="train", transform=TRAIN_TRANSFORM)
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
    criterion = DiceCELoss(
        num_classes=NUM_CLASSES,
        class_weights=CLASS_WEIGHTS.to(device),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch}/{EPOCHS}] train loss: {train_loss:.4f}  val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, CHECKPOINT_PATH, epoch, val_loss)
            print(f"  Saved best model to {CHECKPOINT_PATH}")

    print(f"Training finished. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
