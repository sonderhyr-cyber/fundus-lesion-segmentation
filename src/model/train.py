from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

SRC_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paths import CHECKPOINT_PATH, OUTPUT_DIR
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


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _load_cfg(model_name: str) -> dict:
    """Load configs/{model_name}.yaml if it exists; return empty dict otherwise."""
    cfg_path = PROJECT_ROOT / "configs" / f"{model_name}.yaml"
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def build_model(
    model_name: str,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[nn.Module, Path]:
    """
    Instantiate a model by name.

    Returns (model, checkpoint_path).
    Supported names: 'unet', 'hrdecoder'.
    """
    cfg = _load_cfg(model_name)
    model_cfg = cfg.get("model", {})
    ckpt_cfg  = cfg.get("checkpoint", {})

    if model_name == "unet":
        model = UNet(in_channels=3, num_classes=NUM_CLASSES)
        ckpt_path = CHECKPOINT_PATH

    elif model_name == "hrdecoder":
        from baselines.hrdecoder import HRDecoder
        model = HRDecoder(
            num_classes=model_cfg.get("num_classes", NUM_CLASSES),
            pretrained=model_cfg.get("pretrained", True),
            hr_scale=tuple(model_cfg.get("hr_scale", [512, 512])),
            scale_ratio=tuple(model_cfg.get("scale_ratio", [0.75, 1.25])),
            hr_loss_weight=model_cfg.get("hr_loss_weight", 0.1),
            crop_num=model_cfg.get("crop_num", 2),
            divisible=model_cfg.get("divisible", 8),
            criterion=criterion,        # needed for HR auxiliary loss
        )
        ckpt_dir = PROJECT_ROOT / ckpt_cfg.get("dir", "outputs/checkpoints")
        ckpt_file = ckpt_cfg.get("filename", "hrdecoder_best.pth")
        ckpt_path = ckpt_dir / ckpt_file

    elif model_name == "m2mrf":
        from models.m2mrf import M2MRFNet
        ckpt_dir  = PROJECT_ROOT / ckpt_cfg.get("dir",      "outputs/checkpoints")
        ckpt_file = ckpt_cfg.get("filename", "m2mrf_best.pth")
        ckpt_path = ckpt_dir / ckpt_file
        model = M2MRFNet(
            num_classes=model_cfg.get("num_classes", NUM_CLASSES),
            variant=model_cfg.get("variant", "A"),
            stage2_channels=tuple(model_cfg.get("stage2_channels", [18, 36])),
            stage3_channels=tuple(model_cfg.get("stage3_channels", [18, 36, 72])),
            stage4_channels=tuple(model_cfg.get("stage4_channels", [18, 36, 72, 144])),
            m2mrf_patch_size=tuple(model_cfg.get("m2mrf_patch_size", [8, 8])),
            m2mrf_encode_channels_rate=model_cfg.get("m2mrf_encode_channels_rate", 4),
            m2mrf_fc_channels_rate=model_cfg.get("m2mrf_fc_channels_rate", 64),
        )

    else:
        raise ValueError(
            f"Unknown model '{model_name}'. Supported: 'unet', 'hrdecoder', 'm2mrf'."
        )

    return model.to(device), ckpt_path


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

    # Detect whether the model uses HR auxiliary training (e.g. HRDecoder).
    # Such models expose `hr_loss_weight` and expect `gt=masks` in forward().
    uses_hr_training: bool = hasattr(model, "hr_loss_weight")

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        if uses_hr_training:
            # HRDecoder: forward(x, gt=masks) → (fuse_logit, aux_loss)
            outputs = model(images, gt=masks)
            fuse_logit, aux_loss = outputs
            loss = criterion(fuse_logit, masks) + aux_loss
        else:
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
    parser = argparse.ArgumentParser(description="Fundus lesion segmentation training")
    parser.add_argument(
        "--model",
        type=str,
        default="unet",
        choices=["unet", "hrdecoder", "m2mrf"],
        help="Model architecture to train (default: unet)",
    )
    args = parser.parse_args()
    model_name = args.model

    # Load per-model config overrides (if configs/{model}.yaml exists)
    cfg = _load_cfg(model_name)
    train_cfg = cfg.get("training", {})

    batch_size          = train_cfg.get("batch_size", BATCH_SIZE)
    epochs              = train_cfg.get("epochs", EPOCHS)
    lr                  = train_cfg.get("learning_rate", LEARNING_RATE)
    epoch_size          = train_cfg.get("epoch_size", EPOCH_SIZE)
    fg_ratio            = train_cfg.get("fg_ratio", FG_RATIO)
    early_stop_patience = train_cfg.get("early_stop_patience", EARLY_STOP_PATIENCE)
    reset_training      = train_cfg.get("reset_training", RESET_TRAINING)

    device = get_device()
    print(f"Model: {model_name}  |  Device: {device}")

    train_dataset = ForegroundPatchDataset(
        split="train",
        epoch_size=epoch_size,
        fg_ratio=fg_ratio,
        transform=None,
    )
    val_dataset = FundusSegmentationDataset(split="valid")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    criterion = FocalTverskyLoss(num_classes=NUM_CLASSES)
    model, ckpt_path = build_model(model_name, criterion, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    print(f"Checkpoint path: {ckpt_path}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Resume from checkpoint unless reset_training is set.
    if not reset_training and ckpt_path.exists():
        start_epoch, best_val_loss = load_checkpoint(
            ckpt_path, model, optimizer, scheduler, device
        )
    else:
        if reset_training and ckpt_path.exists():
            print("reset_training=True — ignoring existing checkpoint, starting fresh.")
        start_epoch = 1
        best_val_loss = float("inf")

    if start_epoch > epochs:
        print(f"Already trained for {epochs} epochs. Done.")
        return

    epochs_no_improve = 0

    for epoch in range(start_epoch, epochs + 1):
        # Re-sample the patch index each epoch so the model sees different
        # crops every epoch rather than the same epoch_size patches repeatedly.
        train_dataset._build_epoch_index()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch}/{epochs}] "
            f"train loss: {train_loss:.4f}  val loss: {val_loss:.4f}  lr: {current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, scheduler, ckpt_path, epoch, val_loss)
            print(f"  Saved best model to {ckpt_path}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve}/{early_stop_patience} epochs.")
            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    print(f"Training finished. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
