from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paths import CHECKPOINT_PATH, OUTPUT_DIR
from dataset import NUM_CLASSES, FundusSegmentationDataset
from model.unet import UNet

PREDICT_OUTPUT_DIR = OUTPUT_DIR / "predictions"
NUM_SAMPLES = 8


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


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    colors = {
        0: (0, 0, 0),
        1: (255, 0, 0),
        2: (0, 255, 0),
        3: (0, 0, 255),
        4: (255, 255, 0),
    }
    colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_id, color in colors.items():
        colored[mask == cls_id] = color
    return colored


def save_prediction(image: torch.Tensor, pred_mask: np.ndarray, gt_mask: torch.Tensor, output_path: Path) -> None:
    image_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    pred_color = mask_to_color(pred_mask)
    gt_color = mask_to_color(gt_mask.numpy())

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_np)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(pred_color)
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    axes[2].imshow(gt_color)
    axes[2].set_title("Ground Truth")
    axes[2].axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def main() -> None:
    device = get_device()
    dataset = FundusSegmentationDataset(split="valid")
    model = load_model(CHECKPOINT_PATH, device)

    PREDICT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for idx in range(min(NUM_SAMPLES, len(dataset))):
        image, mask = dataset[idx]
        output = model(image.unsqueeze(0).to(device))
        pred = output.argmax(dim=1).squeeze(0).cpu().numpy()

        output_path = PREDICT_OUTPUT_DIR / f"prediction_{idx + 1}.png"
        save_prediction(image, pred, mask, output_path)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
