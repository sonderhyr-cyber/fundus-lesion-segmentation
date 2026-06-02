from __future__ import annotations

import random
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

_SRC_DIR = Path(__file__).resolve().parent.parent.parent  # src/
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from paths import OUTPUT_DIR
from dataset import CLASS_NAMES, NUM_CLASSES, FundusSegmentationDataset
from baselines.smp.evaluate import load_model
from baselines.smp.model import normalize_imagenet

BEST_MODEL_PATH = OUTPUT_DIR / "smp" / "checkpoints" / "best_model.pth"
PREDICTIONS_DIR = OUTPUT_DIR / "smp" / "predictions"
N_SAMPLES = 8
RANDOM_SEED = 42
OVERLAY_ALPHA = 0.5

# RGB colors per semantic class (matches visualize.py's BGR palette, converted)
CLASS_COLORS_RGB: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),       # background — black
    1: (255, 0, 0),     # MA         — red
    2: (0, 255, 0),     # HE         — green
    3: (0, 0, 255),     # EX         — blue
    4: (255, 255, 0),   # SE         — yellow
}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert an (H, W) integer mask to an (H, W, 3) RGB color image."""
    colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c, color in CLASS_COLORS_RGB.items():
        colored[mask == c] = color
    return colored


def make_overlay(image_rgb: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """Blend colorized prediction onto the original image over lesion pixels."""
    colored = colorize_mask(pred_mask)
    has_lesion = pred_mask > 0
    overlay = image_rgb.copy()
    blended = (
        image_rgb.astype(float) * (1 - OVERLAY_ALPHA)
        + colored.astype(float) * OVERLAY_ALPHA
    ).astype(np.uint8)
    overlay[has_lesion] = blended[has_lesion]
    return overlay


def save_prediction_figure(
    image_rgb: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(image_rgb)
    axes[0].set_title("Image", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(colorize_mask(gt_mask))
    axes[1].set_title("Ground Truth", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(colorize_mask(pred_mask))
    axes[2].set_title("Prediction", fontsize=11)
    axes[2].axis("off")

    axes[3].imshow(make_overlay(image_rgb, pred_mask))
    axes[3].set_title("Overlay", fontsize=11)
    axes[3].axis("off")

    legend_patches = [
        mpatches.Patch(
            color=tuple(c / 255 for c in CLASS_COLORS_RGB[cls_id]),
            label=CLASS_NAMES[cls_id],
        )
        for cls_id in range(1, NUM_CLASSES)
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=NUM_CLASSES - 1, frameon=False)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=(0, 0.06, 1, 0.95))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path.name}")


@torch.no_grad()
def main() -> None:
    device = get_device()
    model = load_model(BEST_MODEL_PATH, device)
    print(f"Loaded checkpoint: {BEST_MODEL_PATH}")

    dataset = FundusSegmentationDataset(split="valid")
    random.seed(RANDOM_SEED)
    indices = random.sample(range(len(dataset)), min(N_SAMPLES, len(dataset)))

    print(f"Generating {len(indices)} prediction visualizations → {PREDICTIONS_DIR}\n")

    for i, idx in enumerate(indices, start=1):
        image_tensor, mask_tensor = dataset[idx]

        # Original image as uint8 RGB (no normalization applied — display-ready)
        image_rgb = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        gt_mask = mask_tensor.numpy().astype(np.uint8)

        # Normalize and infer
        inp = normalize_imagenet(image_tensor.unsqueeze(0).to(device))
        pred_mask = model(inp).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        stem = dataset.samples[idx][0].stem
        output_path = PREDICTIONS_DIR / f"pred_{i:02d}_{stem}.png"
        save_prediction_figure(image_rgb, gt_mask, pred_mask, output_path, title=stem)

    print(f"\nAll predictions saved to {PREDICTIONS_DIR}")


if __name__ == "__main__":
    main()
