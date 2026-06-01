import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from paths import DATA_ROOT, OUTPUT_DIR

IMAGE_DIR = DATA_ROOT / "images" / "train"
LABEL_DIR = DATA_ROOT / "labels" / "train"

NUM_SAMPLES = 8
RANDOM_SEED = 42
OVERLAY_ALPHA = 0.45

CLASS_NAMES = {0: "MA", 1: "HE", 2: "EX", 3: "SE"}
CLASS_COLORS_BGR = {
    0: (0, 0, 255),      # red
    1: (0, 255, 0),      # green
    2: (255, 0, 0),      # blue
    3: (0, 255, 255),    # yellow
}


def parse_yolo_segmentation(label_path: Path, img_h: int, img_w: int) -> list[tuple[int, np.ndarray]]:
    """Parse YOLO segmentation label: class_id x1 y1 x2 y2 ... (normalized)."""
    polygons = []
    with open(label_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            cls_id = int(float(parts[0]))
            coords = list(map(float, parts[1:]))
            if len(coords) % 2 != 0:
                coords = coords[:-1]
            points = np.array(
                [[int(coords[i] * img_w), int(coords[i + 1] * img_h)] for i in range(0, len(coords), 2)],
                dtype=np.int32,
            )
            if len(points) >= 3:
                polygons.append((cls_id, points))
    return polygons


def polygons_to_masks(
    polygons: list[tuple[int, np.ndarray]], img_h: int, img_w: int
) -> tuple[np.ndarray, np.ndarray]:
    """Convert polygons to class index mask and colored BGR mask."""
    class_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    colored_mask = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    for cls_id, points in polygons:
        if cls_id not in CLASS_COLORS_BGR:
            continue
        cv2.fillPoly(class_mask, [points], cls_id + 1)
        cv2.fillPoly(colored_mask, [points], CLASS_COLORS_BGR[cls_id])

    return class_mask, colored_mask


def create_overlay(image_bgr: np.ndarray, colored_mask: np.ndarray, alpha: float = OVERLAY_ALPHA) -> np.ndarray:
    """Blend colored mask onto the original image."""
    mask_region = colored_mask.sum(axis=2) > 0
    overlay = image_bgr.copy()
    blended = cv2.addWeighted(image_bgr, 1 - alpha, colored_mask, alpha, 0)
    overlay[mask_region] = blended[mask_region]
    return overlay


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def collect_image_label_pairs(image_dir: Path, label_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for image_path in sorted(image_dir.glob("*.jpg")):
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            pairs.append((image_path, label_path))
    return pairs


def save_visualization(
    image_bgr: np.ndarray,
    colored_mask: np.ndarray,
    overlay_bgr: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(bgr_to_rgb(image_bgr))
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(bgr_to_rgb(colored_mask))
    axes[1].set_title("Mask")
    axes[1].axis("off")

    axes[2].imshow(bgr_to_rgb(overlay_bgr))
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    legend_patches = [
        mpatches.Patch(
            color=tuple(c / 255 for c in reversed(CLASS_COLORS_BGR[cls_id])),
            label=f"{cls_id}: {CLASS_NAMES[cls_id]}",
        )
        for cls_id in sorted(CLASS_NAMES)
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=(0, 0.06, 1, 0.95))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    pairs = collect_image_label_pairs(IMAGE_DIR, LABEL_DIR)
    if not pairs:
        raise FileNotFoundError(f"No image-label pairs found under {IMAGE_DIR}")

    random.seed(RANDOM_SEED)
    samples = random.sample(pairs, min(NUM_SAMPLES, len(pairs)))

    for idx, (image_path, label_path) in enumerate(samples, start=1):
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"Skip {image_path.name}: failed to read image")
            continue

        img_h, img_w = image_bgr.shape[:2]
        polygons = parse_yolo_segmentation(label_path, img_h, img_w)
        _, colored_mask = polygons_to_masks(polygons, img_h, img_w)
        overlay_bgr = create_overlay(image_bgr, colored_mask)

        output_path = OUTPUT_DIR / f"sample_{idx}_visualization.png"
        save_visualization(
            image_bgr=image_bgr,
            colored_mask=colored_mask,
            overlay_bgr=overlay_bgr,
            output_path=output_path,
            title=image_path.name,
        )
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
