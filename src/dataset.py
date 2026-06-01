from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from paths import DATA_ROOT
IMAGE_SIZE = (512, 512)

NUM_CLASSES = 5  # background + MA/HE/EX/SE
CLASS_NAMES = {
    0: "background",
    1: "MA",
    2: "HE",
    3: "EX",
    4: "SE",
}
YOLO_TO_MASK = {0: 1, 1: 2, 2: 3, 3: 4}


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


def yolo_polygons_to_semantic_mask(
    polygons: list[tuple[int, np.ndarray]], img_h: int, img_w: int
) -> np.ndarray:
    """
    Convert YOLO polygons to a semantic segmentation mask.
    background=0, MA=1, HE=2, EX=3, SE=4
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for yolo_cls_id, points in polygons:
        mask_cls_id = YOLO_TO_MASK.get(yolo_cls_id)
        if mask_cls_id is None:
            continue
        cv2.fillPoly(mask, [points], mask_cls_id)
    return mask


def collect_samples(image_dir: Path, label_dir: Path) -> list[tuple[Path, Path]]:
    samples = []
    for image_path in sorted(image_dir.glob("*.jpg")):
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            samples.append((image_path, label_path))
    return samples


class FundusSegmentationDataset(Dataset):
    """PyTorch Dataset for fundus lesion semantic segmentation."""

    def __init__(
        self,
        root: str | Path = DATA_ROOT,
        split: str = "train",
        transform=None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform

        image_dir = self.root / "images" / split
        label_dir = self.root / "labels" / split
        self.samples = collect_samples(image_dir, label_dir)

        if not self.samples:
            raise FileNotFoundError(f"No image-label pairs found under {image_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, label_path = self.samples[idx]

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = image_rgb.shape[:2]

        polygons = parse_yolo_segmentation(label_path, img_h, img_w)
        mask = yolo_polygons_to_semantic_mask(polygons, img_h, img_w)

        image_rgb = cv2.resize(image_rgb, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)

        if self.transform is not None:
            transformed = self.transform(image=image_rgb, mask=mask)
            image_rgb = transformed["image"]
            mask = transformed["mask"]

        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask).long()

        return image_tensor, mask_tensor


if __name__ == "__main__":
    dataset = FundusSegmentationDataset(split="train")
    image, mask = dataset[0]

    print(f"Dataset size: {len(dataset)}")
    print(f"Image shape: {tuple(image.shape)}, dtype: {image.dtype}")
    print(f"Mask shape: {tuple(mask.shape)}, dtype: {mask.dtype}")
    print(f"Mask unique classes: {sorted(mask.unique().tolist())}")

