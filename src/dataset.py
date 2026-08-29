from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from paths import DATA_ROOT

IMAGE_SIZE = (512, 512)

# CLAHE preprocessing parameters
# Applied to the L (luminance) channel in LAB space before training.
# clip_limit controls contrast amplification ceiling (prevents noise blow-up).
# tile_size defines the local block size for adaptive equalization.
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = 8
GAMMA = 0.8  # < 1 brightens dark regions, enhancing low-contrast lesions (MA)
BLACK_BORDER_THRESHOLD = 10
BLACK_BORDER_PADDING_RATIO = 0.02


def find_fundus_bbox(
    image_rgb: np.ndarray,
    threshold: int = BLACK_BORDER_THRESHOLD,
    padding_ratio: float = BLACK_BORDER_PADDING_RATIO,
) -> tuple[int, int, int, int]:
    """Return the padded bounding box of the largest non-black image region.

    The maximum RGB channel is thresholded so dim red retinal pixels are kept.
    Selecting the largest connected component rejects isolated text/camera
    artefacts outside the circular field of view.  Coordinates are returned as
    ``(x1, y1, x2, y2)`` with exclusive ``x2/y2`` bounds.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image [H,W,3], got {image_rgb.shape}")

    height, width = image_rgb.shape[:2]
    non_black = (image_rgb.max(axis=2) > threshold).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(non_black, connectivity=8)
    if count <= 1:
        return 0, 0, width, height

    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x = int(stats[component, cv2.CC_STAT_LEFT])
    y = int(stats[component, cv2.CC_STAT_TOP])
    box_w = int(stats[component, cv2.CC_STAT_WIDTH])
    box_h = int(stats[component, cv2.CC_STAT_HEIGHT])

    # Reject implausibly small components instead of producing a destructive crop.
    if box_w * box_h < 0.1 * width * height:
        return 0, 0, width, height

    pad = int(round(max(box_w, box_h) * padding_ratio))
    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + box_w + pad, width)
    y2 = min(y + box_h + pad, height)
    return x1, y1, x2, y2


def crop_black_border(
    image_rgb: np.ndarray,
    mask: np.ndarray | None = None,
    threshold: int = BLACK_BORDER_THRESHOLD,
    padding_ratio: float = BLACK_BORDER_PADDING_RATIO,
) -> tuple[np.ndarray, np.ndarray | None, tuple[int, int, int, int]]:
    """Crop image and optional mask to the same automatically detected fundus ROI."""
    x1, y1, x2, y2 = find_fundus_bbox(image_rgb, threshold, padding_ratio)
    image_crop = image_rgb[y1:y2, x1:x2]
    mask_crop = None if mask is None else mask[y1:y2, x1:x2]
    return image_crop, mask_crop, (x1, y1, x2, y2)


def _shift_polygons_to_crop(
    polygons: list[tuple[int, np.ndarray]],
    bbox: tuple[int, int, int, int],
) -> list[tuple[int, np.ndarray]]:
    """Translate polygon coordinates into a cropped image coordinate system."""
    x1, y1, x2, y2 = bbox
    crop_w, crop_h = x2 - x1, y2 - y1
    shifted: list[tuple[int, np.ndarray]] = []
    for class_id, points in polygons:
        new_points = points.copy()
        new_points[:, 0] -= x1
        new_points[:, 1] -= y1
        new_points[:, 0] = np.clip(new_points[:, 0], 0, max(crop_w - 1, 0))
        new_points[:, 1] = np.clip(new_points[:, 1], 0, max(crop_h - 1, 0))
        if len(np.unique(new_points, axis=0)) >= 3:
            shifted.append((class_id, new_points))
    return shifted


def apply_clahe_gamma(image_rgb: np.ndarray) -> np.ndarray:
    """
    Enhance retinal fundus image contrast.

    Steps (following the preprocessing pipeline from the literature):
    1. Convert to LAB and apply CLAHE to the L channel only —
       improves local contrast without distorting colour.
    2. Apply gamma correction (γ=0.8) to brighten dark lesion regions.

    Input/output: uint8 RGB image, same shape.
    """
    # Step 1: CLAHE on luminance channel
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=(CLAHE_TILE_SIZE, CLAHE_TILE_SIZE),
    )
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Step 2: Gamma correction — output = (input/255)^γ * 255
    enhanced = np.clip(
        ((enhanced / 255.0) ** GAMMA) * 255.0, 0, 255
    ).astype(np.uint8)

    return enhanced


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
        native_mask: bool = False,
        remove_black_border: bool = False,
        black_border_threshold: int = BLACK_BORDER_THRESHOLD,
        black_border_padding_ratio: float = BLACK_BORDER_PADDING_RATIO,
    ) -> None:
        """
        Args:
            native_mask: if True, the returned mask keeps the image's ORIGINAL
                resolution (no 512x512 resize) while the image tensor is still
                resized to 512x512 for the network. This is the evaluation
                protocol used by M2MRF / HRDecoder: predictions are upsampled
                back to the original resolution before metrics are computed,
                so that micro-lesions (MA) are not destroyed by downsampling.

                NOTE: masks then have varying shapes across samples, so a
                DataLoader using this mode MUST use batch_size=1 (the default
                collate cannot stack ragged tensors).
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.native_mask = native_mask
        self.remove_black_border = remove_black_border
        self.black_border_threshold = black_border_threshold
        self.black_border_padding_ratio = black_border_padding_ratio

        if native_mask and transform is not None:
            raise ValueError(
                "native_mask=True is for evaluation only and cannot be combined "
                "with a transform (image and mask no longer share a shape)."
            )

        split_dir = "val" if split == "valid" and not (self.root / "images" / split).exists() else split
        image_dir = self.root / "images" / split_dir
        label_dir = self.root / "labels" / split_dir
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

        if self.remove_black_border:
            image_rgb, cropped_mask, _ = crop_black_border(
                image_rgb,
                mask,
                threshold=self.black_border_threshold,
                padding_ratio=self.black_border_padding_ratio,
            )
            assert cropped_mask is not None
            mask = cropped_mask

        image_rgb = apply_clahe_gamma(image_rgb)
        image_rgb = cv2.resize(image_rgb, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)

        if self.native_mask:
            # Evaluation protocol: keep GT at original resolution.
            # The model still sees a 512x512 input; evaluate.py upsamples the
            # logits back to (img_h, img_w) before computing metrics.
            image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
            mask_tensor = torch.from_numpy(mask).long()
            return image_tensor, mask_tensor

        mask = cv2.resize(mask, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)

        if self.transform is not None:
            transformed = self.transform(image=image_rgb, mask=mask)
            image_rgb = transformed["image"]
            mask = transformed["mask"]

        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask).long()

        return image_tensor, mask_tensor


class GeometricTransform:
    """
    Lightweight geometric augmentation using numpy only (no albumentations).

    Applied identically to image and mask to keep them in sync.
    Operations: horizontal flip, vertical flip, random 90-degree rotation.
    Each applied independently with probability p.
    """

    def __init__(self, p: float = 0.5, seed: int | None = None) -> None:
        self.p = p
        self.rng = np.random.default_rng(seed)

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
        if self.rng.random() < self.p:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
        if self.rng.random() < self.p:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()
        if self.rng.random() < self.p:
            k = self.rng.integers(1, 4)          # rotate 90, 180, or 270 degrees
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()
        return {"image": image, "mask": mask}


class ForegroundPatchDataset(Dataset):
    """
    Patch-based dataset that crops 512×512 patches from images at their native
    resolution, with foreground-biased centre sampling.

    Why patches instead of full-image resize?
    - Native resolution preserves micro-lesion detail (MA ≈ 5–20 px in a
      ~1500 px image; after resize to 512 it can collapse to 0–1 px).
    - Foreground bias forces the model to see lesion pixels on most steps.

    Sampling strategy per patch:
    - With probability `fg_ratio`: pick a random foreground polygon centroid
      as the patch centre (weighted by polygon area so larger lesions are
      sampled proportionally more).
    - Otherwise: pick a uniformly random centre in the image.
    The centre is clamped so the 512×512 window never goes out of bounds.
    """

    PATCH_SIZE = 512

    def __init__(
        self,
        root: str | Path = DATA_ROOT,
        split: str = "train",
        epoch_size: int = 1500,
        fg_ratio: float = 0.8,
        transform=None,
        remove_black_border: bool = False,
        black_border_threshold: int = BLACK_BORDER_THRESHOLD,
        black_border_padding_ratio: float = BLACK_BORDER_PADDING_RATIO,
        seed: int = 42,
    ) -> None:
        self.root       = Path(root)
        self.split      = split
        self.epoch_size = epoch_size
        self.fg_ratio   = fg_ratio
        self.transform  = transform
        self.remove_black_border = remove_black_border
        self.black_border_threshold = black_border_threshold
        self.black_border_padding_ratio = black_border_padding_ratio
        self.rng = np.random.default_rng(seed)
        self.half       = self.PATCH_SIZE // 2

        split_dir = "val" if split == "valid" and not (self.root / "images" / split).exists() else split
        image_dir = self.root / "images" / split_dir
        label_dir = self.root / "labels" / split_dir
        raw_samples = collect_samples(image_dir, label_dir)
        if not raw_samples:
            raise FileNotFoundError(f"No image-label pairs found under {image_dir}")

        # Pre-parse polygons so __getitem__ only needs to read the image.
        # fg_index: flat list of (img_path, lbl_path, polygon_idx) for each
        # foreground polygon across all images.
        self._samples: list[tuple[Path, Path]] = []
        self._polygons: list[list[tuple[int, np.ndarray]]] = []
        self._fg_index: list[tuple[int, int]] = []  # (sample_idx, polygon_idx)

        # Cache all images and masks in memory so __getitem__ never hits disk.
        # 338 train images at native resolution ≈ 1–2 GB RAM — acceptable on AutoDL.
        self._image_cache: list[np.ndarray] = []  # RGB uint8
        self._mask_cache:  list[np.ndarray] = []  # uint8 semantic mask

        print(f"Loading {len(raw_samples)} images into memory cache...")
        for s_idx, (img_path, lbl_path) in enumerate(raw_samples):
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            polygons = parse_yolo_segmentation(lbl_path, h, w)
            mask = yolo_polygons_to_semantic_mask(polygons, h, w)
            if self.remove_black_border:
                img_rgb, cropped_mask, bbox = crop_black_border(
                    img_rgb,
                    mask,
                    threshold=self.black_border_threshold,
                    padding_ratio=self.black_border_padding_ratio,
                )
                assert cropped_mask is not None
                mask = cropped_mask
                polygons = _shift_polygons_to_crop(polygons, bbox)
            img_rgb = apply_clahe_gamma(img_rgb)
            self._samples.append((img_path, lbl_path))
            self._polygons.append(polygons)
            self._image_cache.append(img_rgb)
            self._mask_cache.append(mask)
            for p_idx, _ in enumerate(polygons):
                self._fg_index.append((len(self._samples) - 1, p_idx))
        print(f"Cache ready: {len(self._image_cache)} images loaded.")

        if not self._fg_index:
            raise RuntimeError("No foreground polygons found — check label files.")

        # Pre-build the epoch index (length = epoch_size) so DataLoader's
        # shuffle gives consistent batches without re-sampling every step.
        self._build_epoch_index()

    # ------------------------------------------------------------------
    def _build_epoch_index(self) -> None:
        """Build (sample_idx, centre_x, centre_y | None) list for one epoch."""
        n_fg = int(self.epoch_size * self.fg_ratio)
        n_bg = self.epoch_size - n_fg

        epoch: list[tuple[int, int | None, int | None]] = []

        # foreground entries
        chosen_fg = self.rng.choice(len(self._fg_index), size=n_fg, replace=True)
        for ci in chosen_fg:
            s_idx, p_idx = self._fg_index[ci]
            _, pts = self._polygons[s_idx][p_idx]
            cx = int(pts[:, 0].mean())
            cy = int(pts[:, 1].mean())
            epoch.append((s_idx, cx, cy))

        # background entries (random crop from any image)
        chosen_bg = self.rng.integers(0, len(self._samples), size=n_bg)
        for s_idx in chosen_bg:
            image = self._image_cache[int(s_idx)]
            img_h, img_w = image.shape[:2]
            cx = int(self.rng.integers(self.half, max(self.half + 1, img_w - self.half)))
            cy = int(self.rng.integers(self.half, max(self.half + 1, img_h - self.half)))
            epoch.append((int(s_idx), cx, cy))

        self.rng.shuffle(epoch)
        self._epoch_index = epoch

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.epoch_size

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s_idx, hint_cx, hint_cy = self._epoch_index[idx]

        image_rgb = self._image_cache[s_idx]
        mask      = self._mask_cache[s_idx]
        img_h, img_w = image_rgb.shape[:2]

        # Determine patch centre
        half = self.half
        if hint_cx is not None:
            cx = int(np.clip(hint_cx, half, img_w - half))
            cy = int(np.clip(hint_cy, half, img_h - half))
        else:
            cx = int(np.random.randint(half, max(half + 1, img_w - half)))
            cy = int(np.random.randint(half, max(half + 1, img_h - half)))

        x1, y1 = cx - half, cy - half
        x2, y2 = x1 + self.PATCH_SIZE, y1 + self.PATCH_SIZE

        img_patch  = image_rgb[y1:y2, x1:x2]
        mask_patch = mask[y1:y2, x1:x2]

        if self.transform is not None:
            transformed = self.transform(image=img_patch, mask=mask_patch)
            img_patch  = transformed["image"]
            mask_patch = transformed["mask"]

        image_tensor = torch.from_numpy(img_patch).permute(2, 0, 1).float() / 255.0
        mask_tensor  = torch.from_numpy(mask_patch).long()
        return image_tensor, mask_tensor


if __name__ == "__main__":
    dataset = FundusSegmentationDataset(split="train")
    image, mask = dataset[0]

    print(f"Dataset size: {len(dataset)}")
    print(f"Image shape: {tuple(image.shape)}, dtype: {image.dtype}")
    print(f"Mask shape: {tuple(mask.shape)}, dtype: {mask.dtype}")
    print(f"Mask unique classes: {sorted(mask.unique().tolist())}")
