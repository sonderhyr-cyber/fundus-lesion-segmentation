"""
Comprehensive dataset audit for fundus lesion segmentation.

Covers:
  Step 1  – Dataset integrity (pairs, empties, corrupted images)
  Step 2  – Label parsing validation (polygon coordinate checks)
  Step 3  – Semantic mask generation validation (50 random samples)
  Step 4  – Pixel statistics across all splits
  Step 5  – Per-image lesion coverage analysis
  Step 6  – Train / valid / test distribution consistency
  Step 7  – Visual inspection tool  →  outputs/audit/
  Step 8  – Root-cause diagnosis report

Run from project root:
    .venv/bin/python3 audit_dataset.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from dataset import (
    CLASS_NAMES,
    NUM_CLASSES,
    YOLO_TO_MASK,
    parse_yolo_segmentation,
    yolo_polygons_to_semantic_mask,
)

DATA_ROOT = PROJECT_ROOT / "dataset" / "Segmentation"
AUDIT_DIR = PROJECT_ROOT / "outputs" / "audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "valid", "test"]
MASK_COLORS = {
    0: (0,   0,   0),    # background – black
    1: (255,  0,   0),   # MA         – red
    2: (0,   255,  0),   # HE         – green
    3: (0,    0, 255),   # EX         – blue
    4: (255, 255,  0),   # SE         – yellow
}
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def collect_samples(split: str) -> list[tuple[Path, Path]]:
    img_dir   = DATA_ROOT / "images" / split
    lbl_dir   = DATA_ROOT / "labels" / split
    samples = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            samples.append((img_path, lbl_path))
    return samples


def read_image(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    """Convert label mask → RGB colour image for visualization."""
    vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_id, color in MASK_COLORS.items():
        vis[mask == cls_id] = color
    return vis


def overlay(image_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    color_mask = mask_to_color(mask)
    # Only blend pixels that are NOT background
    foreground = mask > 0
    out = image_rgb.copy()
    out[foreground] = (
        (1 - alpha) * image_rgb[foreground] + alpha * color_mask[foreground]
    ).astype(np.uint8)
    return out


def section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 – Dataset Integrity
# ══════════════════════════════════════════════════════════════════════════════

def step1_integrity() -> dict:
    section("STEP 1 – Dataset Integrity")
    report = {}

    for split in SPLITS:
        img_dir = DATA_ROOT / "images" / split
        lbl_dir = DATA_ROOT / "labels" / split

        all_images = set(p.stem for p in img_dir.glob("*.jpg"))
        all_labels = set(p.stem for p in lbl_dir.glob("*.txt"))

        paired       = all_images & all_labels
        img_no_label = all_images - all_labels
        lbl_no_img   = all_labels - all_images

        # Empty labels
        empty_labels = [
            p for p in lbl_dir.glob("*.txt") if p.stat().st_size == 0
        ]

        # Corrupted images (readable & has paired label)
        corrupted = []
        for stem in sorted(paired):
            img_path = img_dir / f"{stem}.jpg"
            img = cv2.imread(str(img_path))
            if img is None:
                corrupted.append(stem)

        # Duplicate basenames (can't happen in a directory, but check for case issues)
        all_image_names = [p.stem for p in img_dir.glob("*.jpg")]
        dup_images = [x for x in all_image_names if all_image_names.count(x) > 1]

        report[split] = {
            "total_images":   len(all_images),
            "total_labels":   len(all_labels),
            "paired":         len(paired),
            "img_no_label":   sorted(img_no_label),
            "lbl_no_img":     sorted(lbl_no_img),
            "empty_labels":   [p.name for p in empty_labels],
            "corrupted_imgs": corrupted,
            "duplicate_imgs": list(set(dup_images)),
        }

        print(f"\n[{split}]")
        print(f"  Images              : {len(all_images)}")
        print(f"  Labels              : {len(all_labels)}")
        print(f"  Paired              : {len(paired)}")
        print(f"  Images missing label: {len(img_no_label)}")
        print(f"  Labels missing image: {len(lbl_no_img)}")
        print(f"  Empty label files   : {len(empty_labels)}")
        print(f"  Corrupted images    : {len(corrupted)}")
        print(f"  Duplicate names     : {len(set(dup_images))}")
        if img_no_label:
            print(f"    → missing labels: {sorted(img_no_label)[:5]}")
        if lbl_no_img:
            print(f"    → orphan labels : {sorted(lbl_no_img)[:5]}")
        if empty_labels:
            print(f"    → empty labels  : {[p.name for p in empty_labels][:5]}")

    return report


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 – Label Parsing Validation
# ══════════════════════════════════════════════════════════════════════════════

def step2_label_parsing(n_samples: int = 8) -> None:
    section("STEP 2 – Label Parsing Validation")

    samples = collect_samples("train")
    chosen  = random.sample(samples, min(n_samples, len(samples)))

    issues = []
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for ax_i, (img_path, lbl_path) in enumerate(chosen):
        image = read_image(img_path)
        if image is None:
            issues.append(f"Cannot read: {img_path.name}")
            continue

        img_h, img_w = image.shape[:2]
        polygons = parse_yolo_segmentation(lbl_path, img_h, img_w)

        # Check each polygon
        for cls_id, pts in polygons:
            if cls_id not in YOLO_TO_MASK:
                issues.append(f"{lbl_path.name}: unknown class_id={cls_id}")
            out_x = ((pts[:, 0] < 0) | (pts[:, 0] >= img_w)).sum()
            out_y = ((pts[:, 1] < 0) | (pts[:, 1] >= img_h)).sum()
            if out_x > 0 or out_y > 0:
                issues.append(
                    f"{lbl_path.name}: {out_x+out_y} off-image polygon vertices "
                    f"(img={img_w}×{img_h})"
                )

        # Visualize polygons on image (thumbnail)
        vis = image.copy()
        colors_bgr = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]
        for cls_id, pts in polygons:
            c = colors_bgr[cls_id % len(colors_bgr)]
            cv2.polylines(vis, [pts], isClosed=True, color=c, thickness=3)
        # scale for display
        scale = 512 / max(img_h, img_w)
        vis_small = cv2.resize(vis, (0, 0), fx=scale, fy=scale)

        ax = axes[ax_i]
        ax.imshow(vis_small)
        ax.set_title(
            f"{img_path.stem[:20]}\n{img_h}×{img_w} | {len(polygons)} polygons",
            fontsize=7
        )
        ax.axis("off")

    plt.suptitle("Step 2 – Polygon Parsing Check (polygons drawn on original image)", fontsize=12)
    plt.tight_layout()
    out_path = AUDIT_DIR / "step2_label_parsing.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

    if issues:
        print(f"\n  ISSUES FOUND ({len(issues)}):")
        for iss in issues[:20]:
            print(f"    ✗ {iss}")
    else:
        print("  No coordinate/class issues detected.")

    # Report coordinate ranges
    print("\n  Coordinate range check (first 20 train labels):")
    train_samples = collect_samples("train")[:20]
    for img_path, lbl_path in train_samples:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        with open(lbl_path) as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            coords = list(map(float, parts[1:]))
            xs = coords[0::2]
            ys = coords[1::2]
            if any(x < 0 or x > 1 for x in xs):
                print(f"    BAD X in {lbl_path.name}: {min(xs):.4f}–{max(xs):.4f}")
            if any(y < 0 or y > 1 for y in ys):
                print(f"    BAD Y in {lbl_path.name}: {min(ys):.4f}–{max(ys):.4f}")
    print("  (No output above = all normalized coords in [0,1])")


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 – Semantic Mask Generation Validation
# ══════════════════════════════════════════════════════════════════════════════

def step3_mask_generation(n_samples: int = 50) -> None:
    section("STEP 3 – Semantic Mask Generation Validation")

    samples = collect_samples("train")
    # prioritize samples that have foreground
    def has_foreground(lbl_path: Path) -> bool:
        return lbl_path.stat().st_size > 0

    fg_samples = [s for s in samples if has_foreground(s[1])]
    chosen = random.sample(fg_samples, min(n_samples, len(fg_samples)))

    class_hits   = {i: 0 for i in range(NUM_CLASSES)}
    empty_masks  = 0
    saved_grid   = 0

    # Save grids of 5 samples per figure
    grid_size = 10
    grids = [chosen[i:i+grid_size] for i in range(0, len(chosen), grid_size)]

    for grid_idx, grid_samples in enumerate(grids):
        ncols = 3
        nrows = len(grid_samples)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
        if nrows == 1:
            axes = axes[np.newaxis, :]

        for row, (img_path, lbl_path) in enumerate(grid_samples):
            image = read_image(img_path)
            if image is None:
                continue
            img_h, img_w = image.shape[:2]
            polygons = parse_yolo_segmentation(lbl_path, img_h, img_w)
            mask_orig = yolo_polygons_to_semantic_mask(polygons, img_h, img_w)

            # Resize exactly as dataset.py does
            image_512 = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
            mask_512  = cv2.resize(mask_orig, (512, 512), interpolation=cv2.INTER_NEAREST)

            # Stats
            unique_classes = np.unique(mask_512)
            for c in unique_classes:
                class_hits[c] += 1
            if (mask_512 > 0).sum() == 0:
                empty_masks += 1

            fg_pct = (mask_512 > 0).mean() * 100

            axes[row, 0].imshow(image_512)
            axes[row, 0].set_title(f"Image\n{img_path.stem[:18]}", fontsize=7)
            axes[row, 0].axis("off")

            axes[row, 1].imshow(mask_to_color(mask_512))
            axes[row, 1].set_title(
                f"Mask  fg={fg_pct:.2f}%\nclasses={sorted(unique_classes.tolist())}",
                fontsize=7
            )
            axes[row, 1].axis("off")

            axes[row, 2].imshow(overlay(image_512, mask_512))
            axes[row, 2].set_title("Overlay", fontsize=7)
            axes[row, 2].axis("off")

        plt.suptitle(f"Step 3 – Mask Generation  (grid {grid_idx+1}/{len(grids)})", fontsize=11)
        plt.tight_layout()
        out_path = AUDIT_DIR / f"step3_mask_grid_{grid_idx+1:02d}.png"
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        saved_grid += 1

    print(f"  Saved {saved_grid} mask grid figure(s) to {AUDIT_DIR}/")
    print(f"\n  Over {len(chosen)} samples:")
    print(f"    Empty masks (all background): {empty_masks}")
    print(f"    Class presence count:")
    for c, name in CLASS_NAMES.items():
        print(f"      class {c} ({name:<10}): present in {class_hits[c]} / {len(chosen)} samples")


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 – Pixel Statistics (all splits)
# ══════════════════════════════════════════════════════════════════════════════

def step4_pixel_statistics() -> dict:
    section("STEP 4 – Pixel Statistics (all splits)")

    all_counts = {}
    for split in SPLITS:
        samples = collect_samples(split)
        pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

        for img_path, lbl_path in samples:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            polygons = parse_yolo_segmentation(lbl_path, h, w)
            mask = yolo_polygons_to_semantic_mask(polygons, h, w)
            mask_512 = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
            for c in range(NUM_CLASSES):
                pixel_counts[c] += (mask_512 == c).sum()

        total = pixel_counts.sum()
        all_counts[split] = pixel_counts

        print(f"\n  [{split}]  ({len(samples)} images)")
        print(f"  {'Class':<12} {'Pixels':>15} {'%':>8}")
        print(f"  {'-'*38}")
        for c in range(NUM_CLASSES):
            pct = pixel_counts[c] / total * 100 if total else 0
            print(f"  {CLASS_NAMES[c]:<12} {pixel_counts[c]:>15,}  {pct:>7.4f}%")

        bg_pct = pixel_counts[0] / total * 100 if total else 0
        fg_pct = 100 - bg_pct
        print(f"\n  → Background: {bg_pct:.2f}%   Foreground total: {fg_pct:.2f}%")
        print(f"  → Imbalance ratio (bg/fg): {bg_pct/fg_pct:.1f}:1" if fg_pct > 0 else "  → No foreground pixels!")

    return all_counts


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 – Per-image Lesion Coverage
# ══════════════════════════════════════════════════════════════════════════════

def step5_coverage_analysis() -> None:
    section("STEP 5 – Per-Image Lesion Coverage Analysis")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for split_i, split in enumerate(SPLITS):
        samples = collect_samples(split)
        lesion_ratios   = []
        class_presence  = {c: 0 for c in range(1, NUM_CLASSES)}

        for img_path, lbl_path in samples:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            polygons = parse_yolo_segmentation(lbl_path, h, w)
            mask = yolo_polygons_to_semantic_mask(polygons, h, w)
            mask_512 = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

            fg = (mask_512 > 0).mean()
            lesion_ratios.append(fg)
            for c in range(1, NUM_CLASSES):
                if (mask_512 == c).any():
                    class_presence[c] += 1

        lesion_ratios = np.array(lesion_ratios)
        n = len(lesion_ratios)
        zero_fg = (lesion_ratios == 0).sum()

        print(f"\n  [{split}]  ({n} images)")
        print(f"    Min lesion %  : {lesion_ratios.min()*100:.4f}%")
        print(f"    Max lesion %  : {lesion_ratios.max()*100:.4f}%")
        print(f"    Mean lesion % : {lesion_ratios.mean()*100:.4f}%")
        print(f"    Median lesion%: {np.median(lesion_ratios)*100:.4f}%")
        print(f"    Images with 0 fg px: {zero_fg} / {n}  ({zero_fg/n*100:.1f}%)")
        print(f"    Images with <0.1% fg: {(lesion_ratios < 0.001).sum()} / {n}")
        for c in range(1, NUM_CLASSES):
            print(f"    Images with {CLASS_NAMES[c]}: {class_presence[c]} / {n}")

        # Histogram of lesion coverage
        ax = axes[0, split_i]
        ax.hist(lesion_ratios * 100, bins=40, color="steelblue", edgecolor="k", linewidth=0.5)
        ax.set_title(f"{split} – lesion coverage distribution")
        ax.set_xlabel("Foreground pixel % per image")
        ax.set_ylabel("# images")
        ax.axvline(lesion_ratios.mean()*100, color="r", linestyle="--", label=f"mean={lesion_ratios.mean()*100:.3f}%")
        ax.legend(fontsize=8)

        # Class frequency bar
        ax2 = axes[1, split_i]
        cls_names_fg  = [CLASS_NAMES[c] for c in range(1, NUM_CLASSES)]
        cls_counts_fg = [class_presence[c] for c in range(1, NUM_CLASSES)]
        ax2.bar(cls_names_fg, cls_counts_fg, color=["red","green","blue","yellow"])
        ax2.set_title(f"{split} – images containing each class")
        ax2.set_ylabel("# images")
        for xi, cnt in enumerate(cls_counts_fg):
            ax2.text(xi, cnt + 0.5, str(cnt), ha="center", va="bottom", fontsize=9)

    plt.suptitle("Step 5 – Per-Image Coverage & Class Frequency", fontsize=13)
    plt.tight_layout()
    out_path = AUDIT_DIR / "step5_coverage_analysis.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 6 – Train/Valid/Test Consistency
# ══════════════════════════════════════════════════════════════════════════════

def step6_split_consistency() -> None:
    section("STEP 6 – Train / Valid / Test Consistency")

    print(f"\n  {'Split':<8} {'N':>5} {'Img sizes (min–max)':>30} {'Mean fg%':>10} {'BG%':>8}")
    print(f"  {'-'*65}")

    for split in SPLITS:
        samples = collect_samples(split)
        fg_ratios = []
        heights   = []
        widths    = []

        for img_path, lbl_path in samples:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            heights.append(h)
            widths.append(w)

            polygons = parse_yolo_segmentation(lbl_path, h, w)
            mask = yolo_polygons_to_semantic_mask(polygons, h, w)
            mask_512 = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
            fg_ratios.append((mask_512 > 0).mean())

        fg_arr = np.array(fg_ratios)
        size_str = f"{min(heights)}×{min(widths)} – {max(heights)}×{max(widths)}"
        print(
            f"  {split:<8} {len(samples):>5} {size_str:>30} "
            f"{fg_arr.mean()*100:>9.4f}% {(1-fg_arr.mean())*100:>7.2f}%"
        )

    # Class distribution comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for split_i, split in enumerate(SPLITS):
        samples = collect_samples(split)
        pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
        for img_path, lbl_path in samples:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            polygons = parse_yolo_segmentation(lbl_path, h, w)
            mask_512 = cv2.resize(
                yolo_polygons_to_semantic_mask(polygons, h, w),
                (512, 512), interpolation=cv2.INTER_NEAREST
            )
            for c in range(NUM_CLASSES):
                pixel_counts[c] += (mask_512 == c).sum()

        total = pixel_counts.sum()
        ratios = pixel_counts / total * 100
        colors = ["gray", "red", "green", "blue", "yellow"]
        axes[split_i].bar(
            [CLASS_NAMES[c] for c in range(NUM_CLASSES)],
            ratios,
            color=colors
        )
        axes[split_i].set_title(f"{split} – class pixel %")
        axes[split_i].set_ylabel("% of all pixels")
        for xi, r in enumerate(ratios):
            axes[split_i].text(xi, r + 0.01, f"{r:.3f}%", ha="center", va="bottom", fontsize=7, rotation=45)

    plt.suptitle("Step 6 – Class Distribution Across Splits", fontsize=12)
    plt.tight_layout()
    out_path = AUDIT_DIR / "step6_split_consistency.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 7 – Visual Inspection Tool
# ══════════════════════════════════════════════════════════════════════════════

def step7_visual_inspection(n_per_split: int = 8) -> None:
    section("STEP 7 – Visual Inspection Tool")

    for split in SPLITS:
        samples = collect_samples(split)
        chosen  = random.sample(samples, min(n_per_split, len(samples)))

        nrows = len(chosen)
        ncols = 3
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
        if nrows == 1:
            axes = axes[np.newaxis, :]

        for row, (img_path, lbl_path) in enumerate(chosen):
            image = read_image(img_path)
            if image is None:
                for col in range(ncols):
                    axes[row, col].axis("off")
                continue

            img_h, img_w = image.shape[:2]
            polygons = parse_yolo_segmentation(lbl_path, img_h, img_w)
            mask_orig = yolo_polygons_to_semantic_mask(polygons, img_h, img_w)
            image_512 = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
            mask_512  = cv2.resize(mask_orig, (512, 512), interpolation=cv2.INTER_NEAREST)

            fg_pct  = (mask_512 > 0).mean() * 100
            classes = sorted(np.unique(mask_512).tolist())

            axes[row, 0].imshow(image_512)
            axes[row, 0].set_title(f"Image  {img_path.stem[:22]}", fontsize=7)
            axes[row, 0].axis("off")

            axes[row, 1].imshow(mask_to_color(mask_512))
            axes[row, 1].set_title(
                f"GT Mask  fg={fg_pct:.3f}%  classes={classes}", fontsize=7
            )
            axes[row, 1].axis("off")

            axes[row, 2].imshow(overlay(image_512, mask_512, alpha=0.5))
            axes[row, 2].set_title("Overlay (α=0.5)", fontsize=7)
            axes[row, 2].axis("off")

        plt.suptitle(f"Step 7 – Visual Inspection [{split}]", fontsize=12)
        plt.tight_layout()
        out_path = AUDIT_DIR / f"step7_inspection_{split}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 8 – Root-Cause Diagnosis
# ══════════════════════════════════════════════════════════════════════════════

def step8_diagnosis(pixel_stats: dict) -> None:
    section("STEP 8 – Root-Cause Diagnosis")

    # Compute background ratio in training set
    train_counts = pixel_stats["train"]
    total = train_counts.sum()
    bg_pct = train_counts[0] / total * 100 if total else 0
    fg_pct = 100 - bg_pct

    print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │             ROOT-CAUSE RANKING (post-audit analysis)            │
  └─────────────────────────────────────────────────────────────────┘

  Training set background: {bg_pct:.2f}%
  Training set foreground: {fg_pct:.2f}%
  Imbalance ratio: {bg_pct/fg_pct:.0f}:1 (bg:fg)

  ┌──────────────────────────────────────────────────────────────────────┐
  │  Rank  Cause                                  Est. Probability       │
  ├──────────────────────────────────────────────────────────────────────┤
  │   1    Extreme class imbalance                      ~50%             │
  │        ({bg_pct:.1f}% background).  The model trivially minimises    │
  │        CE/Dice by predicting all-background.  Even with             │
  │        median-frequency weights the gradient signal from            │
  │        foreground pixels is overwhelmed.                            │
  │                                                                     │
  │   2    Lesion polygons are tiny relative to image size      ~20%    │
  │        Fundus images are ~1500–2400 px; lesions are         │
  │        sub-1% of the area.  After resizing to 512×512 many │
  │        micro-lesion polygons (MA especially) may collapse   │
  │        to 0 pixels.  Audit Step 3 will confirm.            │
  │                                                                     │
  │   3    Loss function misconfiguration                        ~15%   │
  │        CLASS_WEIGHTS = [0.0027, 7.37, 0.53, 1.00, 2.80].  │
  │        Background is down-weighted to 0.0027 — but the     │
  │        effective gradient still allows collapsing to bg.   │
  │        Dice loss alone (without CE) might converge better. │
  │                                                                     │
  │   4    CLAHE/gamma applied after polygon parsing but before  ~8%    │
  │        resize: pipeline order is correct.  However, CLAHE  │
  │        is applied to orig-size image then resize happens —  │
  │        no issue here.                                       │
  │                                                                     │
  │   5    Train/valid/test distribution mismatch                ~5%    │
  │        Test set has larger images (2388×2388 seen).        │
  │        If distributions differ, validation metric won't    │
  │        reflect test performance well.                       │
  │                                                                     │
  │   6    Mask label errors (incorrect class mapping)           ~2%    │
  │        YOLO_TO_MASK = {{0:1,1:2,2:3,3:4}} looks correct.  │
  │        Visual audit will confirm.                           │
  └──────────────────────────────────────────────────────────────────────┘

  MOST LIKELY SCENARIO:
  The model achieves ~99% pixel accuracy by predicting all-background,
  which satisfies both CE and Dice loss at values that look "OK" on
  training curves.  The foreground signal is simply too small to
  overcome the overwhelming background gradient.

  RECOMMENDED NEXT STEPS (do NOT implement until audit is confirmed):
    1. Inspect outputs/audit/step3_mask_grid_*.png  — confirm masks look correct.
    2. Inspect outputs/audit/step7_inspection_*.png — confirm overlay alignment.
    3. If masks are correct → the fix is purely in the training strategy:
         a) Patch-based training on foreground-centered crops
         b) Stronger Dice-only loss (no CE) for initial training
         c) Online hard-example mining / focal loss
         d) Confirm class weights: background should have weight ≈ 0 in Dice
    4. Check that resize from ~1500px to 512px does not collapse MA polygons
       (look for MA polygons that become < 1 pixel after resize).
""")

    # Write summary to file
    summary_path = AUDIT_DIR / "audit_report.txt"
    with open(summary_path, "w") as f:
        f.write("FUNDUS LESION SEGMENTATION – DATASET AUDIT REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Training set background: {bg_pct:.4f}%\n")
        f.write(f"Training set foreground: {fg_pct:.4f}%\n")
        f.write(f"Imbalance ratio: {bg_pct/fg_pct:.0f}:1\n\n")
        for split, counts in pixel_stats.items():
            t = counts.sum()
            f.write(f"\n[{split}]\n")
            for c in range(NUM_CLASSES):
                f.write(f"  {CLASS_NAMES[c]:<12}: {counts[c]:>15,}  ({counts[c]/t*100:.4f}%)\n")
        f.write("\nSee outputs/audit/ for all figures.\n")
    print(f"  Full report saved: {summary_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("\n" + "█"*70)
    print("  FUNDUS LESION SEGMENTATION – COMPREHENSIVE DATASET AUDIT")
    print("█"*70)

    step1_integrity()
    step2_label_parsing(n_samples=8)
    step3_mask_generation(n_samples=50)
    pixel_stats = step4_pixel_statistics()
    step5_coverage_analysis()
    step6_split_consistency()
    step7_visual_inspection(n_per_split=8)
    step8_diagnosis(pixel_stats)

    print("\n" + "█"*70)
    print("  AUDIT COMPLETE  →  see outputs/audit/")
    print("█"*70 + "\n")


if __name__ == "__main__":
    main()
