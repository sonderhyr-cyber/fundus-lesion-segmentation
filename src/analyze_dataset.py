from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import CLASS_NAMES, FundusSegmentationDataset, NUM_CLASSES
from paths import OUTPUT_DIR
TABLE_PATH = OUTPUT_DIR / "dataset_class_stats.csv"
CHART_PATH = OUTPUT_DIR / "dataset_class_distribution.png"


def count_class_pixels(dataset: FundusSegmentationDataset, batch_size: int = 8) -> np.ndarray:
    """Count pixel totals for each class across the entire dataset."""
    pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    for _, mask_batch in loader:
        for mask in mask_batch:
            counts = torch.bincount(mask.flatten(), minlength=NUM_CLASSES)
            pixel_counts += counts.cpu().numpy()

    return pixel_counts


def build_stats_table(pixel_counts: np.ndarray) -> list[dict[str, str | int | float]]:
    total_pixels = int(pixel_counts.sum())
    rows = []
    for cls_id in range(NUM_CLASSES):
        count = int(pixel_counts[cls_id])
        ratio = count / total_pixels if total_pixels > 0 else 0.0
        rows.append(
            {
                "class_id": cls_id,
                "class_name": CLASS_NAMES[cls_id],
                "pixel_count": count,
                "ratio": ratio,
                "ratio_percent": ratio * 100,
            }
        )
    return rows


def print_table(rows: list[dict[str, str | int | float]]) -> None:
    header = f"{'Class':<12} {'Pixels':>15} {'Ratio':>10}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['class_name']:<12} "
            f"{row['pixel_count']:>15,} "
            f"{row['ratio_percent']:>9.4f}%"
        )


def save_table_csv(rows: list[dict[str, str | int | float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("class_id,class_name,pixel_count,ratio,ratio_percent\n")
        for row in rows:
            f.write(
                f"{row['class_id']},{row['class_name']},{row['pixel_count']},"
                f"{row['ratio']:.8f},{row['ratio_percent']:.4f}\n"
            )


def save_bar_chart(rows: list[dict[str, str | int | float]], output_path: Path) -> None:
    class_names = [str(row["class_name"]) for row in rows]
    pixel_counts = [int(row["pixel_count"]) for row in rows]
    ratios = [float(row["ratio_percent"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(class_names, pixel_counts, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"])
    axes[0].set_title("Class Pixel Count")
    axes[0].set_ylabel("Pixels")
    axes[0].ticklabel_format(style="plain", axis="y")
    for idx, count in enumerate(pixel_counts):
        axes[0].text(idx, count, f"{count:,}", ha="center", va="bottom", fontsize=9)

    axes[1].bar(class_names, ratios, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"])
    axes[1].set_title("Class Pixel Ratio")
    axes[1].set_ylabel("Percentage (%)")
    for idx, ratio in enumerate(ratios):
        axes[1].text(idx, ratio, f"{ratio:.2f}%", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Training Set Class Distribution (512x512)", fontsize=14)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    dataset = FundusSegmentationDataset(split="train")
    pixel_counts = count_class_pixels(dataset)
    rows = build_stats_table(pixel_counts)

    print(f"Training set size: {len(dataset)} images")
    print(f"Mask size: 512 x 512")
    print()
    print_table(rows)

    save_table_csv(rows, TABLE_PATH)
    save_bar_chart(rows, CHART_PATH)

    print()
    print(f"Saved table: {TABLE_PATH}")
    print(f"Saved chart: {CHART_PATH}")


if __name__ == "__main__":
    main()
