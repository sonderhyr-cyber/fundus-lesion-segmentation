#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ddr_split.py - Re-split the DDR segmentation dataset into train:val:test = 6:1:3,
stratified by SE (Soft Exudate) lesion presence and pixel ratio.

This is a custom experimental design (NOT the original paper split):
  - All 757 DDR images (original train 383 / val 149 / test 225) are merged
    into one pool, then re-split with stratified sampling.
  - Stratification key: SE lesion.
      Tier A: no SE pixels
      Tier B: SE pixels present, SE pixel ratio < median (median over SE+ images)
      Tier C: SE pixels present, SE pixel ratio >= median
    Each tier is split 6:1:3 internally (largest-remainder allocation, then a
    deterministic adjustment pass on the largest tier to hit the exact global
    targets 454/76/227). The tier results are merged per split.
  - Random seed fixed to 42. This split is meant to be used as THE fixed split
    for all follow-up experiments (板块二/三), never re-generated per run.

Usage:
    python ddr_split.py --root <DDR root> --out <output root>

Outputs (under --out):
    images/{train,val,test}/          symlinks to the .jpg files
    labels/{train,val,test}/          symlinks to fused .png labels + per-class
                                      EX/HE/SE/MA .tif dirs (official README layout)
    train.txt / val.txt / test.txt    image file name lists (one per line)
    split_assignment.json             per-image tier/split/pixel stats (reproducibility)
    split_stats.csv                   per-split class statistics
    split_report.md                   human-readable statistical report

All outputs are fully reproducible from the original DDR root with the fixed
seed (42) and sorted file ordering.
"""

import argparse
import csv
import json
import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

CLASSES = ["EX", "HE", "SE", "MA"]
CLASS_IDS = {"EX": 1, "HE": 2, "SE": 3, "MA": 4}
SEED = 42
TARGET = {"train": 454, "val": 76, "test": 227}
SPLITS = ["train", "val", "test"]
RATIO = {"train": 0.6, "val": 0.1, "test": 0.3}


def link(src, dst):
    """Symlink src -> dst, falling back to a copy if symlinks are unavailable."""
    if os.path.lexists(dst):
        os.remove(dst)
    try:
        os.symlink(src, dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def collect_images(root):
    """Return sorted list of (image_name, original_subset) and anomaly messages."""
    images = []
    anomalies = []
    for subset in SPLITS:
        img_dir = os.path.join(root, "images", subset)
        label_dir = os.path.join(root, "labels", subset)
        if not os.path.isdir(img_dir):
            raise SystemExit("Missing image dir: %s" % img_dir)
        names = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(".jpg"))
        for name in names:
            stem = os.path.splitext(name)[0]
            if not os.path.isfile(os.path.join(label_dir, stem + ".png")):
                anomalies.append("%s: missing fused png label" % name)
            for c in CLASSES:
                if not os.path.isfile(os.path.join(label_dir, c, stem + ".tif")):
                    anomalies.append("%s: missing %s tif label" % (name, c))
            images.append((name, subset))
    return images, anomalies


def image_stats(args):
    root, subset, name = args
    stem = os.path.splitext(name)[0]
    label_dir = os.path.join(root, "labels", subset)
    tif_counts = {}
    total = None
    for c in CLASSES:
        arr = cv2.imread(os.path.join(label_dir, c, stem + ".tif"),
                         cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise IOError("Cannot read %s" % os.path.join(label_dir, c, stem + ".tif"))
        tif_counts[c] = int((arr == 255).sum())
        if total is None:
            total = arr.size
    png_counts = {}
    png_path = os.path.join(label_dir, stem + ".png")
    parr = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if parr is not None:
        for c in CLASSES:
            png_counts[c] = int((parr == CLASS_IDS[c]).sum())
    return {"name": name, "subset": subset, "tif": tif_counts,
            "png": png_counts, "total": total}


def largest_remainder_counts(n):
    """Split n images into (train, val, test) by ratio 0.6/0.1/0.3."""
    raw = {s: n * RATIO[s] for s in SPLITS}
    counts = {s: int(raw[s]) for s in SPLITS}
    remaining = n - sum(counts.values())
    order = sorted(SPLITS, key=lambda s: raw[s] - counts[s], reverse=True)
    for i in range(remaining):
        counts[order[i % len(order)]] += 1
    return counts


def split_tier(items, rng):
    """Shuffle items in place and split into 6:1:3; returns {split: [items]}."""
    rng.shuffle(items)
    counts = largest_remainder_counts(len(items))
    result = {}
    pos = 0
    for s in SPLITS:
        result[s] = items[pos:pos + counts[s]]
        pos += counts[s]
    return result


def adjust_to_target(allocation):
    """
    allocation: {tier: {split: [names]}}.
    Per-tier largest-remainder can drift a few images from the global target;
    fix it deterministically by moving images inside the largest tier.
    """
    totals = {s: sum(len(allocation[t][s]) for t in allocation) for s in SPLITS}
    diff = {s: TARGET[s] - totals[s] for s in SPLITS}
    tiers = sorted(allocation, key=lambda t: -len(allocation[t]["train"]))
    while any(diff.values()):
        surplus = [s for s in SPLITS if diff[s] < 0]
        deficit = [s for s in SPLITS if diff[s] > 0]
        if not surplus or not deficit:
            raise RuntimeError("Cannot adjust split counts: %s" % diff)
        src, dst = surplus[0], deficit[0]
        moved = False
        for t in tiers:
            if allocation[t][src]:
                item = allocation[t][src].pop()
                allocation[t][dst].append(item)
                diff[src] += 1
                diff[dst] -= 1
                moved = True
                break
        if not moved:
            raise RuntimeError("Adjustment failed: %s" % diff)
    return allocation


def write_outputs(root, out_root, images, assignment, stats, anomalies):
    for s in SPLITS:
        os.makedirs(os.path.join(out_root, "images", s), exist_ok=True)
        lbl = os.path.join(out_root, "labels", s)
        os.makedirs(lbl, exist_ok=True)
        for c in CLASSES:
            os.makedirs(os.path.join(lbl, c), exist_ok=True)

    with open(os.path.join(out_root, "train.txt"), "w") as f:
        pass
    list_files = {s: [] for s in SPLITS}
    for name, subset in images:
        split = assignment[name]["split"]
        stem = os.path.splitext(name)[0]
        img_src = os.path.join(root, "images", subset, name)
        img_dst = os.path.join(out_root, "images", split, name)
        link(img_src, img_dst)
        png_src = os.path.join(root, "labels", subset, stem + ".png")
        png_dst = os.path.join(out_root, "labels", split, stem + ".png")
        link(png_src, png_dst)
        for c in CLASSES:
            tif_src = os.path.join(root, "labels", subset, c, stem + ".tif")
            tif_dst = os.path.join(out_root, "labels", split, c, stem + ".tif")
            link(tif_src, tif_dst)
        list_files[split].append(name)
        assignment[name]["tier"] = assignment[name].get("tier")

    for s in SPLITS:
        with open(os.path.join(out_root, "%s.txt" % s), "w") as f:
            f.write("\n".join(sorted(list_files[s])) + "\n")

    with open(os.path.join(out_root, "split_assignment.json"), "w") as f:
        json.dump({"seed": SEED, "anomalies": anomalies, "images": assignment},
                  f, indent=2)


def compute_split_stats(stats, split_of):
    """Return per-split + overall class statistics dict."""
    split_names = {img: split_of[img] for img in split_of}
    groups = {s: [] for s in SPLITS}
    for rec in stats:
        groups[split_names[rec["name"]]].append(rec)
    groups["all"] = stats

    report = {}
    for gname, recs in groups.items():
        n = len(recs)
        class_img_count = {c: 0 for c in CLASSES}
        class_pixel = {c: 0 for c in CLASSES}
        total_pixel = sum(r["total"] for r in recs)
        for r in recs:
            for c in CLASSES:
                if r["tif"][c] > 0:
                    class_img_count[c] += 1
                class_pixel[c] += r["tif"][c]
        tier_counts = {t: 0 for t in ("A", "B", "C")}
        for r in recs:
            tier_counts[r["tier"]] += 1
        report[gname] = {
            "n": n,
            "total_pixel": total_pixel,
            "tiers": tier_counts,
            "class_img_count": class_img_count,
            "class_pixel": class_pixel,
        }
    return report


def write_stats(out_root, report):
    with open(os.path.join(out_root, "split_stats.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "n_images", "class", "pos_images", "image_pct",
                    "pos_pixels", "pixel_pct"])
        for gname in SPLITS + ["all"]:
            g = report[gname]
            for c in CLASSES:
                img_pct = 100.0 * g["class_img_count"][c] / g["n"]
                pix_pct = 100.0 * g["class_pixel"][c] / g["total_pixel"]
                w.writerow([gname, g["n"], c, g["class_img_count"][c],
                            round(img_pct, 2), g["class_pixel"][c],
                            round(pix_pct, 4)])
    # Markdown report
    lines = []
    lines.append("# DDR 6:1:3 re-split report (stratified by SE, seed=42)\n")
    lines.append("Custom experimental split (not the original paper split). "
                 "All 757 DDR images merged and re-split train:val:test = "
                 "454:76:227.\n")
    lines.append("| split | n | tierA | tierB | tierC |")
    lines.append("|---|---|---|---|---|")
    for gname in SPLITS + ["all"]:
        g = report[gname]
        lines.append("| %s | %d | %d | %d | %d |" % (
            gname, g["n"], g["tiers"]["A"], g["tiers"]["B"], g["tiers"]["C"]))
    lines.append("")
    lines.append("| split | class | pos_images | image_pct | pixel_pct |")
    lines.append("|---|---|---|---|---|")
    for gname in SPLITS + ["all"]:
        g = report[gname]
        for c in CLASSES:
            img_pct = 100.0 * g["class_img_count"][c] / g["n"]
            pix_pct = 100.0 * g["class_pixel"][c] / g["total_pixel"]
            lines.append("| %s | %s | %d | %.2f | %.4f |" % (
                gname, c, g["class_img_count"][c], img_pct, pix_pct))
    lines.append("")
    lines.append("Note: image_pct = share of images containing the lesion; "
                 "pixel_pct = lesion pixels / total pixels over the split.")
    with open(os.path.join(out_root, "split_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="DDR 6:1:3 re-split (SE stratified)")
    parser.add_argument("--root", default="/root/HRDecoder/data/DDR",
                        help="original DDR root with images/ and labels/")
    parser.add_argument("--out", default="/root/HRDecoder/data/DDR_split_613",
                        help="output root")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    images, anomalies = collect_images(args.root)
    print("[1/5] Collected %d images; %d anomalies" % (len(images), len(anomalies)))
    for a in anomalies:
        print("  ANOMALY: %s" % a)

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        stats = list(ex.map(image_stats,
                            [(args.root, s, n) for n, s in images]))
    print("[2/5] Computed per-image class pixel stats")

    for rec in stats:
        for c in CLASSES:
            if rec["png"][c] != rec["tif"][c]:
                anomalies.append(
                    "%s: fused png vs tif mismatch for %s (png=%d tif=%d)"
                    % (rec["name"], c, rec["png"][c], rec["tif"][c]))

    # SE ratios over SE-positive images
    se_ratio = {}
    for rec in stats:
        se_ratio[rec["name"]] = rec["tif"]["SE"] / float(rec["total"])
    pos_ratios = [r for r in se_ratio.values() if r > 0]
    median_se = float(np.median(pos_ratios)) if pos_ratios else 0.0
    print("[3/5] SE-positive images: %d, median SE pixel ratio: %.6f"
          % (len(pos_ratios), median_se))

    tiers = {"A": [], "B": [], "C": []}
    tier_of = {}
    for rec in stats:
        r = se_ratio[rec["name"]]
        if r == 0:
            tier = "A"
        elif r < median_se:
            tier = "B"
        else:
            tier = "C"
        tiers[tier].append(rec["name"])
        tier_of[rec["name"]] = tier

    rng = random.Random(SEED)
    allocation = {}
    for t in ("A", "B", "C"):
        allocation[t] = split_tier(tiers[t], rng)
    allocation = adjust_to_target(allocation)

    split_of = {}
    for t in allocation:
        for s in SPLITS:
            for name in allocation[t][s]:
                split_of[name] = s

    for s in SPLITS:
        n = sum(len(allocation[t][s]) for t in allocation)
        assert n == TARGET[s], "target mismatch %s: %d != %d" % (s, n, TARGET[s])
    print("[4/5] Final split sizes: %s"
          % {s: sum(len(allocation[t][s]) for t in allocation) for s in SPLITS})

    assignment = {}
    for rec in stats:
        assignment[rec["name"]] = {
            "original_subset": rec["subset"],
            "split": split_of[rec["name"]],
            "tier": tier_of[rec["name"]],
            "se_ratio": round(se_ratio[rec["name"]], 8),
            "total_pixels": rec["total"],
            "class_pixels_tif": rec["tif"],
        }
    for rec in stats:
        rec["tier"] = tier_of[rec["name"]]

    write_outputs(args.root, args.out, images, assignment, stats, anomalies)
    report = compute_split_stats(stats, split_of)
    write_stats(args.out, report)
    print("[5/5] Output written to %s" % args.out)

    # Console validation table: SE / EX / HE / MA image% & pixel% per split
    print("\n=== Validation (image% / pixel%) ===")
    header = "%-5s" % "split"
    for c in CLASSES:
        header += " | %s img%% / pix%%" % c
    print(header)
    for gname in SPLITS + ["all"]:
        g = report[gname]
        row = "%-5s" % gname
        for c in CLASSES:
            img_pct = 100.0 * g["class_img_count"][c] / g["n"]
            pix_pct = 100.0 * g["class_pixel"][c] / g["total_pixel"]
            row += " | %5.2f / %6.4f" % (img_pct, pix_pct)
        print(row)
    if anomalies:
        print("\n%d anomaly records (see split_assignment.json):" % len(anomalies))
        for a in anomalies[:30]:
            print("  %s" % a)


if __name__ == "__main__":
    main()
