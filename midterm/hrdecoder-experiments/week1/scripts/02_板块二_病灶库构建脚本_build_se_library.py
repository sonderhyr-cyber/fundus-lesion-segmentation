"""Build the SE lesion library for the SECopyPaste augmentation.

Scans the fixed 6:1:3 train split (DDR_split_613), extracts every SE
connected component (from the SE .tif label) as an image+mask patch and
stores it in a pickled library used by mmseg's SECopyPaste transform.

Usage:
    python build_se_library.py --root /root/HRDecoder/data/DDR_split_613 \
        --out /root/HRDecoder/data/DDR_split_613/se_lesion_library.pkl \
        --min-area 50
"""

import argparse
import os
import pickle

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/root/HRDecoder/data/DDR_split_613')
    parser.add_argument(
        '--out',
        default='/root/HRDecoder/data/DDR_split_613/se_lesion_library.pkl')
    parser.add_argument('--min-area', type=int, default=50,
                        help='minimum SE component area (pixels)')
    parser.add_argument('--max-patches-per-image', type=int, default=20)
    parser.add_argument('--pad', type=int, default=4,
                        help='padding around each component bbox')
    args = parser.parse_args()

    img_dir = os.path.join(args.root, 'images', 'train')
    se_dir = os.path.join(args.root, 'labels', 'train', 'SE')
    names = sorted(os.listdir(img_dir))
    library = []
    stats = {'images_with_se': 0, 'components': 0, 'total_se_px': 0}
    sizes = []

    for name in names:
        if not name.endswith('.jpg'):
            continue
        stem = os.path.splitext(name)[0]
        img = cv2.imread(os.path.join(img_dir, name))
        se = cv2.imread(os.path.join(se_dir, stem + '.tif'),
                        cv2.IMREAD_GRAYSCALE)
        if img is None or se is None:
            print('skip %s (missing file)' % name)
            continue
        se_mask = (se == 255).astype(np.uint8)
        if se_mask.sum() == 0:
            continue
        stats['images_with_se'] += 1
        stats['total_se_px'] += int(se_mask.sum())
        n, labels, ccstats, _ = cv2.connectedComponentsWithStats(se_mask, 8)
        per_img = 0
        for i in range(1, n):
            area = ccstats[i, cv2.CC_STAT_AREA]
            if area < args.min_area:
                continue
            x, y, w, h = (ccstats[i, cv2.CC_STAT_LEFT],
                          ccstats[i, cv2.CC_STAT_TOP],
                          ccstats[i, cv2.CC_STAT_WIDTH],
                          ccstats[i, cv2.CC_STAT_HEIGHT])
            y0, y1 = max(y - args.pad, 0), min(y + h + args.pad, se.shape[0])
            x0, x1 = max(x - args.pad, 0), min(x + w + args.pad, se.shape[1])
            crop_img = img[y0:y1, x0:x1].copy()
            crop_mask = (se_mask[y0:y1, x0:x1] * 3).astype(np.uint8)
            library.append({'img': crop_img, 'mask': crop_mask, 'src': name})
            sizes.append((crop_img.shape[0], crop_img.shape[1]))
            stats['components'] += 1
            per_img += 1
            if per_img >= args.max_patches_per_image:
                break

    with open(args.out, 'wb') as f:
        pickle.dump(library, f, protocol=pickle.HIGHEST_PROTOCOL)

    sizes = np.array(sizes)
    print('library written to %s' % args.out)
    print('images with SE: %d' % stats['images_with_se'])
    print('components: %d' % stats['components'])
    print('total SE px: %d' % stats['total_se_px'])
    if len(sizes):
        print('patch size h: min=%d med=%d max=%d' % (
            sizes[:, 0].min(), int(np.median(sizes[:, 0])), sizes[:, 0].max()))
        print('patch size w: min=%d med=%d max=%d' % (
            sizes[:, 1].min(), int(np.median(sizes[:, 1])), sizes[:, 1].max()))
    print('library size: %.1f MB' % (os.path.getsize(args.out) / 1e6))


if __name__ == '__main__':
    main()
