"""SE-targeted augmentation transforms for the HRDecoder experiments.

板块二 (custom extension, NOT part of the original paper method):
  - SECopyPaste: paste real SE lesion patches (image + GT) sampled from a
    prebuilt lesion library onto the retinal FOV of other training images.
"""

import pickle

import cv2
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class SECopyPaste(object):
    """Copy-paste SE lesion patches from a prebuilt library onto the image.

    The transform must be placed right after LoadAnnotations and before any
    geometric transform (Resize), so patches stay at native resolution.
    """

    def __init__(self,
                 library_path,
                 prob=0.5,
                 max_patches=2,
                 fov_thresh=10,
                 fov_erode=3,
                 flip_prob=0.5,
                 max_patch_ratio=0.5):
        self.prob = float(prob)
        self.max_patches = int(max_patches)
        self.fov_thresh = int(fov_thresh)
        self.fov_erode = int(fov_erode)
        self.flip_prob = float(flip_prob)
        self.max_patch_ratio = float(max_patch_ratio)
        with open(library_path, 'rb') as f:
            self.library = pickle.load(f)
        assert len(self.library) > 0, 'empty SE lesion library'
        self._fov_cache = {}

    def _fov_integral(self, img, cache_key=None):
        if cache_key is not None and cache_key in self._fov_cache:
            small = self._fov_cache[cache_key]
            mask = cv2.resize(
                small, (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_NEAREST) > 0
        else:
            gray = img.mean(axis=2).astype(np.float32)
            mask = gray > self.fov_thresh
            if self.fov_erode > 0:
                k = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (self.fov_erode * 2 + 1, self.fov_erode * 2 + 1))
                mask = cv2.erode(mask.astype(np.uint8), k) > 0
            if cache_key is not None:
                small = cv2.resize(
                    mask.astype(np.uint8),
                    (max(img.shape[1] // 4, 1), max(img.shape[0] // 4, 1)),
                    interpolation=cv2.INTER_NEAREST)
                self._fov_cache[cache_key] = small
        h, w = mask.shape
        ii = np.zeros((h + 1, w + 1), dtype=np.int64)
        ii[1:, 1:] = mask.astype(np.int64).cumsum(0).cumsum(1)
        return ii

    def _sample_offset(self, ii, ph, pw, max_tries=200):
        h, w = ii.shape[0] - 1, ii.shape[1] - 1
        if ph > h or pw > w:
            return None
        y_hi, x_hi = h - ph + 1, w - pw + 1
        for _ in range(max_tries):
            y = np.random.randint(0, y_hi)
            x = np.random.randint(0, x_hi)
            area = (ii[y + ph, x + pw] - ii[y, x + pw]
                    - ii[y + ph, x] + ii[y, x])
            if area == ph * pw:
                return (int(y), int(x))
        return None

    def __call__(self, results):
        if np.random.random() > self.prob:
            return results
        img = results['img']
        gt = results['gt_semantic_seg']
        h, w = gt.shape[:2]
        cache_key = results.get('img_info', {}).get('filename')
        ii = self._fov_integral(img, cache_key=cache_key)

        n_patches = min(self.max_patches, len(self.library))
        idx = np.random.permutation(len(self.library))[:n_patches]
        pasted = 0
        for i in idx:
            pimg = self.library[i]['img']
            pmask = self.library[i]['mask']
            ph, pw = pimg.shape[:2]
            if ph > h * self.max_patch_ratio or pw > w * self.max_patch_ratio:
                continue
            if ph > h or pw > w:
                continue
            if np.random.random() < self.flip_prob:
                pimg = pimg[:, ::-1]
                pmask = pmask[:, ::-1]
            offset = self._sample_offset(ii, ph, pw)
            if offset is None:
                continue
            y, x = offset
            img[y:y + ph, x:x + pw] = pimg
            gt[y:y + ph, x:x + pw] = np.maximum(
                gt[y:y + ph, x:x + pw], pmask)
            pasted += 1
            if pasted >= self.max_patches:
                break

        results['img'] = img
        results['gt_semantic_seg'] = gt
        return results

    def __repr__(self):
        return ('SECopyPaste(prob={}, max_patches={}, library_size={})').format(
            self.prob, self.max_patches, len(self.library))
