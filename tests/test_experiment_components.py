from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import crop_black_border
from model.loss import CompositeLoss, build_loss
from models.m2mrf.reassembly import M2MRFCascadeUpsampler


class BlackBorderCropTest(unittest.TestCase):
    def test_image_and_mask_use_identical_crop(self) -> None:
        image = np.zeros((100, 140, 3), dtype=np.uint8)
        image[20:80, 30:120] = (40, 80, 120)
        mask = np.zeros((100, 140), dtype=np.uint8)
        mask[40:50, 60:70] = 4

        image_crop, mask_crop, bbox = crop_black_border(
            image,
            mask,
            threshold=10,
            padding_ratio=0.0,
        )

        self.assertEqual(bbox, (30, 20, 120, 80))
        self.assertEqual(image_crop.shape, (60, 90, 3))
        self.assertIsNotNone(mask_crop)
        assert mask_crop is not None
        self.assertTrue(np.all(mask_crop[20:30, 30:40] == 4))


class CompositeLossTest(unittest.TestCase):
    def test_weighted_loss_is_finite_and_differentiable(self) -> None:
        criterion = build_loss(
            {
                "name": "composite",
                "components": [
                    {"name": "focal_tversky", "weight": 0.5},
                    {"name": "dice", "weight": 0.3},
                    {"name": "cross_entropy", "weight": 0.2},
                ],
            },
            num_classes=5,
        )
        self.assertIsInstance(criterion, CompositeLoss)

        logits = torch.randn(2, 5, 16, 16, requires_grad=True)
        targets = torch.randint(0, 5, (2, 16, 16))
        loss = criterion(logits, targets)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())


class M2MRFReassemblyTest(unittest.TestCase):
    def test_three_step_cascade_upsamples_by_eight(self) -> None:
        upsampler = M2MRFCascadeUpsampler(
            channels=8,
            steps=3,
            patch_size=4,
            encode_channels_rate=4,
            fc_channels_rate=8,
        )
        output = upsampler(torch.randn(1, 8, 8, 8))
        self.assertEqual(tuple(output.shape), (1, 8, 64, 64))


if __name__ == "__main__":
    unittest.main()
