from __future__ import annotations

import torch
import segmentation_models_pytorch as smp

ENCODER_NAME = "resnet34"
ENCODER_WEIGHTS = "imagenet"

# ImageNet normalization constants for the pretrained ResNet34 encoder.
# Must be applied to every image before passing through the model.
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def build_smp_unet(num_classes: int = 5) -> smp.Unet:
    """SMP UNet with ResNet34 encoder pretrained on ImageNet, raw logit output."""
    return smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=num_classes,
        activation=None,
    )


def normalize_imagenet(images: torch.Tensor) -> torch.Tensor:
    """Normalize a [B, 3, H, W] float tensor in [0, 1] to ImageNet statistics."""
    mean = _MEAN.to(images.device)
    std = _STD.to(images.device)
    return (images - mean) / std
